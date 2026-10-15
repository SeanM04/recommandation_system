import pytest
from pydantic import ValidationError

from recommender.algorithms.epsilon_greedy import EpsilonGreedyPolicy
from recommender.algorithms.factory import build_policies, build_policy, policy_rng
from recommender.algorithms.random_policy import RandomPolicy
from recommender.algorithms.rewards import binary_reward, make_reward, margin_reward
from recommender.algorithms.static import StaticPolicy
from recommender.algorithms.thompson import ThompsonSamplingPolicy
from recommender.config import PolicySpec, Settings, load_offers, load_settings
from recommender.domain import IGNORED, TAKEN_UP, VIEWED
from recommender.store.options_store import InMemoryOptionsStore

OFFERS = {offer.offer_id: offer for offer in load_offers()}


# --- reward functions -----------------------------------------------------------------


def test_binary_reward() -> None:
    offer = OFFERS["data_1gb"]
    assert binary_reward(offer, TAKEN_UP) == 1.0
    assert binary_reward(offer, VIEWED) == 0.0
    assert binary_reward(offer, IGNORED) is None  # no update: probably never seen


def test_margin_reward_scales_by_the_top_margin() -> None:
    reward = margin_reward(list(OFFERS.values()))
    top = max(offer.margin_zar for offer in OFFERS.values())
    assert reward(OFFERS["data_20gb"], TAKEN_UP) == pytest.approx(150 / top)
    assert reward(OFFERS["airtime_r50_bonus"], TAKEN_UP) == pytest.approx(4 / top)
    assert reward(OFFERS["data_20gb"], VIEWED) == 0.0
    assert reward(OFFERS["data_20gb"], IGNORED) is None


def test_learning_from_ignored_offers_is_configurable() -> None:
    offers = list(OFFERS.values())
    assert make_reward("binary", offers)(OFFERS["data_1gb"], IGNORED) is None
    assert make_reward("binary", offers, learn_from_ignored=True)(OFFERS["data_1gb"], IGNORED) == 0
    assert make_reward("margin", offers)(OFFERS["data_20gb"], TAKEN_UP) == pytest.approx(1.0)


def test_reward_rejects_unknown_names_and_outcomes() -> None:
    with pytest.raises(ValueError, match="unknown reward"):
        make_reward("profit", list(OFFERS.values()))
    with pytest.raises(ValueError, match="unknown outcome"):
        make_reward("binary", list(OFFERS.values()))(OFFERS["data_1gb"], "clicked")


# --- policy factory -------------------------------------------------------------------


def test_every_configured_policy_is_built() -> None:
    settings = load_settings()
    policies = build_policies(settings, InMemoryOptionsStore())
    assert set(policies) == set(settings.policies)
    assert settings.recommendation.active_policy in policies
    assert isinstance(policies["random"], RandomPolicy)
    assert isinstance(policies["static"], StaticPolicy)
    assert isinstance(policies["epsilon_greedy_segment"], EpsilonGreedyPolicy)
    assert policies["epsilon_greedy_segment"].context == "segment"
    assert isinstance(policies["thompson_segment_decay"], ThompsonSamplingPolicy)
    assert policies["thompson_segment_decay"].decay < 1


def test_static_policy_gets_its_batch_trained_table() -> None:
    policy = build_policy(
        "static", PolicySpec(type="static"), InMemoryOptionsStore(), seed=1, popularity={"a": 0.3}
    )
    assert policy.popularity == {"a": 0.3}


def test_each_policy_has_its_own_reproducible_random_stream() -> None:
    assert policy_rng(42, "thompson").random() == policy_rng(42, "thompson").random()
    assert policy_rng(42, "thompson").random() != policy_rng(42, "random").random()


def test_settings_reject_bad_policy_config() -> None:
    raw = load_settings().model_dump()
    with pytest.raises(ValidationError, match="active_policy"):
        Settings.model_validate(raw | {"recommendation": {"active_policy": "nope"}})
    raw["policies"]["bad"] = {"type": "thompson", "context": "province"}
    with pytest.raises(ValidationError):
        Settings.model_validate(raw)
