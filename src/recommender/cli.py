"""``python manage.py <command>``: run and administer the project, Django style.

    migrate           create the tables in the configured database and load the customers
    createsuperuser   create an admin account (asks for the password)
    createuser NAME   create an account (viewer by default; --role admin)
    changepassword NAME
    users             list the accounts
    createapikey      create an API key (shown once)
    runserver         start the API and the dashboard together; Ctrl+C stops both
    simulate          customers, a 3-seed comparison, 1,000 app opens (what the dashboard shows)
    compare           the 10-seed comparison behind the published results
    nightly           roll the event log up into daily_offer_stats (--all for every day)
    test              run the test suite

The database comes from .env (DB_* or RECOMMENDER_DATABASE_URL), as everywhere else.
"""

from __future__ import annotations

import argparse
import getpass
import os
import socket
import subprocess
import sys
import time

import requests

from recommender.accounts import AccountError, Accounts
from recommender.config import PROJECT_ROOT, load_settings

SCRIPTS = PROJECT_ROOT / "scripts"


def _engine():
    """An engine for the database configured in .env."""
    from recommender.db import make_engine

    return make_engine(load_settings().database_url)


def _where(engine) -> str:
    """'postgresql database X on host' or the SQLite file, for messages."""
    url = engine.url
    return f"{url.get_backend_name()} database {url.database}" + (
        f" on {url.host}" if url.host else ""
    )


def _ask_password(noinput: bool) -> str:
    """Ask for a password twice, or read it from the environment with --noinput."""
    if noinput:
        password = os.environ.get("RECOMMENDER_SUPERUSER_PASSWORD")
        if not password:
            raise SystemExit("--noinput needs RECOMMENDER_SUPERUSER_PASSWORD in the environment")
        return password
    password = getpass.getpass("Password: ")
    if password != getpass.getpass("Password (again): "):
        raise SystemExit("The passwords differ.")
    return password


# --- database and accounts ------------------------------------------------------------------


def migrate(args: argparse.Namespace) -> None:
    """Create the tables and load the customers if the table is empty."""
    from recommender.customers import CustomerRepository, read_customers
    from recommender.db import create_tables

    settings, engine = load_settings(), _engine()
    tables = create_tables(engine)
    print(f"Tables ready in the {_where(engine)}: {', '.join(tables)}")
    customers = CustomerRepository(engine)
    if customers.count() == 0:
        if not settings.paths.customers.exists():
            subprocess.run([sys.executable, str(SCRIPTS / "generate_customers.py")], check=True)
        else:
            customers.load(read_customers(settings.paths.customers))
    print(f"Customers: {customers.count():,}")
    users = Accounts(engine).user_count()
    if users == 0:
        print("No accounts yet: python manage.py createsuperuser")


def _create(username: str, role: str, noinput: bool) -> None:
    """Create an account with a password asked for (or from the environment)."""
    engine = _engine()
    try:
        user = Accounts(engine).create_user(username, _ask_password(noinput), role)
    except AccountError as error:
        raise SystemExit(str(error)) from None
    print(f"Created {user.role} {user.username!r} in the {_where(engine)}.")


def createsuperuser(args: argparse.Namespace) -> None:
    """Create an admin account; the username defaults to 'admin'."""
    username = args.username or (
        "admin" if args.noinput else input("Username [admin]: ").strip() or "admin"
    )
    _create(username, "admin", args.noinput)


def createuser(args: argparse.Namespace) -> None:
    """Create an account with the given role (viewer by default)."""
    _create(args.username, args.role, noinput=False)


def changepassword(args: argparse.Namespace) -> None:
    """Set a new password for an existing account."""
    try:
        Accounts(_engine()).set_password(args.username, _ask_password(noinput=False))
    except AccountError as error:
        raise SystemExit(str(error)) from None
    print(f"Password changed for {args.username!r}.")


def users(args: argparse.Namespace) -> None:
    """Print every account with its role and last sign-in."""
    rows = Accounts(_engine()).list_users()
    if not rows:
        print("No accounts yet: python manage.py createsuperuser")
    for row in rows:
        last = row["last_login_at"] or "never"
        print(f"{row['username']:<24} {row['role']:<8} last sign-in: {last}")


def createapikey(args: argparse.Namespace) -> None:
    """Create an API key and print it once, with how to require keys."""
    key = Accounts(_engine()).create_api_key(args.label)
    print(f"API key ({args.label}). Copy it now; only its hash is stored:\n\n  {key}\n")
    print("Callers send it in the X-API-Key header. To require keys, add to .env:")
    print("  RECOMMENDER_REQUIRE_API_KEY=true")
    print("  RECOMMENDER_API_KEY=<a key>    (the one the dashboard and simulator send)")


# --- servers ----------------------------------------------------------------------------------


def port_in_use(port: int) -> bool:
    """Whether something already listens on ``port`` on this machine."""
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_healthy(url: str, process: subprocess.Popen, timeout_s: float = 60) -> bool:
    """Poll ``url`` until it answers, the process dies, or ``timeout_s`` passes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and process.poll() is None:
        try:
            if requests.get(url, timeout=1).ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def _stop(processes: list[subprocess.Popen]) -> None:
    """Ask each process to stop, then kill any still running after 10 seconds."""
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def runserver(args: argparse.Namespace) -> None:
    """Start the API, wait until it answers, then the dashboard; Ctrl+C stops both."""
    if hasattr(sys.stdout, "reconfigure"):  # the URLs appear at once, even in a log file
        sys.stdout.reconfigure(line_buffering=True)
    settings = load_settings()
    for port, name, variable in [
        (args.api_port, "API", "API_PORT"),
        (args.dashboard_port, "dashboard", "DASHBOARD_PORT"),
    ]:
        if name == "dashboard" and args.no_dashboard:
            continue
        if port_in_use(port):
            raise SystemExit(
                f"Port {port} is already in use (another app?). Pick another for the {name}: "
                f"--{name.lower()}-port, or {variable}=... in .env"
            )
    if settings.require_api_key and not os.environ.get("RECOMMENDER_API_KEY"):
        print(
            "Note: the API requires keys but RECOMMENDER_API_KEY is not set, so the "
            "dashboard's Try it page will be refused (python manage.py createapikey)."
        )

    api_url = f"http://localhost:{args.api_port}"
    env = os.environ | {"RECOMMENDER_API_URL": api_url}
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "recommender.api.main:app",
                "--host",
                args.host,
                "--port",
                str(args.api_port),
                "--no-access-log",
            ],
            cwd=PROJECT_ROOT,
            env=env,
        )
    ]
    try:
        if not _wait_healthy(f"{api_url}/health", processes[0]):
            raise SystemExit("The API did not start; see the messages above.")
        if not args.no_dashboard:
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "streamlit",
                        "run",
                        "dashboard/app.py",
                        "--server.port",
                        str(args.dashboard_port),
                        "--server.address",
                        args.host,
                        "--server.headless",
                        "true",
                    ],
                    cwd=PROJECT_ROOT,
                    env=env,
                )
            )
        engine = _engine()
        print(f"\nDatabase:  {_where(engine)}")
        print(f"API:       {api_url}   (interactive docs: {api_url}/docs)")
        if not args.no_dashboard:
            print(f"Dashboard: http://localhost:{args.dashboard_port}")
            if Accounts(engine).user_count() == 0:
                print("           no accounts yet: the dashboard will ask for the admin account")
        print("Quit with Ctrl+C.\n")
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        print("A server stopped; stopping the other.")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        _stop(processes)


# --- the pipeline scripts -------------------------------------------------------------------


def _script(*command: str):
    """A command that runs these scripts in turn, passing extra arguments to the last."""

    def run(args: argparse.Namespace) -> None:
        """Run each script; stop with its exit code if one fails."""
        for step in command:
            arguments = step.split()
            if step == command[-1]:
                arguments += args.extra
            result = subprocess.run([sys.executable, *arguments], cwd=PROJECT_ROOT)
            if result.returncode:
                raise SystemExit(result.returncode)

    return run


simulate = _script(
    "scripts/generate_customers.py",
    "scripts/compare_algorithms.py --seeds 3 --no-docs",
    "scripts/run_simulation.py --mode api --in-process --rounds 1000",
)
compare = _script("scripts/compare_algorithms.py")
nightly = _script("scripts/nightly_aggregate.py")
test = _script("-m pytest")


def parser() -> argparse.ArgumentParser:
    """The manage.py command-line parser with every command and its arguments."""
    root = argparse.ArgumentParser(
        prog="manage.py",
        description="Run and administer the Telco Dynamic Recommender.",
    )
    commands = root.add_subparsers(dest="command", required=True, metavar="command")

    def add(name: str, handler, help: str) -> argparse.ArgumentParser:
        """Register one command with its handler and help text."""
        command = commands.add_parser(name, help=help, description=help)
        command.set_defaults(handler=handler)
        return command

    add("migrate", migrate, "create the tables and load the customers")
    command = add("createsuperuser", createsuperuser, "create an admin account")
    command.add_argument("--username")
    command.add_argument(
        "--noinput", action="store_true", help="password from RECOMMENDER_SUPERUSER_PASSWORD"
    )
    command = add("createuser", createuser, "create an account")
    command.add_argument("username")
    command.add_argument("--role", choices=["viewer", "admin"], default="viewer")
    command = add("changepassword", changepassword, "set a new password")
    command.add_argument("username")
    add("users", users, "list the accounts")
    command = add("createapikey", createapikey, "create an API key (shown once)")
    command.add_argument("--label", default="unnamed")
    command = add("runserver", runserver, "start the API and the dashboard; Ctrl+C stops both")
    command.add_argument("--api-port", type=int, default=int(os.environ.get("API_PORT", 8000)))
    command.add_argument(
        "--dashboard-port", type=int, default=int(os.environ.get("DASHBOARD_PORT", 8501))
    )
    command.add_argument(
        "--host", default="127.0.0.1", help="0.0.0.0 to accept other machines on the network"
    )
    command.add_argument("--no-dashboard", action="store_true", help="the API only")
    for name, handler, help in [
        ("simulate", simulate, "customers, a 3-seed comparison and 1,000 app opens"),
        ("compare", compare, "the 10-seed comparison (about 9 minutes)"),
        ("nightly", nightly, "roll the event log up into daily_offer_stats"),
        ("test", test, "run the test suite"),
    ]:
        command = add(name, handler, help)
        command.add_argument("extra", nargs=argparse.REMAINDER, help="passed on to the script")
    return root


def main(argv: list[str] | None = None) -> None:
    """Parse the command line and run the chosen command."""
    args = parser().parse_args(argv)
    args.handler(args)
