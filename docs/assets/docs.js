// Theme toggle: system -> light -> dark. Remembered per browser when storage is
// available; the docs render correctly without it.
(function () {
  var root = document.documentElement;
  var order = ["system", "light", "dark"];
  var theme = "system";
  try {
    theme = localStorage.getItem("docs-theme") || "system";
  } catch (error) {
    theme = "system";
  }
  apply();

  document.addEventListener("DOMContentLoaded", function () {
    var button = document.querySelector(".theme-toggle");
    if (!button) return;
    button.textContent = "Theme: " + theme;
    button.addEventListener("click", function () {
      theme = order[(order.indexOf(theme) + 1) % order.length];
      apply();
      button.textContent = "Theme: " + theme;
      try {
        localStorage.setItem("docs-theme", theme);
      } catch (error) {
        /* storage blocked: the choice lasts for this page view only */
      }
    });
  });

  function apply() {
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
  }
})();
