// Theme toggle: system -> light -> dark. Remembered per browser when storage is
// available; the docs render correctly without it.
// Mermaid diagrams (pages that load the library) follow the theme. Without the library,
// for example offline, each diagram's source stays on the page as readable text.
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
    renderDiagrams();
    if (window.matchMedia) {
      var scheme = window.matchMedia("(prefers-color-scheme: dark)");
      var onChange = function () {
        if (theme === "system") renderDiagrams();
      };
      if (scheme.addEventListener) scheme.addEventListener("change", onChange);
    }
    var button = document.querySelector(".theme-toggle");
    if (!button) return;
    button.textContent = "Theme: " + theme;
    button.addEventListener("click", function () {
      theme = order[(order.indexOf(theme) + 1) % order.length];
      apply();
      renderDiagrams();
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

  function isDark() {
    if (theme !== "system") return theme === "dark";
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }

  function renderDiagrams() {
    var nodes = document.querySelectorAll("pre.mermaid");
    if (!nodes.length || !window.mermaid) return;
    // Keep each diagram's source, and stop Mermaid drawing on its own at page load: both
    // must happen now, before anything has been drawn.
    for (var i = 0; i < nodes.length; i++) {
      if (!nodes[i].getAttribute("data-source")) {
        nodes[i].setAttribute("data-source", nodes[i].textContent);
      }
    }
    window.mermaid.initialize({ startOnLoad: false });
    // Mermaid sizes each box from its text, so draw once the web font is in: measured in a
    // fallback font, labels come out clipped when IBM Plex arrives.
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () {
        draw(nodes);
      });
    } else {
      draw(nodes);
    }
  }

  function draw(nodes) {
    for (var i = 0; i < nodes.length; i++) {
      nodes[i].removeAttribute("data-processed");
      nodes[i].textContent = nodes[i].getAttribute("data-source");
    }
    window.mermaid.initialize({
      startOnLoad: false,
      theme: isDark() ? "dark" : "neutral",
      securityLevel: "strict",
      fontFamily: '"IBM Plex Sans", system-ui, sans-serif',
      flowchart: { curve: "basis", htmlLabels: true },
    });
    window.mermaid.run({ nodes: nodes });
  }
})();
