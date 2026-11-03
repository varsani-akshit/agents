/* Open a candidate row to see its chart.
 *
 * Fetched on the click, not with the page. A simulation run holds over a
 * thousand candidates; building every chart up front would cost a thousand
 * requests to show the two or three anybody actually opens, and would undo
 * the point of paging the table in the first place.
 *
 * Two figures per row: the price as candles with the setup's own levels drawn
 * across it, and the strategy's record — because "buy at 3,156.76" and "this
 * strategy makes +0.08R a trade" are the two halves of the same decision and
 * reading them in separate places is how you talk yourself into a bad one.
 */
(function () {
  "use strict";

  function fill(fig, spec) {
    fig.querySelector(".chart-head .t").textContent = spec.title || "";
    fig.querySelector(".chart-head .s").textContent = spec.subtitle || "";
    // charts.js owns drawing, theming and the tooltip convention.
    window.AlfredCharts.draw(fig, spec);
  }

  function fail(fig, message) {
    fig.querySelector(".chart-head .t").textContent = message;
    fig.querySelector(".chart-head .s").textContent = "";
    fig.querySelector(".chart-canvas").style.height = "0px";
  }

  function load(fig, url, emptyMessage) {
    return fetch(url, { headers: { Accept: "application/json" } })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(r.status); })
      .then(function (spec) { fill(fig, spec); })
      .catch(function () { fail(fig, emptyMessage); });
  }

  document.addEventListener("click", function (e) {
    var btn = e.target.closest && e.target.closest(".setup-open");
    if (!btn) return;
    var row = btn.closest("tr");
    var panel = row && row.nextElementSibling;
    if (!panel || !panel.classList.contains("setup-row")) return;

    var open = btn.getAttribute("aria-expanded") === "true";
    btn.setAttribute("aria-expanded", open ? "false" : "true");
    btn.textContent = open ? "chart ▾" : "chart ▴";
    panel.hidden = open;
    if (open || panel.dataset.loaded) return;
    panel.dataset.loaded = "1";

    var figs = panel.querySelectorAll("figure.chart");
    var d = btn.dataset;
    var e = encodeURIComponent;
    var run = d.run ? "?run_id=" + e(d.run) : "";
    load(figs[0],
         "/api/chart/setup/" + e(d.symbol) + "/" + e(d.strategy) + run,
         "No bars stored for this name yet.");
    load(figs[1],
         "/api/chart/record/" + e(d.strategy) + "/" + e(d.region) + run,
         "No stored record for this strategy here.");
  });
})();
