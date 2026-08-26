/* Charts page behaviour: tabs, search, the price explorer, period controls.
 * Rendering itself lives in charts.js — this file only decides what to draw
 * and fetches fresh specs from the live API when a control changes. */
(function () {
  "use strict";

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  };

  /* ── tabs ── */
  function showTab(id) {
    $$(".tab").forEach(function (t) { t.classList.toggle("on", t.dataset.tab === id); });
    $$(".tabpane").forEach(function (p) {
      p.classList.toggle("on", p.dataset.pane === id);
    });
    // Lazy renderer in charts.js watches viewport intersection; nudge it after
    // a pane becomes visible.
    window.dispatchEvent(new Event("scroll"));
  }
  $$(".tab").forEach(function (t) {
    t.addEventListener("click", function () {
      $("#chartsearch").value = "";
      applySearch("");
      showTab(t.dataset.tab);
    });
  });

  /* ── deep links: /charts#rolling_correlations lands on the right tab ── */
  if (location.hash) {
    var target = document.getElementById(location.hash.slice(1));
    if (target) {
      var pane = target.closest(".tabpane");
      if (pane) {
        showTab(pane.dataset.pane);
        setTimeout(function () { target.scrollIntoView({ block: "start" }); }, 60);
      }
    }
  }

  /* ── search: matches across every tab, flattening the tab structure ── */
  function applySearch(q) {
    q = q.trim().toLowerCase();
    var any = false;
    if (!q) {
      $$(".tabpane").forEach(function (p) {
        $$(".chart", p).forEach(function (c) { c.style.display = ""; });
      });
      var on = $(".tab.on");
      showTab(on ? on.dataset.tab : "price");
      $("#nosearch").style.display = "none";
      return;
    }
    $$(".tabpane").forEach(function (p) {
      var hit = false;
      $$(".chart", p).forEach(function (c) {
        var match = (c.dataset.title || "").indexOf(q) !== -1;
        c.style.display = match ? "" : "none";
        hit = hit || match;
      });
      p.classList.toggle("on", hit && p.dataset.pane !== "price");
      any = any || (hit && p.dataset.pane !== "price");
    });
    $$(".tab").forEach(function (t) { t.classList.remove("on"); });
    $("#nosearch").style.display = any ? "none" : "";
    window.dispatchEvent(new Event("scroll"));
  }
  $("#chartsearch").addEventListener("input", function (e) { applySearch(e.target.value); });

  /* ── price explorer ── */
  var state = { symbol: "GOLD", days: "1y", ccy: "USD" };

  function fetchPrice() {
    var node = $("#pricechart");
    node.querySelector(".t").textContent = "Loading…";
    fetch("/api/chart/price?symbol=" + state.symbol + "&days=" + state.days + "&ccy=" + state.ccy)
      .then(function (r) { return r.json(); })
      .then(function (spec) {
        if (spec.error) { node.querySelector(".t").textContent = spec.error; return; }
        node.querySelector(".t").textContent = spec.title;
        node.querySelector(".s").textContent = spec.subtitle || "";
        window.AlfredCharts.draw(node, spec);
      })
      .catch(function () { node.querySelector(".t").textContent = "Fetch failed."; });
  }

  $("#pricectl").addEventListener("click", function (e) {
    var b = e.target.closest(".opt");
    if (!b) return;
    state[b.dataset.k] = b.dataset.v;
    $$('.opt[data-k="' + b.dataset.k + '"]', $("#pricectl")).forEach(function (o) {
      o.classList.toggle("on", o === b);
    });
    fetchPrice();
  });
  fetchPrice();

  /* ── period controls on rebuildable standing figures ── */
  $$(".chartperiods").forEach(function (row) {
    row.addEventListener("click", function (e) {
      var b = e.target.closest(".opt");
      if (!b) return;
      $$(".opt", row).forEach(function (o) { o.classList.toggle("on", o === b); });
      var key = row.dataset.key;
      var node = document.getElementById(key);
      fetch("/api/chart/" + key + "?days=" + b.dataset.days)
        .then(function (r) { return r.json(); })
        .then(function (spec) {
          if (!spec.error) window.AlfredCharts.draw(node, spec);
        });
    });
  });
})();
