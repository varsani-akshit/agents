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

  /* ── deep links: /charts#rolling_correlations lands on the right tab ──
     The browser scrolls to the hash before this runs, while the target's pane
     is still hidden; showing the pane then reflows everything and leaves the
     figure hundreds of pixels above the viewport. So: switch the tab, wait for
     layout, scroll deliberately, and draw the figure — the lazy observer never
     fires for a node that is off-screen, which left deep links on a blank
     frame. */
  function openHash() {
    if (!location.hash) return;
    var target = document.getElementById(location.hash.slice(1));
    if (!target) return;
    var pane = target.closest(".tabpane");
    if (pane) showTab(pane.dataset.pane);
    var settle = function () {
      if (window.AlfredCharts && window.AlfredCharts.ensure) {
        // Draw the neighbours too: each figure sizes itself to its content when
        // drawn, so an undrawn one above the target changes height later and
        // slides the target back out of view.
        var pane = target.closest(".tabpane") || document;
        pane.querySelectorAll("[data-chart]").forEach(window.AlfredCharts.ensure);
      }
      target.scrollIntoView({ block: "start" });
    };
    requestAnimationFrame(function () { requestAnimationFrame(settle); });
    // One correction after ECharts has laid out, for any residual shift.
    setTimeout(settle, 260);
  }
  openHash();
  window.addEventListener("hashchange", openHash);

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
