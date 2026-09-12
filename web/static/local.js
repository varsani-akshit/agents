/* Timestamps in the reader's own timezone.
 *
 * Alfred stores and reasons in UTC — a brief covering "the last 12 hours"
 * needs a fixed boundary, not one that moves with whoever opens it. But the
 * reader should not do the arithmetic, so every timestamp ships as a <time>
 * element carrying the exact instant, and this rewrites the visible text to
 * their local clock. With JavaScript off the UTC text stands, correctly
 * labelled, which is why the label is inside the element and not beside it. */
(function () {
  "use strict";
  var OPTS = {
    datetime: { day: "2-digit", month: "short", year: "numeric",
                hour: "2-digit", minute: "2-digit", hour12: false },
    daytime:  { day: "2-digit", month: "short", hour: "2-digit",
                minute: "2-digit", hour12: false },
    date:     { day: "2-digit", month: "long", year: "numeric" },
    dayshort: { day: "2-digit", month: "short" },
    time:     { hour: "2-digit", minute: "2-digit", hour12: false },
  };

  function localise(root) {
    var zone;
    try { zone = Intl.DateTimeFormat().resolvedOptions().timeZone; } catch (e) {}
    (root || document).querySelectorAll("time[datetime]").forEach(function (el) {
      if (el.dataset.localised) return;
      var d = new Date(el.getAttribute("datetime"));
      if (isNaN(d)) return;
      var fmt = OPTS[el.dataset.fmt] || OPTS.datetime;
      try {
        el.textContent = new Intl.DateTimeFormat(undefined, fmt).format(d);
        el.dataset.localised = "1";
        if (zone) el.title = zone + " · " + d.toISOString().replace("T", " ").slice(0, 16) + " UTC";
      } catch (e) {}
    });
    // "UTC" printed next to a converted time is now a lie; the element's own
    // tooltip carries the zone instead.
    (root || document).querySelectorAll(".facts span, .eyebrow, .when").forEach(function (n) {
      if (n.querySelector("time[data-localised]")) {
        n.innerHTML = n.innerHTML.replace(/\s*UTC\b/g, "");
      }
    });
  }

  localise();
  window.AlfredLocal = { localise: localise };
  document.addEventListener("alfred:content", function (e) { localise(e.detail || document); });
})();
