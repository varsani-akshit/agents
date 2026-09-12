/* Wide tables get an edge fade only while there is more to scroll to.
 * A permanent gradient reads as decoration and hides the last column; one
 * that disappears when you reach the end is a scroll affordance. */
(function () {
  "use strict";
  function mark(wrap) {
    var more = wrap.scrollWidth - wrap.clientWidth - wrap.scrollLeft > 4;
    wrap.classList.toggle("scrolls", more);
  }
  function wire(root) {
    (root || document).querySelectorAll(".tablewrap").forEach(function (w) {
      if (w.dataset.wired) return;
      w.dataset.wired = "1";
      mark(w);
      w.addEventListener("scroll", function () { mark(w); }, { passive: true });
    });
  }
  wire();
  window.addEventListener("resize", function () {
    document.querySelectorAll(".tablewrap").forEach(mark);
  });
  document.addEventListener("alfred:content", function (e) { wire(e.detail || document); });
})();
