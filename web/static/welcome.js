/* The brief's arrival card.
 * A brief is a delivery, not a page you browsed to — so every visit opens with
 * a moment to land before the reading starts. Shown on each visit (the reader
 * asked for it); a click dismisses it early for anyone in a hurry. */
(function () {
  "use strict";
  var el = document.getElementById("welcome");
  if (!el) return;

  var h = new Date().getHours();
  var greeting = h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
  var g = el.querySelector(".wg");
  if (g) g.textContent = greeting + ".";

  var drop = function () {
    if (el.classList.contains("done")) return;
    el.classList.add("done");
    setTimeout(function () { el.remove(); }, 520);
  };
  el.addEventListener("click", drop);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" || e.key === "Enter") drop();
  }, { once: true });
  // Matches the CSS fade, which begins at 3.1s.
  setTimeout(drop, 3350);
})();
