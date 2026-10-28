/* Navigation: on narrow screens the rail is a drawer.
 * Opens from the menu button; closes on the scrim, the close button, Escape,
 * and when a link inside it is followed. Rows marked data-href open their
 * page when clicked anywhere, like the property manager's tables. */
(function () {
  "use strict";
  var rail = document.getElementById("rail");
  var scrim = document.getElementById("scrim");
  var burger = document.getElementById("burger");
  var closeBtn = document.getElementById("railclose");

  function set(open) {
    if (!rail) return;
    rail.classList.toggle("open", open);
    if (scrim) scrim.classList.toggle("open", open);
    if (burger) burger.setAttribute("aria-expanded", open ? "true" : "false");
  }

  if (burger) burger.addEventListener("click", function () { set(true); });
  if (closeBtn) closeBtn.addEventListener("click", function () { set(false); });
  if (scrim) scrim.addEventListener("click", function () { set(false); });
  if (rail) rail.addEventListener("click", function (e) {
    if (e.target.closest("a")) set(false);
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") set(false);
  });

  document.addEventListener("click", function (e) {
    var row = e.target.closest("[data-href]");
    if (!row || e.target.closest("a, button, input, summary")) return;
    if (e.metaKey || e.ctrlKey) window.open(row.dataset.href, "_blank");
    else window.location.href = row.dataset.href;
  });
})();
