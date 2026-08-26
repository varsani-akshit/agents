/* Navigation: mobile panel and the account menu.
 * Both close on outside click and on Escape — a <details> menu left open when
 * you click elsewhere is the classic half-built dropdown. */
(function () {
  "use strict";
  var burger = document.getElementById("burger");
  var panel = document.getElementById("navpanel");
  var acct = document.getElementById("acct");

  function closePanel() {
    if (!panel) return;
    panel.classList.remove("open");
    burger.setAttribute("aria-expanded", "false");
  }

  if (burger && panel) {
    burger.addEventListener("click", function (e) {
      e.stopPropagation();
      var open = panel.classList.toggle("open");
      burger.setAttribute("aria-expanded", open ? "true" : "false");
      if (open && acct) acct.removeAttribute("open");
    });
  }

  document.addEventListener("click", function (e) {
    if (acct && acct.hasAttribute("open") && !acct.contains(e.target)) {
      acct.removeAttribute("open");
    }
    if (panel && panel.classList.contains("open") &&
        !panel.contains(e.target) && e.target !== burger) {
      closePanel();
    }
  });

  document.addEventListener("keydown", function (e) {
    if (e.key !== "Escape") return;
    if (acct) acct.removeAttribute("open");
    closePanel();
  });
})();
