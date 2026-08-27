/* The Share control on a brief.
 * Creating a link is a POST rather than a GET so it cannot be triggered by
 * something merely prefetching or crawling the page. */
(function () {
  "use strict";
  var bar = document.querySelector(".sharebar");
  if (!bar) return;
  var id = bar.dataset.id;

  bar.addEventListener("click", function (e) {
    var b = e.target.closest("[data-act]");
    if (!b) return;
    var act = b.dataset.act;

    if (act === "copy") {
      var f = document.getElementById("shareurl");
      f.select();
      navigator.clipboard.writeText(f.value).then(function () {
        var was = b.textContent;
        b.textContent = "Copied";
        setTimeout(function () { b.textContent = was; }, 1600);
      });
      return;
    }

    b.disabled = true;
    fetch("/digest/" + id + "/" + (act === "share" ? "share" : "unshare"), { method: "POST" })
      .then(function (r) { return r.json(); })
      .then(function () { location.reload(); })
      .catch(function () { b.disabled = false; b.textContent = "Failed — retry"; });
  });
})();
