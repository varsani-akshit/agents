/* The chat surface: custom pickers, an honest thinking state, and answers
 * rendered in place. Posts to /api/ask rather than submitting the form, so the
 * question appears immediately and the reader watches progress instead of a
 * frozen page for the minute or more an investigation takes. */
(function () {
  "use strict";

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) {
    return Array.prototype.slice.call((r || document).querySelectorAll(s));
  };

  var form = $("#composer");
  if (!form) return;
  var box = $("#q"), scroll = $("#chatscroll"), send = $("#send");

  /* ── branded pickers ── native selects cannot be themed, and a Windows
     dropdown in the middle of this page reads as someone else's software. */
  $$(".pick").forEach(function (pick) {
    var btn = $(".pick-btn", pick), menu = $(".pick-menu", pick);
    btn.addEventListener("click", function (e) {
      e.stopPropagation();
      var open = pick.classList.contains("open");
      $$(".pick").forEach(function (p) { p.classList.remove("open"); });
      pick.classList.toggle("open", !open);
    });
    $$("button[data-v]", menu).forEach(function (opt) {
      opt.addEventListener("click", function () {
        pick.dataset.value = opt.dataset.v;
        $(".pick-label", pick).textContent = $(".t", opt).textContent;
        pick.classList.remove("open");
      });
    });
  });
  document.addEventListener("click", function () {
    $$(".pick").forEach(function (p) { p.classList.remove("open"); });
  });

  /* ── composer: grows with the question, Enter sends ── */
  function grow() {
    box.style.height = "auto";
    box.style.height = Math.min(box.scrollHeight, 200) + "px";
  }
  box.addEventListener("input", grow);
  box.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
  });
  grow();
  box.focus();

  $$(".sg").forEach(function (b) {
    b.addEventListener("click", function () {
      box.value = b.textContent.trim();
      grow(); box.focus();
    });
  });

  function el(cls, html) {
    var d = document.createElement("div");
    d.className = cls;
    if (html) d.innerHTML = html;
    return d;
  }
  function toBottom() { scroll.scrollTop = scroll.scrollHeight; }

  /* ── thinking state ── the stage labels are the real pipeline, so a long
     wait reads as work in progress rather than a hang. */
  var QUICK = ["Reading the question", "Searching the corpus",
               "Measuring against stored data", "Checking the live web",
               "Composing the answer"];
  var DEEP = ["Planning the investigation", "Dispatching researchers",
              "Gathering sources in parallel", "Computing the evidence",
              "Cross-checking findings", "Writing the note"];

  function thinking(depth) {
    var node = el("msg alfred thinking",
      '<div class="who"><span class="mark"></span>Alfred</div>' +
      '<div class="think"><span class="dots"><i></i><i></i><i></i></span>' +
      '<span class="stage"></span></div>');
    scroll.appendChild(node);
    toBottom();
    var stages = depth === "deep" ? DEEP : QUICK;
    var i = 0, label = $(".stage", node);
    label.textContent = stages[0];
    var timer = setInterval(function () {
      i = Math.min(i + 1, stages.length - 1);
      label.style.opacity = "0";
      setTimeout(function () {
        label.textContent = stages[i];
        label.style.opacity = "";
      }, 180);
    }, depth === "deep" ? 16000 : 9000);
    return { node: node, stop: function () { clearInterval(timer); } };
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var q = box.value.trim();
    if (!q) return;
    var depth = $('.pick[data-pick="depth"]').dataset.value || "quick";
    var model = $('.pick[data-pick="model"]').dataset.value || "";

    var empty = $("#chatempty");
    if (empty) empty.remove();

    var user = el("msg user", '<div class="bubble"></div>');
    $(".bubble", user).textContent = q;
    scroll.appendChild(user);
    box.value = ""; grow();
    send.disabled = true;
    var think = thinking(depth);

    fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, depth: depth, model: model })
    })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, d: d }; }); })
      .then(function (res) {
        think.stop(); think.node.remove();
        send.disabled = false;
        if (!res.ok || res.d.error) {
          var err = el("msg alfred",
            '<div class="who"><span class="mark"></span>Alfred</div>' +
            '<div class="failed"></div>');
          $(".failed", err).textContent = res.d.error || "That did not go through. Try again.";
          scroll.appendChild(err); toBottom();
          return;
        }
        var a = el("msg alfred reveal",
          '<div class="who"><span class="mark"></span>Alfred' +
          (res.d.depth === "deep"
            ? ' <a class="when" href="' + res.d.url + '">open the full note →</a>' : "") +
          '</div><article class="mbody"></article>');
        $(".mbody", a).innerHTML = res.d.html;
        scroll.appendChild(a);
        toBottom();
        box.focus();
      })
      .catch(function () {
        think.stop(); think.node.remove();
        send.disabled = false;
        var err = el("msg alfred",
          '<div class="who"><span class="mark"></span>Alfred</div>' +
          '<div class="failed">Connection lost before the answer arrived.</div>');
        scroll.appendChild(err); toBottom();
      });
  });

  toBottom();
})();
