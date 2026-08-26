/* The knowledge graph: documents, concepts, and the links between them.
 *
 * Nodes come from /api/graph and are laid out by force simulation. Clicking a
 * document loads its detail and nearest neighbours into the side panel, and the
 * neighbours are themselves clickable — so the graph is a way to read the
 * corpus by association rather than a picture of it. */
(function () {
  "use strict";

  var COLOR = { document: "#EA580C", entity: "#111111", theme: "#0F766E" };
  var EDGE = { similar: "#F0A87C", mentions: "#D8DCE2", relation: "#111111" };

  var el = document.getElementById("graphcanvas");
  var panel = document.getElementById("gpanel");
  var stats = document.getElementById("gstats");
  var search = document.getElementById("gsearch");
  if (!el || typeof echarts === "undefined") return;

  var chart = echarts.init(el);
  var state = { days: 7, urgency: "Medium", q: "" };

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  /* ── data ── */
  function load() {
    chart.showLoading({ text: "", maskColor: "rgba(250,250,250,.7)", color: "#EA580C" });
    fetch("/api/graph?days=" + state.days + "&urgency=" + state.urgency +
          "&q=" + encodeURIComponent(state.q))
      .then(function (r) { return r.json(); })
      .then(render)
      .catch(function () {
        chart.hideLoading();
        stats.textContent = "Could not load the graph.";
      });
  }

  function render(g) {
    chart.hideLoading();
    stats.textContent = g.stats.documents + " documents · " + g.stats.concepts +
      " concepts · " + g.stats.links + " links";
    if (!g.nodes.length) {
      chart.clear();
      panel.innerHTML = '<p class="hint">Nothing in this window. Widen it, or lower the urgency floor.</p>';
      return;
    }

    var nodes = g.nodes.map(function (n) {
      return {
        id: n.id, name: n.label, raw: n,
        symbolSize: n.kind === "document" ? 7 + n.weight * 3 : 12 + n.weight * 2.5,
        itemStyle: { color: COLOR[n.kind] || COLOR.theme },
        // Labelling 220 documents is unreadable; the concepts carry the map.
        label: { show: n.kind !== "document" }
      };
    });
    var links = g.edges.map(function (e) {
      return {
        source: e.source, target: e.target, raw: e,
        lineStyle: {
          color: EDGE[e.kind] || EDGE.mentions,
          width: e.kind === "similar" ? 0.5 + e.weight * 2.2 : 0.6,
          opacity: e.kind === "mentions" ? 0.5 : 0.85,
          curveness: 0.08
        }
      };
    });

    chart.setOption({
      tooltip: {
        confine: true, backgroundColor: "#fff", borderColor: "#E5E7EB", borderWidth: 1,
        extraCssText: "max-width:320px;white-space:normal;border-radius:0;",
        textStyle: { fontSize: 12, color: "#111" },
        formatter: function (p) {
          if (p.dataType === "edge") {
            var e = p.data.raw;
            return e.kind === "similar" ? "similarity " + e.weight : esc(e.label || e.kind);
          }
          var n = p.data.raw;
          if (n.kind !== "document") {
            return "<b>" + esc(n.label) + "</b><br>" + n.mentions + " documents";
          }
          return "<b>" + esc(n.label) + "</b><br><span style='color:#6B7280'>" +
                 esc(n.source) + " · " + esc(n.urgency) + "</span>";
        }
      },
      series: [{
        type: "graph", layout: "force", roam: true, data: nodes, links: links,
        // Tuned so 250-odd nodes settle inside the frame rather than sprawling
        // past its edges: stronger gravity pulls the mass inward, and a shorter
        // edge range keeps clusters compact enough to read.
        force: { repulsion: 90, edgeLength: [22, 80], gravity: 0.22, friction: 0.16 },
        scaleLimit: { min: 0.4, max: 8 },
        label: {
          fontSize: 11, color: "#111", position: "right",
          // Concept labels collided into unreadable stacks at this density.
          overflow: "truncate", width: 96
        },
        labelLayout: { hideOverlap: true },
        emphasis: { focus: "adjacency", label: { show: true }, lineStyle: { width: 3 } }
      }]
    }, true);
  }

  /* ── detail panel ── */
  function showDocument(id) {
    panel.innerHTML = '<p class="hint">Loading…</p>';
    fetch("/api/graph/document/" + id)
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d || !d.document) {
          panel.innerHTML = '<p class="hint">Not found.</p>';
          return;
        }
        var doc = d.document;
        var tags = (doc.entities || []).concat(doc.themes || []);
        panel.innerHTML =
          '<div class="gp-meta">' + esc(doc.source) + " · tier " + doc.tier +
            " · " + esc(doc.urgency || "—") + "</div>" +
          "<h4>" + esc(doc.title) + "</h4>" +
          (doc.summary ? "<p>" + esc(doc.summary) + "</p>" : "") +
          (tags.length
            ? '<div class="gp-tags">' + tags.map(function (t) {
                return '<span class="chip">' + esc(t) + "</span>";
              }).join("") + "</div>"
            : "") +
          (doc.url
            ? '<p><a href="' + esc(doc.url) + '" target="_blank" rel="noopener">Open source ↗</a></p>'
            : "") +
          (d.neighbours.length
            ? "<h5>Most related</h5>" + d.neighbours.map(function (x) {
                return '<a class="gp-rel" href="#" data-id="' + x.id + '">' +
                  '<span class="sim">' + x.similarity.toFixed(2) + "</span>" +
                  esc(x.title) + "</a>";
              }).join("")
            : '<p class="hint">No neighbours above the similarity floor.</p>');
      })
      .catch(function () { panel.innerHTML = '<p class="hint">Could not load that document.</p>'; });
  }

  chart.on("click", function (p) {
    if (p.dataType !== "node") return;
    var n = p.data.raw;
    if (n.kind === "document") {
      showDocument(n.id.slice(1));
      return;
    }
    // A concept node filters the graph to the documents that mention it.
    state.q = n.label;
    search.value = n.label;
    panel.innerHTML = "<h4>" + esc(n.label) + '</h4><p class="hint">' +
      n.mentions + " documents mention this. Graph filtered to them.</p>";
    load();
  });

  panel.addEventListener("click", function (e) {
    var a = e.target.closest(".gp-rel");
    if (!a) return;
    e.preventDefault();
    showDocument(a.dataset.id);
  });

  /* ── controls ── */
  document.querySelector(".gctl").addEventListener("click", function (e) {
    var b = e.target.closest(".opt");
    if (!b) return;
    state[b.dataset.k] = b.dataset.v;
    document.querySelectorAll('.gctl .opt[data-k="' + b.dataset.k + '"]').forEach(function (o) {
      o.classList.toggle("on", o === b);
    });
    load();
  });

  var timer;
  search.addEventListener("input", function (e) {
    clearTimeout(timer);
    var v = e.target.value;
    timer = setTimeout(function () { state.q = v; load(); }, 350);
  });

  window.addEventListener("resize", function () { chart.resize(); });
  load();
})();
