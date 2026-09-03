/* The knowledge graph: documents, concepts, and the links between them.
 *
 * Nodes come from /api/graph and are laid out by force simulation. The
 * interaction model is focus, not navigation: clicking a node selects it and
 * dims everything unrelated, so the neighbourhood stands out without the map
 * being rebuilt underneath the reader. Filtering is a deliberate second step,
 * and it can always be undone — an empty result never replaces a good graph
 * with a blank frame. */
(function () {
  "use strict";

  var COLOR = { document: "#EA580C", entity: "#111111", theme: "#0F766E",
                security: "#1D4ED8" };
  var EDGE = { similar: "#F0A87C", mentions: "#D8DCE2", relation: "#111111",
               covers: "#93B4F5" };
  var DIM = "#E8E8E8";

  var el = document.getElementById("graphcanvas");
  var panel = document.getElementById("gpanel");
  var stats = document.getElementById("gstats");
  var search = document.getElementById("gsearch");
  if (!el || typeof echarts === "undefined") return;

  var chart = echarts.init(el);
  var state = { days: 7, urgency: "Medium", q: "", concept: "" };
  var current = null;     // last good graph payload
  var selected = null;    // selected node id
  var adjacency = {};     // node id -> Set of connected node ids

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  /* ── data ─────────────────────────────────────────────────────────────── */
  function load(opts) {
    opts = opts || {};
    chart.showLoading({ text: "", maskColor: "rgba(250,250,250,.55)", color: "#EA580C" });
    var url = "/api/graph?days=" + state.days + "&urgency=" + state.urgency +
              "&q=" + encodeURIComponent(state.q) +
              "&concept=" + encodeURIComponent(state.concept);
    fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (g) {
        chart.hideLoading();
        if (!g.nodes || !g.nodes.length) {
          // A filter that matches nothing must not destroy the map the reader
          // was reading. Keep it, say so, and offer the way back.
          if (opts.revert) opts.revert();
          note("Nothing matched that. Showing the previous view.");
          if (!current) emptyState();
          return;
        }
        current = g;
        render(g);
      })
      .catch(function () {
        chart.hideLoading();
        note("Could not load the graph.");
      });
  }

  function note(msg) {
    var n = document.getElementById("gnote");
    if (!n) return;
    n.textContent = msg;
    n.classList.add("on");
    setTimeout(function () { n.classList.remove("on"); }, 3200);
  }

  function emptyState() {
    chart.clear();
    panel.innerHTML = '<p class="hint">Nothing in this window. Widen it, or lower the urgency floor.</p>';
  }

  function render(g) {
    stats.textContent = g.stats.documents + " documents · " + g.stats.concepts +
      " concepts · " + g.stats.links + " links";

    adjacency = {};
    g.edges.forEach(function (e) {
      (adjacency[e.source] = adjacency[e.source] || new Set()).add(e.target);
      (adjacency[e.target] = adjacency[e.target] || new Set()).add(e.source);
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
            return "<b>" + esc(n.label) + "</b><br>" + n.mentions +
                   " documents<br><span style='color:#9CA3AF'>click to focus</span>";
          }
          return "<b>" + esc(n.label) + "</b><br><span style='color:#6B7280'>" +
                 esc(n.source) + " · " + esc(n.urgency) + "</span>";
        }
      },
      series: [{
        type: "graph", layout: "force", roam: true,
        data: buildNodes(g), links: buildLinks(g),
        // Nodes can be dragged: pulling a cluster apart is how a dense patch
        // becomes readable, and the simulation settles around the new position.
        draggable: true,
        force: { repulsion: 90, edgeLength: [22, 80], gravity: 0.22, friction: 0.16 },
        scaleLimit: { min: 0.4, max: 8 },
        label: { fontSize: 11, color: "#111", position: "right",
                 overflow: "truncate", width: 96 },
        labelLayout: { hideOverlap: true },
        emphasis: { focus: "adjacency", label: { show: true }, lineStyle: { width: 3 } }
      }]
    }, true);
  }

  /* Selection is expressed in the data itself rather than by re-querying:
     unrelated nodes fade, the neighbourhood keeps its colour. */
  function related(id) {
    if (!id) return null;
    var set = new Set(adjacency[id] || []);
    set.add(id);
    return set;
  }

  function buildNodes(g) {
    var keep = related(selected);
    return g.nodes.map(function (n) {
      var on = !keep || keep.has(n.id);
      return {
        id: n.id, name: n.label, raw: n,
        symbolSize: (n.kind === "document" ? 7 + n.weight * 3 : 12 + n.weight * 2.5) *
                    (n.id === selected ? 1.6 : 1),
        itemStyle: {
          color: on ? (COLOR[n.kind] || COLOR.theme) : DIM,
          borderColor: n.id === selected ? "#111" : "transparent",
          borderWidth: n.id === selected ? 2 : 0,
          opacity: on ? 1 : 0.35
        },
        label: { show: n.kind !== "document" && on, color: on ? "#111" : "#B9BDC4" }
      };
    });
  }

  function buildLinks(g) {
    var keep = related(selected);
    return g.edges.map(function (e) {
      var on = !keep || (keep.has(e.source) && keep.has(e.target));
      return {
        source: e.source, target: e.target, raw: e,
        lineStyle: {
          color: on ? (EDGE[e.kind] || EDGE.mentions) : DIM,
          width: e.kind === "similar" ? 0.5 + e.weight * 2.2 : 0.6,
          opacity: on ? (e.kind === "mentions" ? 0.5 : 0.85) : 0.12,
          curveness: 0.08
        }
      };
    });
  }

  function repaint() {
    if (!current) return;
    chart.setOption({ series: [{ data: buildNodes(current), links: buildLinks(current) }] });
  }

  function select(id) {
    selected = id;
    repaint();
    document.getElementById("gclear").classList.toggle("on", !!(id || state.q || state.concept));
  }

  /* ── detail panel ─────────────────────────────────────────────────────── */
  function showDocument(id) {
    panel.innerHTML = '<p class="hint">Loading…</p>';
    fetch("/api/graph/document/" + id)
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d || !d.document) { panel.innerHTML = '<p class="hint">Not found.</p>'; return; }
        var doc = d.document;
        var tags = (doc.entities || []).concat(doc.themes || []);
        panel.innerHTML =
          '<div class="gp-meta">' + esc(doc.source) + " · tier " + doc.tier +
            " · " + esc(doc.urgency || "—") + "</div>" +
          "<h4>" + esc(doc.title) + "</h4>" +
          (doc.summary ? "<p>" + esc(doc.summary) + "</p>" : "") +
          (tags.length
            ? '<div class="gp-tags">' + tags.map(function (t) {
                // Tags are live: each one focuses that concept in the graph.
                return '<button class="chip" data-concept="' + esc(t) + '">' + esc(t) + "</button>";
              }).join("") + "</div>"
            : "") +
          (doc.url
            ? '<p><a href="' + esc(doc.url) + '" target="_blank" rel="noopener">Open source ↗</a></p>'
            : "") +
          (d.neighbours && d.neighbours.length
            ? "<h5>Most related</h5>" + d.neighbours.map(function (x) {
                return '<a class="gp-rel" href="#" data-id="' + x.id + '">' +
                  '<span class="sim">' + x.similarity.toFixed(2) + "</span>" +
                  esc(x.title) + "</a>";
              }).join("")
            : '<p class="hint">No neighbours above the similarity floor.</p>');
      })
      .catch(function () { panel.innerHTML = '<p class="hint">Could not load that document.</p>'; });
  }

  function showSecurity(n) {
    var docs = (current ? current.nodes : []).filter(function (x) {
      return x.kind === "document" && (adjacency[n.id] || new Set()).has(x.id);
    });
    panel.innerHTML =
      '<div class="gp-meta">' + esc(n.exchange || "") +
        (n.sector ? " · " + esc(n.sector) : "") + "</div>" +
      "<h4>" + esc(n.label) + "</h4>" +
      "<p>Named in " + n.mentions + " document" + (n.mentions === 1 ? "" : "s") +
        " in this window.</p>" +
      '<p><a class="gp-act" href="/markets/' + esc(n.symbol) +
        '">Open the company →</a></p>' +
      (docs.length
        ? "<h5>Coverage on the map</h5>" + docs.slice(0, 12).map(function (x) {
            return '<a class="gp-rel" href="#" data-id="' + x.id.slice(1) + '">' +
                   esc(x.label) + "</a>";
          }).join("")
        : "");
  }

  function showConcept(n) {
    var docs = (current ? current.nodes : []).filter(function (x) {
      return x.kind === "document" && (adjacency[n.id] || new Set()).has(x.id);
    });
    panel.innerHTML =
      '<div class="gp-meta">' + esc(n.kind) + "</div>" +
      "<h4>" + esc(n.label) + "</h4>" +
      "<p>" + n.mentions + " documents mention this in the corpus; " +
      docs.length + " are on the map.</p>" +
      '<button class="gp-act" id="gfocus">Filter the graph to this concept</button>' +
      (docs.length
        ? "<h5>On the map</h5>" + docs.slice(0, 14).map(function (x) {
            return '<a class="gp-rel" href="#" data-id="' + x.id.slice(1) + '">' +
                   esc(x.label) + "</a>";
          }).join("")
        : "");
    var btn = document.getElementById("gfocus");
    if (btn) btn.addEventListener("click", function () {
      var prev = { concept: state.concept, q: state.q };
      state.concept = n.label;
      state.q = "";
      search.value = "";
      selected = null;
      load({ revert: function () { state.concept = prev.concept; state.q = prev.q; } });
      document.getElementById("gclear").classList.add("on");
    });
  }

  /* ── interaction ──────────────────────────────────────────────────────── */
  chart.on("click", function (p) {
    if (p.dataType !== "node") return;
    var n = p.data.raw;
    select(n.id);
    if (n.kind === "document") showDocument(n.id.slice(1));
    else if (n.kind === "security") showSecurity(n);
    else showConcept(n);
  });

  // Clicking empty canvas clears the selection — the way out of a focus.
  chart.getZr().on("click", function (e) {
    if (e.target) return;   // a node or edge handled it
    select(null);
    panel.innerHTML = '<p class="hint">Click a node to inspect it.</p>';
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") { select(null); }
  });

  panel.addEventListener("click", function (e) {
    var a = e.target.closest(".gp-rel");
    if (a) {
      e.preventDefault();
      select("d" + a.dataset.id);
      showDocument(a.dataset.id);
      return;
    }
    var chip = e.target.closest(".chip[data-concept]");
    if (chip) {
      var id = "e:" + chip.dataset.concept;
      var node = (current ? current.nodes : []).find(function (x) {
        return x.label === chip.dataset.concept;
      });
      if (node) { select(node.id); showConcept(node); }
      else note("That concept is not on the current map.");
    }
  });

  /* ── controls ─────────────────────────────────────────────────────────── */
  document.querySelector(".gctl").addEventListener("click", function (e) {
    var b = e.target.closest(".opt");
    if (!b) return;
    var prev = state[b.dataset.k];
    state[b.dataset.k] = b.dataset.v;
    document.querySelectorAll('.gctl .opt[data-k="' + b.dataset.k + '"]').forEach(function (o) {
      o.classList.toggle("on", o === b);
    });
    selected = null;
    load({ revert: function () { state[b.dataset.k] = prev; } });
  });

  document.getElementById("gclear").addEventListener("click", function () {
    state.q = ""; state.concept = ""; selected = null;
    search.value = "";
    this.classList.remove("on");
    load();
  });

  document.getElementById("greset").addEventListener("click", function () {
    // Re-running the layout is the reliable way back from a tangled drag.
    if (current) render(current);
  });

  var timer;
  search.addEventListener("input", function (e) {
    clearTimeout(timer);
    var v = e.target.value, prev = state.q;
    timer = setTimeout(function () {
      state.q = v;
      state.concept = "";
      selected = null;
      document.getElementById("gclear").classList.toggle("on", !!v);
      load({ revert: function () { state.q = prev; } });
    }, 350);
  });

  window.addEventListener("resize", function () { chart.resize(); });
  load();
})();
