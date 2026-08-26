/* Knowledge graph: entities and edges as a force layout.
 * Node size follows mention count, edge width follows confirmed strength, and
 * colour follows entity kind — so the shape of what Alfred knows is visible at
 * a glance, and a hub that keeps appearing in the corpus looks like one. */
(function () {
  "use strict";

  var KIND_COLOR = {
    institution: "#EA580C", asset: "#B45309", country: "#0F766E",
    person: "#64748B", policy: "#111111", concept: "#9CA3AF"
  };

  fetch("/api/graph")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      var el = document.getElementById("graphcanvas");
      if (!data.nodes.length) {
        el.outerHTML = '<p class="empty">The graph is empty — it fills as documents are analysed.</p>';
        return;
      }
      var maxMention = Math.max.apply(null, data.nodes.map(function (n) { return n.mention_count || 1; }));
      var nodes = data.nodes.map(function (n) {
        return {
          id: n.canonical, name: n.canonical, category: n.kind || "concept",
          symbolSize: 10 + 26 * Math.sqrt((n.mention_count || 1) / maxMention),
          itemStyle: { color: KIND_COLOR[n.kind] || KIND_COLOR.concept },
          label: { show: (n.mention_count || 1) > maxMention * 0.15 }
        };
      });
      var have = {};
      nodes.forEach(function (n) { have[n.id] = true; });
      var links = data.edges
        .filter(function (e) { return have[e.source_entity] && have[e.target_entity]; })
        .map(function (e) {
          return {
            source: e.source_entity, target: e.target_entity,
            lineStyle: { width: 0.8 + 2.4 * (e.strength || 0.5), curveness: 0.12 },
            _meta: e
          };
        });

      var chart = echarts.init(el);
      chart.setOption({
        tooltip: {
          formatter: function (p) {
            if (p.dataType === "edge") {
              var m = p.data._meta;
              return "<b>" + m.source_entity + "</b> —" + m.relation + " (" + m.direction +
                     ")→ <b>" + m.target_entity + "</b><br>strength " + m.strength +
                     " · seen " + m.confirm_count + "×" +
                     (m.rationale ? "<br><span style='color:#6B7280'>" + m.rationale + "</span>" : "");
            }
            return "<b>" + p.data.name + "</b> · " + p.data.category;
          },
          textStyle: { fontSize: 12 }, extraCssText: "max-width:340px;white-space:normal;"
        },
        series: [{
          type: "graph", layout: "force", data: nodes, links: links, roam: true,
          force: { repulsion: 220, edgeLength: [60, 160], gravity: 0.08 },
          label: { fontSize: 11, color: "#111" },
          edgeSymbol: ["none", "arrow"], edgeSymbolSize: 6,
          emphasis: { focus: "adjacency", lineStyle: { width: 4 } },
          lineStyle: { color: "#C7CDD6" }
        }]
      });
      window.addEventListener("resize", function () { chart.resize(); });
    });
})();
