/* Renders the JSON chart specs from signals/chartdata.py with ECharts.
 *
 * One library, one theme, one tooltip convention. Charts are drawn on demand as
 * they scroll into view — a brief can carry a dozen figures and initialising
 * them all at once is visibly slow on a phone.
 */
(function () {
  "use strict";

  var CSS = getComputedStyle(document.documentElement);
  var tok = function (name, fallback) {
    return (CSS.getPropertyValue(name) || "").trim() || fallback;
  };

  var INK = tok("--t1", "#111111");
  var T2 = tok("--t2", "#6B7280");
  var T3 = tok("--t3", "#9CA3AF");
  var BORDER = tok("--border", "#E5E7EB");
  var ACCENT = tok("--accent", "#EA580C");
  var SURFACE = tok("--surface", "#FFFFFF");
  var UP = "#15803D";
  var DOWN = "#B91C1C";

  var SANS = "Inter, system-ui, -apple-system, sans-serif";
  var MONO = "'JetBrains Mono', ui-monospace, monospace";

  var isPhone = function () { return window.innerWidth < 640; };

  var base = function () {
    return {
      backgroundColor: "transparent",
      animationDuration: 320,
      textStyle: { fontFamily: SANS, color: T2, fontSize: 12 },
      grid: {
        left: 4, right: 8, top: 12, bottom: 4,
        containLabel: true
      },
      tooltip: {
        backgroundColor: SURFACE,
        borderColor: BORDER,
        borderWidth: 1,
        padding: [8, 11],
        extraCssText: "box-shadow:0 1px 3px rgba(0,0,0,.07);border-radius:6px;",
        textStyle: { color: INK, fontSize: 12, fontFamily: SANS }
      }
    };
  };

  var axisCommon = {
    axisLine: { lineStyle: { color: BORDER } },
    axisTick: { show: false },
    axisLabel: { color: T3, fontSize: 10, fontFamily: MONO },
    splitLine: { lineStyle: { color: BORDER, opacity: 0.55, type: [3, 4] } }
  };

  function fmt(v, unit) {
    if (v === null || v === undefined) return "–";
    var n = Number(v);
    var s = Math.abs(n) >= 1000 ? n.toLocaleString(undefined, { maximumFractionDigits: 0 })
                                : n.toFixed(Math.abs(n) < 10 ? 2 : 1);
    return s + (unit || "");
  }

  /* ── line and dual-axis line ─────────────────────────────────────────── */
  function lineOption(spec) {
    var dual = spec.type === "dual_line";
    var yAxes = [Object.assign({}, axisCommon, {
      type: "value", name: spec.yLabel || "", nameTextStyle: { color: T3, fontSize: 10 },
      scale: true, inverse: !!spec.invertY,
      min: spec.yMin !== undefined ? spec.yMin : undefined,
      max: spec.yMax !== undefined ? spec.yMax : undefined
    })];
    if (dual) {
      yAxes.push(Object.assign({}, axisCommon, {
        type: "value", name: spec.y2Label || "", scale: true,
        nameTextStyle: { color: T3, fontSize: 10 },
        splitLine: { show: false }
      }));
    }

    var series = spec.series.map(function (s) {
      var conf = {
        name: s.name, type: "line", data: s.data,
        yAxisIndex: s.axis || 0,
        showSymbol: false, symbol: "circle", symbolSize: 6,
        lineStyle: { width: s.width || 1.7, color: s.color },
        itemStyle: { color: s.color },
        emphasis: { focus: "series" },
        connectNulls: false
      };
      if (s.dashed) conf.lineStyle.type = "dashed";
      return conf;
    });

    if (spec.markZero) {
      series[0].markLine = {
        silent: true, symbol: "none",
        lineStyle: { color: T3, width: 1, type: "solid", opacity: 0.6 },
        data: [{ yAxis: 0 }], label: { show: false }
      };
    }
    if (spec.band) {
      series.push({
        type: "line", data: [], silent: true,
        markArea: {
          silent: true,
          itemStyle: { color: ACCENT, opacity: 0.06 },
          data: [[{ yAxis: spec.band.lower }, { yAxis: spec.band.upper }]]
        },
        markLine: {
          silent: true, symbol: "none",
          lineStyle: { color: T3, type: "dashed", width: 1 },
          label: { show: false }, data: [{ yAxis: spec.band.mean }]
        }
      });
    }

    return Object.assign(base(), {
      tooltip: Object.assign(base().tooltip, {
        trigger: "axis",
        axisPointer: { type: "line", lineStyle: { color: T3, width: 1, type: "dashed" } },
        formatter: function (ps) {
          var out = '<div style="font-family:' + MONO + ';font-size:10px;color:' + T3 +
                    ';margin-bottom:4px">' + ps[0].axisValue + "</div>";
          ps.forEach(function (p) {
            if (p.value === null || p.value === undefined) return;
            out += '<div style="display:flex;gap:10px;justify-content:space-between">' +
                   '<span><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:' +
                   p.color + ';margin-right:6px"></span>' + p.seriesName + "</span>" +
                   '<span style="font-family:' + MONO + '">' + fmt(p.value) + "</span></div>";
          });
          return out;
        }
      }),
      legend: {
        show: spec.series.length > 1, top: 0, right: 0, icon: "roundRect",
        itemWidth: 9, itemHeight: 2, itemGap: 14,
        textStyle: { color: T2, fontSize: 11 }
      },
      grid: { left: 4, right: dual ? 12 : 8, top: spec.series.length > 1 ? 34 : 12,
              bottom: 4, containLabel: true },
      xAxis: Object.assign({}, axisCommon, {
        type: "category", data: spec.x, boundaryGap: false,
        axisLabel: {
          color: T3, fontSize: 10, fontFamily: MONO, hideOverlap: true,
          formatter: function (v) {
            var d = new Date(v);
            return isNaN(d) ? v : d.toLocaleDateString(undefined, { month: "short", year: "2-digit" });
          }
        }
      }),
      yAxis: yAxes,
      series: series
    });
  }

  /* ── horizontal bars, including the diverging regime votes ───────────── */
  function barOption(spec) {
    var diverging = spec.type === "diverging";
    var colors = spec.values.map(function (v) {
      if (spec.signColour === false && spec.colour) return spec.colour;
      return v >= 0 ? (diverging ? UP : ACCENT) : DOWN;
    });
    return Object.assign(base(), {
      tooltip: Object.assign(base().tooltip, {
        trigger: "item",
        formatter: function (p) {
          var note = spec.notes && spec.notes[p.dataIndex];
          var w = spec.weights && spec.weights[p.dataIndex];
          return '<div style="font-weight:500">' + p.name + "</div>" +
                 '<div style="font-family:' + MONO + ';margin-top:2px">' +
                 fmt(p.value, spec.unit) + "</div>" +
                 (note ? '<div style="color:' + T2 + ';margin-top:4px;max-width:230px;' +
                         'white-space:normal">' + note + "</div>" : "") +
                 (w ? '<div style="color:' + T3 + ';font-family:' + MONO +
                      ';font-size:10px;margin-top:3px">weight ' + w + "</div>" : "");
        }
      }),
      grid: { left: 4, right: 26, top: 6, bottom: 4, containLabel: true },
      xAxis: Object.assign({}, axisCommon, {
        type: "value",
        min: spec.min !== undefined ? spec.min : undefined,
        max: spec.max !== undefined ? spec.max : undefined,
        axisLabel: { color: T3, fontSize: 10, fontFamily: MONO,
                     formatter: function (v) { return v + (spec.unit || ""); } }
      }),
      yAxis: Object.assign({}, axisCommon, {
        type: "category", data: spec.categories,
        splitLine: { show: false },
        axisLabel: { color: T2, fontSize: isPhone() ? 9 : 10.5, fontFamily: MONO }
      }),
      series: [{
        type: "bar", data: spec.values,
        barMaxWidth: 13,
        itemStyle: {
          color: function (p) { return colors[p.dataIndex]; },
          borderRadius: 2
        },
        label: {
          show: !isPhone(), position: "right", color: T3,
          fontFamily: MONO, fontSize: 9.5,
          formatter: function (p) { return (p.value > 0 ? "+" : "") + p.value; }
        }
      }]
    });
  }

  /* ── heatmap ─────────────────────────────────────────────────────────── */
  function heatmapOption(spec) {
    return Object.assign(base(), {
      tooltip: Object.assign(base().tooltip, {
        formatter: function (p) {
          return '<div style="font-weight:500">' + spec.yLabels[p.value[1]] + "</div>" +
                 '<div style="color:' + T2 + '">' + spec.xLabels[p.value[0]] + "</div>" +
                 '<div style="font-family:' + MONO + ';margin-top:3px">' +
                 fmt(p.value[2], spec.unit) + "</div>";
        }
      }),
      grid: { left: 4, right: 4, top: 6, bottom: 4, containLabel: true },
      xAxis: Object.assign({}, axisCommon, {
        type: "category", data: spec.xLabels, splitLine: { show: false },
        axisLabel: { color: T2, fontSize: 10, fontFamily: MONO,
                     rotate: spec.xLabels.length > 8 ? 45 : 0 }
      }),
      yAxis: Object.assign({}, axisCommon, {
        type: "category", data: spec.yLabels, splitLine: { show: false },
        axisLabel: { color: T2, fontSize: isPhone() ? 8.5 : 10, fontFamily: MONO }
      }),
      visualMap: {
        min: spec.min, max: spec.max, calculable: false, show: false,
        inRange: { color: ["#B91C1C", "#F0A38C", "#F7F7F6", "#9CC9AE", "#15803D"] }
      },
      series: [{
        type: "heatmap", data: spec.cells,
        itemStyle: { borderColor: SURFACE, borderWidth: 1 },
        label: {
          show: !isPhone(), color: INK, fontFamily: MONO, fontSize: 8.5,
          formatter: function (p) { return p.value[2]; }
        },
        emphasis: { itemStyle: { borderColor: INK, borderWidth: 1.5 } }
      }]
    });
  }

  /* ── yield curve: numeric x, one point per tenor ─────────────────────── */
  function curveOption(spec) {
    return Object.assign(base(), {
      tooltip: Object.assign(base().tooltip, {
        trigger: "axis",
        formatter: function (ps) {
          var out = '<div style="font-family:' + MONO + ';font-size:10px;color:' + T3 + '">' +
                    ps[0].value[0] + "-year</div>";
          ps.forEach(function (p) {
            out += '<div style="display:flex;gap:12px;justify-content:space-between">' +
                   "<span>" + p.seriesName + "</span><span style='font-family:" + MONO + "'>" +
                   p.value[1].toFixed(2) + "%</span></div>";
          });
          return out;
        }
      }),
      legend: { show: true, top: 0, right: 0, icon: "roundRect", itemWidth: 9,
                itemHeight: 2, itemGap: 14, textStyle: { color: T2, fontSize: 11 } },
      grid: { left: 4, right: 10, top: 32, bottom: 4, containLabel: true },
      xAxis: Object.assign({}, axisCommon, {
        type: "value", name: spec.xLabel, nameLocation: "middle", nameGap: 26,
        nameTextStyle: { color: T3, fontSize: 10 }
      }),
      yAxis: Object.assign({}, axisCommon, { type: "value", scale: true, name: spec.yLabel,
                                             nameTextStyle: { color: T3, fontSize: 10 } }),
      series: spec.series.map(function (s) {
        return {
          name: s.name, type: "line", data: s.data,
          symbol: "circle", symbolSize: 7,
          lineStyle: { width: 2, color: s.color, type: s.dashed ? "dashed" : "solid" },
          itemStyle: { color: s.color }
        };
      })
    });
  }

  /* ── small multiples: one ECharts instance, N stacked grids ──────────── */
  function smallMultipleOption(spec) {
    var n = spec.panels.length;
    var cols = isPhone() ? 1 : 2;
    var rows = Math.ceil(n / cols);
    var grids = [], xs = [], ys = [], series = [], titles = [];
    var gapX = 8, gapY = 13;

    spec.panels.forEach(function (p, i) {
      var r = Math.floor(i / cols), c = i % cols;
      var w = (100 - gapX * (cols + 1)) / cols;
      var h = (100 - gapY * (rows + 1)) / rows;
      var left = gapX * (c + 1) + w * c;
      var top = gapY * (r + 1) + h * r;

      grids.push({ left: left + "%", top: top + "%", width: w + "%", height: h + "%",
                   containLabel: true });
      titles.push({
        text: p.name, subtext: String(p.last),
        left: left + "%", top: (top - 9.5) + "%",
        textStyle: { fontSize: 11, fontWeight: 400, color: T2, fontFamily: SANS },
        subtextStyle: { fontSize: 12, color: INK, fontFamily: MONO }
      });
      xs.push(Object.assign({}, axisCommon, {
        type: "category", data: p.x, gridIndex: i, boundaryGap: false,
        axisLabel: { color: T3, fontSize: 9, fontFamily: MONO, hideOverlap: true,
          formatter: function (v) {
            var d = new Date(v);
            return isNaN(d) ? v : d.toLocaleDateString(undefined, { month: "short", year: "2-digit" });
          } }
      }));
      ys.push(Object.assign({}, axisCommon, { type: "value", gridIndex: i, scale: true,
                                              axisLabel: { color: T3, fontSize: 9, fontFamily: MONO } }));
      series.push({
        name: p.name, type: "line", data: p.data, xAxisIndex: i, yAxisIndex: i,
        showSymbol: false, lineStyle: { width: 1.6, color: p.color },
        areaStyle: { color: p.color, opacity: 0.07 }
      });
    });

    return Object.assign(base(), {
      tooltip: Object.assign(base().tooltip, { trigger: "axis" }),
      title: titles, grid: grids, xAxis: xs, yAxis: ys, series: series
    });
  }

  var BUILDERS = {
    line: lineOption, dual_line: lineOption, bar_h: barOption,
    diverging: barOption, heatmap: heatmapOption, curve: curveOption,
    small_multiple: smallMultipleOption
  };

  /* Height is driven by content: a 44-row bar chart needs more room than a
     4-series line, and a fixed height would either crush one or waste space. */
  function heightFor(spec) {
    var phone = isPhone();
    switch (spec.type) {
      case "bar_h":
      case "diverging":
        return Math.max(200, (spec.categories.length * (phone ? 15 : 19)) + 60);
      case "heatmap":
        return Math.max(220, (spec.yLabels.length * (phone ? 14 : 17)) + 70);
      case "small_multiple":
        return phone ? 130 * spec.panels.length : 165 * Math.ceil(spec.panels.length / 2);
      default:
        return phone ? 240 : 320;
    }
  }

  function draw(node, spec) {
    var canvas = node.querySelector(".chart-canvas");
    canvas.style.height = heightFor(spec) + "px";
    var chart = echarts.init(canvas, null, { renderer: "canvas" });
    chart.setOption(BUILDERS[spec.type](spec));
    var resize = function () {
      canvas.style.height = heightFor(spec) + "px";
      chart.resize();
    };
    var timer;
    window.addEventListener("resize", function () {
      clearTimeout(timer);
      timer = setTimeout(resize, 140);
    });
    node.dataset.drawn = "1";
    node._chart = chart;
  }

  /* Exposed for the charts page: redraws a mount with a freshly fetched spec
     (period/currency controls) without re-running the lazy-load machinery. */
  window.AlfredCharts = {
    draw: function (node, spec) {
      if (node._chart) { node._chart.dispose(); node._chart = null; }
      delete node.dataset.drawn;
      draw(node, spec);
    }
  };

  function init() {
    var packs = window.MIA_CHARTS || {};
    var nodes = Array.prototype.slice.call(document.querySelectorAll("[data-chart]"));
    if (!nodes.length) return;

    if (typeof echarts === "undefined") {
      nodes.forEach(function (n) {
        n.innerHTML = '<p class="chart-missing">Chart library unavailable offline.</p>';
      });
      return;
    }

    var render = function (node) {
      if (node.dataset.drawn) return;
      var spec = packs[node.dataset.chart];
      if (!spec) {
        node.innerHTML = '<p class="chart-missing">No data for this figure.</p>';
        node.dataset.drawn = "1";
        return;
      }
      draw(node, spec);
    };

    if (!("IntersectionObserver" in window)) {
      nodes.forEach(render);
      return;
    }
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { render(e.target); obs.unobserve(e.target); }
      });
    }, { rootMargin: "300px 0px" });
    nodes.forEach(function (n) { obs.observe(n); });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
