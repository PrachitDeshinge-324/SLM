/* Calibration chart: accuracy of the majority answer against how strongly the samples agree.
   Plain SVG so the page needs no charting library. One line per precision of one model and dataset. */
(function () {
    'use strict';

    var NS = 'http://www.w3.org/2000/svg';
    var W = 520, H = 420, M = { left: 52, right: 16, top: 16, bottom: 46 };

    function svg(tag, attrs, text) {
        var n = document.createElementNS(NS, tag);
        Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
        if (text != null) n.textContent = text;
        return n;
    }

    /* series: [{ precision, label, points: [{x, accuracy, count}], note }] */
    function calibrationChart(series) {
        var root = svg('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'class': 'chart',
            'aria-label': 'Calibration: accuracy of the majority answer by agreement' });
        var x = function (v) { return M.left + v * (W - M.left - M.right); };
        var y = function (v) { return H - M.bottom - v * (H - M.top - M.bottom); };

        for (var t = 0; t <= 10; t += 2) {
            var v = t / 10;
            root.appendChild(svg('line', { x1: x(0), x2: x(1), y1: y(v), y2: y(v), 'class': 'chart-grid' }));
            root.appendChild(svg('text', { x: M.left - 8, y: y(v) + 4, 'text-anchor': 'end', 'class': 'chart-tick' }, Math.round(v * 100) + '%'));
            root.appendChild(svg('text', { x: x(v), y: H - M.bottom + 18, 'text-anchor': 'middle', 'class': 'chart-tick' }, v.toFixed(1)));
        }
        root.appendChild(svg('line', { x1: x(0), y1: y(0), x2: x(1), y2: y(1), 'class': 'chart-diagonal' }));
        root.appendChild(svg('text', { x: (M.left + W - M.right) / 2, y: H - 8, 'text-anchor': 'middle', 'class': 'chart-axis' },
            'Agreement among the samples (share that gave the majority answer)'));
        var yl = svg('text', { transform: 'translate(14 ' + (H - M.bottom) / 2 + ') rotate(-90)', 'text-anchor': 'middle', 'class': 'chart-axis' },
            'Accuracy of the majority answer');
        root.appendChild(yl);

        series.forEach(function (s) {
            var cls = 'series-' + s.precision;
            if (s.points.length > 1) {
                var d = s.points.map(function (p, i) { return (i ? 'L' : 'M') + x(p.x).toFixed(1) + ' ' + y(p.accuracy).toFixed(1); }).join(' ');
                root.appendChild(svg('path', { d: d, 'class': 'chart-line ' + cls }));
            }
            s.points.forEach(function (p) {
                var dot = svg('circle', { cx: x(p.x), cy: y(p.accuracy), r: 4, 'class': 'chart-dot ' + cls });
                dot.appendChild(svg('title', {}, s.label + ': agreement up to ' + p.x.toFixed(1) + ', accuracy ' +
                    (p.accuracy * 100).toFixed(1) + '% (' + p.count + ' questions)'));
                root.appendChild(dot);
            });
        });
        return root;
    }

    window.Charts = { calibrationChart: calibrationChart };
})();
