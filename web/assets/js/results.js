/* Shared helpers for the pages that draw from window.RESULTS (assets/data/results.js).

   RESULTS is written by scripts/export_web_data.py from the run files in results/; nothing on the
   result pages is typed in by hand. This file only formats it: labels, percentages, signed changes
   with their intervals, and small DOM builders. */
(function () {
    'use strict';

    var R = window.RESULTS || { datasets: [], models: [], precisions: [], generator: [], verifier: [], source: {} };

    var labels = { datasets: {}, models: {}, precisions: {} };
    R.datasets.forEach(function (d) { labels.datasets[d.id] = d.label; });
    R.models.forEach(function (m) { labels.models[m.id] = m.label; });
    R.precisions.forEach(function (p) { labels.precisions[p.id] = p.label; });

    var MINUS = '−';

    function el(tag, cls, text) {
        var n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text != null) n.textContent = text;
        return n;
    }

    /* 0.6543 -> "65.4%" */
    function pct(x, digits) {
        if (x == null) return '–';
        return (x * 100).toFixed(digits == null ? 1 : digits) + '%';
    }

    /* 0.0321 -> "+3.2", -0.0721 -> "−7.2" (percentage points) */
    function points(x, digits) {
        if (x == null) return '–';
        var v = x * 100, d = digits == null ? 1 : digits;
        var s = Math.abs(v).toFixed(d);
        if (Number(s) === 0) return (0).toFixed(d);
        return (v < 0 ? MINUS : '+') + s;
    }

    function signed(x, digits) {
        if (x == null) return '–';
        var d = digits == null ? 3 : digits, s = Math.abs(x).toFixed(d);
        if (Number(s) === 0) return (0).toFixed(d);
        return (x < 0 ? MINUS : '+') + s;
    }

    function mean(values) {
        var v = values.filter(function (x) { return x != null && isFinite(x); });
        return v.length ? v.reduce(function (a, b) { return a + b; }, 0) / v.length : null;
    }

    /* does the 95% interval {mean, lo, hi} exclude zero, and on which side */
    function side(interval) {
        if (!interval) return 0;
        if (interval.lo > 0) return 1;
        if (interval.hi < 0) return -1;
        return 0;
    }

    /* cell for a change with its interval; `goodWhen` is 'up' or 'down': which direction is an improvement */
    function changeCell(interval, options) {
        var o = options || {};
        var td = el('td', 'num');
        if (!interval) { td.textContent = '–'; td.className += ' muted'; return td; }
        var fmt = o.format || points;
        /* the interval sits on its own small line under the mean, which keeps change columns narrow */
        td.textContent = fmt(interval.mean);
        td.appendChild(el('span', 'ci', '[' + fmt(interval.lo) + ', ' + fmt(interval.hi) + ']'));
        var s = side(interval);
        if (s !== 0) {
            var better = (o.goodWhen === 'down') ? s < 0 : s > 0;
            td.className += better ? ' positive' : ' negative';
        } else {
            td.className += ' muted';
        }
        td.title = '95% paired bootstrap interval over questions' + (o.n ? ' (' + o.n + ' questions)' : '');
        return td;
    }

    function cell(text, cls) {
        var td = el('td', cls || null, text);
        return td;
    }

    function headRow(columns) {
        var tr = el('tr');
        columns.forEach(function (c) {
            /* the label wraps inside a narrow box so a long heading does not widen a column of short numbers;
               its last word and the arrow stay together on one line */
            var th = el('th', c.num ? 'num' : null);
            var box = el('span', 'th-label');
            var words = String(c.label).split(' ');
            var tail = el('span', 'th-tail', words.pop());
            if (words.length) box.appendChild(document.createTextNode(words.join(' ') + ' '));
            box.appendChild(tail);
            th.appendChild(box);
            var hint = null;
            if (c.better) {
                /* a column's `better` is 'up' (higher is better) or 'down' (lower is better); none means neither */
                var up = c.better === 'up';
                hint = up ? 'Higher is better.' : 'Lower is better.';
                var mark = el('span', 'better-mark', up ? '↑' : '↓');
                mark.setAttribute('role', 'img');
                mark.setAttribute('aria-label', hint.slice(0, -1).toLowerCase());
                tail.appendChild(mark);
            }
            if (c.title || hint) th.title = [c.title, hint].filter(Boolean).join(' ');
            tr.appendChild(th);
        });
        return tr;
    }

    function table(columns) {
        var t = el('table', 'dense');
        var thead = el('thead');
        thead.appendChild(headRow(columns));
        t.appendChild(thead);
        t.appendChild(el('tbody'));
        return t;
    }

    function groupRow(text, span) {
        var tr = el('tr', 'group-row');
        var td = el('td', null, text);
        td.colSpan = span;
        tr.appendChild(td);
        return tr;
    }

    /* the table sits inside the same scrolling wrapper the shell gives hand-written tables */
    function wrapped(t) {
        var w = el('div', 'table-wrap table-scroll');
        w.appendChild(t);
        return w;
    }

    function datasetsWith(rows) {
        var present = {};
        rows.forEach(function (r) { present[r.dataset] = true; });
        return R.datasets.filter(function (d) { return present[d.id]; });
    }

    function sourceLine(kind) {
        var s = R.source || {};
        var n = kind === 'verifier' ? s.verifier_runs : s.generator_runs;
        var noun = kind === 'verifier' ? 'verifier rows' : 'generator runs';
        var bits = [n + ' ' + noun + ' re-scored from the run files in ' + (s.results_dir || 'results') + '/'];
        if (s.latest_generator_run) bits.push('latest generation run ' + s.latest_generator_run);
        if (R.generated_at) bits.push('exported ' + R.generated_at);
        return bits.join(' · ');
    }

    function showMissing(target) {
        target.textContent = '';
        target.appendChild(el('p', 'callout warning',
            'No results data was found. Run python scripts/export_web_data.py from the repository root to create assets/data/results.js.'));
    }

    window.Results = {
        data: R, labels: labels, el: el, pct: pct, points: points, signed: signed, mean: mean, side: side,
        changeCell: changeCell, cell: cell, table: table, groupRow: groupRow, wrapped: wrapped,
        datasetsWith: datasetsWith, sourceLine: sourceLine, showMissing: showMissing,
        has: function () { return R.generator.length + R.verifier.length > 0; },
        modelLabel: function (id) { return labels.models[id] || id.split('/').pop(); },
        datasetLabel: function (id) { return labels.datasets[id] || id; },
        precisionLabel: function (id) { return labels.precisions[id] || id; }
    };
})();
