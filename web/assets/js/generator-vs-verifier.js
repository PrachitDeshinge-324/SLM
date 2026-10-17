/* Generator vs Verifier page: one table, from window.GEN_VS_VER (scripts/export_generator_vs_verifier.py). */
(function () {
    'use strict';

    var D = window.GEN_VS_VER;
    var root = document.getElementById('gv-table');
    if (!D || !root) return;

    var DATASETS = { gsm8k: 'GSM8K', math500: 'MATH-500', ChilleD_SVAMP: 'SVAMP', cqa: 'CommonsenseQA' };
    var DATASET_ORDER = ['gsm8k', 'math500', 'ChilleD_SVAMP', 'cqa'];
    var PRECISIONS = { '16bit': '16-bit', '8bit': '8-bit', '4bit': '4-bit' };

    function el(tag, cls, text) {
        var e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text !== undefined) e.textContent = text;
        return e;
    }
    function pct(x) { return (100 * x).toFixed(1); }
    function signed(x) { var v = 100 * x; return (v > 0 ? '+' : '') + v.toFixed(1); }
    function modelLabel(id) { return id.indexOf('_') >= 0 ? id.slice(id.indexOf('_') + 1) : id; }

    function deltaCell(d) {
        var td = el('td', 'num', signed(d.mean));
        td.appendChild(el('span', 'ci', '[' + signed(d.lo) + ', ' + signed(d.hi) + ']'));
        if (d.mean > 1e-9) td.className += ' positive';
        else if (d.mean < -1e-9) td.className += ' negative';
        else td.className += ' muted';
        td.title = 'Verifier minus generator, points; 95% bootstrap interval over questions';
        return td;
    }

    var columns = [
        ['Model'], ['Precision'],
        ['Generator single', 1], ['Verifier single', 1], ['Δ single', 1],
        ['Generator voted', 1], ['Verifier voted', 1], ['Δ voted', 1],
        ['Verdicts parsed', 1]
    ];
    var table = el('table', 'dense');
    var thead = el('thead'), hr = el('tr');
    columns.forEach(function (c) { hr.appendChild(el('th', c[1] ? 'num' : null, c[0])); });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = el('tbody');

    var datasets = DATASET_ORDER.filter(function (ds) { return D.rows.some(function (r) { return r.dataset === ds; }); });
    D.rows.forEach(function (r) { if (datasets.indexOf(r.dataset) < 0) datasets.push(r.dataset); });

    datasets.forEach(function (ds) {
        var group = el('tr', 'group-row');
        var gtd = el('td', null, DATASETS[ds] || ds);
        gtd.colSpan = columns.length;
        group.appendChild(gtd);
        tbody.appendChild(group);
        D.rows.filter(function (r) { return r.dataset === ds; }).forEach(function (r) {
            var tr = el('tr');
            tr.appendChild(el('td', null, modelLabel(r.model)));
            tr.appendChild(el('td', null, PRECISIONS[r.precision] || r.precision));
            tr.appendChild(el('td', 'num', pct(r.single)));
            tr.appendChild(el('td', 'num', pct(r.ver_single)));
            tr.appendChild(deltaCell(r.delta_single));
            tr.appendChild(el('td', 'num', pct(r.voted)));
            tr.appendChild(el('td', 'num', pct(r.ver_voted)));
            tr.appendChild(deltaCell(r.delta_voted));
            tr.appendChild(el('td', 'num' + (r.parse_coverage < 0.8 ? ' negative' : ''), Math.round(100 * r.parse_coverage) + '%'));
            tbody.appendChild(tr);
        });
    });
    table.appendChild(tbody);

    var wrap = el('div', 'table-wrap table-scroll');
    wrap.appendChild(table);
    root.appendChild(wrap);

    var n = D.rows.length ? D.rows[0].n_questions : 0;
    document.getElementById('gv-source').textContent =
        D.rows.length + ' runs, ' + n + ' questions per model, 8 samples each. Generated ' + D.generated_at + '.';
})();
