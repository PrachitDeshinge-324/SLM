/* Phase 2 page: verifier results drawn from window.RESULTS.verifier (see results.js). */
(function () {
    'use strict';

    var Rz = window.Results;
    var COMPARED = ['8bit', '4bit'];
    var ALWAYS_APPROVES = 0.95;
    var NEVER_APPROVES = 0.05;

    function onReady(fn) {
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
        else fn();
    }

    function notes(r) {
        var out = [];
        if (r.approval_rate >= ALWAYS_APPROVES) out.push('approves almost every trace');
        if (r.approval_rate <= NEVER_APPROVES) out.push('rejects almost every trace');
        if (r.delta_note) out.push(r.delta_note);
        return out;
    }

    /* the export pairs a row with its 16-bit run only when the two can be compared */
    function comparable(row) {
        return !!row.delta;
    }

    /* ---------- what every row covers ---------- */
    function renderScope(target) {
        var rows = Rz.data.verifier;
        var questions = Rz.data.verifier_questions;
        if (!target || !rows.length || !questions) return;
        var sizes = {};
        rows.forEach(function (r) { sizes[r.n_samples] = r.total_traces; });
        var parts = Object.keys(sizes).map(function (n) {
            return questions + ' × ' + n + ' solutions = ' + sizes[n] + ' traces';
        });
        target.textContent = 'Every row is scored on the same ' + questions + ' questions of its model, and on every solution to them: '
            + parts.join(', or ') + ' in each row.';
    }

    function renderLeftOut(target) {
        var runs = Rz.data.verifier_left_out || [];
        if (!target || !runs.length) return;
        var groups = {};
        runs.forEach(function (r) {
            var key = Rz.modelLabel(r.model) + ' on ' + Rz.datasetLabel(r.dataset) + ' with ' + r.n_samples + ' solutions per question';
            (groups[key] = groups[key] || []).push(Rz.precisionLabel(r.precision) + ': ' + r.n_questions);
        });
        target.textContent = 'Left out because they hold fewer than ' + Rz.data.verifier_questions + ' complete questions: '
            + Object.keys(groups).map(function (key) { return key + ' (' + groups[key].join(', ') + ')'; }).join('; ') + '.';
    }

    /* ---------- at a glance ---------- */
    function renderGlance(target) {
        var rows = Rz.data.verifier;
        var t = Rz.table([
            { label: 'Dataset' },
            { label: 'Precision' },
            { label: 'Cells', num: true, title: 'Models with a comparable 16-bit run' },
            { label: 'Δ voted accuracy', num: true, title: 'Generator majority, mean change in points' },
            { label: 'Δ guided accuracy', num: true, title: 'Verifier-filtered majority, mean change in points' },
            { label: 'Δ false positive rate', num: true, title: 'Mean change in points; a rise means more wrong traces approved' },
            { label: 'Δ MCC', num: true, title: 'Mean change; a fall means worse discrimination' },
            { label: 'FPR intervals off zero', num: true, title: 'Cells where the 95% interval of the FPR change excludes zero' }
        ]);
        var body = t.tBodies[0];
        var any = false;

        Rz.datasetsWith(rows).forEach(function (d) {
            COMPARED.forEach(function (p) {
                var cells = rows.filter(function (r) { return r.dataset === d.id && r.precision === p && comparable(r); });
                if (!cells.length) return;
                any = true;
                var pick = function (key) { return Rz.mean(cells.map(function (r) { return r.delta[key].mean; })); };
                var off = cells.filter(function (r) { return Rz.side(r.delta.trace_fpr) !== 0; }).length;
                var tr = document.createElement('tr');
                tr.appendChild(Rz.cell(d.label));
                tr.appendChild(Rz.cell(Rz.precisionLabel(p)));
                tr.appendChild(Rz.cell(String(cells.length), 'num'));
                tr.appendChild(Rz.cell(Rz.points(pick('gen_maj_acc')), 'num'));
                tr.appendChild(Rz.cell(Rz.points(pick('ver_maj_acc')), 'num'));
                tr.appendChild(Rz.cell(Rz.points(pick('trace_fpr')), 'num'));
                tr.appendChild(Rz.cell(Rz.signed(pick('mcc')), 'num'));
                tr.appendChild(Rz.cell(off + ' of ' + cells.length, 'num'));
                body.appendChild(tr);
            });
        });

        target.textContent = '';
        if (!any) { target.appendChild(Rz.el('p', 'muted', 'No quantized verifier run has a comparable 16-bit run yet.')); return; }
        target.appendChild(Rz.wrapped(t));
    }

    /* ---------- results by dataset ---------- */
    var COLUMNS = [
        { label: 'Precision' },
        { label: 'N', num: true, title: 'Solutions per question' },
        { label: 'Traces judged', num: true, title: 'Every solution to the questions of this row' },
        { label: 'Cut off', num: true, title: 'Solutions that hit the token limit: counted as wrong, and they do not vote' },
        { label: 'No verdict', num: true, title: 'Traces where the verifier\'s reply held no verdict; the rates leave them out' },
        { label: 'Voted accuracy', num: true, title: 'Generator majority answer is correct' },
        { label: 'Guided accuracy', num: true, title: 'Majority among the traces the verifier approved' },
        { label: 'Guided − voted', num: true, title: 'Selection utility, points' },
        { label: 'Approval rate', num: true, title: 'Share of traces the verifier approved' },
        { label: 'False positive rate', num: true, title: 'Share of incorrect traces the verifier approved' },
        { label: 'MCC', num: true },
        { label: 'Balanced accuracy', num: true },
        { label: 'Δ FPR', num: true, title: 'Change against 16-bit, points, 95% interval' },
        { label: 'Δ MCC', num: true, title: 'Change against 16-bit, 95% interval' },
        { label: 'Notes' }
    ];

    function renderDataset(target, datasetId) {
        var rows = Rz.data.verifier.filter(function (r) { return r.dataset === datasetId; });
        var t = Rz.table(COLUMNS);
        var body = t.tBodies[0];
        var model = null;

        rows.forEach(function (r) {
            if (r.model !== model) {
                model = r.model;
                body.appendChild(Rz.groupRow(Rz.modelLabel(model), COLUMNS.length));
            }
            var d = r.delta;
            var tr = document.createElement('tr');
            tr.appendChild(Rz.cell(Rz.precisionLabel(r.precision)));
            tr.appendChild(Rz.cell(String(r.n_samples), 'num'));
            tr.appendChild(Rz.cell(String(r.total_traces), 'num'));
            tr.appendChild(Rz.cell(r.truncation_unknown > 0 ? 'unknown' : String(r.truncated), 'num'));
            tr.appendChild(Rz.cell(String(r.unparseable), 'num'));
            tr.appendChild(Rz.cell(Rz.pct(r.gen_maj_acc), 'num'));
            tr.appendChild(Rz.cell(Rz.pct(r.ver_maj_acc), 'num'));
            tr.appendChild(Rz.cell(Rz.points(r.ver_minus_gen), 'num'));
            tr.appendChild(Rz.cell(Rz.pct(r.approval_rate), 'num'));
            tr.appendChild(Rz.cell(Rz.pct(r.trace_fpr), 'num'));
            tr.appendChild(Rz.cell(Rz.signed(r.mcc), 'num'));
            tr.appendChild(Rz.cell(Rz.pct(r.balanced_accuracy), 'num'));
            tr.appendChild(Rz.changeCell(d && d.trace_fpr, { goodWhen: 'down', n: d && d.n_paired }));
            tr.appendChild(Rz.changeCell(d && d.mcc, { goodWhen: 'up', format: Rz.signed, n: d && d.n_paired }));
            tr.appendChild(Rz.cell(notes(r).join('; '), 'small muted'));
            body.appendChild(tr);
        });

        target.textContent = '';
        target.appendChild(Rz.wrapped(t));
    }

    function renderDatasetChooser(chips, target) {
        var datasets = Rz.datasetsWith(Rz.data.verifier);
        function show(id) {
            Array.prototype.forEach.call(chips.querySelectorAll('.chip'), function (b) {
                b.setAttribute('aria-pressed', b.getAttribute('data-id') === id ? 'true' : 'false');
            });
            renderDataset(target, id);
        }
        datasets.forEach(function (d) {
            var b = Rz.el('button', 'chip', d.label);
            b.type = 'button';
            b.setAttribute('data-id', d.id);
            b.addEventListener('click', function () { show(d.id); });
            chips.appendChild(b);
        });
        if (datasets.length) show(datasets[0].id);
    }

    onReady(function () {
        var root = document.getElementById('results-root');
        if (!root) return;
        if (!Rz) { root.textContent = 'The results helpers (assets/js/results.js) did not load.'; return; }
        if (!Rz.has()) { Rz.showMissing(root); return; }
        document.getElementById('source-line').textContent = Rz.sourceLine('verifier');
        renderScope(document.getElementById('scope-line'));
        renderLeftOut(document.getElementById('left-out-line'));
        renderGlance(document.getElementById('glance'));
        renderDatasetChooser(document.getElementById('dataset-chips'), document.getElementById('dataset-table'));
    });
})();
