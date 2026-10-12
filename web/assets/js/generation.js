/* Phase 1 page: generator results drawn from window.RESULTS.generator (see results.js). */
(function () {
    'use strict';

    var Rz = window.Results;
    var HIGH_TRUNCATION = 0.05;
    var MIN_PAIRED = 50;     // a cell needs at least this many shared questions to enter the averages
    var BASELINE = '16bit';
    var COMPARED = ['8bit', '4bit'];

    function onReady(fn) {
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
        else fn();
    }

    function notes(run) {
        var out = [];
        if (run.expected_questions && run.n_questions < run.expected_questions) {
            out.push('incomplete run: ' + run.n_questions + ' of ' + run.expected_questions + ' questions');
        }
        if (run.max_new_tokens == null) out.push('token limit not recorded');
        if (run.delta_note) out.push(run.delta_note);
        return out;
    }

    function inAverages(run) { return run.delta && run.delta.n_paired >= MIN_PAIRED; }

    function whyNot(run) {
        return run.delta ? 'only ' + run.delta.n_paired + ' questions shared with the 16-bit run' : run.delta_note;
    }

    /* ---------- at a glance ---------- */
    function renderGlance(target) {
        var runs = Rz.data.generator;
        var t = Rz.table([
            { label: 'Dataset' },
            { label: 'Precision' },
            { label: 'Cells', num: true, title: 'Model and sample count pairs that have a 16-bit run on the same token limit' },
            { label: 'Voted accuracy', num: true, title: 'Mean change in points over the cells' },
            { label: 'Single-sample accuracy', num: true, title: 'Mean change in points over the cells' },
            { label: 'Range (single sample)', num: true },
            { label: 'Intervals off zero', num: true, title: 'Cells where the 95% interval of the single-sample change excludes zero' }
        ]);
        var body = t.tBodies[0];
        var any = false;

        Rz.datasetsWith(runs).forEach(function (d) {
            COMPARED.forEach(function (p) {
                var cells = runs.filter(function (r) { return r.dataset === d.id && r.precision === p && inAverages(r); });
                if (!cells.length) return;
                any = true;
                var single = cells.map(function (r) { return r.delta.sample_accuracy.mean; });
                var off = cells.filter(function (r) { return Rz.side(r.delta.sample_accuracy) !== 0; }).length;
                var tr = document.createElement('tr');
                tr.appendChild(Rz.cell(d.label));
                tr.appendChild(Rz.cell(Rz.precisionLabel(p)));
                tr.appendChild(Rz.cell(String(cells.length), 'num'));
                tr.appendChild(Rz.cell(Rz.points(Rz.mean(cells.map(function (r) { return r.delta.accuracy.mean; }))), 'num'));
                tr.appendChild(Rz.cell(Rz.points(Rz.mean(single)), 'num'));
                tr.appendChild(Rz.cell(Rz.points(Math.min.apply(null, single)) + ' to ' + Rz.points(Math.max.apply(null, single)), 'num'));
                tr.appendChild(Rz.cell(off + ' of ' + cells.length, 'num'));
                body.appendChild(tr);
            });
        });

        target.textContent = '';
        if (!any) { target.appendChild(Rz.el('p', 'muted', 'No quantized run has a comparable 16-bit run yet.')); return; }
        target.appendChild(Rz.wrapped(t));

        var skipped = runs.filter(function (r) { return r.precision !== BASELINE && !inAverages(r); });
        if (skipped.length) {
            var box = Rz.el('div', 'callout warning');
            box.appendChild(Rz.el('h3', null, skipped.length + ' quantized run' + (skipped.length > 1 ? 's are' : ' is') + ' not in these averages'));
            var ul = Rz.el('ul');
            skipped.forEach(function (r) {
                ul.appendChild(Rz.el('li', null, Rz.modelLabel(r.model) + ', ' + Rz.precisionLabel(r.precision) + ', ' +
                    Rz.datasetLabel(r.dataset) + ', N=' + r.n_samples + ': ' + whyNot(r)));
            });
            box.appendChild(ul);
            target.appendChild(box);
        }
    }

    /* ---------- results by dataset ---------- */
    var COLUMNS = [
        { label: 'Precision' },
        { label: 'N', num: true, title: 'Samples per question' },
        { label: 'Questions', num: true },
        { label: 'Voted accuracy', num: true, title: 'Majority answer is correct' },
        { label: 'Single-sample accuracy', num: true, title: 'Share of all samples that finished and are correct' },
        { label: 'Pass@N', num: true, title: 'At least one of the N samples is correct' },
        { label: 'Agreement', num: true, title: 'Mean share of samples that gave the majority answer' },
        { label: 'Diversity', num: true, title: 'Mean number of distinct answers' },
        { label: 'Truncated', num: true, title: 'Share of samples that hit the token limit (they never vote)' },
        { label: 'Δ voted', num: true, title: 'Change against 16-bit, points, 95% interval' },
        { label: 'Δ single sample', num: true, title: 'Change against 16-bit, points, 95% interval' },
        { label: 'Δ pass@N', num: true, title: 'Change against 16-bit, points, 95% interval' },
        { label: 'Notes' }
    ];

    function renderDataset(target, datasetId) {
        var runs = Rz.data.generator.filter(function (r) { return r.dataset === datasetId; });
        var t = Rz.table(COLUMNS);
        var body = t.tBodies[0];
        var model = null;

        runs.forEach(function (r) {
            if (r.model !== model) {
                model = r.model;
                body.appendChild(Rz.groupRow(Rz.modelLabel(model), COLUMNS.length));
            }
            var tr = document.createElement('tr');
            tr.appendChild(Rz.cell(Rz.precisionLabel(r.precision)));
            tr.appendChild(Rz.cell(String(r.n_samples), 'num'));
            tr.appendChild(Rz.cell(String(r.n_questions), 'num'));
            tr.appendChild(Rz.cell(Rz.pct(r.accuracy), 'num'));
            tr.appendChild(Rz.cell(Rz.pct(r.sample_accuracy), 'num'));
            tr.appendChild(Rz.cell(Rz.pct(r.pass_at_k), 'num'));
            tr.appendChild(Rz.cell(Rz.pct(r.confidence), 'num'));
            tr.appendChild(Rz.cell(r.diversity.toFixed(2), 'num'));
            tr.appendChild(Rz.cell(Rz.pct(r.cutoff_rate), r.cutoff_rate > HIGH_TRUNCATION ? 'num negative' : 'num'));
            var d = r.delta;
            tr.appendChild(Rz.changeCell(d && d.accuracy, { n: d && d.n_paired }));
            tr.appendChild(Rz.changeCell(d && d.sample_accuracy, { n: d && d.n_paired }));
            tr.appendChild(Rz.changeCell(d && d.pass_at_k, { n: d && d.n_paired }));
            var n = notes(r);
            if (d && d.n_paired !== r.n_questions) n.unshift('compared on ' + d.n_paired + ' shared questions');
            tr.appendChild(Rz.cell(n.join('; '), 'small muted'));
            body.appendChild(tr);
        });

        target.textContent = '';
        target.appendChild(Rz.wrapped(t));
    }

    function renderDatasetChooser(chips, target) {
        var datasets = Rz.datasetsWith(Rz.data.generator);
        var current = datasets.length ? datasets[0].id : null;
        function show(id) {
            current = id;
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
        if (current) show(current);
    }

    /* ---------- calibration ---------- */
    function renderCalibration(controls, target, legend) {
        var runs = Rz.data.generator.filter(function (r) { return r.calibration && r.calibration.length; });
        var selects = {};
        ['dataset', 'model', 'n'].forEach(function (key) {
            var field = Rz.el('div', 'field');
            var label = Rz.el('label', null, { dataset: 'Dataset', model: 'Model', n: 'Samples per question' }[key]);
            var select = Rz.el('select', 'select');
            select.id = 'cal-' + key;
            label.htmlFor = select.id;
            field.appendChild(label);
            field.appendChild(select);
            controls.appendChild(field);
            selects[key] = select;
        });

        function fill(select, options, keep) {
            select.textContent = '';
            options.forEach(function (o) {
                var opt = document.createElement('option');
                opt.value = o.value; opt.textContent = o.label;
                select.appendChild(opt);
            });
            if (keep && options.some(function (o) { return String(o.value) === String(keep); })) select.value = keep;
        }

        function refresh(changed) {
            if (changed === 'init' || changed === 'dataset') {
                var ds = Rz.datasetsWith(runs).map(function (d) { return { value: d.id, label: d.label }; });
                if (changed === 'init') fill(selects.dataset, ds);
                var models = {};
                runs.filter(function (r) { return r.dataset === selects.dataset.value; }).forEach(function (r) { models[r.model] = true; });
                fill(selects.model, Object.keys(models).map(function (m) { return { value: m, label: Rz.modelLabel(m) }; }), selects.model.value);
            }
            if (changed !== 'n') {
                var ns = {};
                runs.filter(function (r) { return r.dataset === selects.dataset.value && r.model === selects.model.value; })
                    .forEach(function (r) { ns[r.n_samples] = true; });
                fill(selects.n, Object.keys(ns).sort(function (a, b) { return a - b; })
                    .map(function (n) { return { value: n, label: 'N = ' + n }; }), selects.n.value);
            }
            draw();
        }

        function draw() {
            var chosen = runs.filter(function (r) {
                return r.dataset === selects.dataset.value && r.model === selects.model.value && String(r.n_samples) === String(selects.n.value);
            });
            var series = chosen.map(function (r) {
                return { precision: r.precision, label: Rz.precisionLabel(r.precision), points: r.calibration, run: r };
            });
            target.textContent = '';
            target.appendChild(window.Charts.calibrationChart(series));

            legend.textContent = '';
            series.forEach(function (s) {
                var li = Rz.el('li');
                var swatch = Rz.el('span', 'swatch series-' + s.precision);
                li.appendChild(swatch);
                li.appendChild(document.createTextNode(s.label + ': voted accuracy ' + Rz.pct(s.run.accuracy) + ' on ' + s.run.n_questions + ' questions'));
                legend.appendChild(li);
            });
        }

        selects.dataset.addEventListener('change', function () { refresh('dataset'); });
        selects.model.addEventListener('change', function () { refresh('model'); });
        selects.n.addEventListener('change', function () { refresh('n'); });
        if (runs.length) refresh('init');
    }

    onReady(function () {
        var root = document.getElementById('results-root');
        if (!root) return;
        if (!Rz) { root.textContent = 'The results helpers (assets/js/results.js) did not load.'; return; }
        if (!Rz.has()) { Rz.showMissing(root); return; }
        document.getElementById('source-line').textContent = Rz.sourceLine('generator');
        renderGlance(document.getElementById('glance'));
        renderDatasetChooser(document.getElementById('dataset-chips'), document.getElementById('dataset-table'));
        renderCalibration(document.getElementById('cal-controls'), document.getElementById('cal-chart'), document.getElementById('cal-legend'));
    });
})();
