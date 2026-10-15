/* Phase 3 page: the vote as a label source, and self-training on it.
   Drawn from window.RESULTS (generator, verifier, self_training); see results.js. */
(function () {
    'use strict';

    var Rz = window.Results;
    var USABLE_PARSE = 0.9, ALWAYS_APPROVES = 0.97, NEVER_APPROVES = 0.05;

    function onReady(fn) {
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
        else fn();
    }

    function median(values) {
        var v = values.filter(function (x) { return x != null && isFinite(x); }).sort(function (a, b) { return a - b; });
        if (!v.length) return null;
        var m = Math.floor(v.length / 2);
        return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
    }

    function num(x, digits) { return x == null ? '–' : x.toFixed(digits == null ? 2 : digits); }

    function row(cells) {
        var tr = document.createElement('tr');
        cells.forEach(function (c) { tr.appendChild(c); });
        return tr;
    }

    function note(target, text, cls) {
        target.appendChild(Rz.el('p', cls || 'muted small', text));
    }

    /* ================================================================ Part 1: labels */

    function renderLabelTable(target, datasetId) {
        target.textContent = '';
        var runs = Rz.data.generator.filter(function (r) { return r.dataset === datasetId; });
        var t = Rz.table([
            { label: 'Precision' },
            { label: 'Samples', num: true, title: 'Answers drawn per question (N)' },
            { label: 'Questions', num: true },
            { label: 'Labelled', better: 'up', num: true, title: 'Share of questions where at least 75% of all samples agree' },
            { label: 'Label accuracy', better: 'up', num: true, title: 'Share of those labels that equal the reference answer' },
            { label: 'Δ labelled vs 16-bit', better: 'up', num: true, title: 'Change in points, 95% paired interval' },
            { label: 'Voted', better: 'up', num: true },
            { label: 'Single answer', better: 'up', num: true },
            { label: 'Note', title: 'Why a run is not compared with 16-bit' }
        ]);
        var body = t.tBodies[0], lastGroup = null;
        runs.forEach(function (r) {
            var group = Rz.modelLabel(r.model);
            if (group !== lastGroup) { body.appendChild(Rz.groupRow(group, 9)); lastGroup = group; }
            body.appendChild(row([
                Rz.cell(Rz.precisionLabel(r.precision)),
                Rz.cell(String(r.n_samples), 'num'),
                Rz.cell(String(r.n_questions), 'num'),
                Rz.cell(Rz.pct(r.label_share), 'num'),
                Rz.cell(Rz.pct(r.label_accuracy), 'num'),
                r.precision === '16bit' ? Rz.cell('baseline', 'num muted')
                    : Rz.changeCell(r.delta && r.delta.label_share, { goodWhen: 'up', n: r.delta && r.delta.n_paired }),
                Rz.cell(Rz.pct(r.accuracy), 'num'),
                Rz.cell(Rz.pct(r.sample_accuracy), 'num'),
                Rz.cell(r.delta_note || '', 'small muted')
            ]));
        });
        target.appendChild(Rz.wrapped(t));
    }

    function renderLabelChooser(chips, target) {
        var datasets = Rz.datasetsWith(Rz.data.generator);
        if (!datasets.length) return;
        function select(id) {
            Array.prototype.forEach.call(chips.querySelectorAll('.chip'), function (b) {
                b.setAttribute('aria-pressed', b.getAttribute('data-id') === id ? 'true' : 'false');
            });
            renderLabelTable(target, id);
        }
        datasets.forEach(function (d) {
            var b = Rz.el('button', 'chip', d.label);
            b.type = 'button';
            b.setAttribute('data-id', d.id);
            b.addEventListener('click', function () { select(d.id); });
            chips.appendChild(b);
        });
        select(datasets[0].id);
    }

    function unusable(r) {
        var why = [];
        if (r.parse_coverage < USABLE_PARSE) why.push(Rz.pct(r.parse_coverage, 0) + ' of verdicts parsed');
        if (r.approval_rate >= ALWAYS_APPROVES) why.push('approves almost everything');
        if (r.approval_rate <= NEVER_APPROVES) why.push('approves almost nothing');
        return why;
    }

    function renderVerifier(summaryTarget, tableTarget) {
        var rows = Rz.data.verifier.filter(function (r) { return r.vote_vs_verifier; });
        if (!rows.length) { note(tableTarget, 'No verifier runs in the export.'); return; }
        var usable = rows.filter(function (r) { return !unusable(r).length; });

        var pick = function (key) { return usable.map(function (r) { return r.vote_vs_verifier[key]; }); };
        var bothBeats = usable.filter(function (r) { return r.vote_vs_verifier.both_auroc > r.vote_vs_verifier.vote_auroc; }).length;
        var bothWorse = usable.filter(function (r) { return r.vote_vs_verifier.both_auroc < r.vote_vs_verifier.vote_auroc; }).length;
        var s = Rz.table([
            { label: 'Score' }, { label: 'Median AUROC', better: 'up', num: true }, { label: 'Median label accuracy at the vote\'s number of labels', better: 'up', num: true }
        ]);
        [['Vote (agreement)', 'vote'], ['Verifier (approval)', 'verifier'], ['Both (product)', 'both']].forEach(function (p) {
            s.tBodies[0].appendChild(row([
                Rz.cell(p[0]),
                Rz.cell(num(median(pick(p[1] + '_auroc'))), 'num'),
                Rz.cell(Rz.pct(median(pick(p[1] + '_label_accuracy'))), 'num')
            ]));
        });
        summaryTarget.appendChild(Rz.el('p', null, 'Across the ' + usable.length + ' usable rows of ' + rows.length +
            ', multiplying in the verifier raised AUROC over the vote alone in ' + bothBeats + ' and lowered it in ' + bothWorse + '.'));
        summaryTarget.appendChild(Rz.wrapped(s));

        var t = Rz.table([
            { label: 'Model' }, { label: 'Precision' },
            { label: 'Labels', num: true, title: 'Questions the vote labels; every score is judged on this many' },
            { label: 'AUROC vote', better: 'up', num: true }, { label: 'AUROC verifier', better: 'up', num: true }, { label: 'AUROC both', better: 'up', num: true },
            { label: 'Label acc. vote', better: 'up', num: true }, { label: 'Label acc. verifier', better: 'up', num: true }, { label: 'Label acc. both', better: 'up', num: true },
            { label: 'Approval', num: true, title: 'Share of all solutions the verifier approves' },
            { label: 'Note' }
        ]);
        var body = t.tBodies[0], lastGroup = null;
        rows.forEach(function (r) {
            var v = r.vote_vs_verifier, why = unusable(r);
            var group = Rz.datasetLabel(r.dataset) + ', ' + r.n_samples + ' samples, ' + r.n_questions + ' questions';
            if (group !== lastGroup) { body.appendChild(Rz.groupRow(group, 11)); lastGroup = group; }
            var tr = row([
                Rz.cell(Rz.modelLabel(r.model)), Rz.cell(Rz.precisionLabel(r.precision)),
                Rz.cell(String(v.labels_k), 'num'),
                Rz.cell(num(v.vote_auroc), 'num'), Rz.cell(num(v.verifier_auroc), 'num'), Rz.cell(num(v.both_auroc), 'num'),
                Rz.cell(Rz.pct(v.vote_label_accuracy), 'num'), Rz.cell(Rz.pct(v.verifier_label_accuracy), 'num'),
                Rz.cell(Rz.pct(v.both_label_accuracy), 'num'),
                Rz.cell(Rz.pct(r.approval_rate, 0), 'num'),
                Rz.cell(why.length ? 'unusable: ' + why.join(', ') : '', 'small muted')
            ]);
            if (why.length) tr.className = 'muted';
            body.appendChild(tr);
        });
        tableTarget.appendChild(Rz.wrapped(t));
    }

    /* ================================================================ Part 2: self-training */

    var STATUS = { 'synced': 'green', 'in progress': 'orange', 'not synced': '' };

    /* "synced" only means no file is half-written: say which rounds have test results so far */
    function badge(e) {
        var rounds = (e.test || []).filter(function (r) { return r.round != null; }).map(function (r) { return r.round; });
        var span = rounds.length ? 'rounds ' + Math.min.apply(null, rounds) + '–' + Math.max.apply(null, rounds) : '';
        var text = e.status === 'not synced' ? 'Not synced yet'
            : (e.status === 'in progress' ? 'Still running' : 'Synced') + (span ? ': ' + span + ' tested' : '');
        return Rz.el('span', 'badge ' + (STATUS[e.status] || ''), text);
    }

    function changeOrDash(change, key, goodWhen) {
        if (!change) return Rz.cell('reference', 'num muted');
        return Rz.changeCell(change[key], { goodWhen: goodWhen, n: change.n_paired });
    }

    function testTable(e) {
        var t = Rz.table([
            { label: 'Run' },
            { label: 'Batch · GPU', title: 'Batch size every question ran at, and the GPU' },
            { label: 'Single answer', better: 'up', num: true }, { label: 'Δ single', better: 'up', num: true },
            { label: 'Cut off', better: 'down', num: true }, { label: 'Δ cut off', better: 'down', num: true },
            { label: 'Right among finished', better: 'up', num: true, title: 'Pooled over questions' },
            { label: 'Δ same questions', better: 'up', num: true, title: 'Right among finished, compared question by question' },
            { label: 'Voted', better: 'up', num: true }, { label: 'Δ voted', better: 'up', num: true },
            { label: 'Voted lost / gained', num: true, title: 'Questions whose voted answer went from right to wrong / wrong to right' },
            { label: 'Confident', num: true, title: 'Share of test questions where at least 75% of samples agree' },
            { label: 'No finished answer', better: 'down', num: true, title: 'Questions where every sample was cut off' },
            { label: 'Tokens', num: true, title: 'Mean generated tokens per answer' }
        ]);
        e.test.forEach(function (r) {
            var c = r.change;
            var same = c && c.finished_accuracy_same_questions;
            t.tBodies[0].appendChild(row([
                Rz.cell(r.name),
                Rz.cell((r.batches || []).join(', ') + ' · ' + (r.gpu || 'unknown'), 'small'),
                Rz.cell(Rz.pct(r.single), 'num'), changeOrDash(c, 'single', 'up'),
                Rz.cell(Rz.pct(r.cutoff), 'num'), changeOrDash(c, 'cutoff', 'down'),
                Rz.cell(Rz.pct(r.finished_accuracy), 'num'),
                c ? Rz.changeCell(same, { goodWhen: 'up', n: same && same.n }) : Rz.cell('reference', 'num muted'),
                Rz.cell(Rz.pct(r.voted), 'num'), changeOrDash(c, 'voted', 'up'),
                Rz.cell(c ? c.voted_lost + ' / ' + c.voted_gained : '–', 'num'),
                Rz.cell(Rz.pct(r.confident), 'num'),
                Rz.cell(String(r.unfinished_questions), 'num'),
                Rz.cell(num(r.tokens, 0), 'num')
            ]));
        });
        return t;
    }

    function labelTable(e) {
        var dpo = e.labels.some(function (l) { return l.objective === 'dpo'; });
        var cols = [
            { label: 'Pool answered by' }, { label: 'Pool questions', num: true }, { label: 'Threshold', num: true },
            { label: dpo ? 'Pairs' : 'Labelled', better: 'up', num: true }, { label: 'Label accuracy', better: 'up', num: true, title: 'Checked against the train answers after the run' },
            { label: 'Wrong labels', better: 'down', num: true }, { label: dpo ? 'Training pairs' : 'Training solutions', num: true }
        ];
        if (dpo) cols.push({ label: 'Rejected answer was right', better: 'down', num: true, title: 'Pairs whose minority answer was in fact the right one' });
        cols.push({ label: 'Next round trained', title: 'Objective, updates and learning rate of the adapter trained on these labels' },
                  { label: 'Loss first → last', num: true });
        var t = Rz.table(cols);
        e.labels.forEach(function (l) {
            var tr = l.training;
            var cells = [
                Rz.cell('round ' + l.round), Rz.cell(String(l.pool_questions), 'num'),
                Rz.cell((l.threshold * 8).toFixed(0) + ' of 8', 'num'),
                Rz.cell(String(l.labelled), 'num'), Rz.cell(Rz.pct(l.label_accuracy), 'num'),
                Rz.cell(String(l.wrong_labels), 'num'), Rz.cell(String(l.items), 'num')
            ];
            if (dpo) cells.push(Rz.cell(Rz.pct(l.rejected_right), 'num'));
            cells.push(Rz.cell(tr ? tr.objective.toUpperCase() + ', ' + tr.steps + ' steps' + (tr.learning_rate ? ', learning rate ' + tr.learning_rate.toExponential(0) : '') : 'not yet', tr ? 'small' : 'small muted'),
                       Rz.cell(tr && tr.loss_first != null ? num(tr.loss_first, 3) + ' → ' + num(tr.loss_last, 3) : '–', 'num'));
            t.tBodies[0].appendChild(row(cells));
        });
        return t;
    }

    function renderExperiments(target, part) {
        var exps = ((Rz.data.self_training || {}).experiments || []).filter(function (e) { return e.part === part; });
        exps.forEach(function (e) {
            var h = Rz.el('h3', null, e.label + ' ');
            h.appendChild(badge(e));
            target.appendChild(h);
            if (e.note) note(target, e.note);
            if (e.status === 'not synced') {
                note(target, 'No files for this experiment in results/self_training/' + e.id + '/ yet. It appears here after the next export once Drive has synced.', 'callout warning');
                return;
            }
            (e.in_progress || []).forEach(function (p) {
                note(target, 'Still running: ' + p.stage + ', ' + p.done + ' of ' + p.total + ' questions. Left out until complete.', 'callout');
            });
            if (e.test && e.test.length) {
                note(target, 'Test: ' + e.test_questions + ' questions, every row against ' + e.reference + '.');
                target.appendChild(Rz.wrapped(testTable(e)));
            }
            if (e.labels && e.labels.length) target.appendChild(Rz.wrapped(labelTable(e)));
        });
    }

    function renderComparison(target, label) {
        var c = ((Rz.data.self_training || {}).comparisons || []).filter(function (x) { return x.label === label; })[0];
        if (!c || (!c.before && !c.rounds.length)) {
            note(target, 'Appears once both arms have a synced untrained reference and at least one trained round.', 'muted small');
            return;
        }
        var t = Rz.table([
            { label: 'Round' }, { label: 'Questions', num: true },
            { label: 'Δ16 − Δ4, single', num: true, title: '16-bit change minus 4-bit change; negative means 16-bit gained less' },
            { label: 'Δ16 − Δ4, voted', num: true },
            { label: 'Untrained 16-bit minus 4-bit, single', better: 'down', num: true, title: 'Gap left between the 4-bit model of this round and the untrained 16-bit model' },
            { label: 'Untrained 16-bit minus 4-bit, voted', better: 'down', num: true }
        ]);
        if (c.before) {
            t.tBodies[0].appendChild(row([
                Rz.cell('before training'), Rz.cell(String(c.before.n_paired), 'num'),
                Rz.cell('–', 'num muted'), Rz.cell('–', 'num muted'),
                Rz.changeCell(c.before.single, { goodWhen: 'down', n: c.before.n_paired }),
                Rz.changeCell(c.before.voted, { goodWhen: 'down', n: c.before.n_paired })
            ]));
        }
        c.rounds.forEach(function (r) {
            t.tBodies[0].appendChild(row([
                Rz.cell('round ' + r.round), Rz.cell(String(r.n_paired), 'num'),
                Rz.changeCell(r.difference_single, { n: r.n_paired }), Rz.changeCell(r.difference_voted, { n: r.n_paired }),
                Rz.changeCell(r.gap_single, { goodWhen: 'down', n: r.n_paired }), Rz.changeCell(r.gap_voted, { goodWhen: 'down', n: r.n_paired })
            ]));
        });
        target.appendChild(Rz.wrapped(t));
        note(target, 'Difference columns: green means the 16-bit model gained clearly more than the 4-bit model, red clearly less, grey that the two cannot be told apart. ' +
            'Gap columns: red means the 4-bit model is still clearly behind the untrained 16-bit model, grey that the remaining gap cannot be told apart from zero.');
    }

    function renderControls(target) {
        var c = (Rz.data.self_training || {}).controls;
        if (!c || !c.rows.length) { note(target, 'No untrained control runs in the export.'); return; }
        var t = Rz.table([
            { label: 'Untrained 4-bit run' }, { label: 'GPU' }, { label: 'Batch', num: true }, { label: 'Token limit', num: true },
            { label: 'Single answer', better: 'up', num: true }, { label: 'Cut off', better: 'down', num: true }, { label: 'Right among finished', better: 'up', num: true }, { label: 'Voted', better: 'up', num: true }
        ]);
        c.rows.forEach(function (r) {
            t.tBodies[0].appendChild(row([
                Rz.cell(r.name), Rz.cell(r.gpu || 'unknown', 'small'),
                Rz.cell((r.batches || []).join(', '), 'num'), Rz.cell(String(r.max_new_tokens), 'num'),
                Rz.cell(Rz.pct(r.single), 'num'), Rz.cell(Rz.pct(r.cutoff), 'num'),
                Rz.cell(Rz.pct(r.finished_accuracy), 'num'), Rz.cell(Rz.pct(r.voted), 'num')
            ]));
        });
        note(target, c.questions + ' test questions shared by all rows.');
        target.appendChild(Rz.wrapped(t));
    }

    onReady(function () {
        if (!Rz || !Rz.has()) { Rz && Rz.showMissing(document.getElementById('results-root')); return; }
        document.getElementById('source-line').textContent = Rz.sourceLine('generator');
        renderLabelChooser(document.getElementById('label-chips'), document.getElementById('label-table'));
        renderVerifier(document.getElementById('verifier-summary'), document.getElementById('verifier-table'));
        renderExperiments(document.getElementById('main-runs'), 'main');
        renderComparison(document.getElementById('main-comparison'), 'Main run');
        renderExperiments(document.getElementById('pilot-runs'), 'pilot');
        renderComparison(document.getElementById('pilot-comparison'), 'Pilot');
        renderControls(document.getElementById('controls'));
        renderExperiments(document.getElementById('dpo-runs'), 'dpo');
    });
})();
