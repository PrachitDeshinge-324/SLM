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
                Rz.cell((r.batches || []).join(', ') + ' · ' + (r.gpu || 'unknown'), 'wrap'),
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
            cells.push(Rz.cell(tr ? tr.objective.toUpperCase() + ', ' + tr.steps + ' steps' + (tr.learning_rate ? ', learning rate ' + tr.learning_rate.toExponential(0) : '') : 'not yet', tr ? null : 'muted'),
                       Rz.cell(tr && tr.loss_first != null ? num(tr.loss_first, 3) + ' → ' + num(tr.loss_last, 3) : '–', 'num'));
            t.tBodies[0].appendChild(row(cells));
        });
        return t;
    }

    function experiments(filter) {
        return ((Rz.data.self_training || {}).experiments || []).filter(filter);
    }

    function renderExperiments(target, part) {
        var exps = typeof part === 'string' ? experiments(function (e) { return e.part === part; }) : part;
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
                Rz.cell(r.name), Rz.cell(r.gpu || 'unknown'),
                Rz.cell((r.batches || []).join(', '), 'num'), Rz.cell(String(r.max_new_tokens), 'num'),
                Rz.cell(Rz.pct(r.single), 'num'), Rz.cell(Rz.pct(r.cutoff), 'num'),
                Rz.cell(Rz.pct(r.finished_accuracy), 'num'), Rz.cell(Rz.pct(r.voted), 'num')
            ]));
        });
        note(target, c.questions + ' test questions shared by all rows.');
        target.appendChild(Rz.wrapped(t));
    }

    /* ================================================================ the 16-bit main run, analysed */

    var P = Rz.points;

    function iv(x) { return x ? P(x.mean) + ' [' + P(x.lo) + ', ' + P(x.hi) + ']' : '–'; }
    function range(values) {
        var lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
        return lo === hi ? P(lo) : P(lo) + ' to ' + P(hi);
    }
    function para(target, html, cls) {
        var p = Rz.el('p', cls || null);
        p.innerHTML = html;
        target.appendChild(p);
        return p;
    }
    function finding(target, html) {
        var d = Rz.el('div', 'callout');
        d.innerHTML = html;
        target.appendChild(d);
    }

    function mainArm(precision) {
        return experiments(function (e) { return e.part === 'main' && e.precision === precision; })[0] || null;
    }
    function tested(e) { return e && e.status !== 'not synced' && e.test && e.test.length > 1; }
    function base(e) { return e.test[0]; }
    function trained(e) { return e.test.slice(1); }
    function last(list) { return list[list.length - 1]; }

    /* s = (1 - c) f, so s_k - s_0 = (c_0 - c_k) f_0 + (1 - c_k)(f_k - f_0), exactly */
    function decompose(r0, r) {
        return { finishing: (r0.cutoff - r.cutoff) * r0.finished_accuracy,
                 accuracy: (1 - r.cutoff) * (r.finished_accuracy - r0.finished_accuracy) };
    }

    /* the gain is all finishing: every finishing term positive, no same-question accuracy interval clearly above zero */
    function finishOnly(e) {
        var r0 = base(e);
        return trained(e).every(function (r) {
            return decompose(r0, r).finishing > 0 && Rz.side(r.change.finished_accuracy_same_questions) <= 0;
        });
    }
    function poolLabels(e) { return (e.labels || []).filter(function (l) { return l.label_accuracy != null; }); }
    function labelsFalling(e) {
        var L = poolLabels(e);
        return L.length > 1 && L.every(function (l, i) { return !i || l.label_accuracy < L[i - 1].label_accuracy; });
    }

    function armName(e) { return Rz.precisionLabel(e.precision); }
    function list(values, fmt) { return values.map(function (v) { return (fmt || P)(v); }).join(', '); }
    function steps(values) { return values.slice(1).map(function (v, i) { return v - values[i]; }); }
    function clearRounds(e, key, sign) {
        return trained(e).filter(function (r) { return Rz.side(key(r)) === sign; });
    }
    function sameQ(r) { return r.change.finished_accuracy_same_questions; }
    function armList(target, arms, sentence) {
        var ul = Rz.el('ul');
        arms.forEach(function (e) {
            var li = Rz.el('li');
            li.innerHTML = '<strong>' + armName(e) + ':</strong> ' + sentence(e);
            ul.appendChild(li);
        });
        var d = Rz.el('div', 'callout');
        d.appendChild(Rz.el('strong', null, 'Finding.'));
        d.appendChild(ul);
        target.appendChild(d);
    }
    /* one table, the arms one after another under a group row each */
    function armTable(target, arms, columns, rowsOf) {
        var t = Rz.table(columns);
        arms.forEach(function (e) {
            if (arms.length > 1) t.tBodies[0].appendChild(Rz.groupRow(armName(e), columns.length));
            rowsOf(e).forEach(function (r) { t.tBodies[0].appendChild(r); });
        });
        target.appendChild(Rz.wrapped(t));
    }

    function renderSetup(target, arms, all) {
        if (!all.length) { note(target, 'No main run in the export.'); return; }
        var e = all[0];
        var p = Rz.el('p', null, (e.note ? e.note + ' ' : '') + 'Both arms share the pool, test questions, settings, GPU type, batch size and library versions; only the weight precision differs. ' +
            'Every round is compared with its own untrained round 0' + (e.test_questions ? ' on the same ' + e.test_questions + ' test questions' : '') + '. ');
        all.forEach(function (x) { p.appendChild(Rz.el('strong', null, ' ' + armName(x) + ' ')); p.appendChild(badge(x)); });
        target.appendChild(p);
        all.forEach(function (x) {
            (x.in_progress || []).forEach(function (q) {
                note(target, armName(x) + ' still running: ' + q.stage + ', ' + q.done + ' of ' + q.total + ' questions. Left out until complete.', 'callout');
            });
        });
    }

    function renderDecomposition(findingTarget, tableTarget, arms) {
        armList(findingTarget, arms, function (e) {
            var r0 = base(e), rows = trained(e), k = last(rows);
            var parts = rows.map(function (r) { return decompose(r0, r); });
            var up = clearRounds(e, function (r) { return r.change.single; }, 1).length;
            var sameUp = clearRounds(e, sameQ, 1).length, sameDown = clearRounds(e, sameQ, -1).length;
            return 'single-answer accuracy clearly up in ' + up + ' of ' + rows.length + ' rounds (' + list(rows, function (r) { return P(r.change.single.mean); }) + '). ' +
                'Cut-off ' + Rz.pct(r0.cutoff) + ' → ' + Rz.pct(k.cutoff) + ', so the finishing term is ' + range(parts.map(function (d) { return d.finishing; })) +
                ' points; the pooled accuracy term is ' + range(parts.map(function (d) { return d.accuracy; })) + '. Question by question, accuracy when finished moved ' +
                range(rows.map(function (r) { return sameQ(r).mean; })) + ' points, clearly up in ' + sameUp + ' and clearly down in ' + sameDown + ' of ' + rows.length + ' rounds. ' +
                (finishOnly(e) ? '<strong>The gain is finishing, not accuracy.</strong>'
                    : sameUp === rows.length ? '<strong>The model is right more often when it finishes, not only finishing more.</strong>' : '');
        });
        armTable(tableTarget, arms, [
            { label: 'Round' },
            { label: 'Single answer', better: 'up', num: true },
            { label: 'Δ single', better: 'up', num: true, title: 'Against round 0, points, 95% paired interval' },
            { label: 'Finishing more', better: 'up', num: true, title: '(c₀ − cₖ) f₀, points' },
            { label: 'Right more when finished', better: 'up', num: true, title: '(1 − cₖ)(fₖ − f₀), points, pooled f' },
            { label: 'Cut off', better: 'down', num: true },
            { label: 'Right among finished', better: 'up', num: true, title: 'Pooled over questions' },
            { label: 'Δ right among finished, same questions', better: 'up', num: true, title: 'Question by question, points, 95% paired interval' }
        ], function (e) {
            var r0 = base(e);
            return e.test.map(function (r, i) {
                if (!i) return row([Rz.cell('round 0'), Rz.cell(Rz.pct(r.single), 'num'), Rz.cell('reference', 'num muted'), Rz.cell('–', 'num muted'), Rz.cell('–', 'num muted'),
                    Rz.cell(Rz.pct(r.cutoff), 'num'), Rz.cell(Rz.pct(r.finished_accuracy), 'num'), Rz.cell('reference', 'num muted')]);
                var d = decompose(r0, r);
                return row([Rz.cell('round ' + r.round), Rz.cell(Rz.pct(r.single), 'num'), Rz.changeCell(r.change.single, { goodWhen: 'up', n: r.change.n_paired }),
                    Rz.cell(P(d.finishing), 'num'), Rz.cell(P(d.accuracy), 'num'), Rz.cell(Rz.pct(r.cutoff), 'num'), Rz.cell(Rz.pct(r.finished_accuracy), 'num'),
                    Rz.changeCell(sameQ(r), { goodWhen: 'up', n: sameQ(r) && sameQ(r).n })]);
            });
        });
        note(tableTarget, 'The two terms add up exactly to the change in single-answer accuracy. They use pooled f, which drops when hard questions that used to be cut off start to finish; ' +
            'the last column compares each question with itself and is the firmer test of "right more often".');
    }

    function labelFacts(e) {
        var L = poolLabels(e);
        if (L.length < 2) return null;
        var l0 = L[0], lk = last(L);
        var right = function (l) { return Math.round(l.labelled * l.label_accuracy); };
        return { L: L, l0: l0, lk: lk, right: right,
                 perRound: (lk.label_accuracy - l0.label_accuracy) / (L.length - 1),
                 added: function (l) { return l.labelled > l0.labelled ? (right(l) - right(l0)) / (l.labelled - l0.labelled) : null; } };
    }

    function renderLabelDrift(findingTarget, tableTarget, arms) {
        var withLabels = arms.filter(function (e) { return labelFacts(e); });
        if (!withLabels.length) { note(tableTarget, 'Appears once two rounds of pool labels are synced.'); return; }
        armList(findingTarget, withLabels, function (e) {
            var f = labelFacts(e);
            return 'supply ' + f.l0.labelled + ' → ' + f.lk.labelled + ' labels, accuracy ' + Rz.pct(f.l0.label_accuracy) + ' → ' + Rz.pct(f.lk.label_accuracy) +
                ' (' + (labelsFalling(e) ? 'down every round, ' : '') + 'about ' + P(f.perRound) + ' points per round), wrong labels ' + f.l0.wrong_labels + ' → ' + f.lk.wrong_labels +
                '. The labels added since round 0 are right ' + Rz.pct(f.added(f.lk)) + ' of the time.';
        });
        armTable(tableTarget, withLabels, [
            { label: 'Pool answered by' },
            { label: 'Labelled', num: true, title: 'Questions with at least 6 of 8 samples agreeing' },
            { label: 'Label accuracy', better: 'up', num: true, title: 'Checked against the train answers after the run' },
            { label: 'Right labels', num: true }, { label: 'Wrong labels', better: 'down', num: true },
            { label: 'Accuracy of labels added since round 0', better: 'up', num: true, title: '(nₖaₖ − n₀a₀) / (nₖ − n₀)' }
        ], function (e) {
            var f = labelFacts(e);
            return f.L.map(function (l, i) {
                return row([Rz.cell('round ' + l.round), Rz.cell(String(l.labelled), 'num'), Rz.cell(Rz.pct(l.label_accuracy), 'num'),
                    Rz.cell(String(f.right(l)), 'num'), Rz.cell(String(l.wrong_labels), 'num'),
                    i ? Rz.cell(Rz.pct(f.added(l)), 'num') : Rz.cell('–', 'num muted')]);
            });
        });
    }

    function renderVoteSharpening(findingTarget, tableTarget, arms) {
        armList(findingTarget, arms, function (e) {
            var r0 = base(e), rows = trained(e), k = last(rows);
            var lost = rows.reduce(function (a, r) { return a + r.change.voted_lost; }, 0);
            var gained = rows.reduce(function (a, r) { return a + r.change.voted_gained; }, 0);
            var down = clearRounds(e, function (r) { return r.change.voted; }, -1).length, up = clearRounds(e, function (r) { return r.change.voted; }, 1).length;
            return 'voted accuracy changed by ' + list(rows, function (r) { return P(r.change.voted.mean); }) + ' points (clearly up in ' + up + ', clearly down in ' + down + ' of ' + rows.length + ' rounds); ' +
                'summed over rounds ' + lost + ' voted answers went right → wrong and ' + gained + ' wrong → right. Confident questions ' + Rz.pct(r0.confident) + ' → ' + Rz.pct(k.confident) +
                ', of which right ' + Rz.pct(r0.confident_accuracy) + ' → ' + Rz.pct(k.confident_accuracy) + ' (confident but wrong: ' + r0.confident_wrong + ' → ' + k.confident_wrong + ' questions).' +
                (k.confident_accuracy < r0.confident_accuracy ? ' <strong>The model grows surer, and part of the new certainty is on wrong answers.</strong>' : '');
        });
        armTable(tableTarget, arms, [
            { label: 'Round' },
            { label: 'Voted', better: 'up', num: true }, { label: 'Δ voted', better: 'up', num: true, title: 'Against round 0, points, 95% paired interval' },
            { label: 'Voted lost / gained', num: true, title: 'Questions whose voted answer went from right to wrong / wrong to right' },
            { label: 'Confident', num: true, title: 'Share of test questions where at least 6 of 8 samples agree' },
            { label: 'Confident and right', better: 'up', num: true, title: 'Share of confident questions whose voted answer is right: the label accuracy on unseen questions' },
            { label: 'Confident but wrong', better: 'down', num: true, title: 'Number of test questions' },
            { label: 'Distinct answers', num: true, title: 'Mean number of different answers per question' }
        ], function (e) {
            return e.test.map(function (r) {
                var c = r.change;
                return row([Rz.cell(r.name), Rz.cell(Rz.pct(r.voted), 'num'), changeOrDash(c, 'voted', 'up'),
                    Rz.cell(c ? c.voted_lost + ' / ' + c.voted_gained : '–', c ? 'num' : 'num muted'),
                    Rz.cell(Rz.pct(r.confident), 'num'), Rz.cell(Rz.pct(r.confident_accuracy), 'num'),
                    Rz.cell(String(r.confident_wrong), 'num'), Rz.cell(num(r.distinct), 'num')]);
            });
        });
    }

    function comparison(label) {
        var c = ((Rz.data.self_training || {}).comparisons || []).filter(function (x) { return x.label === label; })[0];
        return c && c.before && c.rounds.length ? c : null;
    }

    function renderCatchUp(findingTarget, tableTarget) {
        var c = comparison('Main run');
        if (!c) { note(tableTarget, 'Appears once both arms have an untrained round 0 and at least one trained round.', 'muted small'); return; }
        var k = last(c.rounds);
        var closed = function (key) { return c.before[key].mean > 0 ? 1 - k['gap_' + key].mean / c.before[key].mean : null; };
        var diffClear = c.rounds.filter(function (r) { return Rz.side(r.difference_single) < 0; }).map(function (r) { return r.round; });
        finding(findingTarget, '<strong>Finding.</strong> Before training the 4-bit model trails the untrained 16-bit model by ' + iv(c.before.single) + ' points in single-answer accuracy and ' +
            iv(c.before.voted) + ' in voted accuracy. After round ' + k.round + ' the gaps are ' + iv(k.gap_single) + ' and ' + iv(k.gap_voted) +
            (closed('single') != null ? ': training closed ' + Rz.pct(closed('single'), 0) + ' of the single-answer gap and ' + Rz.pct(closed('voted'), 0) + ' of the voted gap' : '') + '. ' +
            'The 4-bit model gained clearly more than the 16-bit model in single-answer accuracy in ' + (diffClear.length ? diffClear.length + ' of ' + c.rounds.length + ' rounds (round ' + diffClear.join(', ') + ')' : 'no round') +
            (Rz.side(k.gap_single) === 0 ? '. By round ' + k.round + ' the remaining single-answer gap can no longer be told apart from zero, though its interval only just touches it.' : '.') +
            (Rz.side(k.gap_voted) > 0 ? ' The voted gap is still clearly open.' : ''));
        renderComparison(tableTarget, 'Main run');
    }

    /* Key takeaways: trends across rounds and arms. Every claim is checked against the data before it is shown,
       and labelled by how firm it is. */
    var TAG = { clear: ['Clear', 'green'], steady: ['Steady', 'green'], noise: ['Within noise', 'orange'], none: ['No trend', ''],
                pattern: ['Pattern', 'purple'], anomaly: ['Anomaly', 'red'], next: ['Next', 'blue'], data: ['Phases 1–2', ''] };

    function renderTakeaways(target, full, quant) {
        function item(tag, html) {
            var li = Rz.el('li');
            li.innerHTML = '<span class="badge ' + TAG[tag][1] + '">' + TAG[tag][0] + '</span> ' + html;
            target.appendChild(li);
        }
        function singles(e) { return list(trained(e), function (r) { return P(r.change.single.mean); }); }
        function nClear(e, key, sign) { return clearRounds(e, key, sign).length; }
        var single = function (r) { return r.change.single; }, voted = function (r) { return r.change.voted; };

        item('data', '<strong>Quantization costs labels, not label quality.</strong> It mainly lowers how many questions the vote can label, and the model judging its own answers is a worse label source than the vote (section 2).');

        var both = tested(full) && tested(quant);
        if (!tested(full)) { item('next', 'No trained round of the main run is synced yet.'); return; }
        var arms = [full].concat(tested(quant) ? [quant] : []);
        var K = trained(full).length;

        // 1. which arm improves
        if (both) {
            var q4 = nClear(quant, single, 1), q16 = nClear(full, single, 1), c = comparison('Main run');
            var kd = c && last(c.rounds);
            item(q4 === trained(quant).length && q16 < trained(full).length ? 'clear' : 'noise',
                '<strong>Training on its own votes helps the 4-bit model and barely moves the 16-bit one.</strong> Single-answer accuracy against round 0: 4-bit ' + singles(quant) +
                ' points (clear in ' + q4 + ' of ' + trained(quant).length + ' rounds), 16-bit ' + singles(full) + ' (clear in ' + q16 + ' of ' + K + ').' +
                (kd ? ' The 16-bit change minus the 4-bit change is ' + iv(kd.difference_single) + ' points at round ' + kd.round + '; the two arms differ clearly in ' +
                    c.rounds.filter(function (r) { return Rz.side(r.difference_single) !== 0; }).length + ' of ' + c.rounds.length + ' rounds.' : ''));
            if (kd && c.before.single.mean > 0) {
                item(Rz.side(last(trained(quant)).change.single) > 0 ? 'clear' : 'noise',
                    '<strong>The 4-bit model closes most of its gap to the untrained 16-bit model.</strong> Single-answer gap ' + P(c.before.single.mean) + ' → ' + P(kd.gap_single.mean) +
                    ' points (' + Rz.pct(1 - kd.gap_single.mean / c.before.single.mean, 0) + ' closed); voted gap ' + P(c.before.voted.mean) + ' → ' + P(kd.gap_voted.mean) +
                    (Rz.side(kd.gap_voted) > 0 ? ', still clearly open' : '') + ' (section 4).');
            }
        }

        // 2. where the gain comes from: accuracy when finished, and cut-off
        var cutShare = function (e) {
            var cut = e.test.map(function (r) { return r.cutoff; }), total = cut[0] - last(cut);
            return total > 0 ? (cut[0] - cut[1]) / total : null;
        };
        item(arms.every(function (e) { return nClear(e, function (r) { return r.change.cutoff; }, -1) === trained(e).length; }) ? 'clear' : 'noise',
            '<strong>' + (both && !finishOnly(quant) && finishOnly(full) ? 'At 4-bit the model gets more right when it finishes; at 16-bit it only finishes more.' : 'Where the gain comes from.') + '</strong> ' +
            arms.map(function (e) {
                var cut = e.test.map(function (r) { return r.cutoff; });
                return armName(e) + ': accuracy when finished, question by question, ' + list(trained(e), function (r) { return P(sameQ(r).mean); }) +
                    ' points (clear in ' + nClear(e, sameQ, 1) + ' of ' + trained(e).length + '); cut-off ' + Rz.pct(cut[0]) + ' → ' + Rz.pct(last(cut)) +
                    (cutShare(e) != null ? ', ' + Rz.pct(cutShare(e), 0) + ' of that drop in round 1' : '');
            }).join('. ') + '. <em>So:</em> the 16-bit loop mostly teaches finishing and saturates early; the 4-bit loop keeps cutting off less and is also right more often (3.1).');

        // 3. labels decay in every arm
        var lf = arms.map(labelFacts).filter(Boolean);
        if (lf.length) {
            item(arms.every(labelsFalling) ? 'steady' : 'pattern',
                '<strong>The labels get worse every round in ' + (lf.length > 1 ? 'both arms' : 'the run') + '.</strong> ' +
                arms.filter(labelFacts).map(function (e) { var f = labelFacts(e); return armName(e) + ' ' + Rz.pct(f.l0.label_accuracy) + ' → ' + Rz.pct(f.lk.label_accuracy) + ' (about ' + P(f.perRound) + ' points per round, wrong labels ' + f.l0.wrong_labels + ' → ' + f.lk.wrong_labels + ')'; }).join('; ') +
                '.' + (both && !finishOnly(quant) ? ' <em>So:</em> the 4-bit model improved on labels that were no better than 16-bit\'s, so label accuracy alone does not decide whether training helps; it is still what limits more rounds.' : ''));
        }

        // 4. the vote
        item(arms.every(function (e) { return nClear(e, voted, 1) + nClear(e, voted, -1) === trained(e).length; }) ? 'clear' : 'noise',
            '<strong>' + (both && trained(full).every(function (r) { return r.change.voted.mean < 0; }) && trained(quant).filter(function (r) { return r.change.voted.mean > 0; }).length > trained(quant).length / 2
                ? 'The vote drifts down at 16-bit and mostly up at 4-bit.' : 'The vote.') + '</strong> ' + arms.map(function (e) {
                var rows = trained(e);
                var lost = rows.reduce(function (a, r) { return a + r.change.voted_lost; }, 0), gained = rows.reduce(function (a, r) { return a + r.change.voted_gained; }, 0);
                return armName(e) + ' voted accuracy ' + list(rows, function (r) { return P(r.change.voted.mean); }) + ' points (clearly up in ' + nClear(e, voted, 1) + ', clearly down in ' + nClear(e, voted, -1) +
                    '), ' + lost + ' questions lost and ' + gained + ' gained in total';
            }).join('; ') + '. Most voted changes are within noise at ' + full.test_questions + ' questions.');

        // 5. held-out certainty falls in every arm
        var caFalls = arms.filter(function (e) { return last(e.test).confident_accuracy < base(e).confident_accuracy; });
        if (caFalls.length) {
            item('pattern', '<strong>Every arm becomes confidently wrong more often.</strong> On test questions, the share of confident answers that is right went ' +
                caFalls.map(function (e) { return armName(e) + ' ' + list(e.test, function (r) { return Rz.pct(r.confident_accuracy); }).replace(/, /g, ' → '); }).join('; ') +
                '; confident-but-wrong questions ' + caFalls.map(function (e) { return armName(e) + ' ' + base(e).confident_wrong + ' → ' + last(e.test).confident_wrong; }).join(', ') +
                '. <em>So:</em> the 4-bit gains come with a growing number of confident errors, the thing that would poison later rounds.');
        }

        // 6. the smaller pilot points the same way
        var pilot = comparison('Pilot');
        if (both && pilot && pilot.rounds.every(function (r) { return r.difference_single.mean < 0; })) {
            item('pattern', '<strong>The separate pilot points the same way.</strong> In the smaller pilot (' + pilot.before.n_paired + ' test questions, a smaller pool), the 16-bit model also gained less than the 4-bit model in every round (16-bit minus 4-bit change: ' +
                list(pilot.rounds, function (r) { return P(r.difference_single.mean); }) + ' points), though it was run at different batch sizes (Earlier and supporting runs).');
        }

        // 7. a last round that reverses the one before it on several measures at once
        arms.forEach(function (e) {
            var rows = trained(e);
            if (rows.length < 2) return;
            var k = last(rows), prev = rows[rows.length - 2];
            if (k.single < prev.single && k.confident < prev.confident && k.distinct > prev.distinct) {
                item('anomaly', '<strong>' + armName(e) + ', round ' + k.round + ' reverses round ' + prev.round + '.</strong> Single-answer accuracy ' + Rz.pct(prev.single) + ' → ' + Rz.pct(k.single) +
                    ', confident questions ' + Rz.pct(prev.confident) + ' → ' + Rz.pct(k.confident) + ', distinct answers ' + num(prev.distinct) + ' → ' + num(k.distinct) +
                    '. With one seed and ' + e.test_questions + ' questions this cannot be told apart from noise.');
            }
        });

        // next
        item('next', '<strong>Check the 4-bit gain before building on it.</strong> ' +
            'Rerun the untrained 4-bit round 0 with another seed: a low round 0 would make any later round look like a gain. Add an 8-bit arm to see whether the gain grows with quantization damage. ' +
            'Any guard against label decay must stay label-free, so it cannot use label accuracy.');
    }

    function renderConclusions(target, full, quant) {
        var ul = Rz.el('ul');
        function item(tag, html) { var li = Rz.el('li'); li.innerHTML = '<span class="badge">' + tag + '</span> ' + html; ul.appendChild(li); }
        var arms = [full, quant].filter(tested);
        arms.filter(finishOnly).forEach(function (e) {
            var r0 = base(e), k = last(trained(e));
            item('Interpretation', armName(e) + ': under a token cap the vote can only use finished solutions, and those skew short, so SFT on them mostly teaches the model to finish in time (the brevity trap). ' +
                (k.tokens < r0.tokens ? 'Answers got shorter, ' + num(r0.tokens, 0) + ' → ' + num(k.tokens, 0) + ' tokens, ' : '') + 'the cut-off rate fell, and accuracy when finished did not rise.');
        });
        arms.filter(function (e) { return !finishOnly(e) && e.precision !== '16bit'; }).forEach(function (e) {
            var r0 = base(e), k = last(trained(e)), f = full && tested(full) ? base(full) : null;
            item('Hypothesis', armName(e) + ': training on the vote partly repairs what quantization disturbed. The quantized model starts more scattered and is pulled back towards the untrained 16-bit model: distinct answers per question ' +
                num(r0.distinct) + ' → ' + num(k.distinct) + (f ? ' (16-bit untrained: ' + num(f.distinct) + ')' : '') + ', and most of its single-answer gap closed (section 4). On this view the 16-bit model has little to repair, which would explain why it only learns to finish. Untested: see the Next takeaway.');
        });
        if (arms.some(labelsFalling)) {
            item('Interpretation', 'Each round relabels the pool with the last round\'s model, so wrong labels compound: the loop grows its training set at the margin of agreement, where the vote is least reliable (3.2), and the vote on unseen test questions drifts the same way (3.3).');
        }
        item('Criterion', 'Real self-improvement shows up as accuracy when finished rising clearly, question by question, with label accuracy holding across rounds. ' +
            arms.map(function (e) {
                var first = clearRounds(e, sameQ, 1).length === trained(e).length, second = !labelsFalling(e);
                return armName(e) + (first && second ? ' meets both' : first ? ' meets the first, not the second' : second ? ' meets the second, not the first' : ' meets neither');
            }).join('; ') + '.');
        target.appendChild(ul);
    }

    onReady(function () {
        if (!Rz || !Rz.has()) { Rz && Rz.showMissing(document.getElementById('results-root')); return; }
        var $ = function (id) { return document.getElementById(id); };
        var full = mainArm('16bit'), quant = mainArm('4bit');
        var all = [full, quant].filter(Boolean), arms = all.filter(tested);
        $('source-line').textContent = Rz.sourceLine('generator');
        renderTakeaways($('takeaways'), full, quant);
        renderLabelChooser($('label-chips'), $('label-table'));
        renderVerifier($('verifier-summary'), $('verifier-table'));
        renderSetup($('main-setup'), arms, all);
        if (arms.length) {
            renderDecomposition($('finding-decomposition'), $('decomposition-table'), arms);
            renderLabelDrift($('finding-labels'), $('labels-table'), arms);
            renderVoteSharpening($('finding-vote'), $('vote-table'), arms);
            renderExperiments($('main-full'), all);
        }
        if (tested(full) && tested(quant)) renderCatchUp($('finding-catchup'), $('catchup-table'));
        else note($('catchup-table'), 'Appears once both arms have trained rounds synced.', 'muted small');
        renderConclusions($('conclusions'), full, quant);
        renderExperiments($('pilot-runs'), 'pilot');
        renderComparison($('pilot-comparison'), 'Pilot');
        renderControls($('controls'));
        renderExperiments($('dpo-runs'), 'dpo');
    });
})();
