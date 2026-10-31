/* Site shell. One file every page loads, first thing in <body>.

   It owns everything that used to be copied between pages:
     - page header (title and subtitle) and the navigation bar, from the PAGES registry below
     - notices, an "On this page" list built from the page's h2 headings (pages with toc: true)
     - previous / next links and the footer
     - the card grid on the home page (<div data-page-cards>)
     - MathJax and Mermaid, loaded and configured once, only on pages that list them in libs
     - the page's own scripts (scripts: [...]), loaded in order
     - wrapping every <table> in a scrolling container

   To add a page: create the .html file (see any existing page for the skeleton) and add one entry to PAGES.
   To change a title, subtitle, nav label or card summary: edit it here, once.

   A page with local: true exists only on a developer's machine (localhost or file://). It is left out of the
   navigation, the home cards and the previous / next links when the site is served from anywhere else, so it
   can be kept out of git without leaving dead links. Mark any other link to it with data-local-only. */
(function () {
    'use strict';

    /* ------------------------------------------------------------------ registry */

    var SITE = {
        name: 'Experiment D',
        footer: 'Experiment D: does weight quantization break self-improvement in small language models?'
    };

    var GROUPS = [
        { id: 'background', title: 'Background', intro: 'The question, the data, and what is already known.' },
        { id: 'results', title: 'Results', intro: 'Drawn from the run files in results/ and regenerated on request, so they are always current.' },
        { id: 'direction', title: 'Direction', intro: 'What comes next.' }
    ];

    var NOTICES = {
        'local-only': {
            badge: 'Local only',
            html: '<strong>Local page.</strong> This page is kept out of git and is not part of the published site. ' +
                  'It is a dated snapshot: the dossier (<code>RESEARCH_DOSSIER.md</code>) supersedes parts of it.'
        }
    };

    /* Order here is the order of the navigation bar, the home cards and the previous / next links.
       nav: short label for the bar and the previous / next links. title and subtitle: page header and home card title.
       summary: home card text. group: home section. toc: build "On this page".
       libs: external libraries. scripts: the page's own scripts, in order. notice: key of NOTICES. local: see above. */
    var PAGES = [
        { id: 'home', file: 'index.html', nav: 'Home',
          title: 'Agentic AI: Experiment D',
          subtitle: 'Does quantization break a small model\'s ability to improve itself? Background, results, and what comes next.' },

        { id: 'proposal', file: 'proposal.html', nav: 'Proposal', group: 'background', toc: true,
          title: 'Original Project Proposal',
          subtitle: 'The scoping document that defined the question, the background and the planned experiments.',
          summary: 'The initial scoping document: the problem, how self-improvement and quantization work, and the proposed experiments.' },

        { id: 'problem', file: 'problem_statement.html', nav: 'Problem & Datasets', group: 'background', libs: ['mermaid'],
          title: 'Problem Statement & Datasets',
          subtitle: 'The gap in the literature on quantized self-improvement, and the datasets used to study it.',
          summary: 'Why the question matters, the self-improvement loop under study, and why GSM8K was chosen over SVAMP and CommonsenseQA.' },

        { id: 'previous', file: 'previous_works.html', nav: 'Previous Works', group: 'background',
          title: 'Previous Works',
          subtitle: 'What is already known about quantization and about self-improvement, where the two meet, and what our paper adds.',
          summary: 'A one-minute story, a table placing our paper against prior work, and one idea to remember for each of 84 papers.',
          scripts: ['assets/data/previous-works/themes.js',
                    'assets/data/previous-works/papers-quantization.js',
                    'assets/data/previous-works/papers-reasoning-reliability.js',
                    'assets/data/previous-works/papers-intersection.js',
                    'assets/data/previous-works/papers-consistency-verification.js',
                    'assets/data/previous-works/papers-self-training.js',
                    'assets/data/previous-works/papers-methods.js',
                    'assets/data/previous-works/notes.js',
                    'assets/js/previous-works.js'] },

        { id: 'generation', file: 'generation.html', nav: 'Phase 1: Generation', group: 'results', libs: ['mathjax'], toc: true,
          title: 'Phase 1: Generator Scaling',
          subtitle: 'The impact of quantization on generation accuracy, coverage, diversity and calibration.',
          summary: 'How quantization changes what the base model generates: voted and single-sample accuracy, pass@N, agreement and calibration.',
          scripts: ['assets/data/results.js', 'assets/js/results.js', 'assets/js/charts.js', 'assets/js/generation.js'] },

        { id: 'verifier', file: 'verifier.html', nav: 'Phase 2: Verifier', group: 'results', libs: ['mathjax'], toc: true,
          title: 'Phase 2: Verifier Reliability',
          subtitle: 'Does quantization break the reliability of self-judgment in small language models?',
          summary: 'Whether quantized models can judge their own reasoning: false positive rate, MCC and what verifier-guided voting adds.',
          scripts: ['assets/data/results.js', 'assets/js/results.js', 'assets/js/verifier.js'] },

        { id: 'gen-vs-ver', file: 'generator_vs_verifier.html', nav: 'Generator vs Verifier', group: 'results',
          title: 'Generator vs Verifier',
          subtitle: 'Each model\'s own answers against its own verifier\'s pick, as one answer and as a vote, on the same questions and samples.',
          summary: 'One table across all models, precisions and datasets: generator and verifier accuracy, single and voted, and the difference.',
          scripts: ['assets/data/generator_vs_verifier.js', 'assets/js/generator-vs-verifier.js'] },

        { id: 'labels-training', file: 'labels_training.html', nav: 'Phase 3: Labels & Self-Training', group: 'results', libs: ['mathjax'], toc: true,
          title: 'Phase 3: Labels and Self-Training',
          subtitle: 'Whether a quantized model\'s own votes make good labels, and what happens when it trains on them.',
          summary: 'Label supply and accuracy by precision, the vote against the verifier, and SFT and DPO self-training at 16-bit and 4-bit.',
          scripts: ['assets/data/results.js', 'assets/js/results.js', 'assets/js/labels-training.js'] },

        { id: 'mentor', file: 'mentor_review.html', nav: 'Multi-Round Evaluation', group: 'results', libs: ['mathjax'], toc: true,
          title: 'Multi-Round Self-Training & Verifier Scaling',
          subtitle: 'Empirical results across Qwen3.5-0.8B and 2B: majority voting collapse, critic regularization, and pseudo-label drift.',
          summary: 'Executive presentation for mentor: 4 clean multi-round figures, master scorecards compared to baseline, and root-cause analysis.' },

        { id: 'dynamics', file: 'self_training_dynamics.html', nav: '3-Stage Dynamics', group: 'results', libs: ['mermaid', 'mathjax'], toc: true,
          title: 'Three-Stage Self-Training Dynamics & Literature Context',
          subtitle: 'Visual breakdown of Stage 1 (Generator), Stage 2 (Verifier Filter), and Stage 3 (Test Set), with real examples and literature context.',
          summary: 'What changes each round in SLMs: one-sample accuracy, vote and label quality, real examples, and a comparison with prior literature.' },

        { id: 'audit', file: 'audit.html', nav: 'Audit', group: 'direction', toc: true, local: true, notice: 'local-only',
          title: 'Audit & New Direction',
          subtitle: 'What was wrong, what the data showed at the time, and the one idea worth a paper.',
          summary: 'The early-October audit: bugs found and fixed, and the proposed contribution. A snapshot, kept local.' },

        { id: 'next', file: 'next_steps.html', nav: 'Next Steps', group: 'direction', libs: ['mermaid', 'mathjax'], toc: true,
          title: 'Next Steps',
          subtitle: 'Self-training helped the 4-bit model and not the 16-bit one. Why we think that is, and the experiments that could prove us wrong.',
          summary: 'The precision surprise from Phase 3, the noise-versus-consistent-error explanation, four stages of tests with their failure conditions, and what a paper would need.' }
    ];

    /* Developer machine: localhost, 127.0.0.1, [::1] or a file:// page. */
    var IS_LOCAL = /^(localhost|127\.0\.0\.1|\[::1\]|)$/.test(window.location.hostname);
    if (IS_LOCAL) document.documentElement.setAttribute('data-local', '');
    PAGES = PAGES.filter(function (p) { return !p.local || IS_LOCAL; });

    /* ------------------------------------------------------------------ helpers */

    function esc(s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function slug(s) {
        return s.toLowerCase().replace(/&/g, ' and ').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
    }

    function loadScript(src, onload) {
        var s = document.createElement('script');
        s.src = src;
        s.async = true;
        if (onload) s.onload = onload;
        document.head.appendChild(s);
    }

    /* run in the order given, without blocking the page */
    function loadInOrder(sources) {
        sources.forEach(function (src) {
            var s = document.createElement('script');
            s.src = src;
            s.async = false;
            document.body.appendChild(s);
        });
    }

    var fileName = (window.location.pathname.split('/').pop() || 'index.html');
    var index = 0;
    PAGES.forEach(function (p, i) { if (p.file === fileName) index = i; });
    var page = PAGES[index];

    /* ------------------------------------------------------------------ libraries */

    var LIBS = {
        mathjax: function () {
            window.MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] } };
            loadScript('https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js');
        },
        mermaid: function () {
            loadScript('https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js', function () {
                window.mermaid.initialize({
                    startOnLoad: false,
                    theme: 'base',
                    themeVariables: { primaryColor: '#f1f5f9', primaryBorderColor: '#3182ce', lineColor: '#2d3748', textColor: '#2d3748' }
                });
                /* a cached copy can load before the page is parsed: wait, or later diagrams stay as text */
                var run = function () { window.mermaid.run({ querySelector: '.mermaid' }); };
                if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run);
                else run();
            });
        }
    };
    (page.libs || []).forEach(function (name) { if (LIBS[name]) LIBS[name](); });

    /* ------------------------------------------------------------------ header, navigation, notice */

    document.title = page.id === 'home' ? page.title : page.title + ' | ' + SITE.name;
    if (page.summary) {
        var meta = document.createElement('meta');
        meta.name = 'description';
        meta.content = page.summary;
        document.head.appendChild(meta);
    }

    var nav = PAGES.map(function (p) {
        var current = p === page ? ' aria-current="page"' : '';
        return '<li><a href="' + p.file + '"' + current + '>' + esc(p.nav) + '</a></li>';
    }).join('');

    var shell =
        '<a class="skip-link" href="#main">Skip to content</a>' +
        '<header class="site-header"><h1>' + esc(page.title) + '</h1><p>' + esc(page.subtitle) + '</p></header>' +
        '<nav class="site-nav" aria-label="Primary"><ul>' + nav + '</ul></nav>';

    if (page.notice && NOTICES[page.notice]) {
        shell += '<div class="container page-notice"><div class="callout warning">' + NOTICES[page.notice].html + '</div></div>';
    }

    var anchor = document.currentScript;
    if (anchor) anchor.insertAdjacentHTML('beforebegin', shell);
    else document.write(shell);

    /* ------------------------------------------------------------------ parts that need the page body */

    function wrapTables() {
        Array.prototype.forEach.call(document.querySelectorAll('main table'), function (t) {
            if (t.parentNode.classList.contains('table-wrap')) return;
            var wrap = document.createElement('div');
            wrap.className = 'table-wrap';
            t.parentNode.insertBefore(wrap, t);
            wrap.appendChild(t);
        });
    }

    function buildToc(main) {
        var heads = Array.prototype.filter.call(main.querySelectorAll('h2'), function (h) { return !h.closest('[data-no-toc]'); });
        if (heads.length < 3) return;
        var used = {};
        var items = heads.map(function (h) {
            if (!h.id) {
                var base = slug(h.textContent) || 'section', id = base, n = 2;
                while (used[id] || document.getElementById(id)) id = base + '-' + (n++);
                h.id = id;
            }
            used[h.id] = true;
            return '<li><a href="#' + h.id + '">' + esc(h.textContent) + '</a></li>';
        }).join('');
        main.insertAdjacentHTML('afterbegin', '<nav class="toc" aria-label="On this page"><strong>On this page</strong><ol>' + items + '</ol></nav>');
    }

    function buildCards(target) {
        var html = '';
        GROUPS.forEach(function (g) {
            var cards = PAGES.filter(function (p) { return p.group === g.id; }).map(function (p) {
                var badges = '';
                if (p.notice && NOTICES[p.notice]) badges += ' <span class="badge orange">' + esc(NOTICES[p.notice].badge) + '</span>';
                return '<div class="card">' +
                    '<h3>' + esc(p.title) + badges + '</h3>' +
                    '<p>' + esc(p.summary) + '</p>' +
                    '<a class="card-link" href="' + p.file + '">Open &rarr;</a></div>';
            }).join('');
            html += '<h3>' + esc(g.title) + '</h3><p class="muted">' + esc(g.intro) + '</p><div class="grid">' + cards + '</div>';
        });
        target.innerHTML = html;
    }

    function buildPager() {
        var prev = PAGES[index - 1], next = PAGES[index + 1];
        var html = '<div class="container"><div class="pager">';
        if (prev) html += '<a class="prev" href="' + prev.file + '"><small>&larr; Previous</small>' + esc(prev.nav) + '</a>';
        if (next) html += '<a class="next" href="' + next.file + '"><small>Next &rarr;</small>' + esc(next.nav) + '</a>';
        html += '</div></div>';
        html += '<footer class="site-footer">' + esc(SITE.footer) + '</footer>';
        document.body.insertAdjacentHTML('beforeend', html);
    }

    function enhance() {
        var main = document.querySelector('main');
        if (main) {
            main.id = main.id || 'main';
            wrapTables();
            if (page.toc) buildToc(main);
        }
        var cards = document.querySelector('[data-page-cards]');
        if (cards) buildCards(cards);
        buildPager();
        if (page.scripts) loadInOrder(page.scripts);

        /* the list above shifts the layout after the browser has already jumped to a #section link: jump again */
        if (window.location.hash.length > 1) {
            var target = document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
            if (target) target.scrollIntoView({ behavior: 'instant' });
        }

        /* on narrow screens the nav scrolls sideways: bring the current page into view */
        var active = document.querySelector('.site-nav [aria-current="page"]');
        var strip = active && active.closest('ul');
        if (strip) strip.scrollLeft = Math.max(0, active.offsetLeft - (strip.clientWidth - active.offsetWidth) / 2);
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', enhance);
    else enhance();
})();
