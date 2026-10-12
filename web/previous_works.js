/* Previous Works page: a reading guide to the literature behind the project.
   Data lives in previous_works_data/:
     pw_guide.js       themes, how our paper sits against prior work, and per-paper take / role / start-here
     pw_lens.js        per-paper notes on the quantization side and the self-improvement side
     pw_categories.js  fine-grained research categories (each theme groups one or more)
     pw_*.js           the paper records */
(function () {
    'use strict';

    var THEMES = window.PW_THEMES || [];
    var POSITION = window.PW_POSITION || { add: [], known: [] };
    var NOTES = window.PW_NOTES || {};
    var ROLES = window.PW_ROLES || [];
    var LENS = window.PW_LENS || {};
    var CATS = window.PW_CATEGORIES || [];
    var PAPERS = window.PW_PAPERS || [];

    var STATUS = [
        { id: 'published', label: 'Published' },
        { id: 'accepted', label: 'Accepted' },
        { id: 'workshop', label: 'Workshop' },
        { id: 'preprint', label: 'Preprint' }
    ];
    var STATUS_LABEL = {};
    STATUS.forEach(function (s) { STATUS_LABEL[s.id] = s.label; });

    var READ_LABEL = {
        full: 'Full text read',
        main: 'Main text read (appendices were cut off in retrieval)',
        abstract: 'Abstract and metadata only; full text could not be machine-read',
        record: 'Bibliographic record only (book, not read here)'
    };

    var themeIndex = {}, catTheme = {}, catTitle = {}, byPaperId = {};
    THEMES.forEach(function (t, i) { themeIndex[t.id] = i; t.cats.forEach(function (c) { catTheme[c] = t.id; }); });
    CATS.forEach(function (c) { catTitle[c.id] = c.title; });

    var SORT_DEFAULT = 'theme';
    function freshState() {
        return { q: '', theme: '', start: false, role: '', status: {}, origin: '', yearFrom: '', yearTo: '', sort: SORT_DEFAULT };
    }
    var state = freshState();

    var $ = function (id) { return document.getElementById(id); };
    var els = {};

    /* ---------- helpers ---------- */
    function el(tag, cls, text) {
        var n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text != null) n.textContent = text;
        return n;
    }

    function escapeRegExp(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

    function terms() {
        return state.q.toLowerCase().split(/\s+/).filter(function (t) { return t.length > 0; });
    }

    function haystack(p) {
        if (p._hay) return p._hay;
        var parts = [p.title, p.authors, p.venue, p.year, p.arxiv || '', p.take, p.role, p.summary, p.relevance, p.lq || '', p.ls || '',
            (p.tags || []).join(' '), (p.method || []).join(' '), (p.results || []).join(' '), (p.limits || []).join(' '), p.note || '',
            STATUS_LABEL[p.status] || '', catTitle[p.cat] || '', THEMES[themeIndex[p.theme]].title];
        p._hay = parts.join(' \n ').toLowerCase();
        return p._hay;
    }

    function appendHighlighted(parent, text, ts) {
        if (!ts.length) { parent.appendChild(document.createTextNode(text)); return; }
        var re = new RegExp('(' + ts.map(escapeRegExp).join('|') + ')', 'gi');
        var last = 0, m;
        while ((m = re.exec(text)) !== null) {
            if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
            parent.appendChild(el('mark', 'pw-hl', m[0]));
            last = m.index + m[0].length;
            if (m[0].length === 0) re.lastIndex++;
        }
        if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
    }

    function shortTitle(p) { return p.short || p.title.split(/[:?]/)[0].trim(); }

    /* ---------- filtering and sorting ---------- */
    function activeSet(obj) { return Object.keys(obj).filter(function (k) { return obj[k]; }); }

    function matches(p, skipTheme) {
        var ts = terms();
        if (ts.length) {
            var h = haystack(p);
            for (var i = 0; i < ts.length; i++) if (h.indexOf(ts[i]) === -1) return false;
        }
        if (!skipTheme && state.theme && p.theme !== state.theme) return false;
        if (state.start && !p.start) return false;
        if (state.role && p.role !== state.role) return false;
        var ss = activeSet(state.status);
        if (ss.length && ss.indexOf(p.status) === -1) return false;
        if (state.origin && p.origin !== state.origin) return false;
        if (state.yearFrom && p.year < Number(state.yearFrom)) return false;
        if (state.yearTo && p.year > Number(state.yearTo)) return false;
        return true;
    }

    function titleKey(p) { return p.title.toLowerCase().replace(/^[^a-z0-9]+/, ''); }
    function cmpTitle(a, b) { var x = titleKey(a), y = titleKey(b); return x < y ? -1 : (x > y ? 1 : 0); }

    function sorter(mode) {
        switch (mode) {
            case 'year-desc': return function (a, b) { return b.year - a.year || cmpTitle(a, b); };
            case 'year-asc': return function (a, b) { return a.year - b.year || cmpTitle(a, b); };
            case 'title': return cmpTitle;
            default: return function (a, b) {
                return themeIndex[a.theme] - themeIndex[b.theme] || (b.start ? 1 : 0) - (a.start ? 1 : 0) || a.prio - b.prio || b.year - a.year || cmpTitle(a, b);
            };
        }
    }

    /* ---------- jump to a paper from the tables above ---------- */
    function jumpTo(id) {
        state = freshState();
        syncControls();
        render();
        var card = $('paper-' + id);
        if (!card) return;
        var det = card.querySelector('details');
        if (det) det.open = true;
        card.scrollIntoView({ behavior: 'smooth', block: 'start' });
        card.classList.add('pw-flash');
        setTimeout(function () { card.classList.remove('pw-flash'); }, 2200);
    }

    function paperLink(id) {
        var p = byPaperId[id];
        var b = el('button', 'pw-ref', p ? shortTitle(p) : id);
        b.type = 'button';
        if (p) b.title = p.title + ' (' + p.venue + ')';
        b.addEventListener('click', function () { jumpTo(id); });
        return b;
    }

    function refList(ids) {
        var span = el('span', 'pw-refs');
        ids.forEach(function (id) { span.appendChild(paperLink(id)); });
        return span;
    }

    /* ---------- "where our paper fits" ---------- */
    function renderPosition() {
        if (els.posAdd) {
            POSITION.add.forEach(function (r) {
                var tr = el('tr');
                tr.appendChild(el('th', null, r.topic));
                var prior = el('td');
                prior.setAttribute('data-label', 'Prior work');
                prior.appendChild(document.createTextNode(r.prior + ' '));
                prior.appendChild(refList(r.papers));
                tr.appendChild(prior);
                var ours = el('td', 'pw-ours', r.ours);
                ours.setAttribute('data-label', 'What we add');
                tr.appendChild(ours);
                els.posAdd.appendChild(tr);
            });
        }
        if (els.posKnown) {
            POSITION.known.forEach(function (k) {
                var li = el('li');
                li.appendChild(document.createTextNode(k.text + ' '));
                li.appendChild(refList(k.papers));
                els.posKnown.appendChild(li);
            });
        }
    }

    /* ---------- card ---------- */
    function badge(cls, text) { return el('span', 'pw-badge ' + cls, text); }

    function block(title, child) {
        var s = el('div', 'pw-section');
        s.appendChild(el('h4', null, title));
        s.appendChild(child);
        return s;
    }

    function list(items) {
        var ul = el('ul');
        items.forEach(function (t) { ul.appendChild(el('li', null, t)); });
        return ul;
    }

    function firstAuthor(p) {
        var names = p.authors.split(', ');
        return names.length > 1 ? names[0] + ' et al.' : names[0];
    }

    function sideLine(label, text, cls, ts) {
        var row = el('p', 'pw-side ' + cls);
        row.appendChild(el('strong', null, label + ': '));
        appendHighlighted(row, text, ts);
        return row;
    }

    function card(p, ts) {
        var c = el('article', 'card pw-card' + (p.start ? ' pw-start' : ''));
        c.id = 'paper-' + p.id;

        var h = el('h3');
        var first = (p.links && p.links[0]) ? p.links[0].u : null;
        if (first) {
            var a = el('a');
            a.href = first; a.target = '_blank'; a.rel = 'noopener noreferrer';
            appendHighlighted(a, p.title, ts);
            h.appendChild(a);
        } else {
            appendHighlighted(h, p.title, ts);
        }
        c.appendChild(h);

        var meta = el('div', 'pw-meta');
        meta.appendChild(el('span', null, firstAuthor(p)));
        meta.appendChild(el('span', 'pw-venue', p.venue + ', ' + p.year));
        meta.appendChild(badge(p.status, STATUS_LABEL[p.status]));
        if (p.start) meta.appendChild(badge('start', 'Read first'));
        if (p.origin === 'new') meta.appendChild(badge('newfind', 'Not in dossier'));
        c.appendChild(meta);

        var take = el('p', 'pw-take');
        appendHighlighted(take, p.take, ts);
        c.appendChild(take);

        var use = el('p', 'pw-use');
        use.appendChild(el('span', 'pw-role', p.role));
        appendHighlighted(use, p.relevance, ts);
        c.appendChild(use);

        var det = el('details', 'pw-detail');
        det.appendChild(el('summary', null, 'Method, results, limitations and links'));

        var sum = el('p');
        appendHighlighted(sum, p.summary, ts);
        det.appendChild(block('Summary: problem and contribution', sum));
        det.appendChild(block('Exact methodology', list(p.method)));
        det.appendChild(block('Results and findings', list(p.results)));
        det.appendChild(block('Limitations and open questions', list(p.limits)));

        var sides = el('div');
        if (p.lq) sides.appendChild(sideLine('On quantization', p.lq, 'q', ts));
        if (p.ls) sides.appendChild(sideLine('On self-improvement', p.ls, 's', ts));
        det.appendChild(block('What it covers on each side of our question', sides));

        det.appendChild(block('Authors', el('p', null, p.authors)));

        if (p.note) {
            var note = el('div', 'pw-note');
            note.appendChild(el('strong', null, 'Verification note: '));
            note.appendChild(document.createTextNode(p.note));
            det.appendChild(note);
        }
        det.appendChild(el('div', 'pw-verify', 'Read: ' + (READ_LABEL[p.read] || p.read) + (p.arxiv ? ' · arXiv ' + p.arxiv : '')
            + (p.dossier ? ' · dossier ' + p.dossier : '') + ' · ' + (catTitle[p.cat] || p.cat)));
        c.appendChild(det);

        var links = el('ul', 'pw-links');
        links.setAttribute('aria-label', 'Links');
        (p.links || []).forEach(function (l) {
            var li = el('li');
            var a2 = el('a', l.o ? 'official' : '', l.l);
            a2.href = l.u; a2.target = '_blank'; a2.rel = 'noopener noreferrer';
            li.appendChild(a2);
            links.appendChild(li);
        });
        c.appendChild(links);
        return c;
    }

    function themeIntro(t, n) {
        var box = el('section', 'pw-theme');
        box.id = 'theme-' + t.id;
        var head = el('div', 'pw-theme-head');
        head.appendChild(el('span', 'pw-theme-n', String(t.n)));
        var titles = el('div');
        titles.appendChild(el('h2', null, t.title));
        titles.appendChild(el('p', 'pw-theme-q', t.question + ' (' + n + (n === 1 ? ' paper)' : ' papers)')));
        head.appendChild(titles);
        box.appendChild(head);
        var body = el('div', 'pw-theme-body');
        var know = el('div', 'pw-know');
        know.appendChild(el('h4', null, 'What to remember'));
        know.appendChild(list(t.know));
        body.appendChild(know);
        if (t.open) {
            var open = el('div', 'pw-open');
            open.appendChild(el('h4', null, 'What is still open'));
            open.appendChild(el('p', null, t.open));
            body.appendChild(open);
        }
        box.appendChild(body);
        return box;
    }

    /* ---------- render ---------- */
    function render() {
        var ts = terms();
        var visible = PAPERS.filter(function (p) { return matches(p, false); });
        visible.sort(sorter(state.sort));

        els.list.textContent = '';
        var frag = document.createDocumentFragment();
        if (state.sort === 'theme') {
            var current = null;
            visible.forEach(function (p) {
                if (p.theme !== current) {
                    current = p.theme;
                    var n = visible.filter(function (q) { return q.theme === p.theme; }).length;
                    frag.appendChild(themeIntro(THEMES[themeIndex[p.theme]], n));
                }
                frag.appendChild(card(p, ts));
            });
        } else {
            visible.forEach(function (p) { frag.appendChild(card(p, ts)); });
        }
        els.list.appendChild(frag);

        els.empty.hidden = visible.length !== 0;
        els.count.textContent = visible.length === PAPERS.length
            ? PAPERS.length + ' papers'
            : 'Showing ' + visible.length + ' of ' + PAPERS.length + ' papers';
        renderThemeCounts();
        saveHash();
    }

    function renderThemeCounts() {
        var counts = {}, total = 0;
        PAPERS.forEach(function (p) { if (matches(p, true)) { counts[p.theme] = (counts[p.theme] || 0) + 1; total++; } });
        Array.prototype.forEach.call(els.themes.querySelectorAll('.pw-chip'), function (b) {
            var id = b.getAttribute('data-id');
            b.setAttribute('aria-pressed', state.theme === id ? 'true' : 'false');
            b.querySelector('.pw-count').textContent = String(id ? (counts[id] || 0) : total);
        });
    }

    /* ---------- controls ---------- */
    function buildControls() {
        [{ id: '', title: 'All' }].concat(THEMES).forEach(function (t) {
            var b = el('button', 'pw-chip');
            b.type = 'button'; b.setAttribute('data-id', t.id); b.setAttribute('aria-pressed', 'false');
            if (t.question) b.title = t.question;
            b.appendChild(document.createTextNode((t.n ? t.n + '. ' : '') + t.title));
            b.appendChild(el('span', 'pw-count', '0'));
            b.addEventListener('click', function () { state.theme = (state.theme === t.id) ? '' : t.id; render(); });
            els.themes.appendChild(b);
        });

        ROLES.forEach(function (r) {
            var o = document.createElement('option'); o.value = r; o.textContent = r; els.role.appendChild(o);
        });

        STATUS.forEach(function (s) {
            var lab = el('label');
            var cb = document.createElement('input');
            cb.type = 'checkbox'; cb.value = s.id;
            cb.addEventListener('change', function () { state.status[s.id] = cb.checked; render(); });
            lab.appendChild(cb);
            lab.appendChild(document.createTextNode(s.label));
            els.status.appendChild(lab);
        });

        var years = PAPERS.map(function (p) { return p.year; });
        var minY = Math.min.apply(null, years), maxY = Math.max.apply(null, years);
        [els.yearFrom, els.yearTo].forEach(function (sel) {
            var o = document.createElement('option'); o.value = ''; o.textContent = 'Any'; sel.appendChild(o);
            for (var y = minY; y <= maxY; y++) {
                var oy = document.createElement('option'); oy.value = String(y); oy.textContent = String(y); sel.appendChild(oy);
            }
        });
    }

    function syncControls() {
        els.q.value = state.q;
        els.start.checked = state.start;
        els.role.value = state.role;
        els.origin.value = state.origin;
        els.yearFrom.value = state.yearFrom;
        els.yearTo.value = state.yearTo;
        els.sort.value = state.sort;
        Array.prototype.forEach.call(els.status.querySelectorAll('input'), function (cb) { cb.checked = !!state.status[cb.value]; });
    }

    function bind() {
        var t = null;
        els.q.addEventListener('input', function () {
            clearTimeout(t);
            t = setTimeout(function () { state.q = els.q.value; render(); }, 140);
        });
        els.start.addEventListener('change', function () { state.start = els.start.checked; render(); });
        els.role.addEventListener('change', function () { state.role = els.role.value; render(); });
        els.origin.addEventListener('change', function () { state.origin = els.origin.value; render(); });
        els.yearFrom.addEventListener('change', function () { state.yearFrom = els.yearFrom.value; render(); });
        els.yearTo.addEventListener('change', function () { state.yearTo = els.yearTo.value; render(); });
        els.sort.addEventListener('change', function () { state.sort = els.sort.value; render(); });
        $('pw-expand').addEventListener('click', function () { setAll(true); });
        $('pw-collapse').addEventListener('click', function () { setAll(false); });
        $('pw-reset').addEventListener('click', function () { state = freshState(); syncControls(); render(); });
        var top = $('pw-top');
        top.addEventListener('click', function () { window.scrollTo({ top: 0, behavior: 'smooth' }); });
        window.addEventListener('scroll', function () { top.hidden = window.scrollY < 900; }, { passive: true });
    }

    function setAll(open) {
        Array.prototype.forEach.call(els.list.querySelectorAll('details.pw-detail'), function (d) { d.open = open; });
    }

    /* ---------- URL hash persistence ---------- */
    function saveHash() {
        try {
            var parts = [];
            if (state.q) parts.push('q=' + encodeURIComponent(state.q));
            if (state.theme) parts.push('theme=' + state.theme);
            if (state.start) parts.push('first=1');
            if (state.role) parts.push('role=' + encodeURIComponent(state.role));
            var ss = activeSet(state.status); if (ss.length) parts.push('status=' + ss.join(','));
            if (state.origin) parts.push('src=' + state.origin);
            if (state.yearFrom) parts.push('from=' + state.yearFrom);
            if (state.yearTo) parts.push('to=' + state.yearTo);
            if (state.sort !== SORT_DEFAULT) parts.push('sort=' + state.sort);
            var h = parts.length ? '#' + parts.join('&') : '';
            if (h !== window.location.hash) history.replaceState(null, '', window.location.pathname + window.location.search + h);
        } catch (e) { /* hash persistence is optional */ }
    }

    function loadHash() {
        try {
            var h = window.location.hash.replace(/^#/, '');
            if (!h || h.indexOf('=') === -1) return;
            h.split('&').forEach(function (kv) {
                var i = kv.indexOf('=');
                var k = kv.slice(0, i), v = decodeURIComponent(kv.slice(i + 1));
                if (k === 'q') state.q = v;
                else if (k === 'theme' && v in themeIndex) state.theme = v;
                else if (k === 'first') state.start = v === '1';
                else if (k === 'role' && ROLES.indexOf(v) !== -1) state.role = v;
                else if (k === 'status') v.split(',').forEach(function (x) { if (STATUS_LABEL[x]) state.status[x] = true; });
                else if (k === 'src') state.origin = v;
                else if (k === 'from') state.yearFrom = v;
                else if (k === 'to') state.yearTo = v;
                else if (k === 'sort') state.sort = v;
            });
        } catch (e) { /* ignore malformed hash */ }
    }

    /* ---------- counts used in the prose ---------- */
    function renderCounts() {
        var by = function (f) { return PAPERS.filter(f).length; };
        var values = {
            all: PAPERS.length,
            start: by(function (p) { return p.start; }),
            published: by(function (p) { return p.status === 'published'; }),
            accepted: by(function (p) { return p.status === 'accepted' || p.status === 'workshop'; }),
            preprint: by(function (p) { return p.status === 'preprint'; }),
            dossier: by(function (p) { return p.origin === 'dossier'; }),
            site: by(function (p) { return p.origin === 'site'; }),
            'new': by(function (p) { return p.origin === 'new'; })
        };
        Array.prototype.forEach.call(document.querySelectorAll('[data-pw-count]'), function (n) {
            var v = values[n.getAttribute('data-pw-count')];
            if (v != null) n.textContent = String(v);
        });
    }

    /* ---------- init ---------- */
    function init() {
        els.q = $('pw-q'); els.themes = $('pw-themes'); els.start = $('pw-start'); els.role = $('pw-role');
        els.status = $('pw-status'); els.origin = $('pw-origin');
        els.yearFrom = $('pw-year-from'); els.yearTo = $('pw-year-to'); els.sort = $('pw-sort');
        els.list = $('pw-list'); els.empty = $('pw-empty'); els.count = $('pw-count');
        els.posAdd = $('pw-pos-add'); els.posKnown = $('pw-pos-known');

        /* attach the guide notes to each paper; a paper without notes or a theme is a data error */
        PAPERS = PAPERS.filter(function (p) {
            var note = NOTES[p.id], lens = LENS[p.id] || {};
            if (!note || !(p.cat in catTheme)) { if (window.console) console.warn('No guide notes or theme for', p.id); return false; }
            p.theme = catTheme[p.cat];
            p.take = note.take;
            p.short = note.short;
            p.role = note.role;
            p.start = !!note.start;
            p.lq = lens.q;
            p.ls = lens.s;
            byPaperId[p.id] = p;
            return true;
        });

        buildControls();
        renderPosition();
        loadHash();
        syncControls();
        bind();
        renderCounts();
        render();
    }

    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
