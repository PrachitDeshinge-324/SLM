"""Does quantization move a model along its own accuracy / label-quality curve, or off it?

Three checks on the stored runs (no new generations):
  (a) cell level:     quantized cells against the curve traced by the six 16-bit models on GSM8K;
  (b) question level: at the same agreement, is the voted answer right as often as at 16-bit?
  (c) sample level:   do the wrong samples of a question agree with each other as often as at 16-bit,
                      and are they the same wrong answers?
Reads the cache and uses the definitions of ../reassessment_2026-10-09 (run 00_build_cache.py there first).
Prints the tables and writes the numbers behind 02_figure.py to outputs/01_same_curve.json.
"""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reassessment_2026-10-09"))
from common import *

G = groups()
RNG = np.random.default_rng(0)
B = 2000
GSM = [m for (d, m, n) in sorted(G) if d == "gsm8k" and n == 8]
FIG = {"curve": [], "offcurve": {"4bit": [], "8bit": [], "rerun": [], "models": []}}


def interval(stat, N):
    """95% percentile interval and standard error of stat(indices), resampling questions (paired across runs)."""
    v = np.array([stat(RNG.integers(0, N, N)) for _ in range(B)], float)
    return np.nanpercentile(v, 2.5, axis=0), np.nanpercentile(v, 97.5, axis=0), np.nanstd(v, axis=0)


def fmt(est, lo, hi, scale=100):
    star = "*" if lo > 0 or hi < 0 else " "
    return f"{scale*est:+5.1f} [{scale*lo:+5.1f},{scale*hi:+5.1f}]{star}"


def cells(precision):
    """Model x dataset cells with a valid contrast against 16-bit (same selection as 10_generality.py)."""
    for (d, m, n), P in sorted(G.items()):
        if "16bit" not in P or precision not in P or not comparable(P, precision):
            continue
        if n == 16 and (d, m, 8) in G and "16bit" in G[(d, m, 8)]:
            continue
        common = sorted(set(P["16bit"]["Q"]) & set(P[precision]["Q"]))
        if len(common) >= 100:
            yield d, m, n, P, common


def replicate_pairs():
    """Two independent runs of one model at one precision: the first 8 samples of the n16 run and the n8 run."""
    for (d, m, n), P in sorted(G.items()):
        if n != 8 or (d, m, 16) not in G:
            continue
        for p in PREC:
            if p in P and p in G[(d, m, 16)]:
                Qa, Qb = G[(d, m, 16)][p]["Q"], P[p]["Q"]
                yield d, m, p, Qa, Qb, sorted(set(Qa) & set(Qb))


def model_pairs():
    """Two different models at 16-bit on the GSM8K questions both answered."""
    for ma, mb in itertools.combinations(GSM, 2):
        Qa, Qb = G[("gsm8k", ma, 8)]["16bit"]["Q"], G[("gsm8k", mb, 8)]["16bit"]["Q"]
        yield ma, mb, Qa, Qb, sorted(set(Qa) & set(Qb))


# ------------------------------------------------------------------ (a) cell-level curve
METRICS = {
    "mean agreement": lambda a, ix: a["conf_all"][ix].mean(),
    "yield at agreement>=0.75": lambda a, ix: (a["conf_all"][ix] >= 0.75).mean(),
    "majority accuracy": lambda a, ix: a["correct"][ix].mean(),
    "label precision at >=0.75": lambda a, ix: a["correct"][ix][a["conf_all"][ix] >= 0.75].mean() if (a["conf_all"][ix] >= 0.75).any() else np.nan,
}
print("=== (a) Cell level, GSM8K 8-sample runs. Do quantized cells sit on the curve traced by the 16-bit models?")
print("    x = per-sample accuracy. For each metric a line is fitted through the 16-bit points of the OTHER five models;")
print("    along-curve = slope * change in x; off-curve = actual change - along-curve (0 = the model moved along the curve).")
print("    Intervals resample the model's own questions and hold the slope fixed, so they are too narrow where the change in x is large (Gemma at 4-bit).")
A = {}
for m in GSM:
    P = {p: v for p, v in G[("gsm8k", m, 8)].items() if p == "16bit" or comparable(G[("gsm8k", m, 8)], p)}
    common = sorted(set.intersection(*[set(v["Q"]) for v in P.values()]))
    A[m] = (len(common), {p: question_arrays(P, p, common) for p in P})
allq = lambda N: np.arange(N)
print("\n16-bit reference points: model | nQ | per-sample accuracy | " + " | ".join(METRICS))
for m in GSM:
    N, a = A[m]
    print(f"  {short(m):13s} | {N:4d} | {a['16bit']['single'].mean():.3f} | " + " | ".join(f"{f(a['16bit'], allq(N)):.3f}" for f in METRICS.values()))
    FIG["curve"].append({"model": short(m), "nQ": N, **{p: {"x": a[p]["single"].mean(), "yield": METRICS["yield at agreement>=0.75"](a[p], allq(N))} for p in a}})
for name, f in METRICS.items():
    x16 = {m: A[m][1]["16bit"]["single"].mean() for m in GSM}
    y16 = {m: f(A[m][1]["16bit"], allq(A[m][0])) for m in GSM}
    slope_all, icpt_all = np.polyfit(list(x16.values()), list(y16.values()), 1)
    if name.startswith("yield"):
        FIG["line"] = {"slope": slope_all, "intercept": icpt_all}
    loo = {}
    for m in GSM:
        others = [o for o in GSM if o != m]
        s, i = np.polyfit([x16[o] for o in others], [y16[o] for o in others], 1)
        loo[m] = (s, y16[m] - (s * x16[m] + i))
    scatter = np.sqrt(np.mean([r ** 2 for _, r in loo.values()]))
    print(f"\n--- {name}: line through the six 16-bit points: slope {slope_all:.2f}, intercept {icpt_all:+.3f}; "
          f"between-model scatter (RMS distance of a held-out 16-bit model from the others' line) {100*scatter:.1f} pts")
    print("    model | prec | change in x | actual change | along-curve | off-curve [95% CI] | held-out 16-bit model's own distance from the line")
    for m in GSM:
        N, a = A[m]
        s, own = loo[m]
        for p in ("8bit", "4bit"):
            if p not in a:
                continue
            dx = a[p]["single"].mean() - a["16bit"]["single"].mean()
            dy = f(a[p], allq(N)) - f(a["16bit"], allq(N))
            off = lambda ix: (f(a[p], ix) - f(a["16bit"], ix)) - s * (a[p]["single"][ix].mean() - a["16bit"]["single"][ix].mean())
            lo, hi, _ = interval(off, N)
            print(f"    {short(m):13s} | {p:4s} | {100*dx:+5.1f} | {100*dy:+5.1f} | {100*s*dx:+5.1f} | {fmt(dy - s*dx, lo, hi)} | {100*own:+5.1f}")

# ------------------------------------------------------------------ (b) same agreement, same trustworthiness?
EDGES = np.array([0, 1, 3, 4, 5, 6, 7, 8])   # agreement in eighths, binned 0 | 1-2 | 3 | 4 | 5 | 6 | 7 | 8
HI = 5                                       # bins 5..7 are agreement >= 0.75


def bins(conf):
    return np.searchsorted(EDGES, np.floor(conf * 8 + 1e-9), side="right") - 1


def curve(b, y):
    """P(voted answer correct | agreement bin); an empty bin borrows the nearest bin that has questions."""
    n = np.bincount(b, minlength=len(EDGES))
    f = np.bincount(b, weights=y, minlength=len(EDGES)) / np.maximum(n, 1)
    have = np.flatnonzero(n > 0)
    for i in np.flatnonzero(n == 0):
        f[i] = f[have[np.argmin(np.abs(have - i))]]
    return f, n


def decompose(ba, ya, bb, yb):
    """
    Run a against reference run b. The change in majority accuracy splits exactly into
      distribution part: reference curve applied to a's agreement levels, minus reference accuracy
      curve part:        a's accuracy minus the reference curve applied to a's agreement levels
    and the change in label precision (agreement >= 0.75) splits the same way.
    """
    fb, nb = curve(bb, yb)
    cf = fb[ba]
    ha, hb = ba >= HI, bb >= HI
    pa = ya[ha].mean() if ha.any() else np.nan
    pb = yb[hb].mean() if hb.any() else np.nan
    pcf = cf[ha].mean() if ha.any() else np.nan
    return np.array([ya.mean() - yb.mean(), cf.mean() - yb.mean(), ya.mean() - cf.mean(),
                     pa - pb, pcf - pb, pa - pcf, (nb[ba] == 0).mean()])


def contrast(qa, qb):
    """Decomposition of run a against run b (per-question stats on the same questions), with intervals."""
    ba, bb = bins(np.array([s["conf_all"] for s in qa])), bins(np.array([s["conf_all"] for s in qb]))
    ya, yb = np.array([s["correct"] for s in qa], float), np.array([s["correct"] for s in qb], float)
    est = decompose(ba, ya, bb, yb)
    lo, hi, se = interval(lambda ix: decompose(ba[ix], ya[ix], bb[ix], yb[ix]), len(qa))
    return est, lo, hi, se


def row(est, lo, hi):
    return (f"{100*est[0]:+5.1f} = {100*est[1]:+5.1f} + {fmt(est[2], lo[2], hi[2])} | "
            f"{100*est[3]:+5.1f} = {100*est[4]:+5.1f} + {fmt(est[5], lo[5], hi[5])} | {est[6]:.2f}")


def keep(group, label, nq, est, lo, hi):
    FIG["offcurve"][group].append({"label": label, "nQ": nq, "est": 100 * est[2], "lo": 100 * lo[2], "hi": 100 * hi[2]})


print("\n\n=== (b) Question level. At the same agreement, is the voted answer right as often as at 16-bit?")
print("    change in majority accuracy  = distribution part (the model lands on lower agreement levels) + curve part (same agreement, different accuracy)")
print("    change in label precision at agreement>=0.75 splits the same way. Curve part = 0 means the model moved along its own 16-bit calibration curve.")
print("    Last column: share of the run's questions in an agreement bin that is empty in the reference run (nearest bin used).")
HEAD = "d majority = distribution + curve part [95% CI] | d label precision = mix + curve part [95% CI] | empty-bin share"
for p in ("4bit", "8bit"):
    print(f"\n--- {p} against 16-bit\ncell | nQ | {HEAD}")
    pooled = {}
    for d, m, n, P, common in cells(p):
        est, lo, hi, se = contrast([qstats(P[p]["Q"][q]) for q in common], [qstats(P["16bit"]["Q"][q]) for q in common])
        pooled[(d, short(m))] = (est, se, len(common))
        keep(p, f"{d[:7]} {short(m)}", len(common), est, lo, hi)
        print(f"{d[:7]:7s} {short(m):13s} n{n:<2d} | {len(common):4d} | {row(est, lo, hi)}")
    for label, sub in [("all cells", pooled),
                       ("without Gemma", {k: v for k, v in pooled.items() if "gemma" not in k[1]}),
                       ("full test sets only", {k: v for k, v in pooled.items() if v[2] >= 1000})]:
        for i, what in [(2, "curve part of majority accuracy"), (5, "curve part of label precision")]:
            k, mean, sem, tau, i2 = pool_random_effects([100 * v[0][i] for v in sub.values()], [100 * v[1][i] for v in sub.values()])
            print(f"  pooled {what:32s} ({label}, {k} cells): {mean:+5.2f} [{mean-1.96*sem:+5.2f},{mean+1.96*sem:+5.2f}]  tau {tau:.2f}")

print(f"\n--- the same contrasts in the 16-sample runs (a second pair of runs for cells that already appear above)\ncell | prec | nQ | {HEAD}")
for (d, m, n), P in sorted(G.items()):
    if n != 16 or "16bit" not in P or "16bit" not in G.get((d, m, 8), {}):
        continue
    for p in ("4bit", "8bit"):
        if p not in P or not comparable(P, p):
            continue
        common = sorted(set(P["16bit"]["Q"]) & set(P[p]["Q"]))
        est, lo, hi, _ = contrast([qstats(P[p]["Q"][q]) for q in common], [qstats(P["16bit"]["Q"][q]) for q in common])
        print(f"{d[:7]:7s} {short(m):13s} n16 | {p:4s} | {len(common):4d} | {row(est, lo, hi)}")

print("\n--- yardstick 1: two independent runs of the SAME model at the SAME precision (first 8 samples of the n16 run against the n8 run)")
print(f"cell | prec | nQ | {HEAD}")
for d, m, p, Qa, Qb, common in replicate_pairs():
    est, lo, hi, _ = contrast([qstats(Qa[q], range(8)) for q in common], [qstats(Qb[q]) for q in common])
    keep("rerun", f"{d[:7]} {short(m)} {p}", len(common), est, lo, hi)
    print(f"{d[:7]:7s} {short(m):13s} | {p:5s} | {len(common):4d} | {row(est, lo, hi)}")

print("\n--- yardstick 2: two DIFFERENT models at 16-bit on GSM8K (the less accurate model against the more accurate one, shared questions)")
print(f"weaker model vs stronger model | nQ | {HEAD}")
between = []
for ma, mb, Qa, Qb, common in model_pairs():
    sa, sb = [qstats(Qa[q]) for q in common], [qstats(Qb[q]) for q in common]
    if np.mean([s["single"] for s in sa]) > np.mean([s["single"] for s in sb]):
        ma, mb, sa, sb = mb, ma, sb, sa
    est, lo, hi, se = contrast(sa, sb)
    between.append((est, se))
    keep("models", f"{short(ma)} vs {short(mb)}", len(common), est, lo, hi)
    print(f"{short(ma):13s} vs {short(mb):13s} | {len(common):4d} | {row(est, lo, hi)}")
for i, what in [(2, "curve part of majority accuracy"), (5, "curve part of label precision")]:
    v = np.array([100 * e[i] for e, _ in between])
    k, mean, sem, tau, i2 = pool_random_effects(v, [100 * s[i] for _, s in between])
    print(f"  {what:32s} over {len(v)} model pairs: median absolute {np.nanmedian(np.abs(v)):.1f} pts, range {np.nanmin(v):+.1f}..{np.nanmax(v):+.1f}, tau {tau:.2f}")


# ------------------------------------------------------------------ (c) error structure
def wrong(q, idx=None):
    """Per sample: its answer if the sample is complete, parsed and wrong, otherwise None."""
    S = q["S"] if idx is None else [q["S"][i] for i in idx]
    return [s["ans"] if s["ans"] is not None and not s["cut"] and not s["correct"] else None for s in S]


def within(a):
    """Unbiased estimate of P(two wrong samples of this question give the same answer)."""
    c = collections.Counter(x for x in a if x is not None)
    w = sum(c.values())
    return sum(k * (k - 1) for k in c.values()) / (w * (w - 1))


def across(a, b):
    """
    P(a wrong sample of run a and a wrong sample of run b give the same answer). Pairs with the same
    sample index are left out: precisions share per-question seeds, so those two samples are not independent.
    """
    return np.mean([x == y for i, x in enumerate(a) for j, y in enumerate(b) if i != j and x is not None and y is not None])


def structure_row(Wa, Wb):
    """Runs a and b on the same questions; only questions with at least two wrong samples in both."""
    ok = [i for i in range(len(Wa)) if sum(x is not None for x in Wa[i]) >= 2 and sum(x is not None for x in Wb[i]) >= 2]
    wa, wb = np.array([within(Wa[i]) for i in ok]), np.array([within(Wb[i]) for i in ok])
    x = np.array([across(Wa[i], Wb[i]) for i in ok])
    stat = lambda ix: np.array([wa[ix].mean() - wb[ix].mean(), x[ix].mean() - (wa[ix].mean() + wb[ix].mean()) / 2])
    est = stat(np.arange(len(ok)))
    lo, hi, _ = interval(stat, len(ok))
    same = [x == y for a, b in zip(Wa, Wb) for x, y in zip(a, b) if x is not None and y is not None]
    return (f"{len(ok):4d} | {wb.mean():.3f} | {wa.mean():.3f} | {fmt(est[0], lo[0], hi[0])} | {x.mean():.3f} | "
            f"{fmt(est[1], lo[1], hi[1])} | {np.mean(same):.3f} (n={len(same)})")


print("\n\n=== (c) Sample level. Do wrong samples agree with each other more under quantization, and are they the same wrong answers?")
print("    within = P(two wrong samples of one question give the same answer), averaged over questions with >=2 wrong samples in both runs.")
print("    across = P(a wrong sample from one run and a wrong sample from the other give the same answer), sample pairs with the same index left out.")
print("    If quantization only makes the model wrong more often, within is unchanged and across equals within.")
print("    Last column: the left-out same-index pairs. A share far above 'across' means the two runs followed the same random numbers.")
SHEAD = "questions | within (reference) | within (run) | difference [95% CI] | across | across - mean within [95% CI] | same-index pairs equal"
for p in ("4bit", "8bit"):
    print(f"\n--- {p} (run) against 16-bit (reference)\ncell | {SHEAD}")
    for d, m, n, P, common in cells(p):
        print(f"{d[:7]:7s} {short(m):13s} n{n:<2d} | " + structure_row([wrong(P[p]["Q"][q]) for q in common], [wrong(P["16bit"]["Q"][q]) for q in common]))

print(f"\n--- yardstick 1: two independent runs of the SAME model at the SAME precision (n16 run, first 8 samples = run; n8 run = reference)\ncell | prec | {SHEAD}")
for d, m, p, Qa, Qb, common in replicate_pairs():
    print(f"{d[:7]:7s} {short(m):13s} | {p:5s} | " + structure_row([wrong(Qa[q], range(8)) for q in common], [wrong(Qb[q]) for q in common]))

print(f"\n--- yardstick 2: two DIFFERENT models at 16-bit on GSM8K\nrun vs reference | {SHEAD}")
for ma, mb, Qa, Qb, common in model_pairs():
    print(f"{short(ma):13s} vs {short(mb):13s} | " + structure_row([wrong(Qa[q]) for q in common], [wrong(Qb[q]) for q in common]))

# ------------------------------------------------------------------ (d) size within a family
print("\n\n=== (d) Does the smaller model of a family lose more at 4-bit? GSM8K, questions that all four runs share.")
print("    Only two families have two sizes, so this is two comparisons, not a trend.")
print("family: small vs large | nQ | metric | 4-bit change, small | 4-bit change, large | small minus large [95% CI]")
for small, large in [("Qwen_Qwen3.5-0.8B", "Qwen_Qwen3.5-2B"), ("meta-llama_Llama-3.2-1B-Instruct", "meta-llama_Llama-3.2-3B-Instruct")]:
    runs = {k: G[("gsm8k", m, 8)] for k, m in (("small", small), ("large", large))}
    common = sorted(set.intersection(*[set(P[p]["Q"]) for P in runs.values() for p in ("16bit", "4bit")]))
    arr = {(k, p): question_arrays(P, p, common) for k, P in runs.items() for p in ("16bit", "4bit")}
    for name, f in [("per-sample accuracy", lambda a, ix: a["single"][ix].mean())] + list(METRICS.items())[1:]:
        change = lambda k, ix: f(arr[(k, "4bit")], ix) - f(arr[(k, "16bit")], ix)
        gap = lambda ix: change("small", ix) - change("large", ix)
        lo, hi, _ = interval(gap, len(common))
        ix = allq(len(common))
        print(f"{short(small):13s} vs {short(large):11s} | {len(common):4d} | {name:26s} | {100*change('small', ix):+5.1f} | {100*change('large', ix):+5.1f} | {fmt(gap(ix), lo, hi)}")

json.dump(FIG, open(os.path.join(HERE, "outputs", "01_same_curve.json"), "w"), indent=1)
