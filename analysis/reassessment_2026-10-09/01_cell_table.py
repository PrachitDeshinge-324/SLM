from common import *
print("cell | prec | nQ | single | maj | gain | pass@k | confAll | confVal | AUROC_all | AUROC_val | hi-conf(>=.75) share | err among hi-conf | cut | fmt | none | medlen | dupQ")
for (d, m, n), P in sorted(groups().items()):
    common = sorted(set.intersection(*[set(v["Q"]) for v in P.values()]))
    if len(common) < 20: 
        print(f"-- {d} {short(m)} n{n}: only {len(common)} common questions across {list(P)}; skipped"); continue
    for p in PREC:
        if p not in P: continue
        Q = P[p]["Q"]
        st = [qstats(Q[q]) for q in common]
        y = np.array([s["correct"] for s in st]); ca = np.array([s["conf_all"] for s in st]); cv = np.array([s["conf_valid"] for s in st])
        hi = ca >= 0.75
        allS = [s for q in common for s in Q[q]["S"]]
        comp = [s for s in allS if not s["cut"]]
        fmt = np.mean([s["path"] in ("hash", "boxed") for s in comp]) if comp else float("nan")
        none = np.mean([s["ans"] is None for s in comp]) if comp else float("nan")
        dup = np.mean([len(set(s["h"] for s in Q[q]["S"])) < len(Q[q]["S"]) for q in common])
        print(f"{d[:7]:7s} {short(m):12s} n{n:<2d} | {p:5s} | {len(common):4d} | {np.mean([s['single'] for s in st]):.3f} | {y.mean():.3f} | {y.mean()-np.mean([s['single'] for s in st]):+.3f} | {np.mean([s['pass'] for s in st]):.3f} | {ca.mean():.3f} | {cv.mean():.3f} | {auroc(ca,y):.3f} | {auroc(cv,y):.3f} | {hi.mean():.3f} | {(1-y[hi].mean()) if hi.any() else float('nan'):.3f} ({int((~y[hi]).sum())}/{int(hi.sum())}) | {np.mean([s['cut'] for s in allS]):.3f} | {fmt:.3f} | {none:.3f} | {int(np.median([s['len'] for s in allS])):4d} | {dup:.2f}")
    print()
