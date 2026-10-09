from common import *
G = groups()
print("=== (f) pseudo-label audit on stored data. Filter = keep question if majority share (of all n samples) >= tau; pseudo-label = majority answer.")
print("    yield = share of questions kept; prec = share of kept questions whose label is correct; trace_prec = share of kept traces (those matching the label) that are correct")
for (d, m, n), P in sorted(G.items()):
    if "16bit" not in P: continue
    P = {p: v for p, v in P.items() if p == "16bit" or comparable(P, p)}
    common = sorted(set.intersection(*[set(v["Q"]) for v in P.values()]))
    if len(common) < 100: continue
    print(f"\n{d} {short(m)} n{n} nQ={len(common)}")
    for tau in (0.5, 0.75, 1.0):
        row = f"  tau={tau:.2f} "
        base = None
        for p in PREC:
            if p not in P: continue
            st = [qstats(P[p]["Q"][q]) for q in common]
            keep = np.array([s["conf_all"] >= tau for s in st]); y = np.array([s["correct"] for s in st])
            N = len(common)
            yv, pv = keep.mean(), (y[keep].mean() if keep.any() else np.nan)
            ci = bootstrap_interval(lambda ix: y[ix][keep[ix]].mean() if keep[ix].any() else np.nan, N, resamples=1000)
            row += f"| {p:5s} yield {yv:.3f} prec {pv:.3f} [{ci[0]:.3f},{ci[1]:.3f}] wrong-labels/1000q {1000*(keep & ~y).mean():5.1f} "
        print(row)

print("\n=== (g) label-free accuracy-change estimators per quantized cell (all cells with >=100 common questions)")
print("cell | prec | dAcc(majority) | dConfAll | dConfValid | dSingle | abs err: ConfAll  ConfValid  no-change")
E = []
for (d, m, n), P in sorted(G.items()):
    if "16bit" not in P: continue
    for p in ("8bit", "4bit"):
        if p not in P or not comparable(P, p): continue
        common = sorted(set(P["16bit"]["Q"]) & set(P[p]["Q"]))
        if len(common) < 100: continue
        a = [qstats(P[p]["Q"][q]) for q in common]; b = [qstats(P["16bit"]["Q"][q]) for q in common]
        f = lambda k: np.mean([x[k] for x in a]) - np.mean([x[k] for x in b])
        dacc, dca, dcv, ds = f("correct"), f("conf_all"), f("conf_valid"), f("single")
        E.append((d, dacc, dca, dcv))
        print(f"{d[:7]:7s} {short(m):13s} n{n:<2d} | {p:4s} | {100*dacc:+5.1f} | {100*dca:+5.1f} | {100*dcv:+5.1f} | {100*ds:+5.1f} | {100*abs(dacc-dca):4.1f}  {100*abs(dacc-dcv):4.1f}  {100*abs(dacc):4.1f}")
for name, sel in [("all", lambda d: True), ("gsm8k only", lambda d: d == "gsm8k"), ("cqa only", lambda d: d == "cqa")]:
    X = np.array([e[1:] for e in E if sel(e[0])])
    if len(X) < 3: continue
    print(f"  {name:10s} cells={len(X):2d}  r(dAcc,dConfAll)={np.corrcoef(X[:,0],X[:,1])[0,1]:.2f}  r(dAcc,dConfValid)={np.corrcoef(X[:,0],X[:,2])[0,1]:.2f}  MAE ConfAll={100*np.abs(X[:,0]-X[:,1]).mean():.1f}  ConfValid={100*np.abs(X[:,0]-X[:,2]).mean():.1f}  no-change={100*np.abs(X[:,0]).mean():.1f}")

print("\n=== (h) is truncation tied to question difficulty? GSM8K, 16-bit runs: questions with 0 truncated samples vs >=1")
for (d, m, n), P in sorted(G.items()):
    if d != "gsm8k" or n != 8 or "16bit" not in P: continue
    Q = P["16bit"]["Q"]
    nt = [q for q in Q if not any(s["cut"] for s in Q[q]["S"])]; tr = [q for q in Q if any(s["cut"] for s in Q[q]["S"])]
    if len(tr) < 10: continue
    acc_complete = lambda qs: np.mean([s["correct"] for q in qs for s in Q[q]["S"] if not s["cut"]])
    print(f"  {short(m):13s} no-trunc questions {len(nt):4d}: per-sample acc of complete samples {acc_complete(nt):.3f}, mean question length {np.mean([Q[q]['qlen'] for q in nt]):.0f} chars | questions with truncation {len(tr):4d}: {acc_complete(tr):.3f}, {np.mean([Q[q]['qlen'] for q in tr]):.0f} chars")
