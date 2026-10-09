"""Build a per-sample cache of every generator run (read-only on the repo)."""
import glob, json, os, pickle, re, sys, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from src.experiment_d.answer_extraction import get_extractor, _strip_think, _last_boxed, _NUM
from src.experiment_d.answer_matching import answers_match, normalize_answer
from src.experiment_d.run_files import is_legacy_file, parse_run_name

OUT = os.path.join(os.path.dirname(__file__), "cache.pkl")

def gsm_path(text):
    t = _strip_think(text)
    if re.findall(r'####[^\d\-\n]{0,40}?(' + _NUM + ')', t): return "hash"
    b = _last_boxed(t)
    if b is not None and re.search(r'(' + _NUM + ')', b.replace('\\$', '')): return "boxed"
    if re.findall(r'=\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', t): return "eq"
    if re.search(r'(?:the answer is|final answer is|therefore,?\s*(?:the )?\w+ is|she makes|he makes|profit is|total is)\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', t, flags=re.I): return "text"
    if re.findall(r'\$\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', t): return "dollar"
    if re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', t): return "last"
    return "none"

def cqa_path(text):
    t = _strip_think(text)
    if re.search(r'(?:Answer|####)\s*:?\s*\(?([A-E])\)?', t, flags=re.I): return "hash"
    return "fallback" if re.findall(r'\b([A-E])\b', t) else "none"

def math_path(text):
    t = _strip_think(text)
    return "boxed" if _last_boxed(t) is not None else "fb_" + gsm_path(text)

runs = {}
t0 = time.time()
for path in sorted(glob.glob(os.path.join(REPO, "results", "*", "generator", "*", "*.jsonl"))):
    run = parse_run_name(path)
    if run is None or is_legacy_file(path): continue   # legacy backup files are never analysed
    model, prec, n = run["model"], run["precision"], run["n_samples"]
    dataset = path.split(os.sep)[-4]
    dkey = "ChilleD/SVAMP" if dataset == "ChilleD_SVAMP" else dataset
    ex = get_extractor(dkey)
    pathfn = cqa_path if dataset == "cqa" else math_path if dataset == "math500" else gsm_path
    cfg, Q = None, {}
    for line in open(path):
        r = json.loads(line)
        if r.get("type") == "run_config": cfg = r; continue
        if "raw_samples" not in r: continue
        S = []
        for s in r["raw_samples"]:
            resp = s["response"]
            a = ex(resp)
            S.append({
                "ans": None if a is None else normalize_answer(a),
                "raw": a,
                "correct": bool(a is not None and answers_match(a, r["ground_truth"])),
                "cut": bool(s["cutoff"]), "len": s["length"], "chars": len(resp),
                "path": pathfn(resp), "think": "</think>" in resp,
                "h": hash(resp),
            })
        Q[r["qid"]] = {"gt": r["ground_truth"], "qlen": len(r["question"]), "S": S}
    runs[(dataset, model, n, prec)] = {"cfg": cfg, "Q": Q}
    print(f"{time.time()-t0:6.1f}s {dataset:14s} {model[-26:]:26s} n{n:<2d} {prec:5s} q={len(Q)}", flush=True)
pickle.dump(runs, open(OUT, "wb"))
print("saved", OUT)
