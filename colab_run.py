"""
colab_run.py — Robust all-in-one entrypoint for `colab run`

══════════════════════════════════════════════════════════════
WORKFLOW — run everything through run_experiment.sh, or by hand:
══════════════════════════════════════════════════════════════

`colab run` is NOT used here: it always provisions its own fresh,
separate VM, so anything mounted via `colab drivemount` beforehand
would not be visible to it. Instead this script runs on the SAME
session Drive was mounted on, via `colab exec`:

Step 1 — Create a session:
    colab new -s exp --gpu A100

Step 2 — Mount Google Drive on that session (handles headless OAuth):
    colab drivemount -s exp

Step 3 — Set experiment args on the session (colab exec doesn't
         forward argv the way colab run does, so pass them as an
         env var the kernel will still have set for the next call):
    echo 'import os; os.environ["EXPERIMENT_ARGS"] = \
        "--model Qwen/Qwen3.5-0.8B --precision 8bit --dataset gsm8k \
         --n_samples 16 --batch_size 64 --seed 42"' | colab exec -s exp

Step 4 — Run this script on that same session:
    colab exec -s exp -f colab_run.py

Step 5 — Stop the session (Drive files are already safe):
    colab stop -s exp

══════════════════════════════════════════════════════════════
OPTIONAL: Live monitoring from a second terminal tab:
    colab log -s exp     # Stream/export live stdout (tqdm + resource prints)
    colab status -s exp  # GPU type, uptime info
    colab console -s exp # Full shell → run: watch -n5 nvidia-smi
══════════════════════════════════════════════════════════════

Configurable via env vars:
    REPO_URL, REPO_DIR, DRIVE_DIR, RESOURCE_LOG_INTERVAL, REPO_BRANCH,
    EXPERIMENT_ARGS, EXTRA_PIP_PACKAGES, HF_TOKEN

Fallback:
    If Drive is NOT mounted, results are saved to /content/SLM/results/
    (lost when VM terminates) and a big warning is printed.
"""

import datetime
import io
import json
import os
import shlex
import subprocess
import sys
import threading
import time

# ── Configurable constants ───────────────────────────────────────────────────
REPO_URL              = os.environ.get("REPO_URL",    "https://github.com/PrachitDeshinge-324/SLM.git")
REPO_DIR              = os.environ.get("REPO_DIR",    "/content/SLM")
REPO_BRANCH           = os.environ.get("REPO_BRANCH", "")  # auto-detect if empty
DRIVE_DIR             = os.environ.get("DRIVE_DIR",   "/content/drive/MyDrive/Experiment_D_Results")
RESOURCE_LOG_INTERVAL = int(os.environ.get("RESOURCE_LOG_INTERVAL", "300"))
EXTRA_PIP_PACKAGES    = os.environ.get("EXTRA_PIP_PACKAGES", "flash-linear-attention").split()

_LOGGER_FAIL_CAP = 5
_RUN_START_TIME  = time.time()
_stop_logger     = threading.Event()

# Line-buffer stdout so `colab log` actually streams live
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True, encoding="utf-8", errors="replace")
except Exception:
    pass


# ─── Drive Detection (read-only — mounting is done externally by `colab drivemount`) ─

def _resolve_output_dir() -> str:
    """
    Check if Drive is already mounted (by `colab drivemount` before this script ran).
    If yes → use DRIVE_DIR.
    If no  → fall back to a local path inside the repo and warn loudly.
    """
    print("=" * 60)
    print("STEP 1: Resolving output directory...")
    print("=" * 60)

    if os.path.isdir("/content/drive/MyDrive"):
        try:
            os.makedirs(DRIVE_DIR, exist_ok=True)
            # Quick write test
            test_file = os.path.join(DRIVE_DIR, ".write_test")
            with _open_with_retry(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            print(f"✅ Drive is mounted and writable. Output → {DRIVE_DIR}")
            return DRIVE_DIR
        except Exception as e:
            print(f"⚠️  Drive is mounted but write test failed: {e}")
    else:
        print("⚠️  Drive is NOT mounted.")
        print("   Did you forget to run `colab drivemount` before `colab run`?")

    # Fallback
    fallback = os.path.join(REPO_DIR, "results")
    os.makedirs(fallback, exist_ok=True)
    print(f"")
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  ❌ RESULTS WILL BE LOST WHEN THE VM TERMINATES!            ║")
    print(f"║  Saving locally to: {fallback:<41}║")
    print(f"║  To fix: stop the session, run `colab drivemount`, retry.   ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print(f"")
    return fallback


# ─── Resource Collection ─────────────────────────────────────────────────────

def _collect_resource_snapshot() -> dict:
    elapsed = round(time.time() - _RUN_START_TIME, 1)
    snapshot: dict = {"timestamp": datetime.datetime.now().isoformat(), "elapsed_sec": elapsed}

    # GPU via torch
    try:
        import torch
        if torch.cuda.is_available():
            dev   = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            snapshot["gpu_name"]             = props.name
            snapshot["gpu_vram_used_gb"]     = round(torch.cuda.memory_allocated(dev) / 1e9, 2)
            snapshot["gpu_vram_reserved_gb"] = round(torch.cuda.memory_reserved(dev)   / 1e9, 2)
            snapshot["gpu_vram_total_gb"]    = round(props.total_memory                 / 1e9, 2)
        else:
            snapshot["gpu_name"] = "N/A (no CUDA)"
    except Exception as e:
        snapshot["gpu_torch_error"] = str(e)

    # System RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        snapshot["ram_used_gb"]  = round(vm.used / 1e9, 2)
        snapshot["ram_total_gb"] = round(vm.total / 1e9, 2)
        snapshot["ram_pct"]      = vm.percent
    except ImportError:
        try:
            meminfo: dict = {}
            with open("/proc/meminfo") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 2:
                        try: meminfo[parts[0].rstrip(":")] = int(parts[1])
                        except ValueError: pass
            total_kb = meminfo.get("MemTotal", 0)
            avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0) + meminfo.get("Buffers", 0) + meminfo.get("Cached", 0))
            used_kb  = total_kb - avail_kb
            snapshot["ram_used_gb"]  = round(used_kb  / 1e6, 2)
            snapshot["ram_total_gb"] = round(total_kb / 1e6, 2)
            snapshot["ram_pct"]      = round(used_kb / total_kb * 100, 1) if total_kb else 0.0
        except Exception as e:
            snapshot["ram_error"] = str(e)

    # GPU util/temp/power/VRAM via nvidia-smi (per-field guarded for MIG VMs)
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,power.draw,power.limit,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip().split(", ")
        fields = ["gpu_util_pct", "gpu_temp_c", "gpu_power_w", "gpu_power_limit_w", "gpu_vram_used_mb", "gpu_vram_total_mb"]
        for name, val in zip(fields, smi_out):
            v = val.strip()
            if v and v != "[N/A]":
                try: snapshot[name] = float(v)
                except ValueError: pass
                
        # Override PyTorch VRAM with true system VRAM if nvidia-smi returned it
        if "gpu_vram_used_mb" in snapshot:
            snapshot["gpu_vram_used_gb"] = round(snapshot["gpu_vram_used_mb"] / 1024, 2)
        if "gpu_vram_total_mb" in snapshot:
            snapshot["gpu_vram_total_gb"] = round(snapshot["gpu_vram_total_mb"] / 1024, 2)
            
    except Exception as e:
        snapshot["smi_error"] = str(e)

    return snapshot


def _resource_logger_thread(log_path: str) -> None:
    consecutive_failures = 0
    while not _stop_logger.wait(timeout=RESOURCE_LOG_INTERVAL):
        try:
            snap = _collect_resource_snapshot()
            h, rem = divmod(int(snap["elapsed_sec"]), 3600)
            m, s   = divmod(rem, 60)
            print(
                f"[RESOURCE @{h:02d}h{m:02d}m{s:02d}s] "
                f"GPU {snap.get('gpu_util_pct','?')}% | "
                f"VRAM {snap.get('gpu_vram_used_gb','?')}/{snap.get('gpu_vram_total_gb','?')} GB | "
                f"{snap.get('gpu_temp_c','?')}°C | "
                f"RAM {snap.get('ram_used_gb','?')}/{snap.get('ram_total_gb','?')} GB",
                flush=True,
            )
            with _open_with_retry(log_path, "a") as f:
                f.write(json.dumps(snap) + "\n")
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            print(f"[RESOURCE LOGGER ERROR {consecutive_failures}/{_LOGGER_FAIL_CAP}] {e}", flush=True)
            if consecutive_failures >= _LOGGER_FAIL_CAP:
                return


def _write_final_snapshot(log_path: str) -> None:
    try:
        snap = _collect_resource_snapshot()
        snap["event"] = "run_complete"
        h, rem = divmod(int(snap["elapsed_sec"]), 3600)
        m, s   = divmod(rem, 60)
        print(f"\n[FINAL RESOURCE @{h:02d}h{m:02d}m{s:02d}s]")
        print(f"  GPU VRAM : {snap.get('gpu_vram_used_gb','N/A')} / {snap.get('gpu_vram_total_gb','N/A')} GB")
        print(f"  GPU Util : {snap.get('gpu_util_pct','N/A')}%  |  Temp: {snap.get('gpu_temp_c','N/A')}°C")
        print(f"  RAM      : {snap.get('ram_used_gb','N/A')} / {snap.get('ram_total_gb','N/A')} GB")
        with _open_with_retry(log_path, "a") as f:
            f.write(json.dumps(snap) + "\n")
    except Exception as e:
        print(f"[FINAL SNAPSHOT ERROR] {e}", flush=True)


def _open_with_retry(path: str, mode: str, retries: int = 5, delay: float = 3.0):
    """
    Open a file with retries. Google Drive's FUSE mount can transiently
    report "No such file or directory" for a path moments after it was
    created and verified writable -- a known Drive-mount quirk, not specific
    to this CLI. Retrying with a short backoff, and re-creating the parent
    directory on each attempt, works around it without masking a genuinely
    missing or unwritable directory.
    """
    parent = os.path.dirname(path)
    last_err: Exception = OSError(f"Could not open {path}")
    for attempt in range(1, retries + 1):
        try:
            os.makedirs(parent, exist_ok=True)
            return open(path, mode, encoding="utf-8")
        except OSError as e:
            last_err = e
            print(f"[RETRY {attempt}/{retries}] Could not open {path}: {e}. Retrying in {delay}s...", flush=True)
            time.sleep(delay)
    raise last_err


def _tee_output(proc_stream, log_file, terminal_stream) -> None:
    """Mirror experiment stdout to terminal (with \r for tqdm) and log file (with \r→\n)."""
    import os
    warned = False
    try:
        while True:
            # Read in chunks (unbuffered) so tqdm's \r doesn't hang readline()
            chunk = os.read(proc_stream.fileno(), 1024)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            try:
                terminal_stream.write(text)
                terminal_stream.flush()
            except Exception as e:
                if not warned:
                    print(f"[TEE WARN] terminal write failed: {e}", flush=True)
                    warned = True
            try:
                log_file.write(text)
                log_file.flush()
            except Exception as e:
                if not warned:
                    print(f"[TEE WARN] log write failed: {e}", flush=True)
                    warned = True
    finally:
        try: proc_stream.close()
        except Exception: pass


def _detect_default_branch() -> str:
    if REPO_BRANCH:
        return REPO_BRANCH
    try:
        out = subprocess.check_output(
            ["git", "-C", REPO_DIR, "symbolic-ref", "refs/remotes/origin/HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        ).strip()
        return out.rsplit("/", 1)[-1] or "main"
    except Exception:
        return "main"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:

    # ── STEP 1: Resolve output directory (Drive if mounted, else local fallback) ─
    output_dir = _resolve_output_dir()

    # ── STEP 2: Install Python dependencies ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Installing Python dependencies...")
    print("=" * 60)
    BOOTSTRAP_PACKAGES = [
        "transformers>=4.36.0", "accelerate>=0.25.0", "bitsandbytes>=0.41.0",
        "pandas>=2.0.0", "pyarrow>=14.0.0", "numpy>=1.24.0",
        "matplotlib>=3.7.0", "seaborn>=0.13.0", "tqdm>=4.65.0",
        "python-dotenv>=1.0.0", "datasets>=2.14.0",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--upgrade-strategy", "only-if-needed"] + BOOTSTRAP_PACKAGES,
        check=True,
    )
    print("Bootstrap packages ready.")

    # ── STEP 3: Clone / hard-reset the GitHub repo ───────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Cloning / resetting repo from GitHub...")
    print("=" * 60)
    if os.path.exists(os.path.join(REPO_DIR, ".git")):
        branch = _detect_default_branch()
        print(f"Repo found. Resetting to origin/{branch}...")
        subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin"],                      check=False)
        subprocess.run(["git", "-C", REPO_DIR, "checkout", branch],                     check=False)
        subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{branch}"],  check=False)
        subprocess.run(["git", "-C", REPO_DIR, "clean", "-d"],                          check=False)
    else:
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print(f"Repo ready at {REPO_DIR}")

    req_path = os.path.join(REPO_DIR, "requirements.txt")
    if os.path.isfile(req_path):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "--upgrade-strategy", "only-if-needed", "-r", req_path],
            check=False,
        )

    if EXTRA_PIP_PACKAGES:
        print(f"Installing extra packages: {' '.join(EXTRA_PIP_PACKAGES)}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + EXTRA_PIP_PACKAGES,
            check=False,
        )

    script_path  = os.path.join(REPO_DIR, "scripts", "run_experiment_d.py")
    analyze_path = os.path.join(REPO_DIR, "scripts", "analyze_results.py")
    if not os.path.isfile(script_path):
        print(f"\n[FATAL] Script not found: {script_path}")
        print("Check your repo structure and REPO_URL.")
        return 1

    if not os.environ.get("HF_TOKEN"):
        print("[WARNING] HF_TOKEN is not set — gated Hugging Face models/datasets will fail to download.")

    # ── STEP 4: Start background resource logger ─────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Starting resource logger...")
    print("=" * 60)
    run_ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path   = os.path.join(output_dir, f"resource_log_{run_ts}.jsonl")
    stdout_log = os.path.join(output_dir, f"experiment_stdout_{run_ts}.log")

    logger_thread = threading.Thread(target=_resource_logger_thread, args=(log_path,))
    logger_thread.start()
    print(f"Resource logger  → {log_path}")
    print(f"Stdout tee'd to  → {stdout_log}")

    # ── STEP 5: Run the experiment ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Starting experiment...")
    print("=" * 60)

    env_args = os.environ.get("EXPERIMENT_ARGS", "")
    experiment_args = shlex.split(env_args) if env_args else list(sys.argv[1:])

    # Strip any user-supplied --output_dir — we own this flag
    while "--output_dir" in experiment_args:
        idx = experiment_args.index("--output_dir")
        experiment_args.pop(idx)
        if idx < len(experiment_args):
            experiment_args.pop(idx)
        print("[WARNING] User-supplied --output_dir ignored — Drive path is used instead.")

    cmd = [sys.executable, script_path, "--output_dir", output_dir] + experiment_args
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_DIR
    env["PYTHONUNBUFFERED"] = "1"

    print(f"Command    : {' '.join(cmd)}")
    print(f"PYTHONPATH : {REPO_DIR}")
    print("=" * 60 + "\n")

    returncode = 1
    proc       = None

    try:
        with _open_with_retry(stdout_log, "w") as log_f:
            proc = subprocess.Popen(
                cmd, cwd=REPO_DIR, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            tee_thread = threading.Thread(
                target=_tee_output, args=(proc.stdout, log_f, sys.stdout), daemon=True,
            )
            tee_thread.start()
            proc.wait()
            tee_thread.join()      # No timeout — drain the full pipe tail
            returncode = proc.returncode

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Sending SIGTERM...", flush=True)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("[INTERRUPTED] SIGTERM ignored — sending SIGKILL", flush=True)
                    proc.kill()
                    proc.wait(timeout=5)
            except Exception as e:
                print(f"[INTERRUPTED] Cleanup error: {e}", flush=True)
        returncode = 130

    except Exception as e:
        print(f"\n[FATAL] Failed to launch experiment: {e}", flush=True)
        returncode = 1

    finally:
        _stop_logger.set()
        logger_thread.join(timeout=10)
        _write_final_snapshot(log_path)

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if returncode == 0:
        print("✅ SUCCESS: Experiment complete!")
    else:
        print(f"❌ FAILED : Exit code {returncode}")
    print(f"Results    : {output_dir}")
    print(f"Stdout log : {stdout_log}")
    print(f"Resources  : {log_path}")
    print("=" * 60)

    # ── STEP 6: Analyze results (only if the experiment itself succeeded) ────
    if returncode == 0 and os.path.isfile(analyze_path):
        print("\n" + "=" * 60)
        print("STEP 6: Analyzing results...")
        print("=" * 60)
        analyze_cmd = [sys.executable, analyze_path, "--results_dir", output_dir]
        analyze_env = os.environ.copy()
        analyze_env["PYTHONPATH"] = REPO_DIR
        analyze_result = subprocess.run(analyze_cmd, cwd=REPO_DIR, env=analyze_env)
        if analyze_result.returncode != 0:
            print(f"[WARNING] analyze_results.py exited with code {analyze_result.returncode}")
    elif returncode == 0:
        print(f"[INFO] {analyze_path} not found — skipping analysis step.")

    return returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        # When this file is sent to the kernel via `colab exec -f`, a raised
        # SystemExit can surface as a cell-level error in the CLI's output
        # even though everything above already ran and printed its result.
        # colab run has documented handling that normalizes this; colab exec's
        # handling isn't documented, so swallow it here rather than risk a
        # scary but harmless traceback in the log.
        pass