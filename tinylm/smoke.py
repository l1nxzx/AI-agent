"""End-to-end CPU smoke test on real (small) data. Takes about a minute after the first download.

  python -m tinylm.smoke

Steps: prepare a tiny dataset (real downloads from Hugging Face, which also checks the dataset formats),
train 10 steps, stop, train to 20 steps (this must RESUME from the checkpoint), then sample.
Exits non-zero with a message if anything is off.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile


def run(cmd):
    print("$ " + " ".join(cmd), flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(res.stdout)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise SystemExit(f"FAILED (exit {res.returncode}): {' '.join(cmd)}")
    return res.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default=None, help="default: a fresh temp directory")
    args = ap.parse_args()
    work = args.work_dir or tempfile.mkdtemp(prefix="tinylm_smoke_")
    data_dir, run_dir = os.path.join(work, "data"), os.path.join(work, "run")
    py = [sys.executable, "-u", "-m"]
    cfg = ["--config", "configs/smoke.json", "--data_dir", data_dir, "--out_dir", run_dir]

    run(py + ["tinylm.prepare_data", "--data_dir", data_dir, "--vocab_size", "512", "--smoke"])
    out1 = run(py + ["tinylm.train"] + cfg + ["--set", "train.max_steps=10", "train.warmup_steps=3"])
    out2 = run(py + ["tinylm.train"] + cfg + ["--set", "train.max_steps=20", "train.warmup_steps=3"])
    sample = run(py + ["tinylm.sample", "--ckpt_dir", run_dir, "--chat", "--prompt", "Hello", "--max_new_tokens", "20"])

    problems = []
    if "RESUMED" not in out2 or "at step 10" not in out2:
        problems.append("second run did not resume from step 10")
    first = float(re.search(r"initial val loss stories=([\d.]+)", out1).group(1))
    last = [float(x) for x in re.findall(r"EVAL step 20 .*?mix ([\d.]+)", out2)]
    if not last or not last[-1] < first - 0.3:
        problems.append(f"validation loss did not drop enough (start {first}, end {last})")
    if "grad_norm nan" in out1 or "grad_norm nan" in out2 or "grad_norm inf" in out2:
        problems.append("NaN gradient norm")
    if "--- sample 1" not in sample:
        problems.append("sampling produced no output")
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        n = len([json.loads(line) for line in f])
    if n < 5:
        problems.append("metrics.jsonl has too few records")
    if problems:
        raise SystemExit("SMOKE TEST FAILED:\n  " + "\n  ".join(problems))
    print(f"\nSMOKE TEST PASSED (initial val loss {first:.3f} -> {last[-1]:.3f}; resume worked; work dir {work})")


if __name__ == "__main__":
    main()
