"""Generates colab_train.ipynb. Run:  python notebooks/build_notebook.py"""
import json
import os

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# Train a small language model from scratch (Colab T4)

Run the cells top to bottom (Runtime > Run all works too). If the session disconnects, reconnect and
run all cells again: data preparation is skipped when already done, and training resumes from the
latest checkpoint on your Google Drive.

Before starting: Runtime > Change runtime type > T4 GPU.
""")

code("""
# 1) Settings. Change CONFIG to "small" or "medium" for bigger models (see configs/ in the repo).
CONFIG = "tiny"            # tiny (~2M), small (~11M), medium (~20M)
REPO = "l1nxzx/AI-agent"
BRANCH = "claude/optimistic-faraday-l0rq48"   # change to "main" once merged
DRIVE_DIR = "/content/drive/MyDrive/tinylm"   # data, checkpoints and logs live here
OVERRIDES = ""             # optional, e.g. "train.max_steps=2000 train.batch_size=32"
CHAT_MODE = "dailydialog_identity"  # dailydialog_identity: real dialogues + small Petryk identity layer | dailydialog_only | dailydialog | persona | real
""")

code("""
# 2) Check the GPU and mount Google Drive (a permission prompt appears once).
import subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
from google.colab import drive
drive.mount("/content/drive")
""")

code("""
# 3) Get the code and install dependencies.
# If the repo is private, add a Colab secret named GITHUB_TOKEN (key icon in the left sidebar).
import os, subprocess
try:
    from google.colab import userdata
    token = userdata.get("GITHUB_TOKEN")
except Exception:
    token = None
url = f"https://{token + '@' if token else ''}github.com/{REPO}.git"
if not os.path.isdir("/content/AI-agent/.git"):
    subprocess.run(["git", "clone", "--branch", BRANCH, url, "/content/AI-agent"], check=True)
else:
    subprocess.run(["git", "-C", "/content/AI-agent", "fetch", url, BRANCH], check=True)
    subprocess.run(["git", "-C", "/content/AI-agent", "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", "/content/AI-agent", "reset", "--hard", "FETCH_HEAD"], check=True)
os.chdir("/content/AI-agent")
print(subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True).stdout)
!pip install -q huggingface_hub pyarrow numpy
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU"
""")

code("""
# 4) Download and prepare the data (TinyStories + SmolTalk everyday-conversations).
# The result is stored on Drive, so this only runs once per vocab size.
import json
VOCAB = json.load(open(f"configs/{CONFIG}.json"))["model"]["vocab_size"]
DATA_DIR = f"{DRIVE_DIR}/data_v{VOCAB}_{CHAT_MODE}"
RUN_DIR = f"{DRIVE_DIR}/runs/{CONFIG}_{CHAT_MODE}"
!python -u -m tinylm.prepare_data --data_dir {DATA_DIR} --cache_dir /content/hf_cache --vocab_size {VOCAB} --chat_mode {CHAT_MODE}
""")

code("""
# 5) Train. Safe to run again after a disconnect: it resumes from {RUN_DIR}/ckpt_latest.pt.
# The first lines show the parameter count; then a log line every 50 steps and an EVAL line with sample text.
!python -u -m tinylm.train --config configs/{CONFIG}.json --data_dir {DATA_DIR} --out_dir {RUN_DIR} --set {OVERRIDES}
""")

code("""
# 6) Look at progress: the end of the log, and loss curves (can be run while training is running in another session).
!tail -n 25 {RUN_DIR}/train.log
import json
import matplotlib.pyplot as plt
rows = [json.loads(l) for l in open(f"{RUN_DIR}/metrics.jsonl")]
tr = [r for r in rows if r["type"] == "train"]
ev = [r for r in rows if r["type"] == "eval"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([r["step"] for r in tr], [r["loss"] for r in tr], label="train loss")
ax.plot([r["step"] for r in ev], [r["val_stories"] for r in ev], "o-", label="val stories")
ax.plot([r["step"] for r in ev], [r["val_chat"] for r in ev], "s-", label="val chat")
ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(); ax.grid(alpha=0.3)
plt.show()
""")

code("""
# 7) Generate text from the best checkpoint.
!python -m tinylm.sample --ckpt_dir {RUN_DIR} --prompt "Once upon a time" --num_samples 2 --temperature 0.8 --top_k 40
!python -m tinylm.sample --ckpt_dir {RUN_DIR} --chat --prompt "Hi, how are you today?" --num_samples 2 --temperature 0.5 --top_k 20
""")

code("""
# 8) Chat with the model (type messages in the box; 'quit' to stop, 'reset' to start a new conversation).
# If you updated the code since cell 3, re-run cell 3 first.
!python -u -m tinylm.chat --ckpt_dir {RUN_DIR} --temperature 0.5 --top_k 20
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"}}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_train.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
