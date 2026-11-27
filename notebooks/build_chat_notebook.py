"""Generates colab_chat.ipynb: chat with (or evaluate) a model that is already trained. No GPU needed.
Run:  python notebooks/build_chat_notebook.py"""
import json
import os

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# Chat with a model you already trained

No training and no GPU needed: a normal CPU runtime is fine for these small models. Run the cells in order.
Your trained models are on Google Drive in `MyDrive/tinylm/runs/<config>_<chat mode>/`.
""")

code("""
# 1) Settings. RUN is the folder name under MyDrive/tinylm/runs (cell 4 lists what you have).
RUN = "tiny_dailydialog_only"
REPO = "l1nxzx/AI-agent"
BRANCH = "claude/optimistic-faraday-l0rq48"
DRIVE_DIR = "/content/drive/MyDrive/tinylm"
""")

code("""
# 2) Mount Google Drive and get the code (no GPU check here).
import os, subprocess
from google.colab import drive
drive.mount("/content/drive")
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
!pip install -q huggingface_hub pyarrow numpy
RUN_DIR = f"{DRIVE_DIR}/runs/{RUN}"
""")

code("""
# 3) List the trained models on your Drive: name, training step reached, best validation loss.
import torch
base = f"{DRIVE_DIR}/runs"
for name in sorted(os.listdir(base)):
    p = os.path.join(base, name, "ckpt_best.pt")
    if os.path.exists(p):
        c = torch.load(p, map_location="cpu", weights_only=True)
        n = sum(v.numel() for k, v in c["model"].items())
        print(f"{name:32s} step {c['step']:>6}   best val {c.get('val_loss')}")
""")

code("""
# 4) Chat. Type messages in the box; 'reset' starts a new conversation, 'quit' stops.
!python -u -m tinylm.chat --ckpt_dir {RUN_DIR} --temperature 0.5 --top_k 20
""")

code("""
# 5) Run the fixed standard question list on this model (about 40 conversations). The transcript is also
# saved in the run folder as eval_chat_best.txt, so you can paste it to Claude for the report.
!python -m tinylm.eval_chat --ckpt_dir {RUN_DIR}
""")

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"}}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_chat.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
