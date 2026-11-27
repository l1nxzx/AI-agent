"""Train, resume and sample on a tiny synthetic corpus, with no network access."""
import json
import os

import pytest

from tinylm.bpe import BPETokenizer
from tinylm.config import load_config
from tinylm.data import write_dataset
from tinylm.logutil import setup_logging
from tinylm.sample import main as sample_main
from tinylm.train import PREV, LATEST, lr_at, main as train_main

STORIES = [f"Once upon a time a dog named Rex number {i} ran to the park and played with a ball. It was fun." for i in range(60)]
CONVS = [[("user", f"Hello number {i}"), ("assistant", "Hi! How can I help you today?")] for i in range(30)]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    data = str(root / "data")
    setup_logging()
    tok = BPETokenizer.train(STORIES + [c for conv in CONVS for _, c in conv], 300)
    write_dataset(data, tok, STORIES[:50], STORIES[50:], CONVS[:25], CONVS[25:])
    cfg = {"model": {"vocab_size": 300, "n_layer": 1, "n_head": 2, "dim": 32, "context_len": 32},
           "train": {"batch_size": 4, "max_steps": 8, "warmup_steps": 2, "eval_interval": 4, "eval_batches": 2,
                     "log_interval": 2, "ckpt_interval": 4, "sample_tokens": 5}}
    cfg_path = str(root / "cfg.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    return root, data, cfg_path


def test_train_resume_sample(setup):
    root, data, cfg = setup
    out = str(root / "out")
    base = ["--config", cfg, "--data_dir", data, "--out_dir", out]
    train_main(base + ["--set", "train.max_steps=4"])
    assert os.path.exists(os.path.join(out, LATEST))
    train_main(base)  # resumes from step 4 and runs to 8
    log = open(os.path.join(out, "train.log")).read()
    assert "RESUMED" in log and "at step 4" in log
    assert os.path.exists(os.path.join(out, PREV))
    records = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert records[-1]["step"] == 8
    sample_main(["--ckpt_dir", out, "--prompt", "Once", "--max_new_tokens", "5", "--device", "cpu"])


def test_changed_model_config_refuses_resume(setup):
    root, data, cfg = setup
    out = str(root / "out2")
    base = ["--config", cfg, "--data_dir", data, "--out_dir", out]
    train_main(base + ["--set", "train.max_steps=4"])
    with pytest.raises(ValueError, match="different model config"):
        train_main(base + ["--set", "model.n_layer=2"])


def test_vocab_mismatch_is_caught(setup):
    root, data, cfg = setup
    with pytest.raises(ValueError, match="vocab_size"):
        train_main(["--config", cfg, "--data_dir", data, "--out_dir", str(root / "out3"), "--set", "model.vocab_size=400"])


def test_lr_schedule():
    from tinylm.config import TrainConfig
    tc = TrainConfig(lr=1.0, min_lr_ratio=0.1, warmup_steps=10, max_steps=110)
    assert lr_at(0, tc) == pytest.approx(0.1)          # warmup starts above zero
    assert lr_at(9, tc) == pytest.approx(1.0)          # warmup ends at peak
    assert lr_at(10, tc) == pytest.approx(1.0)         # cosine starts at peak
    assert lr_at(60, tc) == pytest.approx(0.55)        # halfway: midpoint of 1.0 and 0.1
    assert lr_at(110, tc) == pytest.approx(0.1)        # ends at min_lr
    assert lr_at(10_000, tc) == pytest.approx(0.1)     # stays at min_lr


def test_chat_loop_and_history_trim(setup, monkeypatch, capsys):
    from tinylm.chat import main as chat_main, trim_history
    root, data, cfg = setup
    out = str(root / "out")
    if not os.path.exists(os.path.join(out, "ckpt_best.pt")):
        train_main(["--config", cfg, "--data_dir", data, "--out_dir", out])
    inputs = iter(["Hello there", "How are you", "reset", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    chat_main(["--ckpt_dir", out, "--max_new_tokens", "6", "--device", "cpu"])
    printed = capsys.readouterr().out
    assert printed.count("bot:") == 2 and "history cleared" in printed
    tok = BPETokenizer.load(os.path.join(out, "tokenizer.json"))
    u, e, b = tok.token_id("<|user|>"), tok.token_id("<|end|>"), tok.token_id("<|bot|>")
    hist = [u, 1, 2, e, b, 3, e, u, 4, e, b, 5, e]
    assert trim_history(list(hist), tok, 8)[0] == u and len(trim_history(list(hist), tok, 8)) <= 8


def test_eval_chat_runs_standard_set(setup, capsys):
    from tinylm.eval_chat import STANDARD_SET, main as eval_main
    root, data, cfg = setup
    out = str(root / "out")
    if not os.path.exists(os.path.join(out, "ckpt_best.pt")):
        train_main(["--config", cfg, "--data_dir", data, "--out_dir", out])
    eval_main(["--ckpt_dir", out, "--max_new_tokens", "4", "--device", "cpu"])
    printed = capsys.readouterr().out
    assert printed.count("you:") >= len(STANDARD_SET)
    result = json.load(open(os.path.join(out, "eval_chat_best.json")))
    assert len(result["records"]) == sum(r for _, _, r in STANDARD_SET)
    assert "=== identity ===" in open(os.path.join(out, "eval_chat_best.txt")).read()


def test_early_stopping(setup, monkeypatch):
    import torch
    root, data, cfg = setup
    out = str(root / "out_early")
    calls = {"n": 0}
    sequence = [5.0, 4.0, 4.5, 4.6, 4.7, 4.8]   # call 0 is the initial eval; then best at step 2, worse at 4 and 6

    def fake_evaluate(model, data_, tc, device, ctx):
        v = sequence[min(calls["n"], len(sequence) - 1)]
        calls["n"] += 1
        return {"stories": v, "chat": v, "mix": v}
    monkeypatch.setattr("tinylm.train.evaluate", fake_evaluate)
    base = ["--config", cfg, "--data_dir", data, "--out_dir", out]
    over = ["--set", "train.max_steps=30", "train.eval_interval=2", "train.early_stop_patience=2",
            "train.ckpt_interval=100", "train.warmup_steps=2", "train.log_interval=2"]
    train_main(base + over)
    log = open(os.path.join(out, "train.log")).read()
    assert "EARLY STOP at step 6" in log
    latest = torch.load(os.path.join(out, LATEST), weights_only=True)
    assert latest["step"] == 6 and latest["early_stopped"] is True and latest["evals_since_best"] == 2
    assert torch.load(os.path.join(out, "ckpt_best.pt"), weights_only=True)["step"] == 2
    records = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"))]
    assert records[-1]["type"] == "early_stop"
    # running it again does nothing
    train_main(base + over)
    assert "already stopped early at step 6" in open(os.path.join(out, "train.log")).read()
    assert torch.load(os.path.join(out, LATEST), weights_only=True)["step"] == 6
    # with early stopping switched off it carries on from the checkpoint
    train_main(base + ["--set", "train.max_steps=10", "train.eval_interval=2", "train.early_stop_patience=0",
                       "train.ckpt_interval=100", "train.warmup_steps=2"])
    assert torch.load(os.path.join(out, LATEST), weights_only=True)["step"] == 10


def test_early_stop_patience_validation():
    from tinylm.config import TrainConfig
    with pytest.raises(ValueError):
        TrainConfig(early_stop_patience=-1).validate()
