"""Training loop with validation, LR schedule, logging and resumable checkpoints.

Usage:
  python -m tinylm.train --config configs/tiny.json --data_dir data --out_dir runs/tiny
  python -m tinylm.train --config configs/tiny.json --data_dir data --out_dir runs/tiny \
      --set train.max_steps=100 train.batch_size=16

If out_dir already contains ckpt_latest.pt, training resumes from it automatically (this is how a
Colab session that disconnected carries on).

What one optimizer step does:
  1. grad_accum_steps times: draw a batch, run the model forward, compute next-token cross-entropy,
     backpropagate (gradients add up across the micro-steps).
  2. Clip the global gradient norm (guards against rare huge updates).
  3. AdamW update with the learning rate from the warmup + cosine schedule.
"""
import argparse
import json
import math
import os
import platform
import random
import time

import numpy as np
import torch

from .bpe import BPETokenizer
from .config import config_from_dict, load_config
from .data import DataSource
from .logutil import MetricsWriter, get_logger, setup_logging
from .model import GPT, count_parameters
from .sample import chat_prompt_ids, generate, stop_ids_for, story_prompt_ids

LATEST, PREV, BEST = "ckpt_latest.pt", "ckpt_prev.pt", "ckpt_best.pt"


def lr_at(step, tc):
    """Learning rate for optimizer step `step` (0-based): linear warmup, then cosine decay to
    lr * min_lr_ratio. Warmup avoids unstable early updates when Adam's statistics are still noisy;
    decay lets the model settle into a minimum at the end."""
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    progress = min(1.0, (step - tc.warmup_steps) / max(1, tc.max_steps - tc.warmup_steps))
    min_lr = tc.lr * tc.min_lr_ratio
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (tc.lr - min_lr)


def make_optimizer(model, tc, device):
    # Weight decay only on matrices (linear layers, embeddings); not on 1-D norm gains.
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": tc.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    kwargs = {"fused": True} if device.type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=tc.lr, betas=(tc.beta1, tc.beta2), **kwargs)


def save_checkpoint(out_dir, name, payload):
    """Write to a temp file, then rename, so a crash mid-write cannot corrupt the checkpoint."""
    path = os.path.join(out_dir, name)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    if name == LATEST and os.path.exists(path):
        os.replace(path, os.path.join(out_dir, PREV))   # keep one older copy as a fallback
    os.replace(tmp, path)


def load_latest(out_dir, device):
    log = get_logger()
    for name in (LATEST, PREV):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            try:
                return torch.load(path, map_location=device, weights_only=True), path
            except Exception as e:  # truncated file after a crash
                log.warning(f"could not read {path}: {e!r}; trying older checkpoint")
    return None, None


@torch.no_grad()
def evaluate(model, data, tc, device, autocast_ctx):
    """Mean loss over fixed validation batches, separately for stories and chat."""
    model.eval()
    result = {}
    for source in ("stories", "chat"):
        losses = []
        for i in range(tc.eval_batches):
            x, y = data.val_batch(source, i, tc.batch_size)
            with autocast_ctx:
                _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
        result[source] = float(np.mean(losses))
    result["mix"] = (1 - tc.chat_fraction) * result["stories"] + tc.chat_fraction * result["chat"]
    model.train()
    return result


def log_samples(model, tok, tc, device):
    log = get_logger()
    gen = torch.Generator().manual_seed(0)
    stop = stop_ids_for(tok)
    s = generate(model, story_prompt_ids(tok, "Once upon a time"), tc.sample_tokens, 0.8, 40, stop, gen)
    log.info("sample [story] Once upon a time" + tok.decode(s).replace("\n", " "))
    c = generate(model, chat_prompt_ids(tok, "Hi, how are you?"), tc.sample_tokens, 0.8, 40, stop, gen)
    log.info("sample [chat] user: Hi, how are you? | bot:" + tok.decode(c).replace("\n", " "))


def describe_environment(device):
    log = get_logger()
    log.info(f"python {platform.python_version()} | torch {torch.__version__} | numpy {np.__version__} | {platform.platform()}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        log.info(f"device: cuda | {props.name} | {props.total_memory / 1e9:.1f} GB | capability {props.major}.{props.minor}")
    else:
        log.info(f"device: cpu | threads={torch.get_num_threads()}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="runs/run")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.max_steps=100 model.n_layer=2")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    log = setup_logging(os.path.join(args.out_dir, "train.log"))
    metrics = MetricsWriter(os.path.join(args.out_dir, "metrics.jsonl"))
    cfg = load_config(args.config, args.set)
    mc, tc = cfg.model, cfg.train
    log.info("=" * 80)
    log.info(f"training run started | out_dir={args.out_dir} | data_dir={args.data_dir}")
    log.info("config: " + json.dumps(cfg.to_dict()))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    describe_environment(device)
    random.seed(tc.seed); np.random.seed(tc.seed); torch.manual_seed(tc.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- precision --------------------------------------------------------------------------
    dtype_name = tc.dtype if tc.dtype != "auto" else ("fp16" if device.type == "cuda" else "fp32")
    if device.type == "cpu" and dtype_name == "fp16":
        log.warning("fp16 is not supported on CPU; using fp32")
        dtype_name = "fp32"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype_name)
    autocast_ctx = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else torch.autocast(device.type, enabled=False)
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype_name == "fp16"))
    log.info(f"precision: {dtype_name} (loss scaling {'on' if scaler.is_enabled() else 'off'})")

    # ---- data and tokenizer -----------------------------------------------------------------
    tok = BPETokenizer.load(os.path.join(args.data_dir, "tokenizer.json"))
    data = DataSource(args.data_dir, mc.context_len, tc.chat_fraction, tc.seed)
    if tok.vocab_size != mc.vocab_size:
        raise ValueError(f"config vocab_size={mc.vocab_size} but the tokenizer in {args.data_dir} has "
                         f"{tok.vocab_size}. Re-run prepare_data with --vocab_size {mc.vocab_size}, or fix the config.")
    if data.meta["tokenizer_fingerprint"] != tok.fingerprint():
        raise ValueError("tokenizer.json does not match the token files (fingerprint differs); re-run prepare_data --force")
    tok.save(os.path.join(args.out_dir, "tokenizer.json"))  # keep the run directory self-contained
    for name, arr in data.arrays.items():
        log.info(f"data {name}: {len(arr):,} tokens")
    eff_batch = tc.batch_size * tc.grad_accum_steps
    tok_per_step = eff_batch * mc.context_len
    story_tokens = len(data.arrays["stories_train"])
    log.info(f"batch: {tc.batch_size} x {tc.grad_accum_steps} accum = {eff_batch} sequences x {mc.context_len} tokens = "
             f"{tok_per_step:,} tokens/step; chat_fraction={tc.chat_fraction}")
    total_tokens = tok_per_step * tc.max_steps
    log.info(f"plan: {tc.max_steps} steps = {total_tokens:,} tokens "
             f"(~{total_tokens * (1 - tc.chat_fraction) / story_tokens:.1f} passes over the story data, "
             f"~{total_tokens * tc.chat_fraction / len(data.arrays['chat_train']):.0f} passes over the chat data)")
    p95 = data.meta.get("chat_tokens_per_conv_p95", float("nan"))
    log.info(f"chat conversations: p95 length {p95:.0f} tokens (context_len is {mc.context_len})")
    if p95 > mc.context_len:
        log.warning("many conversations are longer than context_len; training windows will cut them (fine, but "
                    "the model will see fewer complete conversations)")

    # ---- model ------------------------------------------------------------------------------
    model = GPT(mc).to(device)
    counts = count_parameters(model)
    log.info(f"PARAMETERS: {counts['total']:,} total ({counts['total'] / 1e6:.2f}M) | "
             f"embedding {counts['embedding(tied with output)']:,} | blocks {counts['transformer blocks']:,} "
             f"({counts['per block']:,} each) | non-embedding {counts['non-embedding']:,}")
    optimizer = make_optimizer(model, tc, device)

    # ---- resume -----------------------------------------------------------------------------
    step, best_val = 0, float("inf")
    ckpt, ckpt_path = load_latest(args.out_dir, device)
    if ckpt is not None:
        if ckpt["config"]["model"] != cfg.to_dict()["model"]:
            raise ValueError(f"{ckpt_path} was trained with a different model config:\n  checkpoint: {ckpt['config']['model']}\n"
                             f"  current:    {cfg.to_dict()['model']}\nUse a new --out_dir or restore the original config.")
        if ckpt.get("tokenizer_fingerprint") != tok.fingerprint():
            raise ValueError("checkpoint was trained with a different tokenizer")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler.is_enabled() and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        step, best_val = ckpt["step"], ckpt["best_val"]
        log.info(f"RESUMED from {ckpt_path} at step {step} (best val so far {best_val:.4f})")
        if ckpt["config"]["train"]["max_steps"] != tc.max_steps:
            log.warning(f"max_steps changed from {ckpt['config']['train']['max_steps']} to {tc.max_steps}; "
                        "the learning-rate schedule now follows the new value")
        if step >= tc.max_steps:
            log.info("already at max_steps; nothing to do")
            return
    else:
        log.info("no checkpoint found: starting from scratch")

    def checkpoint(name, with_optimizer=True, val_loss=None):
        payload = {"model": model.state_dict(), "config": cfg.to_dict(), "step": step, "best_val": best_val,
                   "tokenizer_fingerprint": tok.fingerprint(), "val_loss": val_loss}
        if with_optimizer:
            payload["optimizer"] = optimizer.state_dict()
            payload["scaler"] = scaler.state_dict() if scaler.is_enabled() else None
        t0 = time.time()
        save_checkpoint(args.out_dir, name, payload)
        size = os.path.getsize(os.path.join(args.out_dir, name)) / 1e6
        log.info(f"saved {name} at step {step} ({size:.1f} MB, {time.time() - t0:.1f}s)")

    # ---- first-batch sanity checks ----------------------------------------------------------
    x, y, n_chat = data.train_batch(step, 0, tc.batch_size)
    assert (x[:, 1:] == y[:, :-1]).all(), "targets are not inputs shifted by one"
    log.info(f"sanity: first batch shape {tuple(x.shape)}, {n_chat} chat rows; decoded start of row 0: "
             + repr(tok.decode(x[0, :40].tolist())))
    if step == 0:
        ev = evaluate(model, data, tc, device, autocast_ctx)
        expected = math.log(mc.vocab_size)
        log.info(f"sanity: initial val loss stories={ev['stories']:.3f} chat={ev['chat']:.3f}; "
                 f"an untrained model should be near ln(vocab)={expected:.3f}")
        if abs(ev["mix"] - expected) > 0.6:
            log.warning("initial loss is far from ln(vocab): check initialization, tokenizer and data")
        metrics.write(type="eval", step=0, **{f"val_{k}": v for k, v in ev.items()})

    # ---- training loop ----------------------------------------------------------------------
    model.train()
    t_start = time.time()
    t_last = time.time()
    steps_since_log = 0
    loss_sum = 0.0
    skipped = 0
    gn_value = float("nan")
    log.info(f"training from step {step} to {tc.max_steps}")
    while step < tc.max_steps:
        lr = lr_at(step, tc)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro in range(tc.grad_accum_steps):
            x, y, _ = data.train_batch(step, micro, tc.batch_size)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast_ctx:
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss is {loss.item()} at step {step}; training diverged. "
                                   "Try a lower lr, more warmup, or dtype=fp32. The last checkpoint is still intact.")
            scaler.scale(loss / tc.grad_accum_steps).backward()
            step_loss += loss.item() / tc.grad_accum_steps
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scale_before = scaler.get_scale() if scaler.is_enabled() else None
        scaler.step(optimizer)
        scaler.update()
        if scaler.is_enabled() and scaler.get_scale() < scale_before:
            skipped += 1  # fp16 overflow: GradScaler skipped this update and lowered its scale
        gn_value = float(gn)
        step += 1
        loss_sum += step_loss
        steps_since_log += 1

        if step % tc.log_interval == 0 or step == tc.max_steps:
            now = time.time()
            dt = now - t_last
            tps = steps_since_log * tok_per_step / dt
            remaining = (tc.max_steps - step) * dt / steps_since_log
            mem = f" | gpu mem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB" if device.type == "cuda" else ""
            log.info(f"step {step}/{tc.max_steps} | loss {loss_sum / steps_since_log:.4f} | lr {lr:.2e} | "
                     f"grad_norm {gn_value:.3f} | {tps:,.0f} tok/s | {dt / steps_since_log * 1000:.0f} ms/step | "
                     f"eta {remaining / 60:.1f} min | skipped {skipped}{mem}")
            metrics.write(type="train", step=step, loss=loss_sum / steps_since_log, lr=lr, grad_norm=gn_value,
                          tokens_per_sec=tps, skipped_steps=skipped)
            if gn_value > 10 * tc.grad_clip:
                log.warning(f"grad_norm {gn_value:.1f} is much larger than grad_clip {tc.grad_clip}")
            t_last, steps_since_log, loss_sum = now, 0, 0.0

        if step % tc.eval_interval == 0 or step == tc.max_steps:
            ev = evaluate(model, data, tc, device, autocast_ctx)
            improved = ev["mix"] < best_val
            if improved:
                best_val = ev["mix"]
            log.info(f"EVAL step {step} | val stories {ev['stories']:.4f} (ppl {math.exp(ev['stories']):.1f}) | "
                     f"val chat {ev['chat']:.4f} (ppl {math.exp(ev['chat']):.1f}) | mix {ev['mix']:.4f} | "
                     f"best {best_val:.4f}{' (new best)' if improved else ''}")
            metrics.write(type="eval", step=step, **{f"val_{k}": v for k, v in ev.items()})
            try:
                log_samples(model, tok, tc, device)
            except Exception as e:  # a sampling problem must never kill a long training run
                log.warning(f"sample generation failed ({e!r}); continuing")
            if improved:
                checkpoint(BEST, with_optimizer=False, val_loss=ev["mix"])
            t_last = time.time()  # do not count eval time in throughput

        if step % tc.ckpt_interval == 0 or step == tc.max_steps:
            checkpoint(LATEST)
            t_last = time.time()

    log.info(f"finished {tc.max_steps} steps in {(time.time() - t_start) / 60:.1f} min (this session); best val {best_val:.4f}")


if __name__ == "__main__":
    main()
