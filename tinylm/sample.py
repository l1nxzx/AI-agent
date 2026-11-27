"""Text generation: temperature and top-k sampling.

Usage:
  python -m tinylm.sample --ckpt_dir runs/tiny --prompt "Once upon a time"
  python -m tinylm.sample --ckpt_dir runs/tiny --chat --prompt "Hi, how are you?"

How sampling works: the model outputs a score (logit) for every token in the vocabulary. We
  1. divide the logits by the temperature (low = more predictable, high = more random),
  2. keep only the top_k highest scores (everything else gets probability 0),
  3. softmax into probabilities and draw one token at random,
  4. append it to the sequence and repeat.
Temperature 0 means greedy decoding (always the single most likely token).
"""
import argparse
import os

import torch

from .bpe import BOT, END, ENDOFTEXT, USER, BPETokenizer
from .config import config_from_dict
from .logutil import setup_logging
from .model import GPT


@torch.no_grad()
def generate(model, ids, max_new_tokens, temperature=0.8, top_k=40, stop_ids=(), generator=None):
    """ids: list of ints (the prompt). Returns the list of NEW token ids (stop token excluded)."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    new = []
    for _ in range(max_new_tokens):
        # The model only sees the last context_len tokens. (No KV cache: each step recomputes the
        # whole window. Simple, and fast enough for models this small.)
        logits, _ = model(x[:, -model.cfg.context_len:])
        logits = logits[:, -1, :].float()
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
        tok = int(nxt.item())
        if tok in stop_ids:
            break
        new.append(tok)
        x = torch.cat([x, nxt], dim=1)
    model.train(was_training)
    return new


def story_prompt_ids(tok, text):
    """A story starts after <|endoftext|>, which is how documents were separated in training."""
    return [tok.token_id(ENDOFTEXT)] + tok.encode(text)


def chat_prompt_ids(tok, user_text):
    """<|user|> text <|end|> <|bot|>   -- the model then writes the bot reply and ends with <|end|>."""
    return [tok.token_id(USER)] + tok.encode(user_text) + [tok.token_id(END), tok.token_id(BOT)]


def stop_ids_for(tok):
    return {tok.token_id(END), tok.token_id(ENDOFTEXT)}


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = config_from_dict(ckpt["config"])
    model = GPT(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True, help="run directory containing ckpt_best.pt / ckpt_latest.pt and tokenizer.json")
    p.add_argument("--which", default="best", choices=["best", "latest"])
    p.add_argument("--tokenizer", default=None, help="tokenizer.json (default: <ckpt_dir>/tokenizer.json)")
    p.add_argument("--prompt", default="")
    p.add_argument("--chat", action="store_true", help="wrap the prompt as a user chat turn and generate the bot reply")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    log = setup_logging()
    path = os.path.join(args.ckpt_dir, f"ckpt_{args.which}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    tok = BPETokenizer.load(args.tokenizer or os.path.join(args.ckpt_dir, "tokenizer.json"))
    model, ckpt = load_model(path, args.device)
    if ckpt.get("tokenizer_fingerprint") and ckpt["tokenizer_fingerprint"] != tok.fingerprint():
        log.warning("tokenizer fingerprint differs from the one used in training; output will be garbage")
    log.info(f"loaded {path}: step {ckpt['step']}, val_loss {ckpt.get('val_loss')}, "
             f"temperature={args.temperature}, top_k={args.top_k}")
    prompt_ids = chat_prompt_ids(tok, args.prompt) if args.chat else story_prompt_ids(tok, args.prompt)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    for i in range(args.num_samples):
        out = generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_k,
                       stop_ids_for(tok), gen)
        label = "bot: " if args.chat else ""
        print(f"--- sample {i + 1} ({len(out)} tokens) ---")
        print(("user: " + args.prompt + "\n" if args.chat else args.prompt) + label + tok.decode(out))


if __name__ == "__main__":
    main()
