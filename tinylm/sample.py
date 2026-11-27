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


def apply_repetition_penalty(logits, token_ids, penalty):
    """Make tokens that were already used less likely (the 'CTRL' penalty). logits: (1, vocab). For each token
    in token_ids, a positive logit is divided by `penalty` and a negative one multiplied by it, so the token
    always becomes less likely. penalty 1.0 changes nothing; 1.1 to 1.3 is typical."""
    if penalty == 1.0 or not token_ids:
        return logits
    idx = torch.tensor(sorted(token_ids), dtype=torch.long, device=logits.device)
    vals = logits[0, idx]
    logits = logits.clone()
    logits[0, idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


@torch.no_grad()
def generate(model, ids, max_new_tokens, temperature=0.8, top_k=40, stop_ids=(), generator=None,
             repetition_penalty=1.0, penalize_ids=()):
    """ids: list of ints (the prompt). Returns the list of NEW token ids (stop token excluded).

    repetition_penalty > 1 discourages repeating: tokens already generated in this reply, plus the tokens in
    `penalize_ids` (for chat: the words of the bot's earlier replies), become less likely. The stop tokens
    are never penalized, otherwise replies could not end."""
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
        if repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits, (set(penalize_ids) | set(new)) - set(stop_ids), repetition_penalty)
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            # The generator may live on a different device than the model (e.g. a CPU generator with a
            # GPU model); torch.multinomial needs them to match, so sample where the generator is.
            gdev = generator.device if generator is not None else probs.device
            nxt = torch.multinomial(probs.to(gdev), 1, generator=generator).to(probs.device)
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
    p.add_argument("--repetition_penalty", type=float, default=1.0,
                   help="1.0 = off; 1.1-1.3 discourages repeating words within the reply")
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
                       stop_ids_for(tok), gen, args.repetition_penalty)
        label = "bot: " if args.chat else ""
        print(f"--- sample {i + 1} ({len(out)} tokens) ---")
        print(("user: " + args.prompt + "\n" if args.chat else args.prompt) + label + tok.decode(out))


if __name__ == "__main__":
    main()
