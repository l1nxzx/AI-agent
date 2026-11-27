"""Interactive multi-turn chat with a trained checkpoint.

  python -m tinylm.chat --ckpt_dir runs/tiny

Type a message and press Enter; type 'quit' (or an empty line twice) to stop, 'reset' to clear history.

The conversation is kept as token ids in the same format used in training:
  <|user|> text <|end|> <|bot|> reply <|end|> <|user|> ...
Only the last context_len tokens fit in the model, so old turns are dropped from the front (whole
turns at a time, so the history always starts at a <|user|> marker).
"""
import argparse
import os

import torch

from .bpe import BOT, END, USER, BPETokenizer
from .logutil import setup_logging
from .sample import generate, load_model, stop_ids_for


def trim_history(history, tok, max_tokens):
    """Drop the oldest turns until the history fits in max_tokens, keeping it aligned to a user marker."""
    user_id = tok.token_id(USER)
    while len(history) > max_tokens:
        nxt = history.index(user_id, 1) if user_id in history[1:] else None
        if nxt is None:
            del history[:len(history) - max_tokens]   # a single huge turn: cut from the front
            break
        del history[:nxt]
    return history


def chat_turn(model, tok, history, text, max_new_tokens=120, temperature=0.5, top_k=20, generator=None):
    """Add one user message to `history` (a list of token ids, modified in place), generate the bot reply,
    add it to the history too, and return the reply text. Shared by the interactive loop and eval_chat."""
    stop = stop_ids_for(tok)
    history += [tok.token_id(USER)] + tok.encode(text) + [tok.token_id(END), tok.token_id(BOT)]
    trim_history(history, tok, model.cfg.context_len - max_new_tokens)   # leave room for the reply
    reply = generate(model, history, max_new_tokens, temperature, top_k, stop, generator)
    history += reply + [tok.token_id(END)]
    return tok.decode(reply).strip()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--which", default="best", choices=["best", "latest"])
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    setup_logging()
    tok = BPETokenizer.load(os.path.join(args.ckpt_dir, "tokenizer.json"))
    model, ckpt = load_model(os.path.join(args.ckpt_dir, f"ckpt_{args.which}.pt"), args.device)
    print(f"Loaded {args.which} checkpoint (step {ckpt['step']}). Type 'quit' to exit, 'reset' to start over.")
    gen = torch.Generator().manual_seed(args.seed)
    history = []
    blank = 0
    while True:
        try:
            text = input("you: ").strip()
        except EOFError:
            break
        if text == "":
            blank += 1
            if blank >= 2:
                break
            continue
        blank = 0
        if text.lower() in ("quit", "exit"):
            break
        if text.lower() == "reset":
            history = []
            print("(history cleared)")
            continue
        print("bot:", chat_turn(model, tok, history, text, args.max_new_tokens, args.temperature, args.top_k, gen))


if __name__ == "__main__":
    main()
