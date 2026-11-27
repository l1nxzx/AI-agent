"""Run a fixed list of chat prompts (the "standard set") on a saved model and write the transcript.

  python -m tinylm.eval_chat --ckpt_dir runs/tiny_dailydialog_only

Every model in reports/REPORT.md is judged on this same list, at the same sampling settings and seed, so
the answers are directly comparable between models. Each entry is a short conversation (history is
cleared between entries); the context entries have several turns. Output: <ckpt_dir>/eval_chat_<which>.txt
and .json. Scoring is done by reading the transcript (see the report for the 0/1/2 rubric).
"""
import argparse
import json
import os

import torch

from .bpe import BPETokenizer
from .chat import chat_turn
from .logutil import setup_logging
from .sample import load_model

# (group, [user messages in order], repeats). Repeats re-ask the same thing with fresh sampling, which
# measures consistency (does "who are you" give the same answer each time?).
STANDARD_SET = [
    ("greeting", ["Hi"], 1), ("greeting", ["Hey"], 1), ("greeting", ["Hello"], 1),
    ("greeting", ["Hello, how are you?"], 1), ("greeting", ["whats up"], 1),
    ("greeting", ["Good morning"], 1), ("greeting", ["Thanks, bye"], 1),
    ("identity", ["Who are you?"], 3), ("identity", ["What are you?"], 1),
    ("identity", ["What is your name?"], 3), ("identity", ["Are you a robot?"], 1),
    ("identity", ["Are you smart?"], 1), ("identity", ["What is it like to be you?"], 1),
    ("topic", ["Do you want to go to the movies tonight?"], 1), ("topic", ["What do you do for work?"], 1),
    ("topic", ["Can you recommend a good restaurant?"], 1), ("topic", ["What are you doing this weekend?"], 1),
    ("topic", ["I am looking for a job. Any advice?"], 1), ("topic", ["I am tired.", "Why do I feel this way?"], 1),
    ("context", ["My name is Blake.", "What is my name?"], 1),
    ("context", ["I like dogs.", "What animal do I like?"], 1),
    ("context", ["My favorite color is green.", "What is my favorite color?"], 1),
    ("unknown", ["How many r's are in strawberry?"], 1), ("unknown", ["What is 2 plus 2?"], 1),
    ("unknown", ["What is the capital of France?"], 1), ("unknown", ["Explain gravity."], 1),
    ("unknown", ["Which is bigger, 9 or 12?"], 1), ("unknown", ["What is my job?"], 1),
    ("story", ["Tell me a story."], 1), ("story", ["Tell me a story about a dog."], 1),
    ("social", ["You are cute"], 1), ("social", ["You are stupid"], 1), ("social", ["Thank you"], 1),
    ("robust", ["HELLO"], 1), ("robust", ["hi how r u"], 1), ("robust", ["Why"], 1),
    ("loop", ["I love you", "I love you", "I love you", "I hate you"], 1),
]


def run_eval(model, tok, temperature=0.5, top_k=20, max_new_tokens=80, seed=0, repetition_penalty=1.2):
    gen = torch.Generator().manual_seed(seed)
    records = []
    for group, turns, repeats in STANDARD_SET:
        for _ in range(repeats):
            history, exchange = [], []
            for user in turns:
                exchange.append((user, chat_turn(model, tok, history, user, max_new_tokens, temperature, top_k, gen,
                                                 repetition_penalty)))
            records.append({"group": group, "turns": exchange})
    return records


def format_records(records):
    lines, last = [], None
    for r in records:
        if r["group"] != last:
            lines.append(f"\n=== {r['group']} ===")
            last = r["group"]
        for user, bot in r["turns"]:
            lines.append(f"you: {user}\nbot: {bot}")
        lines.append("--")
    return "\n".join(lines).strip() + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--which", default="best", choices=["best", "latest"])
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--repetition_penalty", type=float, default=1.2, help="1.0 = off (as in the first report transcripts)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    log = setup_logging()
    tok = BPETokenizer.load(os.path.join(args.ckpt_dir, "tokenizer.json"))
    model, ckpt = load_model(os.path.join(args.ckpt_dir, f"ckpt_{args.which}.pt"), args.device)
    header = (f"standard chat set | run {os.path.basename(os.path.abspath(args.ckpt_dir))} | {args.which} checkpoint, step "
              f"{ckpt['step']}, val_loss {ckpt.get('val_loss')} | temperature {args.temperature}, top_k {args.top_k}, "
              f"repetition_penalty {args.repetition_penalty}, seed {args.seed}\n")
    records = run_eval(model, tok, args.temperature, args.top_k, args.max_new_tokens, args.seed, args.repetition_penalty)
    text = header + format_records(records)
    suffix = "" if args.repetition_penalty == 1.2 else f"_rp{args.repetition_penalty}"
    base = os.path.join(args.ckpt_dir, f"eval_chat_{args.which}{suffix}")
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(text)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"header": header, "records": records}, f, indent=1)
    print(text)
    log.info(f"wrote {base}.txt and .json")


if __name__ == "__main__":
    main()
