"""Download TinyStories + SmolTalk everyday-conversations, train the tokenizer, write token files.

Usage:
  python -m tinylm.prepare_data --data_dir data --vocab_size 2000
  python -m tinylm.prepare_data --data_dir /tmp/smoke_data --vocab_size 512 --smoke   # tiny, for tests

It is safe to run again: if meta.json exists and matches the requested vocab size, it does nothing
(use --force to rebuild).
"""
import argparse
import json
import os

# The Xet transfer backend needs extra hosts that locked-down networks may block (the download then
# hangs silently). Plain HTTPS downloads are fast enough for these file sizes. Set before the hub
# library is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from .bpe import BPETokenizer
from .data import (DAILYDIALOG_REPO, DAILYDIALOG_REVISION, NQ_REPO, SMOLTALK_REPO, TINYSTORIES_REPO,
                   check_and_locate_files, download, read_conversations, read_dailydialog, read_nq_open_questions,
                   read_story_texts, write_dataset)
from .logutil import setup_logging
from .persona import IDENTITY_INTENTS, PersonaBuilder, pick_unknown_user_turns


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--cache_dir", default=None, help="where raw downloads go (default: <data_dir>/hf_cache)")
    p.add_argument("--vocab_size", type=int, default=2000)
    p.add_argument("--story_shards", type=int, default=1, help="number of TinyStories train parquet shards (~530k stories each)")
    p.add_argument("--max_stories", type=int, default=300_000, help="cap on training stories encoded")
    p.add_argument("--max_val_stories", type=int, default=3000)
    p.add_argument("--tokenizer_stories", type=int, default=30_000, help="stories used to train the tokenizer (chat text is always included)")
    p.add_argument("--chat_mode", choices=["persona", "real", "dailydialog", "dailydialog_only", "dailydialog_identity"], default="persona",
                   help="persona: hand-written consistent bot voice (+ real stories and real user questions); "
                        "real: the everyday-conversations data as published; "
                        "dailydialog: real human dialogues (DailyDialog) plus everyday-conversations, no authored text; "
                        "dailydialog_only: DailyDialog alone (one consistent style); "
                        "dailydialog_identity: DailyDialog plus a small authored identity layer (Petryk)")
    p.add_argument("--persona_train", type=int, default=30_000, help="persona conversations for training")
    p.add_argument("--persona_val", type=int, default=600)
    p.add_argument("--identity_train", type=int, default=1500,
                   help="identity conversations mixed into DailyDialog in dailydialog_identity mode (about 2%% of chat tokens)")
    p.add_argument("--nq_questions", type=int, default=20_000,
                   help="short real factual questions (NQ-Open) that Petryk declines, in persona and identity modes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="use only the small TinyStories validation file, split into train/val (for quick tests)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    log = setup_logging(os.path.join(args.data_dir, "prepare.log"))
    meta_path = os.path.join(args.data_dir, "meta.json")
    if os.path.exists(meta_path) and not args.force:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta["vocab_size"] == args.vocab_size and meta.get("chat_mode", "real") == args.chat_mode:
            log.info(f"{meta_path} already exists (vocab_size={args.vocab_size}, chat_mode={args.chat_mode}); "
                     "nothing to do (use --force to rebuild)")
            return
        log.info(f"existing data has vocab_size={meta['vocab_size']} chat_mode={meta.get('chat_mode', 'real')}, "
                 f"requested {args.vocab_size}/{args.chat_mode}; rebuilding")

    cache = args.cache_dir or os.path.join(args.data_dir, "hf_cache")
    files = check_and_locate_files(smoke=args.smoke, num_story_shards=args.story_shards,
                                   dailydialog=args.chat_mode in ("dailydialog", "dailydialog_only", "dailydialog_identity"),
                                   nq=args.chat_mode in ("persona", "dailydialog_identity"))

    chat_train = read_conversations(download(SMOLTALK_REPO, files["chat_train"], cache))
    chat_val = read_conversations(download(SMOLTALK_REPO, files["chat_test"], cache))
    valid_texts = read_story_texts(download(TINYSTORIES_REPO, files["story_valid"], cache))

    if args.smoke:
        log.info("SMOKE mode: stories come from the TinyStories validation file only")
        story_val = valid_texts[:300]
        story_train = valid_texts[300:300 + 3000]
    else:
        story_train = []
        for f in files["story_train"]:
            story_train += read_story_texts(download(TINYSTORIES_REPO, f, cache), limit=args.max_stories - len(story_train))
            if len(story_train) >= args.max_stories:
                break
        story_val = valid_texts[:args.max_val_stories]
    log.info(f"stories: {len(story_train)} train, {len(story_val)} val; conversations: {len(chat_train)} train, {len(chat_val)} val")

    short_questions = None
    if args.chat_mode in ("persona", "dailydialog_identity"):
        short_questions = {"train": read_nq_open_questions(download(NQ_REPO, files["nq_train"], cache), limit=args.nq_questions),
                           "val": read_nq_open_questions(download(NQ_REPO, files["nq_val"], cache), limit=1000)}
    identity_convs = []
    if args.chat_mode in ("dailydialog", "dailydialog_only", "dailydialog_identity"):
        log.info(f"chat_mode={args.chat_mode}: chat data is entirely real, human-written dialogue (no authored replies)")
        dd_train = read_dailydialog(download(DAILYDIALOG_REPO, files["dd_train"], cache, DAILYDIALOG_REVISION))
        dd_val = read_dailydialog(download(DAILYDIALOG_REPO, files["dd_val"], cache, DAILYDIALOG_REVISION), both_ways=False)
        if args.chat_mode in ("dailydialog_only", "dailydialog_identity"):
            chat_train, chat_val = dd_train, dd_val
            if args.chat_mode == "dailydialog_identity":
                # Identity data is mixed into TRAINING only. Validation stays DailyDialog-only, so the validation
                # loss is directly comparable with a dailydialog_only run; identity is judged by reading replies.
                builder = PersonaBuilder(
                    stories={"train": story_train, "val": story_val},
                    unknown_user_turns={"train": pick_unknown_user_turns(read_conversations(download(SMOLTALK_REPO, files["chat_train"], cache))),
                                        "val": pick_unknown_user_turns(read_conversations(download(SMOLTALK_REPO, files["chat_test"], cache)))},
                    intents=IDENTITY_INTENTS, short_questions=short_questions)
                identity_convs = builder.build(args.identity_train, "train", args.seed, log)
                chat_train = dd_train + identity_convs
        else:
            chat_train, chat_val = dd_train + chat_train, dd_val + chat_val
        log.info(f"chat: {len(chat_train)} train and {len(chat_val)} val dialogues ({args.chat_mode})")
        for conv in chat_train[:3]:
            log.info("dialogue example:\n" + "\n".join(f"  {'user' if r == 'user' else 'bot '}: {t[:160]}" for r, t in conv[:6]))
    if args.chat_mode == "persona":
        log.info("chat_mode=persona: chat data is the hand-written persona bank (tinylm/persona.py). The real "
                 "everyday-conversations data only supplies the user-side questions the bot answers with an honest 'I am small'.")
        builder = PersonaBuilder(
            stories={"train": story_train, "val": story_val},
            unknown_user_turns={"train": pick_unknown_user_turns(chat_train), "val": pick_unknown_user_turns(chat_val)},
            short_questions=short_questions)
        chat_train = builder.build(args.persona_train, "train", args.seed, log)
        chat_val = builder.build(args.persona_val, "val", args.seed, log)
        for conv in chat_train[:5]:
            log.info("persona example:\n" + "\n".join(f"  {'user' if r == 'user' else 'bot '}: {t[:160]}" for r, t in conv))

    tok_texts = story_train[:args.tokenizer_stories] + [c for conv in chat_train for _, c in conv]
    log.info(f"training tokenizer on {len(tok_texts)} texts, target vocab {args.vocab_size}")
    tok = BPETokenizer.train(tok_texts, args.vocab_size, logger=log)
    big = sorted(range(256, tok.num_base), key=lambda i: -len(tok.vocab[i]))[:8]
    log.info("longest learned tokens: " + ", ".join(repr(tok.vocab[i].decode("utf-8", "replace")) for i in big))

    sources = {"stories": TINYSTORIES_REPO, "chat": SMOLTALK_REPO + " (everyday-conversations)",
               "files": {k: v for k, v in files.items()}, "smoke": args.smoke, "chat_mode": args.chat_mode}
    meta = write_dataset(args.data_dir, tok, story_train, story_val, chat_train, chat_val, sources, chat_mode=args.chat_mode)
    if identity_convs:
        from .data import ROLE_TO_MARKER, encode_conversation
        id_tokens = sum(len(encode_conversation(tok, c)) for c in identity_convs)
        total = meta["splits"]["chat_train"]["tokens"]
        log.info(f"identity layer: {len(identity_convs)} conversations = {id_tokens:,} tokens = {id_tokens / total * 100:.1f}% of "
                 f"the {total:,} chat training tokens ({len(identity_convs) / len(chat_train) * 100:.1f}% of conversations)")
        with open(os.path.join(args.data_dir, "identity_examples.txt"), "w", encoding="utf-8") as f:
            f.write("IDENTITY TRAINING EXAMPLES (first 60 of the authored conversations mixed into DailyDialog)\n\n")
            for conv in identity_convs[:60]:
                f.write("\n".join(f"{'user' if r == 'user' else 'bot '}: {t}" for r, t in conv) + "\n" + "-" * 60 + "\n")
        log.info(f"wrote {os.path.join(args.data_dir, 'identity_examples.txt')} (readable)")


if __name__ == "__main__":
    main()
