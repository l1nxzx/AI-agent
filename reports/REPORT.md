# tinylm: model performance report

Living document. One section per trained model, a comparison table, and cross-cutting findings. New models are
added at the end of section 4 and to the tables in section 5. Evidence (verbatim transcripts and item-level
scores) is stored next to this file in `transcripts/` and `scores/`.

Last updated after: run `tiny_dailydialog_only` (final metrics added). Models covered: 3 full Colab runs (all tiny, 2.16M parameters).
Pending: `small`.

## 1. Summary

* All three trained models are the same 2.16M-parameter transformer. They differ only in the chat data and the
  fraction of each batch spent on it. This isolates the effect of data.
* Story generation is the same in every run: locally fluent short stories with weak global coherence and some
  repetition. Validation loss on stories is 1.81 (perplexity 6.1) in run 1 and 1.84 (perplexity 6.3) in run 3, the
  two runs with a story loss on file. The small gap is consistent with run 3 spending 35% of each batch on chat
  instead of 10%, but one run each cannot confirm that.
* Chat is poor in every run. The model produces grammatical, human-sounding fragments that mostly do not relate
  to the question. No data variant tested so far fixed this.
* The failures differ with the data, which shows the data decides the behavior, not a bug:
  broad assistant-style chat gives confident topical invention (run 1); mixing an LLM-written dataset with
  human dialogue gives an inconsistent voice (run 2); human-human dialogue only gives phone-call and office
  style replies with no stable identity (run 3).
* On the standard rubric, run 3 scores a mean of 0.81 out of 2 on 57 graded replies (23% sensible, 42% nonsense).
  The strongest groups are identity (1.30) and topic questions (1.20). The weakest are unknown or factual
  questions (0.36) and greetings (0.56).
* Open questions: does a larger model (`small`, 11M) help real dialogue; does a small authored identity layer
  fix identity and "I do not know"; how much do the numbers move with sampling settings.

## 2. Common setup (all runs)

| item | value |
|---|---|
| architecture | decoder-only transformer, 4 layers, dim 192, 6 heads (head dim 32), context 256, vocab 2000 |
| components | RMSNorm, rotary position embeddings, SwiGLU MLP, tied input/output embeddings, no biases |
| parameters | 2,155,200 total (384,000 embedding, 1,771,008 in blocks) |
| tokenizer | from-scratch byte-level BPE, 1739 merges + 256 bytes + 5 special tokens = 2000 |
| stories | TinyStories, 300,000 training stories (76,125,704 tokens in run 1, 76,760,774 in run 3; the tokenizer is retrained for each data build, so counts differ slightly); validation 3,000 stories (690,483 / 696,146 tokens) |
| optimizer | AdamW, peak lr 2e-3, 300 warmup steps, cosine decay to 2e-4, weight decay 0.1, grad clip 1.0 |
| schedule | 8,000 steps, batch 64 x 256 tokens = 16,384 tokens/step, 131,072,000 tokens in total |
| hardware | Colab Tesla T4, fp16 with loss scaling, 1.24 GB GPU memory, about 270-290k tokens/s, 57-61 ms/step |
| wall time | 8.7 minutes (run 1), 8.8 minutes (run 3); run 2 not on file |

Chat format in all runs: `<|user|> text <|end|> <|bot|> text <|end|> ... <|endoftext|>`.

Differences between runs:

| run | folder on Drive | chat data | chat share of batch | loss on chat |
|---|---|---|---|---|
| 1 | `runs/tiny` | everyday-conversations (SmolTalk) only | 10% | all tokens |
| 2 | `runs/tiny_dailydialog` | DailyDialog + everyday-conversations | 35% | bot tokens only |
| 3 | `runs/tiny_dailydialog_only` | DailyDialog only | 35% | bot tokens only |

Approximate passes over the data: run 1 saw the story data about 1.5 times and the chat data about 23 times
(from the training log). Runs 2 and 3 saw the story data about 1.1 times; the chat data about 12 times (run 2,
about 3.8M chat tokens) and about 14 times (run 3, 3,245,201 chat tokens, from its log).

## 3. How models are evaluated

### 3.1 Validation loss

Cross-entropy on fixed held-out batches, reported for stories and chat. Lower is better. **Validation losses are
not comparable across runs with different chat data**, because the chat validation set, the share of chat in
the mixed score and (for run 1) the set of tokens counted all differ. They are only comparable within a run
over time. They also say little about chat quality: run 1 reached chat perplexity 8.8 and still produced
confident nonsense, because predicting the next token of a conversation is much easier than answering it.

### 3.2 Chat quality: standard set and rubric

Chat quality is judged by reading answers. A fixed prompt list is defined in `tinylm/eval_chat.py`
(`STANDARD_SET`, 37 entries, 41 conversations once repeats are counted, over the groups greeting, identity, topic, context, unknown, story,
social, robust, loop; "who are you" and "what is your name" are asked 3 times to measure consistency).
Run it on any saved model with `python -m tinylm.eval_chat --ckpt_dir <run folder>`; the Colab chat notebook
has a cell for it.

Rubric, version 1, per reply:

* **2** sensible: answers or responds to what was said, grammatical.
* **1** plausible sentence that does not answer or relate to the question (a non sequitur), or an answer that
  is partly right.
* **0** garbled, ungrammatical or semantically nonsensical.

### 3.3 Limitations of this evaluation

* One rater (the assistant), not blinded, with subjective boundaries between 0 and 1. Differences of a few tenths
  between models should not be trusted.
* Small samples (3 to 14 replies per group), one sampling seed, no confidence intervals.
* Runs 1 and 2 were judged only from ad-hoc transcripts, and run 3's scored items are the user's own prompts, not
  the standard set; they overlap with it but are not identical. Running the standard set on all saved models
  would make the comparison fair. This is the first item in section 7.
* In interactive transcripts the conversation history carried over between prompts until `reset`, so some replies
  are conditioned on earlier turns.
* Sampling settings were not recorded in every paste (see each model).

## 4. Models

### 4.1 Run 1: tiny, everyday-conversations only (`runs/tiny`)

Data and training: see section 2. This was the first run, before chat modes existed.

Training metrics (Colab log):

| step | train loss | val stories | val chat | mix |
|---|---|---|---|---|
| 500 | 2.6090 | 2.5706 (ppl 13.1) | 3.0710 (ppl 21.6) | 2.6206 |
| 7500 | 1.7154 | 1.8140 (ppl 6.1) | 2.1809 (ppl 8.9) | 1.8507 |
| 8000 (best) | 1.7205 | 1.8074 (ppl 6.1) | 2.1776 (ppl 8.8) | 1.8444 |

Training was healthy: gradient norm about 0.5 near the end, 0 skipped fp16 steps in every logged line, loss still
slowly falling at the end. The first launch crashed at the first sample printout (a CPU/GPU random-generator mismatch in the sampling
code, fixed in commit `5cd7f6d`); no checkpoint had been written, so the run was restarted.

Stories (temperature 0.8): coherent paragraphs with names, simple plots and dialogue, for example "Once upon a
time, there was a small boy named Timmy. He loved playing with his red ball. One day, he went to the park with
his mom." Weaknesses: details contradict each other ("The park was tall and shiny"), repetition ("Timmy was sad
and scared", "couldn't play with it anymore because he couldn't play anymore"), and the plot drifts. The second
story contains a semantically odd passage ("a fragile pile of soil that it would make a loud noise").

Chat (temperature 0.7, top-k 40; 5 prompts, not scored item by item): every reply is long, confident and about a
specific topic unrelated to the prompt, with invented names and numbers.

* "who are you?" -> "There are many conductures, such as April, Monday, and Scrabble. Is it yourself normal for you?"
* "I like you" -> "Yes, I like basketball. It's a good idea to check the NBA between $20 and a $50ma for a 1000, while the flute is a 5-D about $1.500."
* "you are stupid" -> "Effect most pourangies, but it's not easy to pick up..."

Analysis: the chat set is broad (travel, hobbies, tech, sports) with assistant replies that state specifics. A
2M model learned that style without any knowledge to fill it, so it invents. It also produced non-words
("pourangies", "Queenstem"), which is a symptom of limited capacity on rare words. Loss was computed on user
turns too in this run, which spends capacity on text the bot never needs to produce. Chat validation
perplexity of 8.8 looks good and is misleading (section 3.1).

Verdict: stories acceptable for the size; chat unusable.

### 4.2 Run 2: tiny, DailyDialog + everyday-conversations (`runs/tiny_dailydialog`)

Data: about 24,000 training dialogues (DailyDialog with each dialogue also used with its first utterance dropped,
plus the 2,260 everyday-conversations). Chat share 35%, loss on bot tokens only.

Training metrics: best mixed validation loss 2.128 at step 8000 (the only number on file; the story and chat
components and timing were not provided).

Stories (temperature 0.8): similar to run 1. They are fluent with simple structure, and show typical small-model
repetition ("Lily felt the wind on her face and the wind on her face") and logic slips ("Your puppy is broken").

Chat (temperature 0.5, top-k 20; 66 replies in the transcript, not scored item by item):

* The greeting prompts "Hi", "hi", "Hey", "hi" gave the exact reply "Hello! How can I help you today?" (4 of 6
  greetings). That sentence comes from the LLM-written everyday-conversations data. The other two greetings
  ("whats up", "hey") produced word salad about geography and woodworking, the topics of that dataset.
* Identity is random: "Do you know what you are?" -> "I am in the car."; "Who are you?" -> "I am the student in Roman."
* Memory failed: after "My name is Blake" the bot said "I'm yours." and for "Who am I?" it answered "I'm Jenny."
* A repetition pattern: "I love you" was answered with "I love you" three times in a row, and "I hate you" got
  "I love you too."
* Sensitive prompts ("Kill yourself", an insult) received irrelevant replies ("I'll come back another day.",
  "I'll be back soon."). The model has no safety training and no behavior designed for these inputs.

Analysis: two datasets with different registers (an LLM-written helpful assistant and two humans talking) gave
the model two competing voices. Greetings that match the assistant data get a clean reply, and nearby prompts
fall into the other data's topics. This motivated the DailyDialog-only run.

Verdict: marginally more human-sounding than run 1 but inconsistent; chat unusable.

### 4.3 Run 3: tiny, DailyDialog only (`runs/tiny_dailydialog_only`)

Data: 21,753 training dialogues and 1,000 validation dialogues, all human-written (DailyDialog, both orientations
for training). Chat share 35%, loss on bot tokens only. Text cleanup removes DailyDialog's pre-tokenized spacing
and curly quotes. The source still contains typos and, by topic, mostly practical adult scenes (banks, jobs,
shopping, phone calls); see `readable_samples.txt` in the data folder.

Training metrics: the log header is on file (run started 15:45:11 on a Tesla T4; fp16; chat_loss bot_only, chat share
0.35; chat training data 3,245,201 tokens, chat validation data 163,444 tokens; story training data 76,760,774
tokens; plan of about 1.1 passes over the stories and 14 over the chat data; the 95th percentile conversation is
365 tokens, longer than the 256 context, so many conversations are cut into windows). **Still missing:** the final
validation losses, training time and sample lines (in `runs/tiny_dailydialog_only/train.log` on Drive).

Final result (step 8000, the best checkpoint; run finished in 8.8 minutes):

| val stories | val chat (bot tokens only) | mixed (65% stories, 35% chat) |
|---|---|---|
| 1.8369 (ppl 6.3) | 2.7513 (ppl 15.7) | 2.1570 |

The end-of-training samples in the log (temperature 0.8, top-k 40) were a coherent but wandering story ("Once upon a
time, there was a girl who loved to play with her toys...") and the chat reply "I'm a party." to "Hi, how are you?".
Chat perplexity 15.7 is higher than run 1's 8.8, but the two are not comparable (different chat data and a
different loss: bot tokens only here, all tokens in run 1). Story loss is close to run 1's.

Sampling settings during chat were not recorded (notebook default 0.5 / 20). Full transcript:
`transcripts/tiny_dailydialog_only.txt`. Item-level scores: `scores/tiny_dailydialog_only.json`
(57 graded replies; the earlier rough per-group scores I gave in chat are superseded by these).

| group | n | mean score (0-2) |
|---|---|---|
| identity | 10 | 1.30 |
| topic | 10 | 1.20 |
| story | 3 | 1.00 |
| social | 5 | 0.80 |
| context | 6 | 0.67 |
| greeting | 9 | 0.56 |
| unknown / factual | 14 | 0.36 |
| all | 57 | 0.81 (23% scored 2, 42% scored 0) |

Findings, with evidence:

1. **Short, grammatical, human-sounding replies; meaning is hit or miss.** Good replies are mostly short and
   generic: "Can you recommend a good restaurant?" -> "I'm not sure yet."; "I am tired." / "Why do I feel this
   way?" -> "I think you'll have to work overtime."; "Are you so smart" -> "I'm not so sure."; "Are you a robot?"
   -> "Yes. I am a robot."
2. **Phone-call and office register.** 4 of 9 greeting-type replies are about calling: "Hi, I'm calling to see if
   you've lost it.", "I'm calling to check out this morning.", "I think we're going to call you tomorrow." This
   matches DailyDialog's content, which contains many phone and workplace scripts.
3. **Identity answers are relevant but not consistent.** Four identity questions got four different names (P,
   Peckhan, Zhang, Robbie) and the odd "I'm a ten-med office." So the relevance score (1.30) overstates
   usefulness: there is no stable self. The model has no data teaching it what it is.
4. **No sense of uncertainty.** It never says it does not know. Unknown or factual prompts scored 0.36 overall.
   "What is the capital of France?" got "It's a wine of us.", "Two of Japanese." and "I am considering that you can
   take the division by the middle of South America." (three samples, none correct).
5. **A repetition loop.** Three consecutive unknown-question replies contained "wine" ("I need to get the wine
   from a couple of days", "It's a wine of us.", "Cool, you need to get the wine from the wine."). The model
   stays stuck on a word it produced earlier because the history is part of the prompt.
6. **Context use is partial and often wrong.** After "My name is Blake." the bot replied "Blake is my friend." (it
   copied the name into a wrong role). After "I like dogs." it stayed on dogs, but answered "I am tired." with
   "How about the dogs?" Topic carry-over exists; reasoning over it does not.
7. **Stories are not a chat skill.** "Make a story" -> "You can always be a good girl."; the earlier runs show
   stories work only when prompted as plain text, not in chat.
8. **Social replies are non sequiturs** ("You are cute" -> "I have a look at it.") with one fair reply ("lol what" ->
   "well, I guess.").

Verdict: the most consistent in style of the three (no mixed voices), and the only run with a measured rubric
score. Still unable to hold a conversation. The best-looking answers are generic hedges that fit anything
("I'm not sure.").

## 5. Comparison

### 5.1 Item-level scores (rubric v1; generated by `python reports/score_summary.py`)

| run | n | overall | share 2 | share 0 | context | greeting | identity | social | story | topic | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tiny_dailydialog_only | 57 | 0.81 | 23% | 42% | 0.67 (6) | 0.56 (9) | 1.30 (10) | 0.80 (5) | 1.00 (3) | 1.20 (10) | 0.36 (14) |

Runs 1 and 2 are not yet scored item by item (see section 3.3).

### 5.2 Training metrics on file

| run | best val stories | best val chat | best mixed val | wall time |
|---|---|---|---|---|
| 1 `tiny` | 1.8074 | 2.1776 | 1.8444 (chat share 10%) | 8.7 min |
| 2 `tiny_dailydialog` | not provided | not provided | 2.1284 (chat share 35%) | not provided |
| 3 `tiny_dailydialog_only` | 1.8369 | 2.7513 (bot tokens only) | 2.1570 (chat share 35%) | 8.8 min |

Mixed losses are not comparable (different chat sets and shares).

### 5.3 Qualitative comparison

| aspect | run 1 | run 2 | run 3 |
|---|---|---|---|
| reply length | long | mixed | short |
| on-topic | no | rarely | sometimes (generic) |
| consistent voice | invents topics | two voices | one voice |
| identity | invents ("many conductures") | random | random names, relevant form |
| invents facts/numbers | yes, heavily | some | little (replies are short) |
| context use | not tested | weak | weak |
| stories (text mode) | OK | OK | not shown |

## 6. Cross-cutting findings

1. **Data decides chat behavior at this size.** With the model fixed, the chat data alone changed the failure mode
   from topical invention to inconsistent voices to phone-call style. (Evidence: runs 1-3.)
2. **Validation loss does not predict chat quality.** Run 1 had the lowest chat perplexity on file (8.8, against
   15.7 for run 3, which are not directly comparable) and the worst replies. Reading answers is required.
3. **Fluency is learned easily; grounding is not.** Grammar, register and local plausibility appear within a
   few thousand steps. Facts, arithmetic, identity and memory do not appear at 2M parameters on this data.
4. **No uncertainty without data for it.** Real dialogue data never says "I do not know I am a small model", so
   the model never does. (Preliminary evidence in the appendix suggests this can be taught with data.)
5. **Repetition under pressure.** Out-of-domain prompts make the model lock onto words ("wine") and loop.
   Sampling-time repetition control has not been tried yet.
6. **Story ability is stable across runs** and mostly comes from the TinyStories mixture, not from the chat data.

## 7. Next experiments

1. Run the standard set on the saved models of runs 1-3 (`tinylm.eval_chat`) and score them, so the comparison in
   5.1 is complete and fair.
2. Train `small` (11.4M parameters) on `dailydialog_only` and compare. Hypothesis: better fluency and context use,
   unchanged identity and uncertainty failures.
3. Try a small authored identity layer on top of real data (a few percent of the chat share, covering greetings,
   identity and "I do not know"). This is a data change only; no rules in the model or code.
4. Add a repetition penalty at sampling time and re-run the standard set to measure its effect on the loops.
5. Record sampling settings and the final training log line for every new run, so the tables have no gaps.

## Appendix A. Preliminary experiments (assistant-run, CPU, not full runs)

These were short runs in the development sandbox to check the code paths and to estimate what each data variant
could do. They are not comparable to the Colab runs: 1,800 steps, batch size 16 (4,096 tokens per step, about 7.4M
tokens in total), chat share 50%, stories limited to 60,000, and trained on a 4-core CPU in about 6.5 minutes.

| variant | val stories | val chat | mixed |
|---|---|---|---|
| persona (authored chat, bot-only loss) | 2.4919 | 0.8357 | 1.6638 |
| DailyDialog + everyday-conversations | 2.5020 | 3.2066 | 2.8543 |

Persona variant: the chat data is hand-written by the assistant (a small, friendly, honest bot with a consistent
voice; real stories and real user questions are mixed in). The replies were consistent and in character:
"who are you?" -> "I can talk with you and try to answer questions. I am very small, so I make mistakes
sometimes."; "how many r in strawberry?" -> "I am sorry, I do not know. I like simple things, like stories,
colors and music." One failure: "my name is Dana" -> "Hello, Leo..." (the training names were a fixed list of
30; the data was since changed to include generated names, untested). The very low chat loss (0.84) mostly
reflects that the data is narrow and repetitive, so it is optimistic. This variant is authored data, not a
downloaded dataset, and the project owner objected that it is not "real" (the model itself has no hard-coded
answers, but the behavior is learned from authored text). It has not been run on Colab.

## Appendix B. Adding a model to this report

1. Train, then record: run folder, config, chat mode, final EVAL line (stories, chat, mixed) and wall time from
   `train.log`.
2. Run `python -m tinylm.eval_chat --ckpt_dir <run>` (Colab chat notebook, last cell) and keep the transcript
   under `transcripts/<run>.txt`.
3. Score each reply 0/1/2 into `scores/<run>.json` (same format as the existing file), run
   `python reports/score_summary.py`, and update sections 1, 4, 5, 6 and 7.
