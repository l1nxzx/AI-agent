# tinylm: model performance report

Living document. One section per trained model, a comparison table, and cross-cutting findings. New models are
added at the end of section 4 and to the tables in section 5. Evidence (verbatim transcripts and item-level
scores) is stored next to this file in `transcripts/` and `scores/`.

Last updated after: run `small_dailydialog_only`. Models covered: 5 full Colab runs (three tiny at 2.16M
parameters, two small at 11.4M).

## 1. Summary

* Runs 1-3 are the same 2.16M-parameter transformer and differ only in the chat data and the fraction of each batch
  spent on it, which isolates the effect of data. Runs 4 and 5 repeat the two mixed/DailyDialog-only data
  settings with an 11.4M-parameter model, which isolates the effect of size (with the caveats in 4.4 and 4.5).
* Story generation is the same in every run: locally fluent short stories with weak global coherence and some
  repetition. Validation loss on stories is 1.81 (perplexity 6.1) in run 1 and 1.84 (perplexity 6.3) in run 3, the
  two runs with a story loss on file. The small gap is consistent with run 3 spending 35% of each batch on chat
  instead of 10%, but one run each cannot confirm that.
* Chat is poor in every run. The model produces grammatical, human-sounding fragments that mostly do not relate
  to the question. No data variant or model size tested so far fixed this.
* The failures differ with the data, which shows the data decides the behavior, not a bug:
  broad assistant-style chat gives confident topical invention (run 1); mixing an LLM-written dataset with
  human dialogue gives an inconsistent voice (run 2); human-human dialogue only gives phone-call and office
  style replies with no stable identity (run 3).
* On the standard rubric, run 3 scores a mean of 0.81 out of 2 on 57 graded replies (23% sensible, 42% nonsense).
  The strongest groups are identity (1.30) and topic questions (1.20). The weakest are unknown or factual
  questions (0.36) and greetings (0.56).
* Size, matched on data twice: going from 2M to 11M parameters improves the mixed validation loss by about 0.17 in both
  pairs (run 2 -> 4: 2.1284 -> 1.9583; run 3 -> 5: 2.1570 -> 1.9859) and the story validation loss from 1.84 to
  about 1.56-1.59 (perplexity 6.3 -> 4.8-4.9).
* Fluency improves much more than correctness. The best small model (run 5, DailyDialog only) gives grammatical replies
  almost everywhere: 11% of its graded replies scored 0, against 42% for run 3. But the share scored 2 (sensible and
  on-topic) barely moved, 26% against 23%. The rubric mean is 1.15 against 0.81 (different prompt lists, so
  tentative). Replies are natural-sounding and often non sequiturs from the shop, office and phone scripts of the
  data ("Hi" -> "Hello. Is this the right building?"). Facts, names and memory are still not learned.
* Both small runs overfit the chat data: the best mixed validation loss came at step 10,500 in run 4 (about 16 passes
  over the chat data) and at step 8,000 in run 5 (about 14 passes), and was worse afterwards. Shortening the
  schedule from 15,000 to 10,000 steps did not remove it. The saved best checkpoint is what chat uses.
* Open questions: does a small authored identity layer fix identity and "I do not know"; can early stopping or a
  smaller chat share keep the best generalization; how much do the numbers move with sampling settings (not yet
  tested).

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

Run 4 uses the `small` config instead: 6 layers, dim 384, 6 heads, context 256, vocab 2000, 11,389,824 parameters
(768,000 embedding, 10,621,440 in blocks); batch 32 x 2 accumulation x 256 tokens = 16,384 tokens/step; 15,000 steps =
245,760,000 tokens; peak lr 1e-3 and 500 warmup steps (from `configs/small.json`, assumed unchanged); chat share 35%,
loss on bot tokens only. On the T4 it ran at about 84,000 tokens/s (194 ms/step, 1.54 GB), in 50.9 minutes.

Run 5 uses the same `small` config on `dailydialog_only` with `train.max_steps=10000` (cosine schedule over 10,000
steps; 163,840,000 tokens; 33.9 minutes; about 203 ms/step).

Chat format in all runs: `<|user|> text <|end|> <|bot|> text <|end|> ... <|endoftext|>`.

Differences between runs:

| run | folder on Drive | chat data | chat share of batch | loss on chat |
|---|---|---|---|---|
| 1 | `runs/tiny` | everyday-conversations (SmolTalk) only | 10% | all tokens |
| 2 | `runs/tiny_dailydialog` | DailyDialog + everyday-conversations | 35% | bot tokens only |
| 3 | `runs/tiny_dailydialog_only` | DailyDialog only | 35% | bot tokens only |
| 4 | `runs/small_dailydialog` | DailyDialog + everyday-conversations (same as run 2) | 35% | bot tokens only |
| 5 | `runs/small_dailydialog_only` | DailyDialog only (same as run 3) | 35% | bot tokens only |

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
* Runs 1 and 2 were judged only from ad-hoc transcripts, and the scored items of runs 3, 4 and 5 are the user's own
  prompts, not the standard set; they overlap with it but are not identical. Running the standard set on all saved models
  would make the comparison fair. This is the first item in section 7.
* In interactive transcripts the conversation history carried over between prompts until `reset`, so some replies
  are conditioned on earlier turns.
* Sampling settings were not recorded in every paste (see each model). All transcripts in this report were
  made **before the repetition penalty existed, so effectively with no penalty (1.0)**. Chat and `eval_chat` now
  default to a penalty of 1.2 (the file header records it), so new transcripts are not directly comparable with these
  unless run with `--repetition_penalty 1.0`.

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
365 tokens, longer than the 256 context, so many conversations are cut into windows).

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

### 4.4 Run 4: small, DailyDialog + everyday-conversations (`runs/small_dailydialog`)

Setup: see section 2. Confirmed by the user: `CONFIG = "small"`, `CHAT_MODE = "dailydialog"`. This was started by
accident (the planned run was `dailydialog_only`), and it has the same chat data as run 2, which makes run 2 the
matched comparison for the effect of model size. Sampling settings in the chat paste were not recorded (the
notebook default is 0.5 / 20).

Training metrics (Colab log, end of run):

| step | val stories | val chat (bot tokens) | mixed | note |
|---|---|---|---|---|
| 10,500 | not provided | not provided | 1.9583 (best) | checkpoint used for the chat below |
| 15,000 (final) | 1.5598 (ppl 4.8) | 2.8048 (ppl 16.5) | 1.9956 | best stays 1.9583 |

Training loss at the end is about 1.32 (mixed batches). Throughput 76-85k tokens/s, three skipped fp16 steps in
total (the loss scaler lowering its scale), gradient norm about 0.55. The end-of-run story samples are coherent
("Once upon a time, there was a little girl named Lily... Her mom said, \"Be careful, Lily.\""); the chat sample lines
are "I'm fine. Look at this guy. I just saw him last week." (step 15000) and "I am fine. What have you been up to
today?" (step 14500).

**Overfitting on the chat data.** At 35% chat share for 15,000 steps the model saw the chat data about 23 times
(about 3.8M chat tokens; the tiny run 2 saw it about 12 times). The mixed validation loss was best at step 10,500 and worse by step 15,000 although the story loss
was still improving, so the chat part must have risen. The matched comparison with run 2 (tiny, same data) is the mixed validation loss:
2.1284 (run 2) against 1.9583 (run 4 best, step 10,500) and 1.9956 (run 4 final). The training script keeps
the best checkpoint, which is why the chat used step 10,500. The full `EVAL` curve (one line every 500 steps) would
show exactly where the chat loss turned; it was not provided.

Chat quality (56 graded replies, same rubric; transcript `transcripts/small.txt`, scores `scores/small.json`):

| group | n | mean | group | n | mean |
|---|---|---|---|---|---|
| greeting | 8 | 1.50 | unknown / factual | 7 | 0.14 |
| topic | 13 | 1.31 | context | 6 | 0.67 |
| loop | 7 | 1.29 | robust | 4 | 0.50 |
| social | 5 | 0.80 | story | 2 | 0.50 |
| identity | 4 | 0.75 | **all** | 56 | **0.95** (25% scored 2, 30% scored 0) |

Findings, with evidence:

1. **Natural replies to simple social exchanges.** "Hi, how are you today?" -> "I'm just fine, thanks. How are you
   today?"; "How are you?" -> "Fine, thank you."; "Thanks, bye" -> "Goodbye." Greeting mean 1.50 (run 3: 0.56), though
   the same prompt can still derail ("Hello, how are you?" gave "I'm going to the hospital to see a doctor. My
   father's in New York." in one sample and "I'm fine, thanks." in another).
2. **Plausible answers to everyday questions.** "What do you do for work?" -> "I'm a teacher."; "What are you doing
   this weekend?" -> "I'm preparing for a course."; "Do you like dogs?" -> "I like dogs."; "Explain gravity." ->
   "What do you mean?" (it can ask back).
3. **Invented numbers and facts, now stated confidently.** "How many r's are in strawberry?" -> "15 round."; "What
   is 2 plus 2?" -> "He's 14." and "15."; "What is the capital of France?" -> "2225." and "The capital of France is
   South Youth, Notice Ryan."; "hi how r u" -> "125,000 dollars." DailyDialog contains many prices and times, so the
   model learned that numbers go in answers, not which ones. Unknown-question mean 0.14, the lowest group.
4. **Identity is invented corporate titles:** "I'm the Corpor Lod.", "I'm your license.", "My name is Laura's
   Connica.", "No, I'm the Corporate Management." This reflects the business-dialogue register of the data and
   there is still no stable self.
5. **The model copies itself.** "I love you" was answered "I love you, too." three times; "I hate you" got the same
   line three times; and then "Tell me a story." also got "I love you, too." In another conversation, once the
   bot said "I like dogs." it answered "You are stupid" and "You are cute" with "I do like dogs." The model's own
   earlier replies dominate its next reply.
6. **Names are not remembered:** "My name is Blake." -> "I'm Blake. I have been looking forward to seeing you."
   (the bot takes the name), "What is my name?" -> "I'm Mega."
7. **Stories are not a chat skill** ("Tell me a story." -> "OK, I'll listen to you.", the roles swapped, as in the
   source dialogues where the user tells the story).

Verdict: the best of runs 1-4 on fluency, on natural short replies and on stories, and a clear step up from
the tiny models on simple dialogue. It is not more grounded: facts, identity and memory are no better and in
places worse (identity 0.75 against 1.30, unknown 0.14 against 0.36, but those are against run 3, which had different chat data
and different prompt lists).

### 4.5 Run 5: small, DailyDialog only (`runs/small_dailydialog_only`)

Setup: the `small` config (see section 2) on `dailydialog_only`, the same data folder as run 3, with
`train.max_steps=10000` (the schedule decays over 10,000 steps instead of 15,000). Settings confirmed by the user:
`CONFIG = "small"`, `OVERRIDES = "train.max_steps=10000"`, `CHAT_MODE = "dailydialog_only"`. This run is the matched
size comparison for run 3 (tiny, same data). Chat sampling settings were not recorded (notebook default 0.5 / 20).

Training metrics (Colab log, end of run; 33.9 minutes):

| step | val stories | val chat (bot tokens) | mixed | note |
|---|---|---|---|---|
| 8,000 | not provided | not provided | 1.9859 (best) | checkpoint used for the chat below |
| 10,000 (final) | 1.5858 (ppl 4.9) | 2.7775 (ppl 16.1) | 2.0029 | best stays 1.9859 |

Overfitting again: with 35% chat share the model had about 14 passes over the chat data (3,245,201 tokens) by the
best step (8,000) and about 18 by step 10,000. The mixed loss was worse at the end although the story loss was
still falling. The tiny model on the same data (run 3) had its best at its last step (8,000, 14 passes) and did not
overfit; the larger model overfits at about the same number of passes. Run 4's best was at about 16 passes. The
chat generalization of the 11M model peaks around 14-16 passes over this chat data in both runs.

Stories (sampling cell, temperature 0.8): coherent openings and dialogue ("Alex said, \"I lost my toy car.\""), with
odd content drift ("a big, bright power... kept meowing and meowing", a "red cord" that "made a loud noise"). The
final story loss (1.5858) is close to run 4's (1.5598, 15,000 steps).

Chat quality (46 graded replies, rubric v1; transcript `transcripts/small_dailydialog_only.txt`, scores
`scores/small_dailydialog_only.json`):

| group | n | mean | group | n | mean |
|---|---|---|---|---|---|
| identity | 4 | 1.50 | context | 6 | 0.83 |
| greeting | 6 | 1.33 | unknown / factual | 4 | 0.75 |
| topic | 9 | 1.33 | robust | 4 | 1.00 |
| loop | 8 | 1.25 | story | 5 | 1.00 |
| **all** | 46 | **1.15** (26% scored 2, 11% scored 0) | | | |

(`social` prompts were not asked in this run.)

Findings, with evidence:

1. **Grammatical almost everywhere, correct only sometimes.** The share of replies scored 0 fell to 11% (run 3: 42%,
   run 4: 30%), but the share scored 2 stayed at 26% (run 3: 23%, run 4: 25%). The model became fluent, not more
   knowledgeable.
2. **Natural short exchanges.** "Hello" -> "Hi there. How are you doing?"; "Hello, how are you?" -> "Fine, thanks.
   And you?"; "Do you want to go to the movies tonight?" -> "Sure. I love movies."; "What are you doing this
   weekend?" -> "I'm going to the cinema on Friday night."; "Tell me a story." -> "So what do you want to hear?" (the
   first run to answer a story request sensibly).
3. **Fluent non sequiturs from the source scripts.** "Hi" -> "Hello. Is this the right building?"; "Good morning" ->
   "I'm afraid we have nobody here."; "hi how r u" -> "that's why we didn't like it."; "Why" -> "well, we don't often
   sell at all." These are lines that fit shop, office and building scenes, not the prompt.
4. **Unknown questions get fluent evasions, not much invented arithmetic.** "How many r's are in strawberry?" ->
   "There's nothing in particular."; "What is the capital of France?" -> "It's only 80 yuan and it's a very popular
   amount."; "What is 2 plus 2?" -> "May I see your 5 -45?" None is correct. The unknown-group mean (0.75) rises
   because grammatical non sequiturs score 1 under the rubric, not because the answers are better.
5. **Identity is relevant, repeated and not grounded.** "Who are you?" and "What are you?" both got "I'm the guy over
   there."; "What is your name?" -> "My name is James."; "Are you a robot?" -> "Yes, I am." The same name came back
   for "What is my name?" ("My name is James."), so the name is a habit, not a memory. "My name is Blake." was
   answered with "How's your friends?"
6. **The repetition loop persists.** "I love you" -> "I love you too." three times (once with an invented name,
   "I love you too, Rose."), and "I hate you" -> "I love you too." three times, until the fourth gave "I'm sorry to
   hear that you're too young."
7. **Context use is weak.** "I like dogs." -> "Me too. Black." -> (asked "What animal do I like?") "Well, do you like
   Chinese food?"; in another conversation "Well, the Reading is a past."

Verdict: the best model so far on every measure in this report (rubric mean 1.15, 11% nonsense, best mixed validation
loss), and a clear step up in fluency, but the same gaps in grounding as the others. Its best replies are short and
generic, and its worst are fluent lines from unrelated scenes.

## 5. Comparison

### 5.1 Item-level scores (rubric v1; generated by `python reports/score_summary.py`)

| run | n | overall | share 2 | share 0 | context | greeting | identity | loop | robust | social | story | topic | unknown |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| small_dailydialog | 56 | 0.95 | 25% | 30% | 0.67 (6) | 1.50 (8) | 0.75 (4) | 1.29 (7) | 0.50 (4) | 0.80 (5) | 0.50 (2) | 1.31 (13) | 0.14 (7) |
| small_dailydialog_only | 46 | 1.15 | 26% | 11% | 0.83 (6) | 1.33 (6) | 1.50 (4) | 1.25 (8) | 1.00 (4) | - | 1.00 (5) | 1.33 (9) | 0.75 (4) |
| tiny_dailydialog_only | 57 | 0.81 | 23% | 42% | 0.67 (6) | 0.56 (9) | 1.30 (10) | - | - | 0.80 (5) | 1.00 (3) | 1.20 (10) | 0.36 (14) |

Runs 1 and 2 are not yet scored item by item (see section 3.3). The prompt lists differ between runs (they are the
user's own prompts), the `social`, `loop` and `robust` groups were not asked in every run, and the mixed run 4 used
different chat data from runs 3 and 5, so only the overlapping groups are comparable and differences of a few
tenths are within the noise of a single rater. The data-matched pairs are run 3 (tiny) with run 5 (small), both
DailyDialog only, and run 2 (tiny, not scored item by item) with run 4 (small), both mixed.

### 5.2 Training metrics on file

| run | best val stories | best val chat | best mixed val | wall time |
|---|---|---|---|---|
| 1 `tiny` | 1.8074 | 2.1776 | 1.8444 (chat share 10%) | 8.7 min |
| 2 `tiny_dailydialog` | not provided | not provided | 2.1284 (chat share 35%) | not provided |
| 3 `tiny_dailydialog_only` | 1.8369 | 2.7513 (bot tokens only) | 2.1570 (chat share 35%) | 8.8 min |
| 4 `small_dailydialog` | 1.5598 (final step; ppl 4.8) | 2.8048 (final step; ppl 16.5) | best 1.9583 at step 10,500; final 1.9956 | 50.9 min |
| 5 `small_dailydialog_only` | 1.5858 (final step; ppl 4.9) | 2.7775 (final step; ppl 16.1) | best 1.9859 at step 8,000; final 2.0029 | 33.9 min |

Mixed losses are comparable only between runs with the same chat data and share: runs 2 and 4 (2.1284 -> 1.9583) and
runs 3 and 5 (2.1570 -> 1.9859).

### 5.3 Qualitative comparison

| aspect | run 1 | run 2 | run 3 | run 4 (small, mixed) | run 5 (small, DD only) |
|---|---|---|---|---|---|
| reply length | long | mixed | short | short | short |
| on-topic | no | rarely | sometimes (generic) | often, simple questions | often, simple questions |
| consistent voice | invents topics | two voices | one voice | mostly one, business register | one voice, shop/office scenes |
| identity | invents ("many conductures") | random | random names | invented corporate titles | repeated "guy over there", name James |
| invents facts/numbers | yes, heavily | some | little | yes, numbers | fluent evasions, wrong numbers |
| context use | not tested | weak | weak | weak, copies its own replies | weak |
| nonsense replies (score 0) | not scored | not scored | 42% | 30% | 11% |
| stories (text mode) | OK | OK | not shown | best (val 1.56) | similar (val 1.59) |

## 6. Cross-cutting findings

1. **Data decides chat behavior at this size.** With the model fixed, the chat data alone changed the failure mode
   from topical invention to inconsistent voices to phone-call style. (Evidence: runs 1-3.)
2. **Validation loss does not predict chat quality.** Run 1 had the lowest chat perplexity on file (8.8, against
   15.7 for run 3, which are not directly comparable) and the worst replies.
3. **Size helps fluency, in a repeatable amount.** Matched on data twice, the mixed validation loss drops by about
   0.17 going from 2M to 11M parameters (2.1284 -> 1.9583 on the mixed data, 2.1570 -> 1.9859 on DailyDialog only),
   and the story loss from 1.84 to about 1.56-1.59.
4. **Fluency is not correctness.** From run 3 to run 5 the share of nonsense replies fell from 42% to 11%, while
   the share of sensible, on-topic replies stayed at 23-26%. Fluent non sequiturs score 1 under the rubric, so
   groups such as `unknown` rise (0.36 -> 0.75) without any answer becoming correct. Facts, arithmetic, identity
   and memory are not learned at 2M or 11M parameters on this data.
5. **No uncertainty without data for it.** Real dialogue data never says "I do not know, I am a small model", so
   the model never does; it gives fluent evasions or invented names and numbers. (Preliminary evidence in the
   appendix suggests this can be taught with data.)
6. **Repetition and self-copying persist at both sizes.** Out-of-domain prompts make the model lock onto words
   ("wine", run 3) and repeat its own earlier replies ("I love you, too." for unrelated prompts in run 4; "I love
   you too." three times for "I hate you" in run 5). Sampling-time repetition control has not been tried yet.
7. **The larger model overfits the small chat set.** Mixed validation loss was best at step 10,500 in run 4 and at
   step 8,000 in run 5, i.e. after about 14-16 passes over the chat data, and worse afterwards, even with the shorter
   schedule of run 5. The tiny model reached 14 passes without overfitting. Saving the best checkpoint by validation
   loss protects the result. More chat data, a lower chat share or early stopping would stop wasted training.
8. **Story ability is stable across chat data variants** and mostly comes from the TinyStories mixture; it improves
   with size.

## 7. Next experiments

1. Paste the `EVAL` lines (one per 500 steps) for runs 4 and 5 to see exactly where the chat loss turned (optional).
2. Run the standard set on the saved models of runs 1-5 (`tinylm.eval_chat`) and score them, so the comparison in
   5.1 uses identical prompts. Run 2 is the most useful one to add, as the partner of run 4.
3. Early stopping is implemented (`train.early_stop_patience`, default 3 validations without a new best mixed
   loss); use it in the next `small` or `medium` run and record where it stops.
4. A repetition penalty is implemented (`--repetition_penalty`, default 1.2 in chat and `eval_chat`). Measure it: run
   the standard set on the saved models with 1.0, 1.1, 1.2 and 1.3 and score the loops and non-sequitur rate. A
   quick test on a weak CPU model (not one of the runs above) showed loops broken at 1.2 and drift at 1.4.
5. Try a small authored identity layer on top of real data (a few percent of the chat share, covering greetings,
   identity and "I do not know"). This is a data change only; no rules in the model or code.
6. Hold off on `medium` (20M) until there is more chat data: it would overfit the chat set even sooner.
7. Record sampling settings and the final training log lines for every new run, so the tables have no gaps.

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
downloaded dataset (and its first version had a data bug: in some intents the reply was chosen at random from a list, so a reply could be paired with the wrong question; fixed since, with question-specific replies stored as pairs, and the whole bank rewritten for the Petryk identity), and the project owner objected that it is not "real" (the model itself has no hard-coded
answers, but the behavior is learned from authored text). It has not been run on Colab.

### A.2 Identity layer (Petryk) on DailyDialog: CPU sanity run

Added after run 5 and not a Colab run. It checks that the `dailydialog_identity` data (DailyDialog plus authored
identity conversations in the voice of Petryk, a calm, plain-spoken small language model) is learned and not broken.
Same CPU setup as above (1,800 steps, batch 16, chat share 50%), but with the identity layer enlarged to 6,000
conversations, 11.7% of the chat tokens, so that a short run could show an effect (the real default is 1,500
conversations, about 3.4%). Validation stays DailyDialog-only: val stories 2.4966, val chat 3.1933, mixed 2.8450.

What it learned within 6.6 minutes (no repetition penalty): the greeting ("Hello. I am Petryk. What would you like to
talk about?"), what it can do (the honest "I cannot look things up, and my knowledge is limited" answer), its
creator ("A person built me and trained me from scratch..."), and honest refusal to guess for questions it cannot
answer ("What is the capital of France?" and the strawberry question both got "I cannot answer that reliably, and
I do not want to invent an answer."). What went wrong: "Who are you?" and "What is your name?" were answered
with the name-memory template ("You told me your name is Zoe."), because the name-memory questions included "Who am
I?"; name copying was not learned yet (a name given in the conversation was not reproduced); "Can you see me?" got
an answer from another question; and an everyday question ("Do you want to go to the movies tonight?") was
answered with a story, a sign the layer was too large at 11.7%. The data was fixed for the first problem (the
confusable question was removed); the others are expected to improve with the longer Colab run and the smaller
real share, and need checking there.

## Appendix B. Adding a model to this report

1. Train, then record: run folder, config, chat mode, final EVAL line (stories, chat, mixed) and wall time from
   `train.log`.
2. Run `python -m tinylm.eval_chat --ckpt_dir <run>` (Colab chat notebook, last cell) and keep the transcript
   under `transcripts/<run>.txt`.
3. Score each reply 0/1/2 into `scores/<run>.json` (same format as the existing file), run
   `python reports/score_summary.py`, and update sections 1, 4, 5, 6 and 7.
