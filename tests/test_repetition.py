import torch

from tinylm.bpe import BOT, END, USER, BPETokenizer
from tinylm.chat import bot_reply_ids, trim_history
from tinylm.config import ModelConfig
from tinylm.model import GPT
from tinylm.sample import apply_repetition_penalty, generate


def test_penalty_math():
    logits = torch.tensor([[2.0, -1.0, 0.5, 3.0]])
    out = apply_repetition_penalty(logits, {0, 1}, 2.0)
    assert torch.allclose(out, torch.tensor([[1.0, -2.0, 0.5, 3.0]]))   # positive divided, negative multiplied
    assert torch.equal(logits, torch.tensor([[2.0, -1.0, 0.5, 3.0]]))   # input is not modified
    assert torch.equal(apply_repetition_penalty(logits, {0}, 1.0), logits)
    assert torch.equal(apply_repetition_penalty(logits, set(), 2.0), logits)


def test_penalty_always_lowers_probability():
    torch.manual_seed(0)
    logits = torch.randn(1, 50) * 3
    ids = {1, 7, 20, 33}
    before = torch.softmax(logits, -1)
    after = torch.softmax(apply_repetition_penalty(logits, ids, 1.5), -1)
    assert all(after[0, i] < before[0, i] for i in ids)


def test_generate_with_penalty_never_repeats_when_forced():
    """A model whose logits strongly prefer one token repeats it; a large penalty must break the loop."""
    torch.manual_seed(0)
    model = GPT(ModelConfig(vocab_size=40, n_layer=1, n_head=2, dim=16, context_len=32)).eval()
    with torch.no_grad():
        model.tok_emb.weight.mul_(8.0)   # sharpen the distribution so greedy decoding is repetitive
    plain = generate(model, [1, 2, 3], 12, temperature=0, stop_ids=())
    penalized = generate(model, [1, 2, 3], 12, temperature=0, stop_ids=(), repetition_penalty=5.0)
    assert len(set(penalized)) >= len(set(plain))
    assert len(set(penalized)) > 1


def test_stop_tokens_are_never_penalized():
    model = GPT(ModelConfig(vocab_size=40, n_layer=1, n_head=2, dim=16, context_len=32)).eval()
    seen = {}
    import tinylm.sample as sample
    original = sample.apply_repetition_penalty

    def spy(logits, token_ids, penalty):
        seen["ids"] = set(token_ids)
        return original(logits, token_ids, penalty)
    sample.apply_repetition_penalty = spy
    try:
        generate(model, [1, 2], 3, temperature=0, stop_ids={5}, repetition_penalty=1.5, penalize_ids={5, 6})
    finally:
        sample.apply_repetition_penalty = original
    assert 5 not in seen["ids"] and 6 in seen["ids"]


def test_bot_reply_ids_only_collects_bot_words():
    tok = BPETokenizer.train(["hello there how are you fine thanks"] * 20, vocab_size=300)
    u, b, e = tok.token_id(USER), tok.token_id(BOT), tok.token_id(END)
    user_words, bot_words = tok.encode("hello there"), tok.encode("fine thanks")
    history = [u] + user_words + [e, b] + bot_words + [e, u] + tok.encode("how are you") + [e, b]
    used = bot_reply_ids(history, tok)
    assert used == set(bot_words)
    assert not (used & {u, b, e})
    assert trim_history(list(history), tok, 100) == history
