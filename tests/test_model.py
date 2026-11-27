import math

import pytest
import torch

from tinylm.config import ModelConfig
from tinylm.model import GPT, count_parameters


def make(**kw):
    base = dict(vocab_size=100, n_layer=2, n_head=2, dim=32, context_len=16)
    base.update(kw)
    return GPT(ModelConfig(**base))


def test_forward_shapes_and_loss():
    m = make()
    idx = torch.randint(0, 100, (3, 10))
    logits, loss = m(idx, idx)
    assert logits.shape == (3, 10, 100)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_initial_loss_near_log_vocab():
    torch.manual_seed(0)
    m = make(vocab_size=500, dim=64)
    idx = torch.randint(0, 500, (16, 16))
    # Targets must be independent of the inputs. (With tied embeddings, targets == inputs would give a
    # lower loss, because a token's embedding has a high dot product with itself.)
    targets = torch.randint(0, 500, (16, 16))
    _, loss = m(idx, targets)
    assert abs(loss.item() - math.log(500)) < 0.3


def test_parameter_count_formula():
    cfg = ModelConfig(vocab_size=100, n_layer=3, n_head=2, dim=32, context_len=16)
    m = GPT(cfg)
    hidden = (int(cfg.mlp_mult * cfg.dim) + 31) // 32 * 32
    per_block = 4 * 32 * 32 + 3 * 32 * hidden + 2 * 32     # qkv+proj, swiglu, two norms
    expected = 100 * 32 + 3 * per_block + 32               # embedding (tied) + blocks + final norm
    assert count_parameters(m)["total"] == expected


def test_causality_future_tokens_do_not_affect_past():
    torch.manual_seed(0)
    m = make().eval()
    a = torch.randint(0, 100, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 100
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-5)
    assert not torch.allclose(la[:, 8:], lb[:, 8:], atol=1e-5)


def test_manual_attention_matches_sdpa():
    torch.manual_seed(0)
    fast = make(attn_impl="sdpa").eval()
    slow = make(attn_impl="manual").eval()
    slow.load_state_dict(fast.state_dict())
    idx = torch.randint(0, 100, (2, 16))
    assert torch.allclose(fast(idx)[0], slow(idx)[0], atol=1e-4)


def test_rope_is_relative():
    """Shifting both query and key positions by the same amount must not change q.k."""
    from tinylm.model import apply_rope, build_rope_cache
    cos, sin = build_rope_cache(8, 32, 10000.0)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    def score(pq, pk):
        qq = apply_rope(q.expand(1, 1, 32, 8), cos, sin)[:, :, pq]
        kk = apply_rope(k.expand(1, 1, 32, 8), cos, sin)[:, :, pk]
        return (qq * kk).sum()
    assert torch.allclose(score(5, 3), score(15, 13), atol=1e-4)


def test_context_overflow_raises():
    m = make()
    with pytest.raises(ValueError):
        m(torch.zeros(1, 17, dtype=torch.long))


def test_config_sizes_near_targets():
    import json, os
    expect = {"tiny": (1.8e6, 2.8e6), "small": (9e6, 13e6), "medium": (17e6, 23e6)}
    for name, (lo, hi) in expect.items():
        with open(os.path.join(os.path.dirname(__file__), "..", "configs", name + ".json")) as f:
            mc = ModelConfig(**json.load(f)["model"])
        n = count_parameters(GPT(mc))["total"]
        assert lo < n < hi, (name, n)
