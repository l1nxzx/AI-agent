"""Configuration: model size and training settings live in a JSON file.

Everything about the size of the model is a setting here (layers, dim, heads,
context length, vocab size), so growing from 2M to 20M parameters means
choosing another config file, not editing code.
"""
import ast
import dataclasses
import json
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 2000      # must equal the tokenizer vocab size (checked at startup)
    n_layer: int = 4            # number of transformer blocks
    n_head: int = 6             # attention heads per block
    dim: int = 192              # residual stream width (a.k.a. d_model)
    context_len: int = 256      # maximum sequence length the model can attend over
    mlp_mult: float = 8 / 3     # SwiGLU hidden width = mlp_mult * dim (rounded to a multiple of 32)
    dropout: float = 0.0        # small models on lots of data do not need dropout
    rope_theta: float = 10000.0  # base frequency for rotary position embeddings
    attn_impl: str = "sdpa"     # "sdpa" (fast, PyTorch kernel) or "manual" (explicit math, for learning/tests)

    def validate(self):
        if self.dim % self.n_head != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by n_head ({self.n_head})")
        head_dim = self.dim // self.n_head
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be even for rotary embeddings")
        if self.attn_impl not in ("sdpa", "manual"):
            raise ValueError(f"attn_impl must be 'sdpa' or 'manual', got {self.attn_impl!r}")


@dataclass
class TrainConfig:
    seed: int = 1337
    batch_size: int = 64            # sequences per micro-step
    grad_accum_steps: int = 1       # micro-steps per optimizer step (effective batch = batch_size * this)
    max_steps: int = 8000           # optimizer steps
    lr: float = 2e-3                # peak learning rate
    min_lr_ratio: float = 0.1       # final lr = lr * min_lr_ratio
    warmup_steps: int = 300         # linear warmup length
    weight_decay: float = 0.1       # applied to weight matrices only
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500        # steps between validation runs (also prints samples)
    eval_batches: int = 25          # batches per validation source per eval
    log_interval: int = 50          # steps between training log lines
    ckpt_interval: int = 500        # steps between checkpoints
    chat_fraction: float = 0.15     # share of each batch drawn from chat data
    chat_loss: str = "bot_only"     # bot_only: learn only bot replies in chat; all: learn every chat token
    dtype: str = "auto"             # auto | fp32 | fp16 | bf16 (auto: fp16 on CUDA, fp32 on CPU)
    sample_tokens: int = 80         # tokens generated for the sample shown at each eval

    def validate(self):
        if not 0.0 <= self.chat_fraction <= 1.0:
            raise ValueError("chat_fraction must be in [0, 1]")
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        if self.chat_loss not in ("bot_only", "all"):
            raise ValueError("chat_loss must be bot_only or all")
        if self.dtype not in ("auto", "fp32", "fp16", "bf16"):
            raise ValueError("dtype must be auto, fp32, fp16 or bf16")


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return dataclasses.asdict(self)


def _build(cls, values, section):
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown keys in '{section}': {sorted(unknown)}; valid keys: {sorted(known)}")
    return cls(**values)


def config_from_dict(d):
    cfg = Config(
        model=_build(ModelConfig, d.get("model", {}), "model"),
        train=_build(TrainConfig, d.get("train", {}), "train"),
    )
    cfg.model.validate()
    cfg.train.validate()
    return cfg


def load_config(path, overrides=()):
    """Load a JSON config and apply overrides such as 'train.max_steps=50'."""
    with open(path) as f:
        d = json.load(f)
    d.setdefault("model", {})
    d.setdefault("train", {})
    for item in overrides:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        section, name = key.split(".", 1)
        if section not in ("model", "train"):
            raise ValueError(f"override section must be 'model' or 'train', got {section!r}")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        d[section][name] = value
    return config_from_dict(d)
