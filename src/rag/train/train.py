"""Fine-tune the dense bi-encoder (bge-m3) with symmetric InfoNCE and mined hard negatives.

bf16 autocast + gradient checkpointing, so bge-m3 (568M params) fits on one 24 GB GPU.
The loss is computed in fp32: logits divided by a small temperature lose too much precision in bf16.
"""

import random

from loguru import logger

from rag.config import TrainConfig
from rag.train.data import build_examples
from rag.train.loss import InfoNCE


def train(cfg: TrainConfig, log_every: int = 50) -> str:
    """Returns the directory the fine-tuned model was saved to."""
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    model = SentenceTransformer(cfg.base_model, device="cuda")
    model.max_seq_length = cfg.max_seq_length

    # hard negatives come from the base model, before any update
    with torch.autocast("cuda", dtype=torch.bfloat16):
        examples = build_examples(lambda texts: model.encode(texts, batch_size=128, normalize_embeddings=True),
                                  cfg.examples_per_source, cfg.mine_top_k, rng)
    logger.info("{} training examples", len(examples))

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    model.train()
    loss_fn = InfoNCE(cfg.init_temperature).to("cuda")
    optimizer = torch.optim.AdamW([{"params": model.parameters(), "lr": cfg.lr},
                                   {"params": loss_fn.parameters(), "lr": cfg.temperature_lr, "weight_decay": 0.0}])
    steps = cfg.epochs * (len(examples) // cfg.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup * steps), steps)

    def embed(texts: list[str]):
        features = {k: v.to("cuda") for k, v in model.preprocess(texts).items()}
        return F.normalize(model(features)["sentence_embedding"].float(), dim=-1)

    step = 0
    for epoch in range(cfg.epochs):
        rng.shuffle(examples)
        for start in range(0, len(examples) - cfg.batch_size + 1, cfg.batch_size):  # drop the last partial batch
            batch = examples[start : start + cfg.batch_size]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = embed([e.query for e in batch])
                c = embed([e.positive for e in batch] + [e.negative for e in batch])
            loss = loss_fn(q, c[: len(batch)], c[len(batch) :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % log_every == 0:
                logger.info("epoch {} step {}/{}: loss {:.4f}, temperature {:.4f}", epoch + 1, step, steps,
                            loss.item(), loss_fn.temperature().item())

    model.save(cfg.output_dir)
    logger.info("saved to {}", cfg.output_dir)
    return cfg.output_dir
