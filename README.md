# MergeDNA

A simplified implementation of [MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging](https://arxiv.org/abs/2511.14806) (Li et al., 2024).

---

## Time investment

| Phase | Time |
|---|---|
| Reading and understanding the paper | ~2.5 hrs |
| Implementation | ~4.5 hrs |
| Fixing bugs and issues | ~1 hr |

---

## What this is

MergeDNA is a hierarchical autoencoder for genomic DNA sequences. The core idea is to learn a *dynamic tokenizer* — rather than chunking DNA into fixed-size k-mers, the model learns which bases are redundant and merges them adaptively using Token Merging (ToMe). Informative, unique regions are preserved at higher resolution; repetitive or low-information regions are compressed more aggressively.

This repo implements the architecture and training procedure described in the paper from scratch in PyTorch, with some simplifications noted below.

---

## Code structure

```
MergeDNA/
├── main.py                  # Entry point: loads config, builds components, runs training
├── config/
│   └── TEMPLATE.yaml        # All hyperparameters
├── dataloader/
│   └── dataloader.py        # FASTA parser, one-hot encoding, PyTorch Dataset
├── components/
│   ├── local.py             # LocalEncoder, LocalDecoder
│   └── latent.py            # LatentEncoder, LatentDecoder
├── utils/
│   ├── attn.py              # LocalWindowSelfAttn, SelfAttn (with DTEM metric projection)
│   └── tome.py              # bipartite_soft_matching, merge_wavg, merge_source, unmerge_source
├── loss/
│   └── loss.py              # LossManager: MTR, MTR-latent, AMTM losses
└── train/
    └── train.py             # Trainer: training loop, validation, checkpointing
```

---

## Architecture

The model is a four-component hierarchical autoencoder:

```
Input (B, N, 4)
    │
    ▼
LocalEncoder   — local-window self-attention with ToMe; compresses N → N/2
    │             source matrix S tracks which original tokens each merged token covers
    ▼
LatentEncoder  — global self-attention with optional ToMe; compresses N/2 → N/4
    │
    ▼
LatentDecoder  — global self-attention; unmerges back to N/2 using latent source
    │
    ▼
LocalDecoder   — local-window self-attention; unmerges back to N using local source S
    │
    ▼
Output (B, N, 4)  — reconstructed one-hot DNA sequence
```

**Token Merging (ToMe):** Within each attention block, a lightweight `metric_proj` layer (the DTEM cosine embedding) computes token similarity. Bipartite soft matching identifies the most similar token pairs and merges them via weighted average. A source matrix accumulates the merge history so tokens can be unmerged in the decoders.

**Three training losses:**

| Loss | Weight | Description |
|---|---|---|
| `L_MTR` | 1.0 | Local encoder (ToMe) → latent encoder → latent decoder → local decoder (unmerge). Full reconstruction. |
| `L_MTR_latent` | 0.25 | Same but also applies ToMe in the latent encoder. Local encoder outputs are detached to keep the two compression stages independent. |
| `L_AMTM` | 1.0 | Adaptive Masked Token Modeling. Uses the latent source matrix to assign masking probabilities — tokens in smaller merged groups (more informative) are more likely to be masked. Runs masked input through the network without latent merging and computes loss only on masked positions. |

**Compression hierarchy** (derived automatically from config, no manual tuning needed):
- Local encoder: 4096 → 2048 tokens
- Latent encoder: 2048 → 1024 tokens (for `L_MTR_latent`)

---

## Data

Human chromosome 1 from the hg38 reference genome, pre-processed into non-overlapping 4096 bp windows using `seqkit`. Bases are one-hot encoded as ACGT → 4-dim vectors; ambiguous bases (N) are all-zeros.

Sequences are split into non-overlapping 4096 bp windows for simplicity. The dataset contains ~56,000 training sequences, with a 90/10 train/val split (deterministic, fixed seed).

---

## What's missing / simplifications

1. **Single-head attention.** The paper uses standard multi-head Transformer blocks. This implementation uses single-head attention throughout for simplicity.

2. **Fixed compression ratio.** The paper uses a compression ratio *sampling* strategy during training — λ is sampled from a distribution each step so the model learns to handle variable compression. Here, `reduce_by` is a fixed value derived from the target compression ratio.

3. **Simplified Transformer blocks.** The paper uses LLaMA-style blocks (pre-norm, RMSNorm, SwiGLU FFN, rotary embeddings). This implementation uses standard post-norm blocks with LayerNorm and GELU.

4. **Partial dataset.** Only chromosome 1 is used. The paper trains on a much larger and more diverse genomic corpus spanning multiple species.

5. **Contiguous token merging.** Standard ToMe (bipartite soft matching) merges tokens based on similarity regardless of their position in the sequence — two bases that are far apart can be merged. The paper appears to enforce contiguous merging (only adjacent tokens are merged), which would be more biologically meaningful. However, the paper does not specify exactly how this is implemented, so this repo uses standard non-contiguous ToMe.

---

## Results

Training for 2 epochs on chromosome 1 (~56k sequences, batch size 4, gradient accumulation steps 16, effective batch size 64):

| Epoch | Global Step | Val MTR | Val MTR-Latent | Val AMTM | Val Total |
|---|---|---|---|---|---|
| 1 | 1,000 | 0.0111 | 0.0111 | 1.3252 | 1.3391 |
| 1 | 3,000 | 0.0099 | 0.0099 | 1.3347 | 1.3471 |
| 1 | 6,000 | 0.0062 | 0.0062 | 1.3107 | 1.3185 |
| 1 | 10,000 | 0.0056 | 0.0056 | 1.3131 | 1.3202 |
| 1 | 12,000 | 0.0148 | 0.0146 | 1.3023 | 1.3207 |
| 2 | 14,000 | 0.0070 | 0.0070 | 1.2994 | 1.3247 |
| 2 | 16,000 | 0.0081 | 0.0079 | 1.2930 | 1.3031 |

**What we can see:** The reconstruction losses (`val_mtr`, `val_mtr_latent`) drop quickly and consistently — from ~0.011 to ~0.006 within the first epoch — indicating the model is learning to compress and reconstruct DNA sequences. The AMTM loss (`val_amtm`) improves more slowly (~1.325 → ~1.293 over two epochs), which is expected: predicting masked tokens from context is a harder task and a better proxy for whether the model is learning meaningful representations.

**Caveats:** The reconstruction losses (`val_mtr`, `val_mtr_latent`) are clearly decreasing, which is encouraging. However, the AMTM loss barely moves from what would be expected near a random baseline (~1.33 for a 4-class prediction task). This is likely because chromosome 1 has low biological context diversity — repetitive regions dominate, so the model has little signal to learn meaningful masked token predictions from context. A more diverse training set spanning multiple chromosomes or species would be needed to see meaningful AMTM improvement. These results are also with single-head attention and a fixed compression ratio, which are known simplifications. Evaluation on a downstream task (promoter prediction, splice sites) would be a more reliable measure of representation quality.

---

## What's next

### Verify and understand current results
- Confirm the model is genuinely learning and not just fitting to repetitive chr1 structure — evaluate on held-out sequences from other chromosomes or species
- Audit tokenization: verify 4096 bp windows are biologically reasonable and understand how the paper handles chunking at sequence boundaries
- Set up a simple downstream eval task (e.g. promoter prediction, splice site detection) to monitor biological representation quality during training

### Architectural improvements
- Multi-head attention
- LLaMA-style blocks: RMSNorm, SwiGLU FFN, rotary position embeddings
- Compression ratio sampling strategy (variable λ during training, as in the paper)
- Train on the full multi-species genomic dataset used in the paper

### Scaling
- Profile I/O and GPU utilization — identify bottlenecks before scaling compute
- Mixed precision training (bfloat16)
- Flash Attention for memory-efficient long-sequence attention
- Multi-GPU training with DDP
- Multi-node training once single-node is saturated
