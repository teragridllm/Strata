# Qwen3.6-35B-A3B on one RTX 5060 Ti: Strata and llama.cpp

**This is a research note, not a benchmark result.** Nothing here was run as a protocol measurement.
It records three things: the state of one machine, the numbers its Strata engine had already written to its log,
and what llama.cpp needs before a fair comparison is possible. Section 5 is an estimate and says so.

## 1. The machine

- GPU: NVIDIA GeForce RTX 5060 Ti, 16,311 MiB, driver 580.178.04, 180 W. Blackwell, sm_120.
- CPU: Intel Core i7-14700, 28 logical CPUs. RAM: 62 GiB. NVMe: 587 GiB free.
- OS: Linux, `strata` 0.2.12 (engine 0.1.58, CUDA 13 build from the engine component).
- Model: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, 22.9 GB, plus `mmproj-F16.gguf` (858 MB).
  The file is the same one llama.cpp would load: architecture `qwen35moe`, 753 tensors, 41 blocks
  (40 layers and one MTP block), 256 experts, top-8.

The weights do not fit in 16 GiB. Both engines must keep part of every layer's experts on the CPU.
That makes host memory bandwidth, PCIe and the thread pool the limiting factors, not compute.

Strata's own layout on this machine, from the engine log:

```text
strata-q36: 753 tensors (layout checked), 4.40 GiB on the GPU in 1.5 s
strata-q36: context 32768 cells, prompt chunk 2048, MTP block loaded, 7.04 GiB of VRAM free
strata-q36: hybrid experts: 70 of 256 per layer in VRAM (5.13 GiB),
            all 18.75 GiB in pinned host memory, the rest of a batch of <= 128 rows
            on 14 CPU threads, bigger prompt chunks streamed (1.10 GiB staging)
```

`nvidia-smi` reported 15,260 of 16,311 MiB in use while that profile served. Load time was 13 to 23 s.

## 2. What the Strata log shows

Profile `ud-q4_k_xl-32k`: `--max-context 32768 --prefill auto --mtp 2 --pool-workers 13 --vision`.
The numbers below are grouped from 1,092 request lines of ordinary chat traffic, not from a fixed prompt set.
They are medians of entries that generated at least 20 tokens. Prompt read is the new tokens only, so a
request that reused a cached prefix reports the speed of the uncached part.

| Prompt tokens | Requests | Prompt read, tok/s | Decode, tok/s | Decode p25 to p75 |
| --- | ---: | ---: | ---: | ---: |
| under 500 | 798 | 360 | 73.9 | 69.3 to 76.9 |
| 1K to 8K | 132 | 686 | 57.6 | 56.5 to 58.3 |
| 8K to 16K | 49 | 2,062 | 43.7 | 41.8 to 46.7 |
| 16K to 25K | 42 | 1,797 | 33.6 | 25.5 to 36.3 |
| over 25K | 34 | 761 | 16.4 | 15.6 to 17.8 |

Two more readings from the same log:

- A short prompt (about 330 tokens) reads at roughly 330 tok/s, so about one second passes before the first
  token. A 13K to 16K prompt reads at 2,200 to 2,260 tok/s.
- 49 requests generated at least 100 tokens with MTP drafts active: median 94.8 tok/s, range 49.8 to 104.5,
  with 237 of 328 drafts accepted in the clearest of them. 657 requests of the same shape ran without
  drafts: median 73.6 tok/s, range 28.6 to 82.7. The two groups are different points in the session, so this
  is an observation, not an A/B measurement. Across all requests the median with drafts (43.5) is *lower*
  than without (72.2), because drafts mostly ran on long-context requests; that comparison says nothing
  about MTP.

The decode fall with context is the expert cache losing slots: at 25K the engine reported
`2123564 of 3103760 batch entries on the GPU (68.4%), 980196 on the CPU pool in 66337 ms`.

## 3. Why the local llama.cpp cannot be compared yet

`/home/onehao/workspace/llama.cpp` is at `f989a6e` (2026-04-10):

| Check | Result |
| --- | --- |
| Build | `GGML_CUDA:BOOL=OFF`; there is no `libggml-cuda.so`. CPU only. |
| CUDA toolkit on the machine | none. The distribution offers `nvidia-cuda-toolkit 12.0`, which does **not** support sm_120. |
| `qwen35moe` architecture | supported. The model loads. |
| MTP speculative decoding | **not present**. `common_speculative_type` has DRAFT, EAGLE3 and the n-gram types only, and `--spec-type` lists no `draft-mtp`. Upstream master has `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` and `--load-mtp`. |
| MoE placement flags | `--cpu-moe`, `--n-cpu-moe`, `-ot`, `--fit` are all present. |

A comparison against this build would be "Strata with CUDA and MTP against llama.cpp on the CPU without MTP".
It would measure the build, not the engines.

## 4. Published numbers worth knowing

### Strata's own same-file comparisons

From [QWEN36.md](../QWEN36.md), llama.cpp on the same GGUF file:

| Hardware | Metric | Strata | llama.cpp | Ratio |
| --- | --- | ---: | ---: | ---: |
| RTX 3060 12 GB, hybrid (37 of 256 experts in VRAM) | decode | 58.5 to 59.7 | 48 to 52 | 1.15 to 1.24 |
| RTX 3060 12 GB | prompt pp2048 | 1,958 to 2,022 | 1,328 | 1.5 |
| RTX 3060 12 GB, 1 / 2 / 4 requests | total tok/s | 64.0 / 85.1 / 99.3 | 41.1 / 52.5 / 63.0 | 1.56 to 1.62 |
| R9700 32 GB, all experts in VRAM | decode | 117.8 | 81.9 to 82.1 | 1.44 |
| R9700 32 GB | prompt pp4096 | 6,220 | 3,794 | 1.64 |

### Someone else's 16 GB cards

Reported by others; not reproduced here.

| Source | Configuration | Result |
| --- | --- | --- |
| Hardware Corner, quoted by [canitrun](https://canitrun.dev/guides/best-llms-for-16gb-vram) | RTX 5060 Ti 16 GB, 16K context | 82.4 tok/s |
| [Rost Glukhov](https://www.glukhov.org/llm-performance/benchmarks/best-llm-on-16gb-vram-gpu) | RTX 4080 16 GB, UD-IQ4_XS 17.7 GB | 95 / 92 / 86 tok/s at 19K / 32K / 64K |
| same | UD-IQ3_XXS 13.2 GB, 96% on the GPU | 147 / 149 / 146 tok/s at 19K / 32K / 64K |
| [hanxiao](https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4) | L4 24 GB, full residency | 80.9 tok/s without MTP, 91 to 99 with |
| [Amine Raji](https://aminrj.com/posts/llamacpp-qwen36-35b) | RTX 3090 24 GB, 65K context | 101.7 / 98.9 / 80.9 tok/s |
| [njannasch](https://njannasch.dev/blog/qwen3-6-turboquant-local-inference) | RTX 5060 Ti 16 GB, q4_0 KV, 262K context | 46 tok/s |
| [Unsloth](https://unsloth.ai/docs/models/qwen3.6) | MoE gain from MTP | 1.15 to 1.25x |

The IQ3_XXS row matters for a different reason: at 13.2 GB it nearly fits a 16 GB card whole. Strata offers
Qwen3.6 as UD-Q4_K_XL, UD-Q6_K_XL and Q8_0 only, so it has no pack in that class. A comparison against
IQ3_XXS is a different quant, not a different engine.

## 5. Expected range on this machine (estimate, not measured)

Same file, 32K context, MTP on both sides:

| Metric | Strata (measured, section 2) | llama.cpp (estimate) |
| --- | ---: | ---: |
| Short-prompt decode | 74 without drafts, up to 105 with | 60 to 80 |
| Prompt read at 13K to 16K | 2,200 to 2,260 tok/s | 1,300 to 1,600 tok/s |
| Decode at 20K and beyond | 25 to 35 tok/s | 20 to 30 tok/s |

The estimate is the RTX 3060 and R9700 ratios applied to this card's Strata numbers. It assumes llama.cpp
keeps a similar share of experts on the GPU, which is what `--n-cpu-moe` is for. It is worth nothing until
someone measures it.

## 6. What differs between the two, in practice

| | Strata | llama.cpp |
| --- | --- | --- |
| Expert placement | automatic (70 of 256 here), with a hot cache and slot replacement | `--n-cpu-moe` or `--fit`; a sweep is needed |
| Host memory | all 18.75 GiB pinned, read by its own pool | mmap and page cache by default |
| MTP | the block is in the file, `--mtp 2` on by default | needs a current build, `--load-mtp --spec-type draft-mtp --spec-draft-n-max 2` |
| Concurrency | `--parallel` up to 8, experts shared inside a window | `-np N` |
| Prompt chunk | 2048 here, chosen by the engine | `-ub`; 512, 1024 and 2048 are worth trying |
| Platforms | NVIDIA and AMD only for this family | everywhere |

## 7. How to measure it properly

1. Update llama.cpp past the commit that adds `draft-mtp`, and build it with CUDA 13 for sm_120.
   The distribution's CUDA 12.0 cannot target this card. Either restore the local checkout and rebuild, or
   use NVIDIA's CUDA 13 repository, the official CUDA binaries, or an `nvidia/cuda:13` container.
2. Fix the shape: the same GGUF file, 32K context, greedy, 256 output tokens, three runs per point, medians,
   a fresh random nonce in every prompt, and the two engines run one after the other because 22.9 GB of
   weights plus both contexts do not share 16 GiB.
3. Sweep only one thing at a time. For llama.cpp, `--n-cpu-moe` first (8 to 14, or `--fit`), then `-ub`.
4. Report MTP on and off as separate rows. Otherwise a 1.2 to 1.3x draft gain reads as an engine difference.
5. Use `tools/ab_engine.py` or `strata bench run --url http://127.0.0.1:PORT` for the Strata arm; both read
   the server's own `timings`, which is what llama-server also reports.

## 8. What this note establishes, and what it does not

Established:

- Strata runs this model on this card at 74 tok/s on short prompts without drafts and 95 to 105 with them,
  2,200 tok/s on a 13K to 16K prompt, and 25 to 35 tok/s once the conversation is past 16K.
- Strata's own same-file comparisons on two other cards put it 1.15 to 1.6x ahead of llama.cpp on decode
  and about 1.5x on prompt read.
- The llama.cpp build on this machine cannot be compared: CPU only, no MTP.

Not established:

- Any llama.cpp number on this machine. Every figure in section 5 is an extrapolation.
- Anything about answer quality. The engines produce the same text on this file
  ([QWEN36.md](../QWEN36.md) records the logit comparison); quality differences come from the quant.
