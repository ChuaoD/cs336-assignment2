# Section 6: Optimizer State Sharding Memory Accounting

All sizes below use decimal GB, where $1\ \mathrm{GB}=10^9$ bytes, to match the profiler output.

## Parameter Count

For the XL model,

$$
P=L(4D^2+3DD_{\mathrm{ff}}+2D)+2VD+D,
$$

where:

- $4D^2$ accounts for the query, key, value, and attention output matrices;
- $3DD_{\mathrm{ff}}$ accounts for the three SwiGLU matrices;
- $2D$ accounts for two RMSNorm weights per Transformer layer;
- $2VD$ accounts for the token embedding and LM head;
- the final $D$ accounts for the final RMSNorm.

Using $L=32$, $D=2560$, $D_{\mathrm{ff}}=10240$, and $V=10000$ gives

$$
P=3{,}406{,}809{,}600.
$$

All parameters and gradients are FP32, so one parameter-sized tensor occupies

$$
4P=13.6272384\ \mathrm{GB}.
$$

Therefore:

- parameters occupy $13.6272384$ GB;
- gradients occupy $13.6272384$ GB;
- each AdamW moment occupies $13.6272384$ GB;
- both AdamW moments occupy $27.2544768$ GB.

## Model Buffers

Each layer stores FP32 RoPE sine and cosine buffers. With context length 512 and head dimension $2560/32=80$, each sine or cosine buffer has shape $(512,40)$. Across 32 layers, the buffers occupy

$$
32\times 2\times 512\times 40\times 4
=5{,}242{,}880\ \mathrm{bytes}
=0.00524288\ \mathrm{GB}.
$$

Thus, the theoretical memory directly after initialization is

$$
13.6272384+0.00524288
=13.63248128\ \mathrm{GB},
$$

which exactly matches the measured $13{,}632.48128$ MB on both ranks.

## Memory Before the First Optimizer Step

Before the first optimizer step, parameters and gradients are allocated, but AdamW has not yet lazily initialized its moment tensors. The persistent model-state memory is therefore

$$
4P_{\mathrm{parameters}}+4P_{\mathrm{gradients}}+\text{buffers}
=8P+\text{buffers}
=27.25971968\ \mathrm{GB}.
$$

The measured current allocation was $27.317736448$ GB. The additional approximately $0.058$ GB comes from live logits, inputs, the loss, DDP bookkeeping, and other small allocations.

The measured forward-backward peak was $32.844754432$ GB. This peak is higher because gradients and saved forward activations coexist during parts of the backward pass. Its precise value depends on tensor lifetimes and cannot be derived from parameter count alone.

## Ordinary AdamW After the First Step

Ordinary AdamW stores two full FP32 moments on every rank. After the first optimizer step, gradients remain allocated, so the persistent model-state memory is

$$
4P_{\mathrm{parameters}}
+4P_{\mathrm{gradients}}
+8P_{\mathrm{Adam\ states}}
+\text{buffers}
=16P+\text{buffers}
=54.51419648\ \mathrm{GB}.
$$

The measured current allocation was $54.572214784$ GB, again approximately $0.058$ GB above the persistent model-state calculation. The optimizer-step peak was $54.879414784$ GB, so temporary AdamW calculations added another $0.3072$ GB at the peak.

## Sharded AdamW After the First Step

The round-robin whole-tensor ownership policy does not split the parameter elements perfectly evenly:

- rank 0 owns $1{,}729{,}003{,}520$ parameter elements;
- rank 1 owns $1{,}677{,}806{,}080$ parameter elements.

Their optimizer-state sizes are therefore

$$
8P_0=13.83202816\ \mathrm{GB}
$$

and

$$
8P_1=13.42244864\ \mathrm{GB}.
$$

Parameters and gradients remain fully replicated. The persistent post-step memory on rank 0 is

$$
2(13.6272384)+13.83202816+0.00524288
=41.09174784\ \mathrm{GB},
$$

and the persistent post-step memory on rank 1 is

$$
2(13.6272384)+13.42244864+0.00524288
=40.68216832\ \mathrm{GB}.
$$

Adding the approximately $0.058$ GB of other live allocations gives $41.149766144$ GB and $40.740186624$ GB, exactly matching the measured current allocations.

## Summary

| Stage | Ordinary AdamW | Sharded rank 0 | Sharded rank 1 |
|---|---:|---:|---:|
| After initialization | 13.6325 GB | 13.6325 GB | 13.6325 GB |
| Before first optimizer step | 27.2597 GB | 27.2597 GB | 27.2597 GB |
| After first optimizer step, persistent model state | 54.5142 GB | 41.0917 GB | 40.6822 GB |
| Measured current memory after step | 54.5722 GB | 41.1498 GB | 40.7402 GB |
| Measured optimizer-step peak | 54.8794 GB | 41.4570 GB | 40.9499 GB |

Using the maximum-memory sharded rank, the theoretical persistent saving is

$$
54.51419648-41.09174784
=13.42244864\ \mathrm{GB}.
$$

A perfectly balanced two-rank partition would save $13.6272384$ GB, exactly 25% of the ordinary persistent total. The whole-tensor round-robin partition saves approximately 24.6% theoretically and 24.5% at the measured optimizer-step peak.

The measured distributed iteration time increased from approximately $1.712$ seconds with ordinary AdamW to $1.748$ seconds with optimizer-state sharding, an overhead of about 2.1%. This indicates that parameter synchronization slightly outweighed the reduction in optimizer computation.
