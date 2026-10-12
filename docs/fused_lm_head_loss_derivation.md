# Fused LM Head and Cross-Entropy

At the 8B training shape, one GPU processes 32,768 tokens against a 151,936-word vocabulary. The complete BF16 logits tensor for those tokens occupies **9.27 GiB**, and a standard softmax cross-entropy also keeps softmax outputs and gradients of similar size. This write-up derives and documents the fused operation in [fused_lm_head_loss.py](../cs336_systems/fused_lm_head_loss.py). It computes the LM-head projection, the summed cross-entropy, and both gradients without ever materializing the logits.

---

## 1. Objective

The operation combines:

1. the LM-head projection;
2. cross-entropy loss;
3. the backward computation of both.

It produces the same summed loss and gradients as:

$$
Z = HW^\top,
\qquad
L = \operatorname{cross\_entropy}(Z,y,\text{reduction=``sum"})
$$

without materializing the complete logits tensor $Z$.

---

## 2. Interface and correctness contract

| Symbol | Meaning | 8B configuration, per GPU |
|---|---|---:|
| $B$ | Local batch size | 1 |
| $S$ | Context length | 32,768 |
| $N_t$ | Number of tokens, $BS$ | 32,768 |
| $D$ | Model dimension, `d_model` | 4,096 |
| $V$ | Vocabulary size | 151,936 |
| $H_{BS}$ | Hidden states before the LM head, shape $[B,S,D]$ | — |
| $H$ | Flattened hidden states, shape $[N_t,D]$ | — |
| $W$ | LM-head weight, shape $[V,D]$ | — |
| $y_{BS}$ | Target IDs, shape $[B,S]$ | — |
| $y$ | Flattened target IDs, shape $[N_t]$ | — |
| $Z=HW^\top$ | Conceptual logits, shape $[N_t,V]$ | Never materialized |
| $L$ | Scalar summed cross-entropy | — |

### Operation boundary

The custom autograd operation is model-facing:

```python
FusedLMHeadCrossEntropyFunction.apply(hidden, weight, targets)
```

- `hidden`: CUDA floating-point tensor shaped `[B, S, D]`;
- `weight`: CUDA floating-point tensor shaped `[V, D]`;
- `targets`: CUDA `torch.int64` tensor shaped `[B, S]`;
- all three tensors are on the same device;
- `hidden` and `weight` have the same final dimension $D$;
- $D$ is a power of two;
- the operation reshapes `hidden` to `[N_t, D]` and `targets` to `[N_t]` internally;
- the output is one FP32 scalar with `reduction="sum"` semantics.

### Saved tensors

The forward pass saves exactly:

- original `hidden`, shape `[B,S,D]`;
- `weight`, shape `[V,D]`;
- original `targets`, shape `[B,S]`;
- flattened FP32 `lse`, shape `[N_t]`.

It does not save logits, probabilities, target masks, or per-tile loss sums.

### Model integration

[`TransformerLM.forward`](../cs336_systems/model_flash.py) keeps two paths through its LM head:

- without targets, it returns logits of shape `[B, S, V]`, which remains the reference path for correctness tests;
- with `targets=...`, it runs the Transformer through the final RMSNorm and passes the `[B,S,D]` hidden states, the `[B,S]` targets, and the real `lm_head.weight` parameter into the fused operation, which returns the summed loss.

The weight enters the operation as an ordinary autograd input, and `backward` returns its gradient rather than assigning to `weight.grad`. This keeps the gradient on PyTorch's normal accumulation path, so FSDP's post-accumulate-gradient hooks still reduce-scatter it.

---

## 3. Mathematical derivation

For token $i$ and vocabulary entry $j$:

$$
z_{ij}=H_i W_j^\top.
$$

The per-token cross-entropy is:

$$
L_i
=
\log\left(\sum_{j=1}^{V}\exp(z_{ij})\right)
-
z_{i,y_i}.
$$

The cross-entropy loss is:

$$
L=\sum_{i=1}^{N_t}L_i.
$$

Therefore, the forward pass needs only:

1. the log-sum-exp of each row;
2. the logit corresponding to the target token.

It does not need to retain every logit.

For numerical stability, define one scalar maximum and one scalar shifted exponential sum for every token:

$$
m_i=\max_j z_{ij}
$$

where $m_i\in\mathbb{R}$. Collect these values into the vector

$$
m=(m_1,\ldots,m_{N_t})\in\mathbb{R}^{N_t}.
$$

For the corresponding shifted exponential sum, define

$$
\ell_i=\sum_j\exp(z_{ij}-m_i)
$$

and

$$
\ell=(\ell_1,\ldots,\ell_{N_t})\in\mathbb{R}^{N_t}.
$$

Then:

$$
\operatorname{LSE}_i=m_i+\log\ell_i
$$

and

$$
L_i=\operatorname{LSE}_i-z_{i,y_i}.
$$

The forward pass saves $\operatorname{LSE}\in\mathbb{R}^{N_t}$, so backward can reconstruct probabilities without a second online reduction.

---

## 4. Online log-sum-exp over vocabulary tiles

Three tile sizes partition the work:

- $B_t$: number of tokens handled by a program;
- $B_v$: tile size along the vocabulary dimension;
- $B_d$: tile size along $D$ (`d_model`).

The $D$ loop advances the hidden and weight block pointers by $B_d$ each iteration and accumulates every partial dot product into the same FP32 logit tile.

Let

$$
K_t=\left\lceil\frac{N_t}{B_t}\right\rceil
\qquad\text{and}\qquad
K_v=\left\lceil\frac{V}{B_v}\right\rceil
$$

be the number of token tiles and vocabulary tiles, respectively. The current kernels use $B_t=256$, $B_v=32$, and $B_d=32$, so the 8B configuration has $K_t=128$ token tiles and $K_v=4{,}748$ vocabulary tiles.

### Algorithm 1: Tiled fused forward pass

**Require:** $H\in\mathbb{R}^{N_t\times D}$, $W\in\mathbb{R}^{V\times D}$, targets $y\in\mathbb{N}^{N_t}$, and tile sizes $B_t,B_v,B_d$.

Split $H$ into $K_t$ tiles $H_1,H_2,\ldots,H_{K_t}$. Token tile $H_i$ has shape $[B_t,D]$.

Split $W$ into $K_v$ tiles $W^{(1)},W^{(2)},\ldots,W^{(K_v)}$. Vocabulary tile $W^{(j)}$ has shape $[B_v,D]$.

For each token tile $H_{i}$ (one Triton program):

1. Load the target IDs for the token tile, $y_{i} \in \mathbb{N}^{B_t}$.
2. Initialize the per-token running vectors:

   $$
   m_i^{(0)}=-\infty\in\mathbb{R}^{B_t},
   \qquad
   \ell_i^{(0)}=0\in\mathbb{R}^{B_t}.
   $$

3. Initialize storage for one target logit per token, $Z_{\text{target},i}=0\in\mathbb{R}^{B_t}$.
4. For each vocabulary tile $W^{(j)}$:

   1. Compute the logit tile:

      $$
      Z_i^{(j)}=H_{i}(W^{(j)})^\top
      \in\mathbb{R}^{B_t\times B_v}.
      $$

      Entries beyond the vocabulary are set to $-\infty$, so they contribute nothing to the maximum or the exponential sum.

   2. Compute the new row maximum:

      $$
      m_i^{(j)}
      =
      \max\left(
      m_i^{(j-1)},
      \operatorname{rowmax}(Z_i^{(j)})
      \right) \in \mathbb{R}^{B_t}.
      $$

   3. Update the exponential sum:

      $$
      \ell_i^{(j)}
      =
      \exp(m_i^{(j-1)}-m_i^{(j)})\ell_i^{(j-1)}
      +
      \operatorname{rowsum}
      \left(
      \exp(Z_i^{(j)}-m_i^{(j)})
      \right) \in \mathbb{R}^{B_t}.
      $$

   4. Let vocabulary tile $j$ contain vocabulary IDs

      $$
      v^{(j)}=(s^{(j)},\ldots,e^{(j)}-1)\in\mathbb{N}^{B_v}.
      $$

      The Boolean mask for tokens whose targets $y_r$ satisfy $s^{(j)}\le y_r<e^{(j)}$ is

      $$
      M^{(j)}_{r,c}=\mathbf{1}[y_r=v^{(j)}_c]
      \in\{0,1\}^{B_t\times B_v},
      $$

      and the target logits accumulate as

      $$
      (Z_{\text{target},i})_r\mathrel{+}=\sum_c
      \operatorname{where}\left(M^{(j)}_{r,c},Z^{(j)}_{r,c},0\right).
      $$

5. Compute:

   $$
   \operatorname{LSE}_{i}
   =
   m_i^{(K_v)}+\log\ell_i^{(K_v)}
   \in\mathbb{R}^{B_t}.
   $$

6. Compute the per-token losses $L_i=\operatorname{LSE}_i-Z_{\text{target},i}$, set entries for tokens beyond $N_t$ to zero, and sum them into one partial loss for the tile.
7. Write $\operatorname{LSE}_i$ to global memory as the $i$-th tile of $\operatorname{LSE}$, and the partial loss as entry $i$ of a $K_t$-element buffer.

After all programs finish, the wrapper sums the $K_t$ partial losses into the scalar loss. With BF16 inputs, each logit tile is rounded to BF16 before the softmax statistics are computed. This matches the materialized reference, which also produces BF16 logits; accumulation and all running statistics stay in FP32.

---

## 5. Backward pass with recomputation

The derivative of the loss with respect to the logits is:

$$
G_{ij}
=
\frac{\partial L}{\partial z_{ij}}
=
\exp(z_{ij}-\operatorname{LSE}_i)
-
\mathbf{1}[j=y_i].
$$

With an upstream gradient $g_L=\partial \mathcal{L}/\partial L$, the scaled logit gradient is $g_L G$. The parameter gradients follow from $Z=HW^\top$:

$$
\nabla_H = g_L\, G W,
\qquad
\nabla_W = g_L\, G^\top H.
$$

### Correctness invariant

For every token row:

$$
\sum_j G_{ij}\approx 0,
$$

because the softmax probabilities sum to one and exactly one target indicator is subtracted.

### Decomposition with a reusable workspace

Materializing $G$ would recreate the $[N_t,V]$ tensor the forward pass avoided. Backward instead processes one token tile at a time with a reusable workspace

$$
G_i\in\mathbb{R}^{B_t\times V}
$$

of $B_t V$ elements: 256 × 151,936 BF16 values, about 74 MiB, rather than 9.27 GiB. For token tile $i$:

1. One Triton launch fills $G_i$ with a one-dimensional grid of $K_v$ programs. Program $j$ owns $G_i^{(j)}\in\mathbb{R}^{B_t\times B_v}$ and loops over $D$ to recompute

   $$
   Z_i^{(j)}=H_i(W^{(j)})^\top.
   $$

2. cuBLAS then consumes the workspace:

   $$
   (\nabla_H)_i=G_iW,
   \qquad
   \nabla_W\mathrel{+}=G_i^\top H_i.
   $$

3. The same workspace is reused for token tile $i+1$.

This tiles $N_t$, $V$, and the reduction over $D$ explicitly. Each logit tile is recomputed exactly once, and both parameter gradients use optimized matrix multiplications.

### Algorithm 2: one Triton program

For token tile $i$ and vocabulary tile $j$:

1. Load the $B_t$ target IDs and $\operatorname{LSE}$ values of token tile $i$.

2. Initialize

   $$
   Z_i^{(j)}=0\in\mathbb{R}^{B_t\times B_v}
   $$

   in FP32.

3. Loop over $D$ in steps of $B_d$: load a hidden tile of shape `[B_t,B_d]` and a weight tile of shape `[B_v,B_d]`, and accumulate `tl.dot(H_tile, W_tile.T, Z_tile)`.

4. Reconstruct the probabilities from the saved log-sum-exp values:

   $$
   P_i^{(j)}=\exp\left(
   Z_i^{(j)}-\operatorname{LSE}_i[:,\mathrm{None}]
   \right).
   $$

5. Build the target mask for vocabulary IDs $v^{(j)}=(s^{(j)},\ldots,e^{(j)}-1)$:

   $$
   M^{(j)}_{r,c}=\mathbf{1}[y_r=v^{(j)}_c]
   \in\{0,1\}^{B_t\times B_v}.
   $$

6. Compute the scaled logit gradient directly:

   $$
   G_i^{(j)}
   =
   g_L\left(P_i^{(j)}-M_i^{(j)}\right).
   $$

7. Write $G_i^{(j)}$ to the workspace as its $j$-th tile.

`tl.program_id(0)` selects vocabulary tile $j$, and the launcher passes the token-tile index $i$, which is constant across that launch. The workspace has the dtype of the hidden states. $\nabla_H$ is computed in that dtype, while $\nabla_W$ accumulates across token tiles in an FP32 buffer that is cast to the weight dtype at the end. With BF16 inputs, the $\nabla_W$ multiplication runs in TF32 on tensor cores: its operands are BF16 values converted to FP32, and TF32's 10 mantissa bits represent BF16's 7 exactly, so no input value changes. FP32 inputs keep full FP32.

---

## 6. What I implemented, and why

I built the operation in stages, each checked against the previous one before moving on:

- **PyTorch tiled reference.** `fused_lm_head_loss_forward_torch` implements Algorithm 1 with explicit token and vocabulary loops. It is slow, but every intermediate (running maxima, exponential sums, target logits) is observable, which made it the oracle for debugging the Triton kernel.
- **Triton forward kernel.** `fused_lm_head_loss_forward_kernel` runs Algorithm 1 with one program per token tile. It writes per-token log-sum-exp values and one partial loss per tile, so no $[N_t]$ loss vector is needed.
- **Autograd integration.** `FusedLMHeadCrossEntropyFunction` takes the model-facing `[B,S,D]` and `[B,S]` shapes, owns the flattening, saves only the four tensors listed in Section 2, and returns `(grad_hidden, grad_weight, None)`. Tile sizes are internal constants, not autograd inputs.
- **Triton backward kernel plus cuBLAS.** `fused_lm_head_loss_backward_kernel` implements Algorithm 2 and fills the reusable $[B_t,V]$ workspace; cuBLAS forms both gradients. I chose this over a fully fused backward pass for two reasons. Programs that each own an output tile of $\nabla_H$ or $\nabla_W$ would have to recompute the same logits for every hidden-dimension tile. Accumulating with atomics instead adds contention and makes the accumulation order nondeterministic. The workspace costs only $O(B_t V)$ memory, so the simpler design was the right first step.
- **Model and FSDP integration.** The fused path connects to `TransformerLM` through the LM head's `targets` argument and runs inside the custom FSDP wrapper in the 8B training step.
- **Tile sizes.** A later tuning pass changed the tile sizes to $B_t=256$, $B_v=32$, $B_d=32$.

---

## 7. Validation

[tests/test_fused_loss.py](../tests/test_fused_loss.py) compares the fused operation against a materialized PyTorch LM head and `F.cross_entropy`.

[tests/test_model_flash.py](../tests/test_model_flash.py) also checks the fused loss inside the Transformer against the materialized model.

Measured forward and backward time at the 8B shape, and the bottlenecks identified by profiling, are reported in the [README](../README.md#8b-training-on-four-a100-gpus) and [results](../results/README.md#8b-training-step-on-four-a100-gpus).
