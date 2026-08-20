# Section 2.1.6(f): Transformer-block memory

The Nsight Systems SQLite export identifies 32 forward ranges containing
`scaled dot product attention`, one for each Transformer block. The block shown
in the screenshot is block 11, spanning 361.242--386.280 ms. Its device-memory
usage increases from 39.910973 GiB to 40.088925 GiB, an exact net increase of
182.223 MiB. This is the memory retained after the block's temporary forward
allocations have been released, so it estimates the tensors saved for backward
by one block. All 32 blocks have the same 182.223 MiB increase.

The five largest retained allocations are all 20 MiB tensors in the SwiGLU
feed-forward path:

| Operation | Role | Retained memory (MiB) | Percentage of block residuals |
|---|---|---:|---:|
| `aten::bmm` | First feed-forward projection | 20.000 | 10.98% |
| `aten::sigmoid` | Sigmoid used to form SiLU | 20.000 | 10.98% |
| `aten::mul` | SiLU multiplication | 20.000 | 10.98% |
| `aten::bmm` | Gating feed-forward projection | 20.000 | 10.98% |
| `aten::mul` | SwiGLU gating multiplication | 20.000 | 10.98% |
| **Top five total** |  | **100.000** | **54.88%** |

Backward executes the blocks in reverse order. Forward block 11 therefore
corresponds to backward occurrence 22, approximately 2.989--3.141 s in the
timeline. Between the low-memory boundaries around that backward block,
allocated memory increases by 217.797 MiB. During the block's backward pass,
182.223 MiB of saved tensors are freed while its parameter gradients are
created. Therefore the inferred gradient size is

$$
G_{\text{block}}
=182.223\ \text{MiB}+217.797\ \text{MiB}
=400.020\ \text{MiB}.
$$

This agrees with the theoretical parameter-gradient size. With
$d=2560$ and $d_{\mathrm{ff}}=10240$, one block has

$$
4d^2+3d d_{\mathrm{ff}}+2d
=104{,}862{,}720
$$

parameters: four attention matrices, three feed-forward matrices, and two
RMSNorm weight vectors. FP32 gradients therefore occupy

$$
104{,}862{,}720\times4/2^{20}
=400.0195\ \text{MiB}.
$$

The trace directly exposes 400 MiB from the seven large matrix gradients; the
remaining approximately 0.0195 MiB comes from the two RMSNorm gradient vectors.
The measured and theoretical results consequently agree to the displayed
precision.
