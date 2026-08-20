import math
import torch
import torch.nn.functional as F
import triton
from cs336_systems.flash_attention import FlashAttentionTriton
from pathlib import Path

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def attention_torch(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal_mask=None
):
    batch_size, n_queries, d = Q.shape
    n_keys = K.shape[-2]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(Q, torch.transpose(K, -2, -1)) * scale
    if causal_mask is not None:
        scores = scores.masked_fill(~causal_mask, -float("inf"))  
    probabilities = F.softmax(scores, dim=-1)
    return torch.matmul(probabilities, V)


flash_triton = FlashAttentionTriton.apply

BATCH = 1
is_causal = True
configs = []
for D in [32, 64, 128]:
    for precision in ["bf16", "fp32"]:
        for mode in ["fwd", "bwd", "fwd-bwd"]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["SEQ_LEN"],
                    x_vals=[2**i for i in range(10, 16)],
                    line_arg="provider",
                    line_vals=['ord_torch', 'flash_triton'],
                    line_names=["Ordinary PyTorch", "FlashAttention Triton"],
                    styles=[("blue", "-"), ("green", "-")],
                    ylabel="Runtime(ms)",
                    plot_name=
                    f"causal-attention-batch{BATCH}-d{D}-{mode}-{precision}",
                    args={
                        "batch_size": BATCH,
                        "D": D,
                        "precision": precision,
                        "mode": mode,
                    },
                ))
                     
@triton.testing.perf_report(configs)
def benchmark_flash(SEQ_LEN, D, precision, provider, mode, batch_size):
    assert mode in ["fwd", "bwd", "fwd-bwd"]
    assert precision in ["bf16", "fp32"]
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    Q = torch.randn(
        batch_size, SEQ_LEN, D,
        device=DEVICE,
        dtype=dtype,
        requires_grad=True,
    )
    K = torch.randn_like(Q, requires_grad=True)
    V = torch.randn_like(Q, requires_grad=True)
    grad_output = torch.randn_like(Q)
    is_causal = True
    causal_mask = None
    if provider == 'ord_torch':
        causal_mask = torch.ones(
            SEQ_LEN,
            SEQ_LEN,
            dtype=torch.bool,
            device=DEVICE,
        ).tril()
        fwd_fn = lambda: attention_torch(Q, K, V, causal_mask)
    if provider == "flash_triton":
        fwd_fn = lambda: flash_triton(Q, K, V, is_causal)
    if mode == "fwd":
        fn = fwd_fn
    elif mode == "bwd":
        output = fwd_fn()
        fn = lambda: torch.autograd.grad(
            output,
            (Q, K, V),
            grad_outputs=grad_output,
            retain_graph=True,
        )
    elif mode == "fwd-bwd":
        def fn():
            output = fwd_fn()
            return torch.autograd.grad(
                output,
                (Q, K, V),
                grad_outputs=grad_output
            )

    fn()  # explicit warm up
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(fn)
    return ms


def main():
    # Test correctness
    torch.manual_seed(0)
    batch_size = 8
    d = 128
    seq_len = 1024
    precision = torch.float32
    is_causal = True
    Q_torch = torch.randn(
        batch_size, seq_len, d,
        device=DEVICE,
        dtype=precision,
        requires_grad=True,
    )
    K_torch = torch.randn_like(Q_torch, requires_grad=True)
    V_torch = torch.randn_like(Q_torch, requires_grad=True)
    causal_mask = torch.ones(
        seq_len,
        seq_len,
        dtype=torch.bool,
        device=DEVICE,
    ).tril()
    output_torch = attention_torch(
        Q_torch,
        K_torch,
        V_torch,
        causal_mask,
    )
    Q_triton = Q_torch.detach().clone().requires_grad_(True)
    K_triton = K_torch.detach().clone().requires_grad_(True)
    V_triton = V_torch.detach().clone().requires_grad_(True)
    output_triton = FlashAttentionTriton.apply(Q_triton, K_triton, V_triton, is_causal)
    torch.testing.assert_close(
        output_triton,
        output_torch,
        rtol=1e-2,
        atol=1e-2,
    )
    print("Passed forward pass!")
    grad_output = torch.randn_like(output_torch)
    output_torch.backward(grad_output)
    output_triton.backward(grad_output)
    torch.testing.assert_close(
        Q_torch.grad,
        Q_triton.grad,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        K_torch.grad,
        K_triton.grad,
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        V_torch.grad,
        V_triton.grad,
        rtol=1e-2,
        atol=1e-2,
    )
    print("Passed backward pass!")

    result_dir = Path("benchmark_flash")
    result_dir.mkdir(parents=True, exist_ok=True)
    benchmark_flash.run(
        show_plots=False,
        print_data=False,
        save_path=result_dir,
    )

if __name__ == "__main__":
    main()
