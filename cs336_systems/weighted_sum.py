import torch
import triton
import triton.language as tl
from einops import rearrange
from pathlib import Path

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def weighted_sum_fwd_kernel(
    x_ptr, weight_ptr,
    output_ptr,
    x_stride_row, x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS, D,
    ROW_TILE_SIZE: tl.constexpr, 
    D_TILE_SIZE: tl.constexpr
):
    row_tile_id = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(ROW_TILE_SIZE * row_tile_id, 0),
        block_shape=(ROW_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )

    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,)
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(ROW_TILE_SIZE * row_tile_id,),
        block_shape=(ROW_TILE_SIZE,),
        order=(0,)
    )

    output = tl.zeros((ROW_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):

        row = tl.load(
            x_block_ptr, 
            boundary_check=(0, 1), 
            padding_option="zero"
        )
        weight = tl.load(
            weight_block_ptr, 
            boundary_check=(0,), 
            padding_option="zero"
        )

        # row:             (ROW_TILE_SIZE, D_TILE_SIZE)
        # weight[None, :]: (1,             D_TILE_SIZE)
        # partial sum:     (ROW_TILE_SIZE,)
        output += tl.sum(row * weight[None, :], axis=1)

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    tl.store(output_block_ptr, output, boundary_check=(0,))


def weighted_sum_fwd(x: torch.Tensor, weight: torch.Tensor):

    assert x.device == DEVICE and weight.device == DEVICE
    assert x.shape[-1] == weight.shape[0]
    output_dims = x.shape[:-1]
    x = rearrange(x, "... d -> (...) d").contiguous()
    NUM_ROWS, D = x.shape
    output = torch.empty((NUM_ROWS,), dtype=x.dtype, device=DEVICE)
    D_TILE_SIZE = max(1, triton.next_power_of_2(D) // 16)
    grid = lambda meta: (triton.cdiv(NUM_ROWS, meta['ROW_TILE_SIZE']), )
    weighted_sum_fwd_kernel[grid](
        x, weight,
        output,
        x.stride(0), x.stride(1),
        weight.stride(0),
        output.stride(0),
        NUM_ROWS,
        D, 
        ROW_TILE_SIZE=16,
        D_TILE_SIZE=D_TILE_SIZE
    )
    return output.view(output_dims)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['NUM_ROWS'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'torch-matmul'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Torch matmul"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="weighted-sum-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'D': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(NUM_ROWS, D, provider):
    x = torch.randn(NUM_ROWS, D, device=DEVICE)
    weight = torch.rand(D, device=DEVICE)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: (x * weight).sum(axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: weighted_sum_fwd(x, weight))
    if provider == "torch-matmul":
        ms = triton.testing.do_bench(lambda: torch.matmul(x, weight))
    gbps = lambda ms: (x.numel() + weight.numel() + NUM_ROWS) * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


def main():

    # Test correctness
    torch.manual_seed(0)
    d = 192
    x = torch.randn(23, 47, d, device=DEVICE)
    weight = torch.rand(d, device=DEVICE)
    output_torch = (x * weight).sum(axis=-1)
    output_triton = weighted_sum_fwd(x, weight)

    torch.testing.assert_close(
        output_triton,
        output_torch,
        rtol=1e-4,
        atol=1e-4,
    )

    print("Passed")

    result_dir = Path("benchmark_weighted_sum")
    result_dir.mkdir(parents=True, exist_ok=True)
    benchmark.run(
        show_plots=False,
        print_data=False,
        save_path=result_dir,
    )

if __name__ == "__main__":
    main()

