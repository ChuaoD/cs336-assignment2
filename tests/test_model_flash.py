import torch

from cs336_basics.model import (
    MultiheadAttention as ReferenceMultiheadAttention,
    TransformerLM as ReferenceTransformerLM,
)

from cs336_systems.model_flash import (
    FlashMultiHeadAttention,
    TransformerLM as FlashTransformerLM,
)

import pytest
from einops import rearrange

from .adapters_flash import (
    run_multihead_self_attention,
    run_multihead_self_attention_with_rope,
    run_transformer_block,
    run_transformer_lm,
)


pytestmark = pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="FlashAttention Triton tests require CUDA",
)

ATTENTION_ATOL = 1e-2
ATTENTION_RTOL = 1e-2

MODEL_ATOL = 2e-2
MODEL_RTOL = 1e-2


def test_multihead_self_attention(
    numpy_snapshot,
    in_embeddings,
    d_model,
    n_heads,
    ts_state_dict,
):
    state_dict, _ = ts_state_dict

    q_proj_weight = state_dict["layers.0.attn.q_proj.weight"]
    k_proj_weight = state_dict["layers.0.attn.k_proj.weight"]
    v_proj_weight = state_dict["layers.0.attn.v_proj.weight"]
    o_proj_weight = state_dict["layers.0.attn.output_proj.weight"]

    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=n_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )

    numpy_snapshot.assert_match(
        actual_output,
        atol=ATTENTION_ATOL,
        rtol=ATTENTION_RTOL,
    )


def test_multihead_self_attention_with_rope(
    numpy_snapshot,
    in_embeddings,
    d_model,
    n_heads,
    ts_state_dict,
    n_keys,
    theta,
    pos_ids,
):
    state_dict, _ = ts_state_dict

    q_proj_weight = state_dict["layers.0.attn.q_proj.weight"]
    k_proj_weight = state_dict["layers.0.attn.k_proj.weight"]
    v_proj_weight = state_dict["layers.0.attn.v_proj.weight"]
    o_proj_weight = state_dict["layers.0.attn.output_proj.weight"]

    # Preserve the position shape used by the original Assignment 1 test.
    pos_ids = rearrange(pos_ids, "seq -> 1 seq")

    actual_output = run_multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=n_heads,
        max_seq_len=n_keys,
        theta=theta,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
        token_positions=pos_ids,
    )

    numpy_snapshot.assert_match(
        actual_output,
        atol=ATTENTION_ATOL,
        rtol=ATTENTION_RTOL,
    )


def test_transformer_block(
    numpy_snapshot,
    ts_state_dict,
    in_embeddings,
    d_model,
    n_heads,
    d_ff,
    n_keys,
    theta,
):
    state_dict, _ = ts_state_dict

    block_weights = {
        key.removeprefix("layers.0."): value
        for key, value in state_dict.items()
        if key.startswith("layers.0.")
    }

    actual_output = run_transformer_block(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=n_keys,
        theta=theta,
        weights=block_weights,
        in_features=in_embeddings,
    )

    numpy_snapshot.assert_match(
        actual_output,
        atol=MODEL_ATOL,
        rtol=MODEL_RTOL,
    )


def test_transformer_lm(
    numpy_snapshot,
    vocab_size,
    n_keys,
    d_model,
    n_layers,
    n_heads,
    d_ff,
    theta,
    ts_state_dict,
    in_indices,
):
    state_dict, _ = ts_state_dict

    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=state_dict,
        in_indices=in_indices,
    )

    numpy_snapshot.assert_match(
        actual_output,
        atol=MODEL_ATOL,
        rtol=MODEL_RTOL,
    )


def test_transformer_lm_truncated_input(
    numpy_snapshot,
    vocab_size,
    n_keys,
    d_model,
    n_layers,
    n_heads,
    d_ff,
    theta,
    ts_state_dict,
    in_indices,
):
    state_dict, _ = ts_state_dict
    truncated_indices = in_indices[..., : in_indices.shape[-1] // 2]

    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        rope_theta=theta,
        weights=state_dict,
        in_indices=truncated_indices,
    )

    numpy_snapshot.assert_match(
        actual_output,
        atol=MODEL_ATOL,
        rtol=MODEL_RTOL,
    )

BACKWARD_ATOL = 1e-3
BACKWARD_RTOL = 1e-3

FULL_MODEL_BACKWARD_ATOL = 5e-3
FULL_MODEL_BACKWARD_RTOL = 5e-3


def _assert_parameter_gradients_close(
    reference_model: torch.nn.Module,
    flash_model: torch.nn.Module,
    *,
    atol: float,
    rtol: float,
) -> None:
    reference_parameters = dict(reference_model.named_parameters())
    flash_parameters = dict(flash_model.named_parameters())

    assert reference_parameters.keys() == flash_parameters.keys(), (
        "Reference and Flash models have different parameter names"
    )

    for name in reference_parameters:
        reference_gradient = reference_parameters[name].grad
        flash_gradient = flash_parameters[name].grad

        assert reference_gradient is not None, (
            f"Reference parameter {name!r} has no gradient"
        )
        assert flash_gradient is not None, (
            f"Flash parameter {name!r} has no gradient"
        )

        try:
            torch.testing.assert_close(
                flash_gradient,
                reference_gradient,
                atol=atol,
                rtol=rtol,
            )
        except AssertionError as error:
            max_absolute_error = (
                flash_gradient - reference_gradient
            ).abs().max().item()

            raise AssertionError(
                f"Gradient mismatch for parameter {name!r}. "
                f"Maximum absolute error: {max_absolute_error}\n{error}"
            ) from error


def test_multihead_self_attention_backward(
    in_embeddings,
    d_model,
    n_heads,
    n_keys,
    theta,
    ts_state_dict,
):
    device = torch.device("cuda")
    state_dict, _ = ts_state_dict

    attention_weights = {
        "q_proj.weight": state_dict["layers.0.attn.q_proj.weight"],
        "k_proj.weight": state_dict["layers.0.attn.k_proj.weight"],
        "v_proj.weight": state_dict["layers.0.attn.v_proj.weight"],
        "output_proj.weight": state_dict[
            "layers.0.attn.output_proj.weight"
        ],
    }

    reference_attention = ReferenceMultiheadAttention(
        d_model=d_model,
        num_heads=n_heads,
        with_rope=True,
        theta=theta,
        max_seq_len=n_keys,
        device=device,
    )

    flash_attention = FlashMultiHeadAttention(
        d_model=d_model,
        num_heads=n_heads,
        with_rope=True,
        theta=theta,
        max_seq_len=n_keys,
        device=device,
    )

    reference_attention.load_state_dict(
        attention_weights,
        strict=True,
    )
    flash_attention.load_state_dict(
        attention_weights,
        strict=True,
    )

    reference_input = (
        in_embeddings
        .to(device)
        .detach()
        .clone()
        .requires_grad_(True)
    )

    flash_input = (
        in_embeddings
        .to(device)
        .detach()
        .clone()
        .requires_grad_(True)
    )

    token_positions = torch.arange(
        reference_input.shape[-2],
        device=device,
    )

    reference_output = reference_attention(
        reference_input,
        token_positions,
    )
    flash_output = flash_attention(
        flash_input,
        token_positions,
    )

    torch.testing.assert_close(
        flash_output,
        reference_output,
        atol=ATTENTION_ATOL,
        rtol=ATTENTION_RTOL,
    )

    # Both models receive exactly the same upstream gradient.
    torch.manual_seed(1234)
    grad_output = torch.randn_like(reference_output)

    reference_output.backward(grad_output)
    flash_output.backward(grad_output)

    assert reference_input.grad is not None
    assert flash_input.grad is not None

    torch.testing.assert_close(
        flash_input.grad,
        reference_input.grad,
        atol=BACKWARD_ATOL,
        rtol=BACKWARD_RTOL,
    )

    _assert_parameter_gradients_close(
        reference_attention,
        flash_attention,
        atol=BACKWARD_ATOL,
        rtol=BACKWARD_RTOL,
    )


def test_transformer_lm_backward(
    vocab_size,
    n_keys,
    d_model,
    n_layers,
    n_heads,
    d_ff,
    theta,
    ts_state_dict,
    in_indices,
):
    device = torch.device("cuda")
    state_dict, _ = ts_state_dict

    reference_model = ReferenceTransformerLM(
        vocab_size=vocab_size,
        context_length=n_keys,
        num_layers=n_layers,
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        theta=theta,
        device=device,
    )

    flash_model = FlashTransformerLM(
        vocab_size=vocab_size,
        context_length=n_keys,
        num_layers=n_layers,
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        theta=theta,
        device=device,
    )

    reference_model.load_state_dict(state_dict, strict=True)
    flash_model.load_state_dict(state_dict, strict=True)

    input_ids = in_indices.to(device)
    token_positions = torch.arange(
        input_ids.shape[-1],
        device=device,
    )

    reference_output = reference_model(
        input_ids,
        token_positions,
    )
    flash_output = flash_model(
        input_ids,
        token_positions,
    )

    torch.testing.assert_close(
        flash_output,
        reference_output,
        atol=MODEL_ATOL,
        rtol=MODEL_RTOL,
    )

    # Use a fixed upstream gradient rather than separate losses whose
    # gradients could differ because their forward outputs differ slightly.
    torch.manual_seed(5678)
    grad_output = torch.randn_like(reference_output)

    reference_output.backward(grad_output)
    flash_output.backward(grad_output)

    _assert_parameter_gradients_close(
        reference_model,
        flash_model,
        atol=FULL_MODEL_BACKWARD_ATOL,
        rtol=FULL_MODEL_BACKWARD_RTOL,
    )