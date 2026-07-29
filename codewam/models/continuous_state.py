from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import masked_mean
from .contracts import ContinuousState, StateInputs


def _encoder_stack(
    *,
    dim: int,
    heads: int,
    layers: int,
    dropout: float,
) -> nn.TransformerEncoder:
    if min(dim, heads, layers) <= 0 or dim % heads:
        raise ValueError("Transformer width, heads, and layers must be compatible.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("Transformer dropout must be in [0,1).")
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=4 * dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer,
        num_layers=layers,
        norm=nn.LayerNorm(dim),
        enable_nested_tensor=False,
    )


class ContinuousStateEncoder(nn.Module):
    """Causal spatiotemporal encoder over frozen Wan-VAE latents."""

    def __init__(
        self,
        *,
        latent_channels: int,
        dim: int,
        heads: int,
        patch_size: int = 2,
        spatial_layers: int = 2,
        temporal_layers: int = 2,
        max_time: int = 32,
        max_cameras: int = 8,
        max_spatial_tokens: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min(
            latent_channels,
            dim,
            heads,
            patch_size,
            spatial_layers,
            temporal_layers,
            max_time,
            max_cameras,
            max_spatial_tokens,
        ) <= 0:
            raise ValueError(
                "Continuous-state dimensions and layer counts must be positive."
            )
        if dim % heads:
            raise ValueError("Continuous-state dim must be divisible by heads.")
        self.latent_channels = int(latent_channels)
        self.dim = int(dim)
        self.patch_size = int(patch_size)
        self.max_time = int(max_time)
        self.max_cameras = int(max_cameras)
        self.max_spatial_tokens = int(max_spatial_tokens)
        self.patch_projection = nn.Conv2d(
            latent_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.camera_embedding = nn.Embedding(max_cameras, dim)
        self.time_embedding = nn.Embedding(max_time, dim)
        self.spatial_embedding = nn.Parameter(
            torch.randn(max_spatial_tokens, dim) * (dim**-0.5)
        )
        self.spatial_encoder = _encoder_stack(
            dim=dim,
            heads=heads,
            layers=spatial_layers,
            dropout=dropout,
        )
        self.temporal_encoder = _encoder_stack(
            dim=dim,
            heads=heads,
            layers=temporal_layers,
            dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward_sequence(
        self,
        state: StateInputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = state.latents
        batch, time, views, channels, height, width = latents.shape
        if channels != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, got {channels}."
            )
        if time > self.max_time or views > self.max_cameras:
            raise ValueError(
                f"Input T/V {(time, views)} exceeds configured maxima "
                f"{(self.max_time, self.max_cameras)}."
            )
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Latent spatial shape {(height, width)} must be divisible by "
                f"patch size {self.patch_size}."
            )

        projected = self.patch_projection(
            latents.reshape(batch * time * views, channels, height, width)
        )
        spatial_tokens = int(projected.shape[-2] * projected.shape[-1])
        if spatial_tokens > self.max_spatial_tokens:
            raise ValueError(
                f"Patch count {spatial_tokens} exceeds {self.max_spatial_tokens}."
            )
        values = projected.flatten(2).transpose(1, 2)
        values = values.reshape(batch, time, views, spatial_tokens, self.dim)

        camera_ids = torch.arange(views, device=latents.device)
        camera = self.camera_embedding(camera_ids)[None, None, :, None]
        spatial = self.spatial_embedding[:spatial_tokens][None, None, None]
        values = values + camera + spatial
        valid = (
            state.latent_valid
            if state.latent_valid is not None
            else torch.ones(
                (batch, time, views),
                dtype=torch.bool,
                device=latents.device,
            )
        )
        values = values * valid[:, :, :, None, None].to(values.dtype)
        values = self.spatial_encoder(
            values.reshape(batch * time * views, spatial_tokens, self.dim)
        ).reshape(batch, time, views, spatial_tokens, self.dim)
        values = values * valid[:, :, :, None, None].to(values.dtype)

        time_ids = torch.arange(time, device=latents.device)
        values = values + self.time_embedding(time_ids)[None, :, None, None]
        temporal = values.permute(0, 2, 3, 1, 4).reshape(
            batch * views * spatial_tokens,
            time,
            self.dim,
        )
        temporal_valid = (
            valid[:, None]
            .expand(batch, spatial_tokens, time, views)
            .permute(0, 3, 1, 2)
            .reshape(batch * views * spatial_tokens, time)
        )
        padding = ~temporal_valid
        all_missing = padding.all(dim=1)
        safe_padding = padding.clone()
        safe_padding[all_missing, 0] = False
        causal_mask = torch.triu(
            torch.ones((time, time), dtype=torch.bool, device=latents.device),
            diagonal=1,
        )
        temporal = self.temporal_encoder(
            temporal,
            mask=causal_mask,
            src_key_padding_mask=safe_padding,
        )
        temporal = temporal * temporal_valid[:, :, None].to(temporal.dtype)
        sequence = temporal.reshape(
            batch,
            views,
            spatial_tokens,
            time,
            self.dim,
        ).permute(0, 3, 1, 2, 4)
        sequence = sequence.reshape(batch, time, views * spatial_tokens, self.dim)
        token_valid = (
            valid[:, :, :, None]
            .expand(batch, time, views, spatial_tokens)
            .reshape(batch, time, views * spatial_tokens)
        )
        return self.output_norm(sequence), token_valid

    def forward(
        self,
        state: StateInputs,
        *,
        time_index: int = -1,
    ) -> ContinuousState:
        sequence, valid = self.forward_sequence(state)
        index = int(time_index)
        if index < 0:
            index += sequence.shape[1]
        if index < 0 or index >= sequence.shape[1]:
            raise IndexError(f"State time index {time_index} is out of range.")
        return ContinuousState(tokens=sequence[:, index], valid=valid[:, index])


class TemporalLatentPredictor(nn.Module):
    """Discardable Stage-0 head that predicts a future frozen latent frame."""

    def __init__(
        self,
        *,
        dim: int,
        latent_channels: int,
        patch_size: int,
        heads: int,
        layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min(dim, latent_channels, patch_size, heads, layers) <= 0:
            raise ValueError(
                "Temporal-predictor dimensions and layer counts must be positive."
            )
        if dim % heads:
            raise ValueError("Temporal predictor dim must be divisible by heads.")
        self.latent_channels = int(latent_channels)
        self.patch_size = int(patch_size)
        self.encoder = _encoder_stack(
            dim=dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
        )
        self.output = nn.Linear(dim, latent_channels * patch_size * patch_size)

    def patchify_target(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 5:
            raise ValueError(
                f"Future latent frame must be [B,V,C,H,W], got {tuple(frame.shape)}."
            )
        batch, views, channels, height, width = frame.shape
        if channels != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} target channels, got {channels}."
            )
        patches = F.unfold(
            frame.reshape(batch * views, channels, height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).transpose(1, 2)
        return patches.reshape(batch, views * patches.shape[1], patches.shape[2])

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        padding = None
        if context_valid is not None:
            if (
                context_valid.dtype != torch.bool
                or tuple(context_valid.shape) != tuple(context_tokens.shape[:2])
            ):
                raise ValueError("Temporal context validity must be bool [B,N].")
            padding = ~context_valid
            all_missing = padding.all(dim=1)
            padding = padding.clone()
            padding[all_missing, 0] = False
        prediction = self.output(
            self.encoder(
                context_tokens,
                src_key_padding_mask=padding,
            )
        )
        if context_valid is not None:
            prediction = prediction * context_valid[:, :, None].to(
                prediction.dtype
            )
        return prediction

    def loss(
        self,
        context: ContinuousState,
        future_frame: torch.Tensor,
        *,
        sample_valid: torch.Tensor | None = None,
        target_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prediction = self(context.tokens, context.valid)
        target = self.patchify_target(future_frame).detach().to(prediction.dtype)
        if prediction.shape != target.shape:
            raise ValueError(
                "State/target patch layouts differ: "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
            )
        valid = context.valid
        if target_valid is not None:
            batch, views = future_frame.shape[:2]
            if (
                target_valid.dtype != torch.bool
                or tuple(target_valid.shape) != (batch, views)
            ):
                raise ValueError("Temporal target validity must be bool [B,V].")
            if target_valid.device != valid.device:
                raise ValueError(
                    "Temporal target validity must share the context device."
                )
            patches_per_view = target.shape[1] // views
            target_token_valid = (
                target_valid[:, :, None]
                .expand(batch, views, patches_per_view)
                .reshape(batch, -1)
            )
            valid = valid & target_token_valid
        if sample_valid is not None:
            if sample_valid.dtype != torch.bool or sample_valid.shape != (valid.shape[0],):
                raise ValueError("Temporal sample supervision must be bool [B].")
            if sample_valid.device != valid.device:
                raise ValueError(
                    "Temporal sample supervision must share the context device."
                )
            valid = valid & sample_valid[:, None]
        squared_error = (prediction - target).square()
        return masked_mean(squared_error, valid)


def temporal_pretraining_loss(
    state_encoder: ContinuousStateEncoder,
    predictor: TemporalLatentPredictor,
    state: StateInputs,
    *,
    context_index: int,
    target_index: int,
    sample_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if target_index <= context_index:
        raise ValueError("Temporal target must be later than the context index.")
    if target_index >= state.latents.shape[1]:
        raise IndexError("Temporal target index exceeds the latent history.")
    context_state = StateInputs(
        latents=state.latents[:, : context_index + 1],
        proprio_history=state.proprio_history,
        past_actions=state.past_actions,
        latent_valid=(
            None
            if state.latent_valid is None
            else state.latent_valid[:, : context_index + 1]
        ),
        proprio_valid=state.proprio_valid,
        past_action_valid=state.past_action_valid,
    )
    context = state_encoder(context_state)
    return predictor.loss(
        context,
        state.latents[:, target_index],
        sample_valid=sample_valid,
        target_valid=(
            None
            if state.latent_valid is None
            else state.latent_valid[:, target_index]
        ),
    )
