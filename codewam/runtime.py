"""CodeWAM runtime factory (hydra `_target_`)。

复用 FastWAM 的底层组件加载(基座专家),组装成 CodeWAM 模型。
与 FastWAM 共享数据/评测管线以便公平对照。
"""
import torch
from omegaconf import DictConfig, OmegaConf

from codewam.model import CodeWAM


def _to_dict(x, name, default=None):
    if isinstance(x, DictConfig):
        x = OmegaConf.to_container(x, resolve=True)
    if x is None:
        x = {} if default is None else default
    if not isinstance(x, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(x)}")
    return x


def create_codewam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = False,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    state_codebook=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    video_dit_config = _to_dict(video_dit_config, "video_dit_config")
    action_dit_config = _to_dict(action_dit_config, "action_dit_config", {})
    video_scheduler = _to_dict(video_scheduler, "video_scheduler", {})
    action_scheduler = _to_dict(action_scheduler, "action_scheduler")
    loss = _to_dict(loss, "loss", {})
    state_codebook = None if state_codebook is None else _to_dict(state_codebook, "state_codebook")

    for k in ("train_shift", "infer_shift", "num_train_timesteps"):
        if k not in action_scheduler:
            raise ValueError(f"`action_scheduler` missing key: {k}")

    return CodeWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        action_dim_loss_weights=loss.get("action_dim_loss_weights", None),
        decision_frame_weights=loss.get("decision_frame_weights", None),
        state_codebook=state_codebook,
    )
