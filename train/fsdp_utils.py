# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import functools
import gc
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.distributed.fsdp._traversal_utils as traversal_utils
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

@contextlib.contextmanager
def _state_dict_type(module, st_type, state_dict_config=None, optim_state_dict_config=None):
    # Replacement for `FSDP.state_dict_type(...)` as a context manager.
    # The upstream context manager exit restores the previous state_dict_type,
    # which is `None` on first entry, triggering `KeyError: None` in
    # `set_state_dict_type` on some torch versions. We explicitly set the type
    # on enter and reset to FULL_STATE_DICT (the library default) on exit.
    kwargs = {}
    if state_dict_config is not None:
        kwargs["state_dict_config"] = state_dict_config
    if optim_state_dict_config is not None:
        kwargs["optim_state_dict_config"] = optim_state_dict_config
    FSDP.set_state_dict_type(module, st_type, **kwargs)
    try:
        yield
    finally:
        FSDP.set_state_dict_type(module, StateDictType.FULL_STATE_DICT)


from modeling.bagel.modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from modeling.bagel.qwen2_navit import (
    Qwen2DecoderLayer, 
    Qwen2MoEDecoderLayer, 
    Qwen2MoTDecoderLayer,
)
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy, 
        backward_prefetch, 
        cpu_offload, 
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=[]):
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD':
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    else:
        device_mesh = None
    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                Qwen2DecoderLayer,
                Qwen2MoEDecoderLayer,
                Qwen2MoTDecoderLayer,
                SiglipEncoderLayer,
                SiglipVisionTransformer,
                MLPconnector,
                TimestepEmbedder,
                PositionEmbedding,
            },
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
    )


class FSDPCheckpoint:
    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir, 
        train_steps, 
        model, 
        ema_model, 
        optimizer, 
        scheduler, 
        data_status,
        logger, 
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")
        dist.barrier()

        # Sharded checkpoints: each rank writes its own shard to disk via
        # torch.distributed.checkpoint. No all-gather to rank 0, so no CPU-RAM
        # spike on rank 0 during save. Produces <name>/ directories containing
        # .distcp files + a .metadata file. Must be reloaded with dcp.load().
        sharded_cfg = ShardedStateDictConfig(offload_to_cpu=False)

        if ema_model is not None:
            with _state_dict_type(ema_model, StateDictType.SHARDED_STATE_DICT, sharded_cfg):
                ema_state_dict = ema_model.state_dict()
                dcp.save(ema_state_dict, checkpoint_id=os.path.join(save_path, "ema"))
                del ema_state_dict
                gc.collect()

        with _state_dict_type(model, StateDictType.SHARDED_STATE_DICT, sharded_cfg):
            model_state_dict = model.state_dict()
            dcp.save(model_state_dict, checkpoint_id=os.path.join(save_path, "model"))
            del model_state_dict
            gc.collect()

        with _state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_save_path = os.path.join(
                save_path, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                opt_sd = optimizer.state_dict()
                torch.save(opt_sd, optimizer_save_path)
                del opt_sd
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                if dist.get_rank() < fsdp_config.num_shard:
                    opt_sd = optimizer.state_dict()
                    torch.save(opt_sd, optimizer_save_path)
                    del opt_sd
            else:
                raise NotImplementedError
            gc.collect()

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if dist.get_rank() == 0 and data_status is not None:
            torch.save(data_status, os.path.join(save_path, "data_status.pt"))

        dist.barrier()
        if dist.get_rank() == 0:
            FSDPCheckpoint._verify_ckpt(save_path, ema_model is not None, fsdp_config, logger)
            logger.info(f"Saved checkpoint to {save_path}.")
        return

    @staticmethod
    def _verify_ckpt(save_path, has_ema, fsdp_config, logger):
        total_shards = (
            fsdp_config.num_shard
            if fsdp_config.sharding_strategy == "HYBRID_SHARD"
            else dist.get_world_size()
        )
        missing = []
        def _check_dcp(name):
            d = os.path.join(save_path, name)
            meta = os.path.join(d, ".metadata")
            if not os.path.isfile(meta) or os.path.getsize(meta) == 0:
                missing.append(f"{name}/.metadata")
            for i in range(total_shards):
                shard = os.path.join(d, f"__{i}_0.distcp")
                if not os.path.isfile(shard) or os.path.getsize(shard) == 0:
                    missing.append(f"{name}/__{i}_0.distcp")
        _check_dcp("model")
        if has_ema:
            _check_dcp("ema")
        for i in range(total_shards):
            p = os.path.join(save_path, f"optimizer.{i:05d}-of-{total_shards:05d}.pt")
            if not os.path.isfile(p) or os.path.getsize(p) == 0:
                missing.append(os.path.basename(p))
        if missing:
            raise RuntimeError(f"Checkpoint incomplete at {save_path}: missing/empty {missing}")
        logger.info(f"Verified checkpoint artifacts at {save_path} ({total_shards} shards, ema={has_ema}).")

    @staticmethod
    def _load_one_sharded(fsdp_model, shard_dir):
        # Load a sharded checkpoint directory (produced by dcp.save) into an
        # FSDP-wrapped model in-place, popping fixed sinusoidal pos embeds so
        # the checkpoint can be reused at different resolutions.
        sharded_cfg = ShardedStateDictConfig(offload_to_cpu=False)
        with _state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT, sharded_cfg):
            state_dict = fsdp_model.state_dict()
            state_dict.pop('latent_pos_embed.pos_embed', None)
            state_dict.pop('vit_pos_embed.pos_embed', None)
            dcp.load(state_dict, checkpoint_id=shard_dir)
            fsdp_model.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()

    @staticmethod
    def _load_full_state_dict(model, state_dict, strict=False):
        if isinstance(model, FSDP):
            with _state_dict_type(model, StateDictType.FULL_STATE_DICT):
                return model.load_state_dict(state_dict, strict=strict)
        return model.load_state_dict(state_dict, strict=strict)

    @staticmethod
    def try_load_ckpt(resume_from, logger, model, ema_model=None, resume_from_ema=False):
        if resume_from is None or not os.path.exists(resume_from):
            logger.info(f"Training from scratch.")
            return model, ema_model

        logger.info(f"Loading checkpoint from {resume_from}.")

        # Sharded checkpoint layout: <resume_from>/model/ and <resume_from>/ema/
        sharded_model_dir = os.path.join(resume_from, "ema" if resume_from_ema else "model")
        sharded_ema_dir = os.path.join(resume_from, "ema")
        if os.path.isdir(sharded_model_dir) and os.path.exists(os.path.join(sharded_model_dir, ".metadata")):
            FSDPCheckpoint._load_one_sharded(model, sharded_model_dir)
            if ema_model is not None:
                ema_src = sharded_ema_dir if os.path.isdir(sharded_ema_dir) and os.path.exists(os.path.join(sharded_ema_dir, ".metadata")) else sharded_model_dir
                if ema_src == sharded_model_dir:
                    logger.info(f"replicating ema model from {sharded_model_dir}.")
                FSDPCheckpoint._load_one_sharded(ema_model, ema_src)
            return model, ema_model

        # Legacy safetensors layout (e.g., the pretrained BAGEL-7B-MoT base).
        if resume_from_ema:
            model_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
        else:
            model_state_dict_path = os.path.join(resume_from, f"model.safetensors")
        model_state_dict = load_file(model_state_dict_path, device="cpu")
        # NOTE position embeds are fixed sinusoidal embeddings, so we can just pop it off,
        # which makes it easier to adapt to different resolutions.
        model_state_dict.pop('latent_pos_embed.pos_embed', None)
        model_state_dict.pop('vit_pos_embed.pos_embed', None)
        msg = FSDPCheckpoint._load_full_state_dict(model, model_state_dict, strict=False)
        logger.info(msg)
        del model_state_dict

        if ema_model is not None:
            ema_state_dict_path = os.path.join(resume_from, f"ema.safetensors")
            if not os.path.exists(ema_state_dict_path):
                logger.info(f"replicaing ema model from {model_state_dict_path}.")
                ema_state_dict_path = model_state_dict_path
            ema_state_dict = load_file(ema_state_dict_path, device="cpu")
            ema_state_dict.pop('latent_pos_embed.pos_embed', None)
            ema_state_dict.pop('vit_pos_embed.pos_embed', None)
            msg = FSDPCheckpoint._load_full_state_dict(ema_model, ema_state_dict, strict=False)
            logger.info(msg)
            del ema_state_dict

        return model, ema_model

    @staticmethod
    def has_compatible_train_state(resume_from, fsdp_config):
        if resume_from is None or not os.path.exists(resume_from):
            return False
        if fsdp_config.sharding_strategy == "FULL_SHARD":
            total_shards = dist.get_world_size()
        elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
            total_shards = fsdp_config.num_shard
        else:
            raise NotImplementedError

        if not os.path.exists(os.path.join(resume_from, "scheduler.pt")):
            return False
        for shard_index in range(total_shards):
            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if not os.path.exists(optimizer_state_dict_path):
                return False
        return True

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            if fsdp_config.sharding_strategy == "FULL_SHARD":
                shard_index = dist.get_rank()
                total_shards = dist.get_world_size()
            elif fsdp_config.sharding_strategy == "HYBRID_SHARD":
                shard_index = dist.get_rank() % fsdp_config.num_shard
                total_shards = fsdp_config.num_shard
            else:
                raise NotImplementedError

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            optimizer_state_dict = torch.load(optimizer_state_dict_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(optimizer_state_dict)
            del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            scheduler_state_dict = torch.load(scheduler_state_dict_path, weights_only=True, map_location="cpu")
            scheduler.load_state_dict(scheduler_state_dict)
            del scheduler_state_dict

            train_steps = int(os.path.basename(os.path.normpath(resume_from))) + 1
            """
            data_status = [
                {
                    dataset_name: {
                        worker_id: [parquet_idx, row_group_id, row_idx],
                    },
                },
            ]
            """
            data_status_path = os.path.join(resume_from, "data_status.pt")
            if os.path.exists(data_status_path):
                data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
                local_rank = dist.get_rank()
                if local_rank < len(data_status):
                    data_status = data_status[local_rank]
                else:
                    data_status = None
            else:
                data_status = None
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen2DecoderLayer, 
        SiglipEncoderLayer, 
        MLPconnector, 
        Qwen2MoEDecoderLayer, 
        Qwen2MoTDecoderLayer
    )
    return isinstance(module, module_options)


def fsdp_ema_setup(ema_model, fsdp_config, ignored_modules=[]):
    for param in ema_model.parameters():
        param.requires_grad = False

    ema_model = fsdp_wrapper(ema_model, fsdp_config, ignored_modules=ignored_modules)
    return ema_model


@torch.no_grad()
def fsdp_ema_update(ema_model, model, decay=0.9999):
    ema_handles = traversal_utils._get_fsdp_handles(ema_model)
    new_handles = traversal_utils._get_fsdp_handles(model)
    assert len(ema_handles) == len(new_handles)
    ema_params = []
    new_params = []

    for ema_handle, new_handle in zip(ema_handles, new_handles):
        if ema_handle.flat_param is not None and new_handle.flat_param.requires_grad:
            ema_params.append(ema_handle.flat_param.data)
            new_params.append(new_handle.flat_param.data.to(dtype=ema_handle.flat_param.dtype))

    torch._foreach_mul_(ema_params, decay)
    torch._foreach_add_(ema_params, new_params, alpha=1 - decay)
