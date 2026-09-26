from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from basicts.utils import RunnerStatus
from .callback import BasicTSCallback
from .factor_cache import FactorCacheManager
from .hilbert import HilbertFactorExtractor, HilbertNoiseGate
from .reservoir import EchoStateNetwork

if TYPE_CHECKING:
    from basicts.runners.basicts_runner import BasicTSRunner


class _DataLoaderWithIndex:
    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self.dataset = dataloader.dataset
        self.collate_fn = dataloader.collate_fn or default_collate
        self.batch_sampler = dataloader.batch_sampler

    def __iter__(self):
        torch.empty((), dtype=torch.int64).random_(
            generator=self._dataloader.generator
        ).item()
        for batch_indices in self.batch_sampler:
            collated = self.collate_fn(
                [self.dataset[index] for index in batch_indices]
            )
            if not isinstance(collated, dict):
                raise TypeError("refinement requires dictionary batches")
            collated = dict(collated)
            collated["idx"] = torch.tensor(batch_indices, dtype=torch.long)
            yield collated

    def __len__(self):
        return len(self._dataloader)

    def __getattr__(self, name):
        return getattr(self._dataloader, name)


class DrafTSCallback(BasicTSCallback):
    persistent_state = True

    def __init__(
        self,
        imf_count: int = 2,
        rc_loss_weight: float = 0.3,
        fused_loss_weight: float = 0.0,
        hilbert_gate_mask_max: float = 0.7,
        hilbert_gate_initial_mask: float = 0.2,
        factor_cache_dir: Optional[str] = None,
        factor_cache_imf_count: Optional[int] = None,
        factor_cache_read_only: bool = False,
        classification_rc: bool = False,
    ):
        super().__init__()
        if rc_loss_weight < 0.0 or fused_loss_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if imf_count <= 0:
            raise ValueError(f"imf_count must be positive, got {imf_count}")
        self.rc_loss_weight = float(rc_loss_weight)
        self.fused_loss_weight = float(fused_loss_weight)
        self.classification_rc = bool(classification_rc)
        self.imf_count = int(imf_count)
        self.hilbert_gate_mask_max = float(hilbert_gate_mask_max)
        self.hilbert_gate_initial_mask = float(hilbert_gate_initial_mask)
        self.rc_node: Optional[EchoStateNetwork] = None
        self.hilbert_gate: Optional[HilbertNoiseGate] = None
        self._extractor = HilbertFactorExtractor(self.imf_count)
        self._factor_cache_manager = FactorCacheManager(
            imf_count=self.imf_count,
            cache_dir=factor_cache_dir,
            cache_imf_count=factor_cache_imf_count,
            read_only=factor_cache_read_only,
        )
        self.factor_cache_dir = self._factor_cache_manager.cache_dir
        self.factor_cache_imf_count = self._factor_cache_manager.cache_imf_count
        self.factor_cache_read_only = self._factor_cache_manager.read_only

    def _validate_saved_configuration(
        self,
        state_dict: Dict[str, Any],
    ) -> None:
        saved = state_dict.get("callback_config")
        if not isinstance(saved, dict):
            raise ValueError(
                "DrafTS checkpoint has no valid configuration metadata"
            )
        saved = dict(saved)
        if saved.get("method") == "Hilbert":
            saved["method"] = "DrafTS"
        current = self._configuration_state()
        mismatches = {
            key: (saved.get(key), value)
            for key, value in current.items()
            if saved.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}: saved={saved_value!r}, active={active_value!r}"
                for key, (saved_value, active_value) in mismatches.items()
            )
            raise ValueError(
                "DrafTS configuration does not match "
                f"the checkpoint ({details})"
            )

    def _wrap_data_loaders(self, runner: "BasicTSRunner") -> None:
        for attribute in (
            "train_data_loader",
            "val_data_loader",
            "test_data_loader",
        ):
            loader = getattr(runner, attribute, None)
            if loader is not None and not isinstance(loader, _DataLoaderWithIndex):
                setattr(runner, attribute, _DataLoaderWithIndex(loader))

    def _add_optimizer_parameters(
        self,
        runner: "BasicTSRunner",
        module: torch.nn.Module,
    ) -> None:
        optimizer = runner.optimizer
        if optimizer is None:
            raise RuntimeError("refinement modules require an initialized optimizer")
        existing = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        parameters = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad and id(parameter) not in existing
        ]
        if parameters:
            optimizer.add_param_group(
                {"params": parameters, "lr": optimizer.param_groups[0]["lr"]}
            )

    def _ensure_rc_node(
        self,
        runner: "BasicTSRunner",
        num_features: int,
        device: torch.device,
        prediction_len: Optional[int] = None,
        output_dim: Optional[int] = None,
    ) -> None:
        if (prediction_len is None) == (output_dim is None):
            raise ValueError("RC requires exactly one output specification")
        created = False
        if self.rc_node is None:
            configured_seed = runner.cfg.seed
            if configured_seed is None:
                raise ValueError("refinement requires cfg.seed")
            devices = []
            if device.type == "cuda":
                device_index = (
                    device.index
                    if device.index is not None
                    else torch.cuda.current_device()
                )
                devices = [device_index]
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(configured_seed))
                self.rc_node = EchoStateNetwork(
                    hidden_dim=64,
                    spectral_radius=0.5,
                    sparsity=0.1,
                    num_features=num_features,
                    prediction_len=prediction_len,
                    device=device,
                    output_dim=output_dim,
                )
            if output_dim is not None:
                torch.nn.init.zeros_(self.rc_node.readout.weight)
                torch.nn.init.zeros_(self.rc_node.readout.bias)
            created = True
        expected_output_dim = (
            int(output_dim)
            if output_dim is not None
            else int(prediction_len) * num_features
        )
        if (
            self.rc_node.W_in.shape[1] != num_features
            or self.rc_node.prediction_len != prediction_len
            or self.rc_node.output_dim != expected_output_dim
        ):
            raise ValueError(
                "RC dimensions changed after initialization: "
                f"expected N={self.rc_node.W_in.shape[1]}, "
                f"output={self.rc_node.output_dim}; got N={num_features}, "
                f"output={expected_output_dim}"
            )
        if created:
            self._add_optimizer_parameters(runner, self.rc_node)

    def on_optimizer_init(self, runner: "BasicTSRunner", **kwargs) -> None:
        loader = runner.train_data_loader
        if loader is None or len(loader.dataset) == 0:
            raise RuntimeError("DrafTS requires training data")
        sample = loader.dataset[0]
        if not isinstance(sample, dict):
            raise RuntimeError("refinement requires dictionary dataset samples")
        if "inputs" not in sample:
            raise RuntimeError("refinement dataset samples must contain 'inputs'")
        inputs = np.asarray(sample["inputs"])
        if inputs.ndim != 2:
            raise RuntimeError(
                f"refinement expects [T,N] inputs, got {inputs.shape}"
            )
        device = next(runner.model.parameters()).device
        self._initialize_gate(runner, device)
        if self.classification_rc:
            self._ensure_rc_node(
                runner,
                num_features=int(inputs.shape[-1]),
                device=device,
                output_dim=int(runner.cfg.model_config.num_classes),
            )
            return
        if "targets" not in sample:
            raise RuntimeError(
                "residual refinement dataset samples must contain 'targets'"
            )
        targets = np.asarray(sample["targets"])
        if targets.ndim != 2:
            raise RuntimeError(
                f"residual refinement expects [H,N] targets, got {targets.shape}"
            )
        self._ensure_rc_node(
            runner,
            num_features=int(inputs.shape[-1]),
            prediction_len=int(targets.shape[0]),
            device=device,
        )

    def state_dict(self) -> Dict[str, Any]:
        if self.hilbert_gate is None or self.rc_node is None:
            raise RuntimeError(
                "refinement modules must be initialized before checkpointing"
            )
        return {
            "callback_config": self._configuration_state(),
            "hilbert_gate": self.hilbert_gate.state_dict(),
            "rc_node": self.rc_node.state_dict(),
        }

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        runner: Optional["BasicTSRunner"] = None,
    ) -> None:
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("refinement callback checkpoint state is empty")
        self._validate_saved_configuration(state_dict)
        gate_state = state_dict.get("hilbert_gate")
        rc_state = state_dict.get("rc_node")
        if not isinstance(gate_state, dict):
            raise ValueError("refinement checkpoint is missing the Hilbert gate state")
        if not isinstance(rc_state, dict):
            raise ValueError("refinement checkpoint is missing the RC state")
        device = (
            next(runner.model.parameters()).device
            if runner is not None
            else torch.device("cpu")
        )
        if self.hilbert_gate is None:
            self.hilbert_gate = HilbertNoiseGate(
                feature_dim=4,
                mask_max=self.hilbert_gate_mask_max,
                initial_mask=self.hilbert_gate_initial_mask,
                initial_weight=1.0,
                device=device,
            )
        self.hilbert_gate.load_state_dict(gate_state)
        if self.rc_node is None:
            hidden_dim, num_features = rc_state["W_in"].shape
            output_dim = rc_state["readout.weight"].shape[0]
            if not self.classification_rc and output_dim % num_features != 0:
                raise ValueError("saved RC output dimension is invalid")
            self.rc_node = EchoStateNetwork(
                hidden_dim=hidden_dim,
                spectral_radius=0.5,
                sparsity=0.1,
                num_features=num_features,
                prediction_len=(
                    None if self.classification_rc else output_dim // num_features
                ),
                device=device,
                output_dim=output_dim if self.classification_rc else None,
            )
        self.rc_node.load_state_dict(rc_state)

    def on_train_start(self, runner: "BasicTSRunner", **kwargs) -> None:
        self._factor_cache_manager.clear()
        self._wrap_data_loaders(runner)
        if self.classification_rc:
            runner.logger.info(
                "DrafTS started with classification RC, "
                f"lambda_rc={self.rc_loss_weight:g}, "
                f"lambda_fused={self.fused_loss_weight:g}"
            )
            return
        runner.logger.info(
            "DrafTS started with residual RC, "
            f"lambda_rc={self.rc_loss_weight:g}, "
            f"lambda_fused={self.fused_loss_weight:g}"
        )

    def on_validate_start(self, runner: "BasicTSRunner", **kwargs) -> None:
        self._wrap_data_loaders(runner)

    def on_test_start(self, runner: "BasicTSRunner", **kwargs) -> None:
        self._wrap_data_loaders(runner)

    def _split(self, runner: "BasicTSRunner") -> str:
        if runner.status == RunnerStatus.TRAINING:
            return "train"
        if runner.status == RunnerStatus.VALIDATING:
            return "val"
        if runner.status in {RunnerStatus.TESTING, RunnerStatus.EVALUATING}:
            return "test"
        raise RuntimeError(f"unsupported runner status for refinement: {runner.status}")

    def on_before_forward(self, runner: "BasicTSRunner", **kwargs) -> None:
        data = kwargs.get("data")
        if not isinstance(data, dict):
            raise TypeError("refinement expects a dictionary batch")
        raw_inputs = data.get("inputs")
        if not torch.is_tensor(raw_inputs) or raw_inputs.ndim != 3:
            raise ValueError("refinement expects inputs with shape [B,T,N]")
        indices = data.get("idx")
        if indices is None:
            raise RuntimeError("indexed data loader did not provide batch indices")
        index_array = self._to_numpy(indices).astype(np.int64).reshape(-1)
        estimated_noise = self._estimate_noise(
            runner,
            self._split(runner),
            index_array,
            raw_inputs,
            data.get("inputs_mask"),
        ).to(device=raw_inputs.device, dtype=raw_inputs.dtype)
        if estimated_noise.shape != raw_inputs.shape:
            raise ValueError(
                f"estimated noise shape {estimated_noise.shape} does not match "
                f"input shape {raw_inputs.shape}"
            )
        clean_inputs = raw_inputs - estimated_noise
        data["refinement_input_raw"] = raw_inputs
        data["refinement_input_noise"] = estimated_noise
        data["refinement_input_clean"] = clean_inputs
        data["inputs"] = clean_inputs

    @staticmethod
    def _standardize_noise(
        noise: torch.Tensor,
        inputs_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if inputs_mask is None:
            mean = noise.mean(dim=(1, 2), keepdim=True)
            centered = noise - mean
            variance = centered.square().mean(dim=(1, 2), keepdim=True)
            std = variance.clamp_min(1e-10).sqrt()
            return centered / std, None
        if inputs_mask.shape != noise.shape:
            raise ValueError(
                "inputs_mask must match refinement noise, got "
                f"{inputs_mask.shape} and {noise.shape}"
            )
        mask = inputs_mask.bool()
        weight = mask.to(noise.dtype)
        count = weight.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        mean = (noise * weight).sum(dim=(1, 2), keepdim=True) / count
        variance = (
            (noise - mean).square() * weight
        ).sum(dim=(1, 2), keepdim=True) / count
        normalized = (noise - mean) / variance.clamp_min(1e-10).sqrt()
        return torch.where(mask, normalized, 0.0), mask

    def on_compute_loss(self, runner: "BasicTSRunner", **kwargs) -> None:
        data = kwargs.get("data")
        forward_return = kwargs.get("forward_return")
        if not isinstance(data, dict) or not isinstance(forward_return, dict):
            raise TypeError(
                "refinement loss requires data and forward_return dictionaries"
            )
        backbone_prediction = forward_return.get("prediction")
        targets = forward_return.get("targets")
        raw_inputs = data.get("refinement_input_raw")
        estimated_noise = data.get("refinement_input_noise")
        clean_inputs = data.get("refinement_input_clean")
        inputs_mask = data.get("inputs_mask")
        tensors = (
            backbone_prediction,
            targets,
            raw_inputs,
            estimated_noise,
            clean_inputs,
        )
        if not all(torch.is_tensor(tensor) for tensor in tensors):
            raise RuntimeError("refinement data flow is incomplete")
        named_tensors = {
            "backbone prediction": backbone_prediction,
            "targets": targets,
            "raw inputs": raw_inputs,
            "estimated noise": estimated_noise,
            "clean inputs": clean_inputs,
        }
        for name, tensor in named_tensors.items():
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"non-finite values in refinement {name}")
        normalized_noise, rc_mask = self._standardize_noise(
            estimated_noise,
            inputs_mask,
        )
        targets_mask = None
        if self.classification_rc:
            if backbone_prediction.ndim != 2 or targets.ndim != 1:
                raise ValueError(
                    "classification RC expects logits [B,C] and targets [B], "
                    f"got {backbone_prediction.shape} and {targets.shape}"
                )
            self._ensure_rc_node(
                runner,
                num_features=int(estimated_noise.shape[-1]),
                device=backbone_prediction.device,
                output_dim=int(backbone_prediction.shape[-1]),
            )
        else:
            if backbone_prediction.shape != targets.shape:
                raise ValueError(
                    f"prediction shape {backbone_prediction.shape} does not match "
                    f"target shape {targets.shape}"
                )
            targets_mask = forward_return.get("targets_mask")
            self._ensure_rc_node(
                runner,
                num_features=int(estimated_noise.shape[-1]),
                device=backbone_prediction.device,
                prediction_len=int(backbone_prediction.shape[1]),
            )
        rc_prediction = self.rc_node(normalized_noise, rc_mask)
        final_prediction = backbone_prediction + rc_prediction
        forward_return.update(
            {
                "prediction": final_prediction,
                "inputs": raw_inputs,
                "backbone_prediction": backbone_prediction,
                "rc_prediction": rc_prediction,
                "refinement_input_raw": raw_inputs,
                "refinement_input_noise": estimated_noise,
                "refinement_input_clean": clean_inputs,
                "refinement_rc_prediction": rc_prediction,
                "refinement_correction": rc_prediction,
            }
        )
        if self.classification_rc:
            backbone_loss = runner._metric_forward(
                runner.loss,
                {"prediction": backbone_prediction, "targets": targets},
            )
            rc_loss = runner._metric_forward(
                runner.loss,
                {
                    "prediction": backbone_prediction.detach() + rc_prediction,
                    "targets": targets,
                },
            )
            fused_loss = runner._metric_forward(
                runner.loss,
                {"prediction": final_prediction, "targets": targets},
            )
        else:
            rc_target = (targets - backbone_prediction).detach()
            forward_return["rc_target"] = rc_target
            forward_return["refinement_rc_target"] = rc_target
            backbone_loss_input = {
                "prediction": backbone_prediction,
                "targets": targets,
            }
            if targets_mask is not None:
                backbone_loss_input["targets_mask"] = targets_mask
            backbone_loss = runner._metric_forward(
                runner.loss,
                backbone_loss_input,
            )
            rc_error = (rc_prediction - rc_target).square()
            if targets_mask is None:
                rc_loss = rc_error.mean()
            else:
                rc_weight = targets_mask.to(rc_error.dtype)
                rc_loss = (
                    (rc_error * rc_weight).sum()
                    / rc_weight.sum().clamp_min(1.0)
                )
            fused_loss = runner._metric_forward(runner.loss, forward_return)
        total_loss = (
            backbone_loss
            + self.rc_loss_weight * rc_loss
            + self.fused_loss_weight * fused_loss
        )
        for name, tensor in {
            "RC prediction": rc_prediction,
            "backbone loss": backbone_loss,
            "RC loss": rc_loss,
            "fused loss": fused_loss,
            "total loss": total_loss,
        }.items():
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"non-finite values in refinement {name}")
        forward_return["loss"] = (
            total_loss
            if runner.status == RunnerStatus.TRAINING
            else fused_loss
        )
        forward_return["backbone_loss"] = backbone_loss
        forward_return["rc_loss"] = rc_loss
        forward_return["fused_reg_loss"] = fused_loss

    @staticmethod
    def _to_numpy(data: Any) -> np.ndarray:
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def _initialize_gate(
        self,
        runner: "BasicTSRunner",
        device: torch.device,
    ) -> None:
        if self.hilbert_gate is None:
            self.hilbert_gate = HilbertNoiseGate(
                feature_dim=4,
                mask_max=self.hilbert_gate_mask_max,
                initial_mask=self.hilbert_gate_initial_mask,
                initial_weight=1.0,
                device=device,
            )
        self._add_optimizer_parameters(runner, self.hilbert_gate)

    def distributed_trainable_modules(self):
        modules = [] if self.rc_node is None else [self.rc_node]
        if self.hilbert_gate is not None:
            modules.append(self.hilbert_gate)
        return tuple(modules)

    def _configuration_state(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "method": "DrafTS",
            "rc_spectral_radius": 0.5,
            "rc_hidden_dim": 64,
            "rc_sparsity": 0.1,
            "rc_loss_weight": self.rc_loss_weight,
            "fused_loss_weight": self.fused_loss_weight,
            "classification_rc": self.classification_rc,
            "imf_count": self.imf_count,
            "emd_spline_kind": "cubic",
            "emd_padding_mode": "reflect",
            "reflect_padding_ratio": 0.25,
            "hilbert_feature_mode": "qaio",
            "hilbert_local_window_periods": 1.5,
            "hilbert_gate_mask_max": self.hilbert_gate_mask_max,
            "hilbert_gate_initial_mask": self.hilbert_gate_initial_mask,
            "hilbert_gate_initial_weight": 1.0,
        }

    def on_epoch_end(self, runner: "BasicTSRunner", **kwargs) -> None:
        self._factor_cache_manager.flush()

    def on_train_end(self, runner: "BasicTSRunner", **kwargs) -> None:
        self._extractor.shutdown()

    def on_test_end(self, runner: "BasicTSRunner", **kwargs) -> None:
        self._factor_cache_manager.flush()
        self._extractor.shutdown()

    def _estimate_noise(
        self,
        runner: "BasicTSRunner",
        split: str,
        indices: np.ndarray,
        inputs: torch.Tensor,
        inputs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.hilbert_gate is None:
            self._initialize_gate(runner, inputs.device)
        candidates_np, features_np, active_np = (
            self._factor_cache_manager.factor_batch(
                runner=runner,
                split=split,
                indices=indices,
                inputs=inputs,
                inputs_mask=inputs_mask,
                extractor=self._extractor,
            )
        )
        candidates = torch.tensor(
            candidates_np,
            device=inputs.device,
            dtype=torch.float32,
        )
        features = torch.tensor(
            features_np,
            device=inputs.device,
            dtype=torch.float32,
        )
        active = torch.tensor(
            active_np,
            device=inputs.device,
            dtype=torch.float32,
        )
        masks = self.hilbert_gate(features, active)
        if inputs_mask is not None:
            masks = masks * inputs_mask.permute(0, 2, 1).unsqueeze(2)
        noise = (masks * candidates).sum(dim=2)
        noise = noise.permute(0, 2, 1).contiguous()
        if inputs_mask is not None:
            noise = torch.where(inputs_mask.bool(), noise, 0.0)
        return noise
