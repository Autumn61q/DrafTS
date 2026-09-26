from typing import Optional

import torch


class EchoStateNetwork(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        spectral_radius: float,
        sparsity: float,
        num_features: int,
        prediction_len: Optional[int],
        device: torch.device,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.prediction_len = prediction_len
        if output_dim is None:
            if prediction_len is None or prediction_len <= 0:
                raise ValueError("prediction_len must be positive for sequence output")
            output_dim = prediction_len * num_features
        elif output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        self.output_dim = int(output_dim)
        fork_devices = []
        if device.type == "cuda":
            device_index = device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            fork_devices.append(device_index)
        with torch.random.fork_rng(devices=fork_devices):
            input_weights = (
                torch.rand(hidden_dim, num_features, device=device) * 2.0 - 1.0
            )
            reservoir = torch.rand(
                hidden_dim, hidden_dim, device=device
            ) * 2.0 - 1.0
            reservoir *= (
                torch.rand(hidden_dim, hidden_dim, device=device) < sparsity
            ).to(reservoir.dtype)
            radius = torch.linalg.eigvals(reservoir).abs().max()
            scaled_reservoir = reservoir * (
                spectral_radius / (radius + 1e-8)
            )
            readout = torch.nn.Linear(
                hidden_dim,
                self.output_dim,
                device=device,
            )
        self.register_buffer("W_in", input_weights)
        self.register_buffer("W_res", scaled_reservoir)
        self.readout = readout

    def forward(
        self,
        noise: torch.Tensor,
        inputs_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, input_len, num_features = noise.shape
        state = noise.new_zeros(batch_size, self.W_res.shape[0])
        for time_index in range(input_len):
            updated_state = torch.tanh(
                noise[:, time_index] @ self.W_in.T + state @ self.W_res.T
            )
            if inputs_mask is None:
                state = updated_state
            else:
                valid_time = inputs_mask[:, time_index].any(dim=-1, keepdim=True)
                state = torch.where(valid_time, updated_state, state)
        prediction = self.readout(state)
        if self.prediction_len is None:
            return prediction
        return prediction.view(batch_size, self.prediction_len, num_features)
