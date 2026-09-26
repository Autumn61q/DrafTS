from concurrent.futures import ProcessPoolExecutor
import os
from typing import Optional, Tuple

import numpy as np
import torch
from PyEMD import EMD


def _extract_emd_factors_worker(
    job: Tuple[np.ndarray, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    signal, imf_count = job
    extractor = HilbertFactorExtractor(imf_count=imf_count, workers=1)
    return extractor.extract_factors(
        np.asarray(signal, dtype=np.float32),
        EMD(spline_kind="cubic"),
    )


class HilbertNoiseGate(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int,
        mask_max: float,
        initial_mask: float,
        initial_weight: float,
        device: torch.device,
    ):
        super().__init__()
        if mask_max < 0.0:
            raise ValueError(
                f"hilbert_gate_mask_max must be non-negative, got {mask_max}"
            )
        if not 0.0 <= initial_mask <= mask_max:
            raise ValueError(
                "hilbert_gate_initial_mask must be in [0, hilbert_gate_mask_max], "
                f"got {initial_mask} and {mask_max}"
            )
        self.mask_max = float(mask_max)
        self.weight = torch.nn.Parameter(
            torch.full((feature_dim,), float(initial_weight), device=device)
        )
        probability = (
            0.5
            if self.mask_max == 0.0
            else float(
                np.clip(initial_mask / self.mask_max, 1e-4, 1.0 - 1e-4)
            )
        )
        bias = float(np.log(probability / (1.0 - probability)))
        self.bias = torch.nn.Parameter(
            torch.tensor(bias, dtype=torch.float32, device=device)
        )

    def forward(
        self,
        features: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        logits = (features * self.weight).sum(dim=-1) + self.bias
        return self.mask_max * torch.sigmoid(logits) * active.unsqueeze(-1)


class HilbertFactorExtractor:
    def __init__(self, imf_count: int, workers: Optional[int] = None):
        if imf_count <= 0:
            raise ValueError(f"imf_count must be positive, got {imf_count}")
        self.imf_count = int(imf_count)
        if workers is None:
            try:
                workers = max(1, int(os.environ.get("DR_EMD_WORKERS", "1")))
            except ValueError as error:
                raise ValueError(
                    "DR_EMD_WORKERS must be a positive integer"
                ) from error
        self.workers = max(1, int(workers))
        self._executor: Optional[ProcessPoolExecutor] = None

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    @staticmethod
    def analytic_signal(signal: np.ndarray) -> np.ndarray:
        length = signal.shape[0]
        if length == 0:
            return signal.astype(np.complex64)
        spectrum = np.fft.fft(signal)
        multiplier = np.zeros(length, dtype=np.float32)
        multiplier[0] = 1.0
        if length % 2 == 0:
            multiplier[length // 2] = 1.0
            multiplier[1 : length // 2] = 2.0
        else:
            multiplier[1 : (length + 1) // 2] = 2.0
        return np.fft.ifft(spectrum * multiplier)

    @staticmethod
    def moving_average(values: np.ndarray, window: int) -> np.ndarray:
        if values.size == 0:
            return values.astype(np.float32, copy=False)
        window = max(1, min(int(window), values.shape[0]))
        if window == 1:
            return values.astype(np.float32, copy=False)
        left = window // 2
        right = window - 1 - left
        padded = np.pad(values, (left, right), mode="edge")
        kernel = np.full(window, 1.0 / window, dtype=np.float64)
        return np.convolve(padded, kernel, mode="valid").astype(
            np.float32, copy=False
        )

    @staticmethod
    def local_complex_innovation(
        analytic: np.ndarray,
        window: int,
    ) -> np.ndarray:
        epsilon = 1e-8
        analytic = np.asarray(analytic, dtype=np.complex128)
        if analytic.size < 2:
            return np.zeros(analytic.size, dtype=np.float32)
        previous = analytic[:-1]
        current = analytic[1:]
        amplitude = np.abs(analytic)
        reference_energy = float(np.median(amplitude) ** 2) + epsilon
        weights = np.minimum(
            1.0,
            np.abs(current) * np.abs(previous) / reference_energy,
        )
        window = max(1, min(int(window), current.size))
        left = window // 2
        right = window - 1 - left
        kernel = np.full(window, 1.0 / window, dtype=np.float64)

        def local_mean(values: np.ndarray) -> np.ndarray:
            return np.convolve(
                np.pad(values, (left, right), mode="edge"),
                kernel,
                mode="valid",
            )

        cross = local_mean(weights * current * np.conj(previous))
        previous_energy = np.real(local_mean(weights * np.abs(previous) ** 2))
        current_energy = np.real(local_mean(weights * np.abs(current) ** 2))
        residual_energy = np.maximum(
            current_energy
            - np.abs(cross) ** 2 / (previous_energy + epsilon),
            0.0,
        )
        pair_innovation = residual_energy / (current_energy + epsilon)
        innovation = np.empty(analytic.size, dtype=np.float32)
        innovation[0] = pair_innovation[0]
        innovation[1:] = pair_innovation.astype(np.float32, copy=False)
        return np.clip(innovation, 0.0, 1.0)

    @staticmethod
    def window_size(phase: np.ndarray) -> int:
        epsilon = 1e-6
        frequency = np.abs(np.gradient(phase))
        positive = frequency[frequency > epsilon]
        median_frequency = float(np.median(positive)) if positive.size else 1.0
        period = 2.0 * np.pi / max(median_frequency, epsilon)
        return int(
            np.clip(
                round(1.5 * period),
                5,
                max(5, phase.shape[0] // 3),
            )
        )

    def hilbert_features(
        self,
        imf: np.ndarray,
        imf_index: int,
    ) -> np.ndarray:
        epsilon = 1e-6
        length = imf.shape[0]
        if length == 0:
            return np.zeros((0, 4), dtype=np.float32)
        order = (
            1.0
            if self.imf_count == 1
            else 1.0 - imf_index / float(self.imf_count - 1)
        )
        zeros = np.zeros(length, dtype=np.float32)
        if length < 3:
            return np.stack(
                (zeros, zeros, zeros, np.full(length, order, dtype=np.float32)),
                axis=-1,
            )
        analytic = self.analytic_signal(imf.astype(np.float32, copy=False))
        amplitude = np.abs(analytic).astype(np.float32, copy=False)
        phase = np.unwrap(np.angle(analytic)).astype(np.float32, copy=False)
        order_feature = np.full(length, order, dtype=np.float32)
        window = self.window_size(phase)
        frequency = np.gradient(phase).astype(np.float32, copy=False)
        frequency_jump = np.abs(np.diff(frequency, prepend=frequency[:1]))
        previous_amplitude = np.concatenate((amplitude[:1], amplitude[:-1]))
        amplitude_reference = float(np.median(amplitude))
        reliability = np.minimum(
            1.0,
            amplitude
            * previous_amplitude
            / (amplitude_reference**2 + epsilon),
        )
        mean_reliability = self.moving_average(reliability, window)
        rms_jump = np.sqrt(
            self.moving_average(
                reliability * frequency_jump**2,
                window,
            )
            / (mean_reliability + epsilon)
        )
        mean_frequency = self.moving_average(
            reliability * np.abs(frequency),
            window,
        ) / (mean_reliability + epsilon)
        frequency_variation = (rms_jump / (mean_frequency + epsilon)).astype(
            np.float32, copy=False
        )
        relative_amplitude_jump = (
            2.0
            * np.abs(amplitude - previous_amplitude)
            / (amplitude + previous_amplitude + epsilon)
        )
        amplitude_variation = np.sqrt(
            self.moving_average(relative_amplitude_jump**2, window)
        ).astype(np.float32, copy=False)
        innovation = self.local_complex_innovation(analytic, window)
        matrix = np.stack(
            (frequency_variation, amplitude_variation, innovation, order_feature),
            axis=-1,
        )
        return np.nan_to_num(
            matrix,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)

    def extract_factors(
        self,
        signal: np.ndarray,
        emd: EMD,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = signal.shape[0]
        candidates = np.zeros((self.imf_count, length), dtype=np.float32)
        features = np.zeros(
            (self.imf_count, length, 4), dtype=np.float32
        )
        active = np.zeros(self.imf_count, dtype=np.float32)
        if length < 2:
            return candidates, features, active
        padding = min(int(round(length * 0.25)), length - 1)
        work_signal = np.pad(
            signal,
            (padding, padding),
            mode="reflect",
        ).astype(np.float32, copy=False)
        imfs = emd.emd(work_signal, max_imf=self.imf_count)
        end = padding + length
        for imf_index, imf in enumerate(imfs[: self.imf_count]):
            candidates[imf_index] = imf[padding:end]
            features[imf_index] = self.hilbert_features(imf, imf_index)[
                padding:end
            ]
            active[imf_index] = 1.0
        return candidates, features, active

    def extract_batch(
        self,
        signals: list[np.ndarray],
    ) -> list[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not signals:
            return []
        if self.workers == 1 or len(signals) == 1:
            emd = EMD(spline_kind="cubic")
            return [self.extract_factors(signal, emd) for signal in signals]
        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self.workers)
        jobs = (
            (
                np.ascontiguousarray(signal, dtype=np.float32),
                self.imf_count,
            )
            for signal in signals
        )
        chunk_size = max(1, len(signals) // (self.workers * 4))
        return list(
            self._executor.map(
                _extract_emd_factors_worker,
                jobs,
                chunksize=chunk_size,
            )
        )
