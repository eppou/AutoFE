import numpy as np
import pandas as pd
from typing import List, Callable, Dict, Optional

class TemporalAggregationEncoder:
    """
    Generic framework for temporal aggregation of time series data.

    This class transforms raw time series into aggregated temporal sequences,
    preserving temporal order while enriching each timestep with statistical summaries.
    """

    def __init__(
        self,
        window_size: int,
        aggregation_windows: List[int],
        aggregation_functions: Dict[str, Callable],
        stride: int = 1,
        drop_incomplete: bool = True,
    ):
        """
        Parameters
        ----------
        window_size : int
            Total length of the input time series (e.g., 180 days)

        aggregation_windows : List[int]
            Sizes of sub-windows used for aggregation (e.g., [3, 5, 7])

        aggregation_functions : Dict[str, Callable]
            Dictionary of aggregation functions.
            Example: {"mean": np.mean, "max": np.max}

        stride : int
            Step size when sliding over the series.

        output_format : str
            "sequence" -> output shape [T, F]
            "flatten"  -> output shape [T*F]

        drop_incomplete : bool
            Whether to drop windows that do not fully fit.
        """

        self.window_size = window_size
        self.aggregation_windows = aggregation_windows
        self.aggregation_functions = aggregation_functions
        self.stride = stride
        self.drop_incomplete = drop_incomplete

        self._validate_parameters()

    def _validate_parameters(self):
        if any(w <= 0 for w in self.aggregation_windows):
            raise ValueError("aggregation_windows must be positive integers")

        if self.stride <= 0:
            raise ValueError("stride must be a positive integer")

    def transform(self, series: np.ndarray) -> np.ndarray:
        """
        Causal and asymmetric temporal feature extraction with non-linear operators.
        """

        series = self._ensure_2d(series)

        if len(series) != self.window_size:
            raise ValueError(
                f"Expected series length {self.window_size}, got {len(series)}"
            )

        aggregated_steps = []
        max_w = max(self.aggregation_windows)

        # hiperparâmetros (podem virar argumentos depois)
        decay = 0.3
        critical_quantile = 0.9

        for t in range(max_w - 1, len(series), self.stride):
            step_features = []

            for w in self.aggregation_windows:
                segment = series[t - w + 1 : t + 1]  # CAUSAL

                # =========================
                # Pesos temporais (assimetria)
                # =========================
                weights = np.exp(-decay * np.arange(w)[::-1])
                weights = weights / weights.sum()

                # =========================
                # Estatísticas básicas
                # =========================
                mean = segment.mean(axis=0)
                std = segment.std(axis=0)
                vmin = segment.min(axis=0)
                vmax = segment.max(axis=0)

                # =========================
                # Estatísticas ponderadas
                # =========================
                wmean = np.sum(segment * weights[:, None], axis=0)
                wmax = np.max(segment * weights[:, None], axis=0)

                # =========================
                # Operadores não lineares
                # =========================
                rms = np.sqrt(np.mean(segment**2, axis=0))
                energy = np.sum(segment**2, axis=0)
                q90 = np.quantile(segment, critical_quantile, axis=0)
                value_range = vmax - vmin

                # =========================
                # Estados críticos / picos
                # =========================
                critical_mask = segment > q90
                time_above_q = critical_mask.sum(axis=0)

                # maior sequência contínua acima do limiar
                longest_run = []
                for j in range(segment.shape[1]):
                    runs = np.diff(
                        np.where(
                            np.concatenate(
                                ([critical_mask[0, j]],
                                critical_mask[:, j] != critical_mask[:-1, j],
                                [True])
                            )
                        )[0]
                    )[::2]
                    longest_run.append(runs.max() if len(runs) > 0 else 0)

                longest_run = np.array(longest_run)

                # =========================
                # Tempo desde último evento
                # =========================
                time_since_last = []
                for j in range(segment.shape[1]):
                    idx = np.where(critical_mask[:, j])[0]
                    if len(idx) == 0:
                        time_since_last.append(w)
                    else:
                        time_since_last.append(w - idx[-1] - 1)

                time_since_last = np.array(time_since_last)

                # =========================
                # Concatenar tudo
                # =========================
                features = np.concatenate([
                    mean, std, vmin, vmax,
                    wmean, wmax,
                    rms, energy, q90, value_range,
                    time_above_q, longest_run, time_since_last
                ])

                step_features.append(features)

            aggregated_steps.append(np.concatenate(step_features))

        return np.array(aggregated_steps)
    
    def _ensure_2d(self, series: np.ndarray) -> np.ndarray:
        if series.ndim == 1:
            return series[:, None]
        return series
