"""Minimal TorchSig PSK generator, extracted from upstream MIT source.

Upstream commit: 1197261d4f0d4c93de18757ee2ceb368ac04104c
Source: https://github.com/TorchDSP/torchsig
Copyright (c) 2022-2025 TorchSig. See TORCHSIG_LICENSE.txt.

Modifications: isolated the baseband helper and four DSP helpers from package
imports; retained only BPSK/QPSK/16QAM constellation maps. Function bodies are
unchanged. No full TorchSig installation is required. This is an upstream
helper extraction, not an independent implementation or a TorchSig benchmark.
"""
from __future__ import annotations
from copy import copy
import numpy as np
import scipy.signal as sp
TorchSigComplexDataType = np.complex64
all_symbol_maps = {
    "bpsk": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 2), 0j))),
    "qpsk": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 2), 1j * np.linspace(-1, 1, 2)))),
    "16qam": np.add(*map(np.ravel, np.meshgrid(np.linspace(-1, 1, 4), 1j * np.linspace(-1, 1, 4)))),
}


def slice_tail_to_length(input_signal: np.ndarray, num_samples: int) -> np.ndarray:
    """Slices the tail of a signal

    Args:
        input_signal (np.ndarray): Input signal
        num_samples (int): Maximum number of samples for a signal

    Raises:
        ValueError: If signal is too short to be sliced

    Returns:
        np.ndarray: Signal with length sliced to `num_samples`
    """
    output_signal = copy(input_signal)

    # too many samples, requires slicing
    if len(output_signal) > num_samples:
        # slice off the tail
        output_signal = output_signal[0:num_samples]

    # too few samples, return error
    elif len(output_signal) < num_samples:
        raise ValueError("signal too short to be sliced")

    # else: valid length, do nothing

    return output_signal

def pad_head_tail_to_length(input_signal: np.ndarray, num_samples: int) -> np.ndarray:
    """Zero pads the head and tail of a signal

    Args:
        input_signal (np.ndarray): Input signal
        num_samples (int): Desired length of signal

    Raises:
        ValueError: If signal is too long to be padded

    Returns:
        np.ndarray: Signal with length padded to `num_samples`
    """
    output_signal = copy(input_signal)

    # signal is too short, so zero pad
    if len(output_signal) < num_samples:
        # zero-pad to proper length
        num_zeros = num_samples - len(output_signal)
        num_zeros_begin = int(np.ceil(num_zeros / 2))
        num_zeros_end = int(np.floor(num_zeros / 2))
        output_signal = np.concatenate((np.zeros(num_zeros_begin), output_signal, np.zeros(num_zeros_end)))

    # signal is too long, throw error
    elif len(output_signal) > num_samples:
        raise ValueError("signal is too long to be zero padded")

    # else: signal correct length, do nothing

    return output_signal

def estimate_filter_length(transition_bandwidth: float, attenuation_db: float, sample_rate: float) -> int:
    """Estimates FIR filter length

    Estimate the length of an FIR filter using fred harris' approximation,
    Multirate Signal Processing for Communication Systems, Second Edition, p.59.

    Args:
        transition_bandwidth (float): The transition bandwidth of the filter,
            0 < `transition_bandwidth` < sample_rate/2.
        attenuation_db (float): Sidelobe attenuation level in dB.
        sample_rate (float): The sampling rate associated with the filter design.

    Returns:
        int: The estimated filter length
    """
    filter_length = int(np.round((sample_rate / transition_bandwidth) * (attenuation_db / 22)))

    # odd-length filters are desirable because they do not introduce a half-sample delay
    if np.equal(np.mod(filter_length, 2), 0):
        filter_length += 1

    return filter_length

def srrc_taps(iq_samples_per_symbol: int, filter_span_in_symbols: int, alpha: float = 0.35) -> np.ndarray:
    """Designs square-root raised cosine (SRRC) pulse shaping filter

    Args:
        iq_samples_per_symbol (int): The samples-per-symbol (SPS) of the underlying modulation,
            equivalent to the oversampling rate.
        filter_span_in_symbols (int): The filter span in number of symbols.
        alpha (float, optional): The alpha roll-off value of the pulse shaping filter, which
            is the amount of excess bandwidth. Defaults to 0.35.

    Returns:
        np.ndarray: SRRC filter weights
    """
    m = filter_span_in_symbols
    n_s = float(iq_samples_per_symbol)
    n = np.arange(-m * n_s, m * n_s + 1)
    taps = np.zeros(int(2 * m * n_s + 1))
    for i in range(int(2 * m * n_s + 1)):
        # handle the discontinuity at t=+-n_s/(4*alpha)
        if n[i] * 4 * alpha == n_s or n[i] * 4 * alpha == -n_s:
            taps[i] = (
                1
                / 2.0
                * (
                    (1 + alpha) * np.sin((1 + alpha) * np.pi / (4.0 * alpha))
                    - (1 - alpha) * np.cos((1 - alpha) * np.pi / (4.0 * alpha))
                    + (4 * alpha) / np.pi * np.sin((1 - alpha) * np.pi / (4.0 * alpha))
                )
            )
        else:
            taps[i] = 4 * alpha / (np.pi * (1 - 16 * alpha**2 * (n[i] / n_s) ** 2))
            taps[i] = taps[i] * (np.cos((1 + alpha) * np.pi * n[i] / n_s) + np.sinc((1 - alpha) * n[i] / n_s) * (1 - alpha) * np.pi / (4.0 * alpha))
    return taps

def constellation_modulator_baseband(
    constellation_name: str,
    pulse_shape_name: str,
    max_num_samples: int,
    oversampling_rate_nominal: int,
    alpha_rolloff: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Modulates constellation based signals (QAM/PSK/ASK/OOK) at complex baseband.

    Args:
        constellation_name: Name of the signal to modulate (e.g., 'qpsk').
        pulse_shape_name: Pulse shaping filter selection ('rectangular' or 'srrc').
        max_num_samples: Maximum number of samples to be produced.
        oversampling_rate_nominal: The amount of oversampling (sampling_rate/bandwidth).
        alpha_rolloff: The alpha-rolloff value for the SRRC filter (0 < alpha_rolloff < 1).
        rng: Random number generator for reproducibility. If None, creates a new default generator.

    Returns:
        np.ndarray: IQ samples of the constellation-modulated complex baseband signal.

    Raises:
        ValueError: If pulse_shape_name is neither 'rectangular' or 'srrc'.
        ValueError: If alpha_rolloff is not defined when selecting 'srrc'.
        ValueError: If max_num_samples or oversampling_rate_nominal are not positive.
    """
    # Input validation
    if max_num_samples <= 0:
        raise ValueError("max_num_samples must be positive")
    if oversampling_rate_nominal <= 0:
        raise ValueError("oversampling_rate_nominal must be positive")

    # Create random number generator if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Get symbol map and normalize to unit power
    symbol_map = all_symbol_maps[constellation_name]
    symbol_map = symbol_map / np.sqrt(np.mean(np.abs(symbol_map) ** 2))

    # Pulse shaping
    samples_per_symbol = oversampling_rate_nominal

    if pulse_shape_name == "rectangular":
        pulse_shape = np.ones(samples_per_symbol)
        pulse_shape_filter_span = 0
    elif pulse_shape_name == "srrc":
        if alpha_rolloff is None:
            raise ValueError("must define an alpha rolloff for SRRC filter")
        if not 0 < alpha_rolloff < 1:
            raise ValueError("alpha_rolloff must be between 0 and 1")

        attenuation_db = 120
        pulse_shape_filter_length = estimate_filter_length(alpha_rolloff, attenuation_db, 1)
        pulse_shape_filter_span = int(np.ceil((pulse_shape_filter_length - 1) / (2 * samples_per_symbol)))
        pulse_shape = srrc_taps(samples_per_symbol, pulse_shape_filter_span, alpha_rolloff)
    else:
        raise ValueError(f"pulse shape {pulse_shape_name} not supported")

    # Calculate number of symbols to generate
    subtract_off_symbols = 2 * pulse_shape_filter_span
    num_symbols = int(np.floor(max_num_samples / samples_per_symbol)) - subtract_off_symbols
    num_symbols = max(num_symbols, 1)

    # Generate symbols (handle OOK case where symbols might be zero)
    symbols = np.zeros(1)
    while np.equal(np.sum(np.abs(symbols)), 0):
        map_index = rng.integers(low=0, high=len(symbol_map), size=num_symbols)
        symbols = symbol_map[map_index]

    # Apply pulse shaping
    constellation_signal_baseband = sp.upfirdn(pulse_shape, symbols, up=samples_per_symbol, down=1)

    # Adjust signal length
    if len(constellation_signal_baseband) < max_num_samples:
        constellation_signal_baseband = pad_head_tail_to_length(constellation_signal_baseband, max_num_samples)
    elif len(constellation_signal_baseband) > max_num_samples:
        constellation_signal_baseband = slice_tail_to_length(constellation_signal_baseband, max_num_samples)

    return constellation_signal_baseband.astype(TorchSigComplexDataType)
