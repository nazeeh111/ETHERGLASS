"""A generated-only example of the fixed synthetic waveform-error bound.

No input capture or selected sample window is accepted here. Coverage is marginal
and conditional on exchangeability with the declared simulator distribution.
"""
from __future__ import annotations
import hashlib
import json
import secrets
from pathlib import Path

import numpy as np

from .synthetic import generate_example
from .separation import separate
from .uncertainty import ensemble_indicator
from .synthetic_bound_guard import verify_binding, record_sha256


def generated_example():
    """Draw a fresh full-frame example; do not accept caller-supplied samples or seed."""
    record = verify_binding()
    # Avoid overlap with the recorded calibration and held-out seed ranges.
    while True:
        seed = secrets.randbits(63)
        if not 900000 <= seed <= 910095:
            break
    example = generate_example(seed, n=4096, split='test')
    x = example['mixture']
    if len(x) != 4096:
        raise ValueError('Synthetic bound unavailable: generated frame length changed.')
    estimate = separate(x, 1e6, target_center=0, symbol_rate=125000, method='hybrid')['components'][0]
    indicator = ensemble_indicator(x, 1e6, 0, 125000, estimate=estimate)
    if indicator['status'] != 'indicator':
        raise ValueError('Synthetic bound unavailable: expected uncalibrated ensemble indicator.')
    scale = indicator['error_scale_power']
    return {'status': 'synthetic-conditional-bound', 'seed': seed, 'sample_count': 4096,
            'mixture_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
            'sample_rate_hz': 1000000, 'target_center_hz': 0, 'symbol_rate_baud': 125000,
            'waveform_error_rms_upper': float(np.sqrt(record['quantile'] * scale)),
            'observed_synthetic_error_rms': float(np.sqrt(np.mean(np.abs(estimate-example['target'])**2))),
            'nominal_marginal_coverage': 0.9, 'calibration_examples': 96,
            'calibration_record_sha256': record_sha256(),
            'scope': 'Fresh full-frame synthetic shifted PSK and chirp generated internally; no imported data or sample selection.',
            'limitations': 'Coverage is marginal under exchangeability with the declared simulator population and fixed models. Not physical RF performance or an identifiability guarantee.'}
