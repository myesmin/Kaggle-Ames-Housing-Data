"""Platform workarounds, isolated here so they are visible rather than scattered.

One entry so far.
"""
from __future__ import annotations

import warnings

import numpy as np

_MATMUL_MESSAGES = ("divide by zero encountered in matmul",
                    "overflow encountered in matmul",
                    "invalid value encountered in matmul")


def silence_spurious_matmul_warnings() -> None:
    """Suppress the false floating-point warnings NumPy raises under Apple Accelerate.

    On macOS, NumPy's default BLAS is Accelerate, whose SIMD kernels leave FPU exception
    flags set even when the computation is exact. NumPy reads those flags after the call
    and reports ``divide by zero``/``overflow``/``invalid value`` for matmuls on entirely
    finite input. The results are correct; only the diagnostic is wrong.

    Verified before suppressing: a 300x300 row-stochastic weights matrix and a 1941x237
    Ridge design matrix, both provably finite, both trigger it (see the note in
    ``avm.morans_i``).  The filters are matmul-specific and the ``np.seterr`` call is
    scoped to the three flags Accelerate corrupts, so real numerical problems elsewhere
    still surface.

    Called at the top of every notebook, and set in ``PYTHONWARNINGS`` where joblib
    spawns worker processes that do not inherit an in-process filter.
    """
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    for message in _MATMUL_MESSAGES:
        warnings.filterwarnings("ignore", message=message, category=RuntimeWarning)


def worker_warning_filter() -> str:
    """The equivalent filter as a ``PYTHONWARNINGS`` value for subprocess workers.

    ``PYTHONWARNINGS`` matches the message as a literal prefix, not a regex, so each
    variant has to be listed in full.
    """
    return ",".join(f"ignore:{m}:RuntimeWarning" for m in _MATMUL_MESSAGES)
