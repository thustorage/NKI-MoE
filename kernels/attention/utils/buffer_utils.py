"""Utility functions for safe tensor buffer checks."""

import nki.language as nl


def is_tensor_in_sbuf(tensor) -> bool:
    """Check if a tensor's buffer is nl.sbuf.

    Args:
        tensor: An NKI tensor.

    Returns:
        True if tensor is allocated in SBUF, False otherwise.
    """
    return tensor.buffer == nl.sbuf