# numpy array to bytes and back

import numpy as np
from io import BytesIO


def numpy_to_bytes(arr: np.ndarray) -> bytes:
    with BytesIO() as f:
        np.save(f, arr)
        return f.getvalue()


def bytes_to_numpy(b: bytes) -> np.ndarray:
    with BytesIO(b) as f:
        return np.load(f)
