# #ckks加密
# import warnings
#
# import tenseal as ts
# import torch
#
# warnings.filterwarnings("ignore", category=UserWarning, module="tenseal")
#
#
# def get_ckks_context() -> ts.Context:
#     poly_modulus_degree = 8192
#     coeff_mod_bit_sizes = [60, 40, 40, 60]
#     context = ts.context(
#         ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, coeff_mod_bit_sizes
#     )
#     context.generate_galois_keys()
#     context.global_scale = 2**40
#     return context
#
#
# def ckks_consine_similarity(
#     context: ts.Context, t1: torch.Tensor, t2: torch.Tensor
# ) -> float:
#     t1_norm = t1 / torch.norm(t1)
#     t2_norm = t2 / torch.norm(t2)
#
#     list1 = t1_norm.flatten().tolist()
#     list2 = t2_norm.flatten().tolist()
#
#     t1_encrypted = ts.ckks_vector(context, list1)
#     t2_encrypted = ts.ckks_vector(context, list2)
#     cosine_similarity_encrypted = t1_encrypted.dot(t2_encrypted)
#
#     return cosine_similarity_encrypted
#
#
# context_ckks = get_ckks_context()













# """Cryptographic helpers with a TenSEAL-backed CKKS context and a safe fallback.
#
# This module preserves the original public interface:
# - get_ckks_context()
# - ckks_consine_similarity(context, t1, t2)
# - context_ckks
#
# When TenSEAL is unavailable, the fallback keeps the code runnable for development
# and unit testing, while preserving the same method signatures.
# """
#
# from __future__ import annotations
#
# import warnings
# from dataclasses import dataclass
# from typing import Iterable, List, Sequence
#
# import numpy as np
# import torch
#
# warnings.filterwarnings("ignore", category=UserWarning, module="tenseal")
#
# try:
#     import tenseal as ts  # type: ignore
# except Exception:  # pragma: no cover
#     ts = None
#
#
# @dataclass
# class _PlainCKKSScalar:
#     value: float
#
#     def decrypt(self):
#         return [float(self.value)]
#
#     def __float__(self):
#         return float(self.value)
#
#
# class _PlainCKKSVector:
#     """Minimal drop-in fallback for testing without TenSEAL.
#
#     It supports the small subset of operations used by util_fusion:
#     addition, scalar multiplication, dot product, size and decrypt.
#     """
#
#     def __init__(self, data: Sequence[float]):
#         self._data = np.asarray(list(data), dtype=np.float64).copy()
#
#     def __add__(self, other):
#         if isinstance(other, _PlainCKKSVector):
#             return _PlainCKKSVector(self._data + other._data)
#         return _PlainCKKSVector(self._data + np.asarray(other, dtype=np.float64))
#
#     def __radd__(self, other):
#         if other == 0:
#             return self
#         return self.__add__(other)
#
#     def __mul__(self, other):
#         if isinstance(other, _PlainCKKSVector):
#             return _PlainCKKSVector(self._data * other._data)
#         return _PlainCKKSVector(self._data * float(other))
#
#     def __rmul__(self, other):
#         return self.__mul__(other)
#
#     def dot(self, other):
#         if isinstance(other, _PlainCKKSVector):
#             value = float(np.dot(self._data, other._data))
#         else:
#             # TenSEAL vectors expose dot over encrypted/plain vector-like objects.
#             # In fallback mode, try to coerce the other operand to a flat array.
#             if hasattr(other, "decrypt"):
#                 other = other.decrypt()
#             value = float(np.dot(self._data, np.asarray(other, dtype=np.float64)))
#         return _PlainCKKSScalar(value)
#
#     def decrypt(self):
#         return self._data.tolist()
#
#     def size(self):
#         return self._data.size
#
#     def flatten(self):
#         return self
#
#     def tolist(self):
#         return self._data.tolist()
#
#
# def get_ckks_context():
#     """Create the CKKS context used by the secure aggregation stage.
#
#     If TenSEAL is installed, a real CKKS context is returned. Otherwise, None is
#     returned and the code will transparently fall back to a plaintext-compatible
#     mock vector implementation.
#     """
#     if ts is None:
#         return None
#
#     poly_modulus_degree = 8192
#     coeff_mod_bit_sizes = [60, 40, 40, 60]
#     context = ts.context(
#         ts.SCHEME_TYPE.CKKS,
#         poly_modulus_degree,
#         -1,
#         coeff_mod_bit_sizes,
#     )
#     context.generate_galois_keys()
#     context.global_scale = 2**40
#     return context
#
#
# def _vector_norm(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#     return t / (torch.norm(t) + eps)
#
#
# def ckks_consine_similarity(
#     context,
#     t1: torch.Tensor,
#     t2: torch.Tensor,
# ):
#     """Compute cosine similarity in encrypted form when possible.
#
#     The original repository spells this function as 'ckks_consine_similarity'.
#     """
#     t1_norm = _vector_norm(t1)
#     t2_norm = _vector_norm(t2)
#     list1 = t1_norm.flatten().detach().cpu().tolist()
#     list2 = t2_norm.flatten().detach().cpu().tolist()
#
#     if ts is None or context is None:
#         vec1 = _PlainCKKSVector(list1)
#         vec2 = _PlainCKKSVector(list2)
#         return vec1.dot(vec2)
#
#     t1_encrypted = ts.ckks_vector(context, list1)
#     t2_encrypted = ts.ckks_vector(context, list2)
#     return t1_encrypted.dot(t2_encrypted)
#
#
# context_ckks = get_ckks_context()

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

warnings.filterwarnings("ignore", category=UserWarning, module="tenseal")

try:
    import tenseal as ts  # type: ignore
except Exception:
    ts = None


@dataclass
class _PlainCKKSScalar:
    value: float

    def decrypt(self):
        return [float(self.value)]


class _PlainCKKSVector:
    def __init__(self, data: Sequence[float]):
        self._data = np.asarray(list(data), dtype=np.float64).copy()

    def __add__(self, other):
        if hasattr(other, "decrypt"):
            other = other.decrypt()
        if isinstance(other, _PlainCKKSVector):
            other = other._data
        return _PlainCKKSVector(self._data + np.asarray(other, dtype=np.float64))

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, _PlainCKKSVector):
            return _PlainCKKSVector(self._data * other._data)
        return _PlainCKKSVector(self._data * float(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def dot(self, other):
        if hasattr(other, "decrypt"):
            other = other.decrypt()
        if isinstance(other, _PlainCKKSVector):
            other = other._data
        return _PlainCKKSScalar(float(np.dot(self._data, np.asarray(other, dtype=np.float64))))

    def decrypt(self):
        return self._data.tolist()


def get_ckks_context():
    if ts is None:
        return None
    poly_modulus_degree = 8192
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree, -1, coeff_mod_bit_sizes)
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context


def ckks_consine_similarity(context, t1: torch.Tensor, t2: torch.Tensor):
    t1 = t1.flatten().float()
    t2 = t2.flatten().float()
    t1 = t1 / (torch.norm(t1) + 1e-12)
    t2 = t2 / (torch.norm(t2) + 1e-12)
    if context is None or ts is None:
        return _PlainCKKSScalar(float(torch.dot(t1, t2).item()))
    v1 = ts.ckks_vector(context, t1.detach().cpu().tolist())
    v2 = ts.ckks_vector(context, t2.detach().cpu().tolist())
    return v1.dot(v2)


context_ckks = get_ckks_context()
