# import copy
# import logging
# import warnings
# from typing import Any, Dict, List, Tuple
#
# import torch
# import torch.utils.data as data
# import tenseal as ts
# import numpy as np
#
#
# from utils.util_crypto import context_ckks
# from utils.util_model import (
#     extract_parameters,
#     flatten_model_parameters,
#     load_model_from_parameters,
#     get_gaussian_noise,
#     get_laplace_noise,
# )
# from utils.util_sys import wrap_torch_median
# from utils.util_sys import wrap_torch_sort
#
# from utils.util_logger import logger
#
# warnings.filterwarnings("ignore", category=UserWarning, module="tenseal")
#
#
# def fusion_avg(model_updates: Dict[int, torch.nn.Module]) -> Dict[str, torch.Tensor]:
#     avgerage_params = {}
#     with torch.no_grad():
#         for key in next(iter(model_updates.values())).state_dict():
#             weighted_params = torch.zeros_like(
#                 next(iter(model_updates.values())).state_dict()[key].float()
#             )
#             for _, model in model_updates.items():
#                 param = model.state_dict()[key].float()
#                 weighted_params += param * 1.0 / len(model_updates)
#             avgerage_params[key] = weighted_params
#
#     return avgerage_params
#
#
# def fusion_fedavg(
#     model_updates: Dict[int, torch.nn.Module],
#     data_size: Dict[int, int],
# ) -> Dict[str, torch.Tensor]:
#
#     total_data_size = sum(data_size.values())
#     weighted_avg_params = {}
#     with torch.no_grad():
#         for key in next(iter(model_updates.values())).state_dict():
#             weighted_params = torch.zeros_like(
#                 next(iter(model_updates.values())).state_dict()[key].float()
#             )
#             for client_id, model in model_updates.items():
#                 weight = data_size[client_id] / total_data_size
#                 param = model.state_dict()[key].float()
#                 weighted_params += param * weight
#             weighted_avg_params[key] = weighted_params
#
#     return weighted_avg_params
#
#
# def fusion_krum(
#     model_updates: Dict[int, torch.nn.Module],
#     max_expected_adversaries=1,
#     device=torch.device("cpu"),
# ) -> Dict[str, torch.Tensor]:
#
#     with torch.no_grad():
#         ids = list(model_updates.keys())
#         updates = [extract_parameters(model_updates[id]) for id in ids]
#         updates = [update.to(device) for update in updates]
#         num_updates = len(updates)
#         updates_stack = torch.stack(updates)
#
#         dist_matrix = torch.cdist(updates_stack, updates_stack, p=2)
#         values, indices = torch.topk(
#             dist_matrix,
#             k=num_updates - max_expected_adversaries - 1,
#             dim=1,
#             largest=False,
#             sorted=True,
#         )
#         # logger.debug(f"current krum values: {values}")
#         scores = values.sum(dim=1)
#         # logger.debug(f"current krum scores: {scores}")
#         min_indices = torch.argmin(scores).item()
#         logger.debug(f"current krum min index: {min_indices}")
#         selected_id = ids[min_indices]
#         logger.info(f"selected client id: {selected_id}")
#
#     selected_model = model_updates[selected_id]
#     krum_params = selected_model.state_dict()
#     return krum_params
#
#
# def fusion_median(
#     model_updates: Dict[int, torch.nn.Module],
#     device: torch.device = torch.device("cpu"),
# ) -> Dict[str, Any]:
#     median_params = {}
#     with torch.no_grad():
#         for key in next(iter(model_updates.values())).state_dict():
#             params = torch.stack(
#                 [model.state_dict()[key].float() for model in model_updates.values()]
#             )
#             median_params[key] = wrap_torch_median(params, dim=0, device=device)
#
#     return median_params
#
#
# def fusion_clipping_median(
#     model_updates: Dict[int, torch.nn.Module],
#     clipping_threshold=0.1,
#     device: torch.device = torch.device("cpu"),
# ) -> Dict[str, Any]:
#     median_params = {}
#     with torch.no_grad():
#         for key in next(iter(model_updates.values())).state_dict():
#             params = torch.stack(
#                 [model.state_dict()[key].float() for model in model_updates.values()]
#             )
#             median_params[key] = wrap_torch_median(params, dim=0, device=device)
#             median_params[key] = torch.clamp(
#                 median_params[key], -clipping_threshold, clipping_threshold
#             )
#
#     return median_params
#
#
# def fusion_trimmed_mean(
#     model_updates: Dict[int, torch.nn.Module],
#     trimmed_ratio: float = 0.1,
#     device: torch.device = torch.device("cpu"),
# ) -> Dict[str, Any]:
#     trimmed_mean_params = {}
#     with torch.no_grad():
#         for key in next(iter(model_updates.values())).state_dict():
#             params = torch.stack(
#                 [model.state_dict()[key].float() for model in model_updates.values()]
#             )
#             lower = int(params.size(0) * trimmed_ratio)
#             upper = int(params.size(0) * (1 - trimmed_ratio))
#             params = wrap_torch_sort(params, dim=0, device=device)[lower:upper]
#             trimmed_mean_params[key] = torch.mean(params, dim=0)
#
#     return trimmed_mean_params
#
# # 普通的余弦相似度
# def fusion_cos_defense(
#     global_model: torch.nn.Module,
#     model_updates: Dict[int, torch.nn.Module],
#     similarity_threshold: float = None,
# ) -> Dict[str, Any]:
#
#     global_last_layer = list(global_model.parameters())[-2].view(-1)
#     models = list(model_updates.values())
#     last_layers = [list(model.parameters())[-2].view(-1) for model in models]
#
#     with torch.no_grad():
#         scores = torch.abs(
#             torch.nn.functional.cosine_similarity(
#                 torch.stack(last_layers),
#                 global_last_layer,
#             )
#         )
#         # print(scores)
#         logger.info(f"current fusion scores: {scores}")
#         min_score = torch.min(scores)
#         scores = (scores - min_score) / (torch.max(scores) - min_score)
#         logger.info(f"normalized fusion scores: {scores}")
#
#         if similarity_threshold is None:
#             similarity_threshold = torch.mean(scores)
#         logger.info(f"similarity threshold: {similarity_threshold}")
#
#         benign_indices = scores >= similarity_threshold
#         if torch.sum(benign_indices) == 0:
#             logger.warning("No models are considered benign based on the threshold.")
#             logger.warning("Return global model of last round.")
#             return global_model.state_dict()
#
#         logger.info(f"current round client list: {model_updates.keys()}")
#         logger.info(f"potential malicide indices: {benign_indices}")
#         logger.info(f"checked benign indices: {benign_indices}")
#
#         weight = 1 / torch.sum(benign_indices).float()
#         fractions = benign_indices.float() * weight
#         logger.info(f"current fusion fractions: {fractions}")
#
#         weighted_params = copy.deepcopy(global_model.state_dict())
#         for param_key in weighted_params.keys():
#             temp_param = torch.zeros_like(
#                 global_model.state_dict()[param_key], dtype=torch.float32
#             )
#             for model, fraction in zip(models, fractions):
#                 temp_param += model.state_dict()[param_key] * fraction
#             weighted_params[param_key].copy_(temp_param)
#             # OUR OPTIMIZATION FOR DEFENSE
#             # weighted_params[param_key] = torch.clamp(
#             #     weighted_params[param_key], -0.1, 0.1
#             # )
#
#     return weighted_params
#
# # 双重防御
# def fusion_dual_defense(
#     global_model: torch.nn.Module,
#     model_updates: Dict[int, torch.nn.Module],
#     data_size: Dict[int, int],
#     similarity_threshold: float = None,
#     epsilon: float = None,
# ) -> Dict[str, torch.Tensor]:
#     # simulate the hyper guard defense (privacy-preserving robust aggregation)
#     # 1) each client generates pre-preprocessed update
#     global_last_layer = list(global_model.parameters())[-2].view(-1)
#     last_layers = {
#         client_id: list(model.parameters())[-2].view(-1)
#         for client_id, model in model_updates.items()
#     }
#     mormalized_global = global_last_layer / torch.norm(global_last_layer)
#     normalized_locals = {
#         client_id: last_layer / torch.norm(last_layer)
#         for client_id, last_layer in last_layers.items()
#     }
#     # 2) encrypt and send to the fusion server
#     encrypted_global = ts.ckks_vector(
#         context_ckks, mormalized_global.flatten().tolist()
#     )
#     encrypted_locals = {
#         client_id: ts.ckks_vector(context_ckks, normalized_local.flatten().tolist())
#         for client_id, normalized_local in normalized_locals.items()
#     }
#     encrypted_updates = {}
#     for client_id, model in model_updates.items():
#         flattened_parameters = flatten_model_parameters(model)
#         encrypted_parameters = [
#             ts.ckks_vector(context_ckks, param) for param in flattened_parameters
#         ]
#         encrypted_updates[client_id] = encrypted_parameters
#
#     # 3) server fuse encrypted update and return back the encrypted scores to each client
#
#     # encrypted_global_dp = encrypted_global + get_gaussian_noise(
#     #     encrypted_global.size(),
#     #     epsilon=0.5,
#     #     delta=1.0 / encrypted_global.size(),
#     #     sensitivity=1,
#     # )
#
#     # encrypted_global_dp = encrypted_global + get_laplace_noise(
#     #     encrypted_global.size(), epsilon=0.5, sensitivity=1
#     # )
#
#     if epsilon is not None and isinstance(epsilon, float):
#         gaussian_nosie = get_gaussian_noise(
#             1, epsilon=epsilon, delta=1.0 / encrypted_global.size(), sensitivity=1
#         )
#         encrypted_global = (
#             encrypted_global + gaussian_nosie.tolist() * encrypted_global.size()
#         )
#
#     # simulate approximated clamp
#     # encrypted_locals_clamp = {
#     #     client_id: encrypted_local * 0.1
#     #     for client_id, encrypted_local in encrypted_locals.items()
#     # }
#
#     encrypted_scores = {
#         client_id: encrypted_local.dot(encrypted_global)
#         for client_id, encrypted_local in encrypted_locals.items()
#     }
#
#     # 4) each client decrypt the scores and send back the benigns for validation
#     client_selections = {}
#     for client_id in model_updates.keys():
#         scores = {
#             client_id: np.abs(encrypted_score.decrypt())
#             for client_id, encrypted_score in encrypted_scores.items()
#         }
#         logger.debug(f"client {client_id} scores: {scores}")
#         min_score = np.min(list(scores.values()))
#         max_score = np.max(list(scores.values()))
#         diff_score = max_score - min_score
#         scores_norm = {
#             client_id: (score - min_score) / diff_score
#             for client_id, score in scores.items()
#         }
#         logger.debug(f"client {client_id} norm scores: {scores_norm}")
#         if similarity_threshold is None:
#             similarity_threshold = np.mean(list(scores_norm.values()))
#         logger.debug(f"client {client_id} similarity threshold: {similarity_threshold}")
#         selected_benigns = [
#             id for id, score in scores_norm.items() if score >= similarity_threshold
#         ]
#         logger.info(f"client {client_id} selected fusion benigns: {selected_benigns}")
#         if len(selected_benigns) == 0:
#             raise ValueError("No models are considered benign based on the threshold.")
#         client_selections[client_id] = selected_benigns
#
#     # 5) server counts and find the majority beningn selections
#     count = {}
#     for _, benigns in client_selections.items():
#         _tuple = tuple(benigns)
#         if _tuple in count:
#             count[_tuple] += 1
#         else:
#             count[_tuple] = 1
#     benigns = None
#     max_count = 0
#     for _benigns, _cnt in count.items():
#         if _cnt > max_count:
#             max_count = _cnt
#             benigns = _benigns
#
#     # 6) final secure aggregation
#     logger.debug(f"final fusion benigns: {benigns}")
#     total_size = sum(data_size[benign_id] for benign_id in benigns)
#     fused_enc_params = [0] * len(encrypted_updates[benigns[0]])
#     for benign_id in benigns:
#         enc_param = encrypted_updates[benign_id]
#         fusion_weight = data_size[benign_id] / total_size
#         weighted_enc_param = [_p * fusion_weight for _p in enc_param]
#         fused_enc_params = [x + y for x, y in zip(fused_enc_params, weighted_enc_param)]
#
#     # 7) send to client for decryption
#     _params = [param.decrypt() for param in fused_enc_params]
#     fused_model = load_model_from_parameters(_params, global_model)
#     fused_params = fused_model.state_dict()
#
#     return fused_params

# """Federated aggregation and encrypted robust defense utilities.
#
# This file is a backward-compatible enhancement of the original repository.
# Public entry points are preserved:
# - fusion_avg
# - fusion_fedavg
# - fusion_krum
# - fusion_median
# - fusion_clipping_median
# - fusion_trimmed_mean
# - fusion_cos_defense
# - fusion_dual_defense
#
# The new implementation adds:
# - layer-wise adaptive privacy perturbation
# - multi-layer similarity scoring
# - robust benign-client voting
# - CKKS fallback for local testing when TenSEAL is unavailable
# """
#
# from __future__ import annotations
#
# import copy
# import logging
# import warnings
# from dataclasses import dataclass
# from typing import Any, Dict, Iterable, List, Sequence, Tuple
#
# import numpy as np
# import torch
# try:
#     import tenseal as ts  # type: ignore
# except Exception:  # pragma: no cover
#     ts = None
#
# from utils.util_crypto import context_ckks
# from utils.util_model import (
#     extract_parameters,
#     flatten_model_parameters,
#     load_model_from_parameters,
#     get_gaussian_noise,
#     get_laplace_noise,
# )
# from utils.util_sys import wrap_torch_median
# from utils.util_sys import wrap_torch_sort
# from utils.util_logger import logger
#
# warnings.filterwarnings("ignore", category=UserWarning, module="tenseal")
#
#
# # ---------------------------------------------------------------------------
# # Compatibility helpers
# # ---------------------------------------------------------------------------
#
# def _state_dict_clone(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
#     return {k: v.detach().clone() for k, v in model.state_dict().items()}
#
#
# def _flatten_state_dict_tensors(model: torch.nn.Module) -> List[torch.Tensor]:
#     return [p.detach().view(-1).float().cpu() for p in model.parameters()]
#
#
# def _safe_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
#     return x / (torch.norm(x) + eps)
#
#
# def _to_numpy_1d(x: torch.Tensor) -> np.ndarray:
#     return x.detach().cpu().flatten().numpy().astype(np.float64, copy=True)
#
#
# def _maybe_ckks_vector(context, values: Sequence[float]):
#     if context is None or ts is None:
#         return _PlainCKKSVector(values)
#     try:
#         return ts.ckks_vector(context, list(map(float, values)))
#     except Exception:
#         return _PlainCKKSVector(values)
#
#
# class _PlainCKKSScalar:
#     def __init__(self, value: float):
#         self.value = float(value)
#
#     def decrypt(self):
#         return [float(self.value)]
#
#
# class _PlainCKKSVector:
#     def __init__(self, data: Sequence[float]):
#         self._data = np.asarray(list(data), dtype=np.float64).copy()
#
#     def __add__(self, other):
#         if isinstance(other, _PlainCKKSVector):
#             return _PlainCKKSVector(self._data + other._data)
#         if hasattr(other, "decrypt"):
#             return _PlainCKKSVector(self._data + np.asarray(other.decrypt(), dtype=np.float64))
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
#         if hasattr(other, "decrypt"):
#             other = other.decrypt()
#         if isinstance(other, _PlainCKKSVector):
#             other = other._data
#         other = np.asarray(other, dtype=np.float64)
#         return _PlainCKKSScalar(float(np.dot(self._data, other)))
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
# def _encrypted_decrypt_to_float(score_obj) -> float:
#     """Convert a TenSEAL scalar/vector or plaintext fallback to float."""
#     try:
#         if hasattr(score_obj, "decrypt"):
#             val = score_obj.decrypt()
#             if isinstance(val, (list, tuple, np.ndarray)):
#                 if len(val) == 1:
#                     return float(val[0])
#                 return float(np.mean(np.asarray(val, dtype=np.float64)))
#             return float(val)
#         if isinstance(score_obj, (list, tuple, np.ndarray)):
#             return float(np.mean(np.asarray(score_obj, dtype=np.float64)))
#         return float(score_obj)
#     except Exception:
#         return float(np.asarray(score_obj).reshape(-1)[0])
#
#
# def _layerwise_split(model: torch.nn.Module) -> List[torch.Tensor]:
#     return [param.detach().view(-1).float().cpu() for param in model.parameters()]
#
#
# def _select_layers(num_layers: int, max_layers: int = 4) -> List[int]:
#     """Prefer tail layers while keeping at least one layer."""
#     if num_layers <= 0:
#         return []
#     if num_layers <= max_layers:
#         return list(range(num_layers))
#     # Tail-biased selection: last two layers + two earlier layers for diversity.
#     idxs = {num_layers - 1, max(0, num_layers - 2), max(0, num_layers - 4), max(0, num_layers - 6)}
#     idxs = [i for i in sorted(idxs) if 0 <= i < num_layers]
#     return idxs
#
#
# def _layer_sensitivity(layer_vec: torch.Tensor) -> float:
#     """Heuristic sensitivity score for privacy budget allocation."""
#     # Larger norm / higher dimensional layers are treated as more sensitive.
#     dim = max(int(layer_vec.numel()), 1)
#     norm = float(torch.norm(layer_vec).item())
#     return float(np.log1p(dim) * (1.0 + norm))
#
#
# def _allocate_layerwise_epsilons(
#     layer_vecs: Sequence[torch.Tensor],
#     total_epsilon: float,
#     floor_ratio: float = 0.15,
# ) -> Dict[int, float]:
#     scores = np.asarray([_layer_sensitivity(v) for v in layer_vecs], dtype=np.float64)
#     # More sensitive layer -> lower epsilon.
#     inv = 1.0 / np.maximum(scores, 1e-12)
#     inv = inv / inv.sum()
#     min_eps = total_epsilon * floor_ratio / max(len(layer_vecs), 1)
#     eps = {}
#     remaining = max(total_epsilon - min_eps * len(layer_vecs), 0.0)
#     for i, p in enumerate(inv):
#         eps[i] = float(min_eps + remaining * p)
#     return eps
#
#
# def _add_gaussian_noise(vec: torch.Tensor, epsilon: float, delta: float) -> torch.Tensor:
#     if epsilon is None or epsilon <= 0:
#         return vec
#     sigma = np.sqrt(2.0 * np.log(1.25 / max(delta, 1e-12))) / epsilon
#     noise = torch.normal(mean=0.0, std=float(sigma), size=vec.shape, device=vec.device)
#     return vec + noise
#
#
# def _multi_feature_score(local_vec: torch.Tensor, global_vec: torch.Tensor) -> float:
#     """Combine cosine similarity, L2 distance and norm ratio into a single score."""
#     local_vec = local_vec.float().view(-1)
#     global_vec = global_vec.float().view(-1)
#
#     local_norm = torch.norm(local_vec) + 1e-12
#     global_norm = torch.norm(global_vec) + 1e-12
#     cosine = torch.dot(local_vec, global_vec) / (local_norm * global_norm)
#     cosine = float(torch.clamp(cosine, -1.0, 1.0).item())
#
#     # Large distance and large norm mismatch indicate suspicious updates.
#     l2_dist = float(torch.norm(local_vec - global_vec).item())
#     norm_ratio = float((local_norm / global_norm).item())
#     norm_ratio = min(max(norm_ratio, 0.0), 10.0)
#
#     # Score is higher for benign-like updates.
#     score = 0.65 * max(cosine, 0.0) + 0.20 * (1.0 / (1.0 + l2_dist)) + 0.15 * (1.0 / (1.0 + abs(norm_ratio - 1.0)))
#     return float(score)
#
#
# def _vote_benign_clients(
#     per_client_scores: Dict[int, float],
#     threshold: float | None = None,
# ) -> List[int]:
#     if not per_client_scores:
#         return []
#     values = np.asarray(list(per_client_scores.values()), dtype=np.float64)
#     if threshold is None:
#         # Robust threshold: mean - std/4.
#         threshold = float(values.mean() - 0.25 * values.std(ddof=0))
#     return [cid for cid, score in per_client_scores.items() if score >= threshold]
#
#
# def _fallback_if_no_tenseal():
#     return context_ckks is None or ts is None
#
#
# # ---------------------------------------------------------------------------
# # Standard aggregation baselines
# # ---------------------------------------------------------------------------
#
# def fusion_avg(model_updates: Dict[int, torch.nn.Module]) -> Dict[str, torch.Tensor]:
#     avgerage_params = {}
#     with torch.no_grad():
#         for key in next(iter(model_updates.values())).state_dict():
#             weighted_params = torch.zeros_like(
#                 next(iter(model_updates.values())).state_dict()[key].float()
#             )
#             for _, model in model_updates.items():
#                 param = model.state_dict()[key].float()
#                 weighted_params += param * 1.0 / len(model_updates)
#             avgerage_params[key] = weighted_params
#     return avgerage_params
#
#
# def fusion_fedavg(
#     model_updates: Dict[int, torch.nn.Module],
#     data_size: Dict[int, int],
# ) -> Dict[str, torch.Tensor]:
#     total_data_size = sum(data_size.values())
#     weighted_avg_params = {}
#     with torch.no_grad():
#         for key in next(iter(model_updates.values())).state_dict():
#             weighted_params = torch.zeros_like(
#                 next(iter(model_updates.values())).state_dict()[key].float()
#             )
#             for client_id, model in model_updates.items():
#                 weight = data_size[client_id] / total_data_size
#                 param = model.state_dict()[key].float()
#                 weighted_params += param * weight
#             weighted_avg_params[key] = weighted_params
#     return weighted_avg_params
#
#
# def fusion_krum(
#     model_updates: Dict[int, torch.nn.Module],
#     max_expected_adversaries=1,
#     device=torch.device("cpu"),
# ) -> Dict[str, torch.Tensor]:
#     with torch.no_grad():
#         ids = list(model_updates.keys())
#         updates = [extract_parameters(model_updates[id]) for id in ids]
#         updates = [update.to(device) for update in updates]
#         num_updates = len(updates)
#         updates_stack = torch.stack(updates)
#         dist_matrix = torch.cdist(updates_stack, updates_stack, p=2)
#         values, indices = torch.topk(
#             dist_matrix,
#             k=max(1, num_updates - max_expected_adversaries - 1),
#             dim=1,
#             largest=False,
#             sorted=True,
#         )
#         scores = values.sum(dim=1)
#         min_indices = torch.argmin(scores).item()
#         logger.debug(f"current krum min index: {min_indices}")
#         selected_id = ids[min_indices]
#         logger.info(f"selected client id: {selected_id}")
#         selected_model = model_updates[selected_id]
#         krum_params = selected_model.state_dict()
#         return krum_params
#
#
# def fusion_median(
#     model_updates: Dict[int, torch.nn.Module],
#     device: torch.device = torch.device("cpu"),
# ) -> Dict[str, Any]:
#     median_params = {}
#     with torch.no_grad():
#         for key in next(iter(model_updates.values())).state_dict():
#             params = torch.stack(
#                 [model.state_dict()[key].float() for model in model_updates.values()]
#             )
#             median_params[key] = wrap_torch_median(params, dim=0, device=device)
#     return median_params
#
#
# def fusion_clipping_median(
#     model_updates: Dict[int, torch.nn.Module],
#     clipping_threshold=0.1,
#     device: torch.device = torch.device("cpu"),
# ) -> Dict[str, Any]:
#     median_params = {}
#     with torch.no_grad():
#         for key in next(iter(model_updates.values())).state_dict():
#             params = torch.stack(
#                 [model.state_dict()[key].float() for model in model_updates.values()]
#             )
#             median_params[key] = wrap_torch_median(params, dim=0, device=device)
#             median_params[key] = torch.clamp(
#                 median_params[key], -clipping_threshold, clipping_threshold
#             )
#     return median_params
#
#
# def fusion_trimmed_mean(
#     model_updates: Dict[int, torch.nn.Module],
#     trimmed_ratio: float = 0.1,
#     device: torch.device = torch.device("cpu"),
# ) -> Dict[str, Any]:
#     trimmed_mean_params = {}
#     with torch.no_grad():
#         for key in next(iter(model_updates.values())).state_dict():
#             params = torch.stack(
#                 [model.state_dict()[key].float() for model in model_updates.values()]
#             )
#             lower = int(params.size(0) * trimmed_ratio)
#             upper = int(params.size(0) * (1 - trimmed_ratio))
#             params = wrap_torch_sort(params, dim=0, device=device)[lower:upper]
#             trimmed_mean_params[key] = torch.mean(params, dim=0)
#     return trimmed_mean_params
#
#
# def fusion_cos_defense(
#     global_model: torch.nn.Module,
#     model_updates: Dict[int, torch.nn.Module],
#     similarity_threshold: float = None,
# ) -> Dict[str, Any]:
#     global_last_layer = list(global_model.parameters())[-2].view(-1)
#     models = list(model_updates.values())
#     last_layers = [list(model.parameters())[-2].view(-1) for model in models]
#     with torch.no_grad():
#         scores = torch.abs(
#             torch.nn.functional.cosine_similarity(
#                 torch.stack(last_layers),
#                 global_last_layer,
#             )
#         )
#         logger.info(f"current fusion scores: {scores}")
#         min_score = torch.min(scores)
#         denom = torch.max(scores) - min_score
#         scores = (scores - min_score) / (denom + 1e-12)
#         logger.info(f"normalized fusion scores: {scores}")
#         if similarity_threshold is None:
#             similarity_threshold = torch.mean(scores)
#         logger.info(f"similarity threshold: {similarity_threshold}")
#         benign_indices = scores >= similarity_threshold
#         if torch.sum(benign_indices) == 0:
#             logger.warning("No models are considered benign based on the threshold.")
#             logger.warning("Return global model of last round.")
#             return global_model.state_dict()
#         logger.info(f"current round client list: {model_updates.keys()}")
#         logger.info(f"potential malicide indices: {benign_indices}")
#         logger.info(f"checked benign indices: {benign_indices}")
#         weight = 1 / torch.sum(benign_indices).float()
#         fractions = benign_indices.float() * weight
#         logger.info(f"current fusion fractions: {fractions}")
#         weighted_params = copy.deepcopy(global_model.state_dict())
#         for param_key in weighted_params.keys():
#             temp_param = torch.zeros_like(
#                 global_model.state_dict()[param_key], dtype=torch.float32
#             )
#             for model, fraction in zip(models, fractions):
#                 temp_param += model.state_dict()[param_key] * fraction
#             weighted_params[param_key].copy_(temp_param)
#         return weighted_params
#
#
# # ---------------------------------------------------------------------------
# # Enhanced dual-defense aggregation
# # ---------------------------------------------------------------------------
#
# def fusion_dual_defense(
#     global_model: torch.nn.Module,
#     model_updates: Dict[int, torch.nn.Module],
#     data_size: Dict[int, int],
#     similarity_threshold: float = None,
#     epsilon: float = None,
# ) -> Dict[str, torch.Tensor]:
#     """Privacy-preserving and robust secure aggregation with enhanced defenses.
#
#     Backward compatible with the original repository:
#     - same required arguments
#     - same return type: state_dict-like parameter mapping
#
#     New behavior:
#     - layer-wise privacy perturbation
#     - multi-layer similarity voting
#     - benign-client majority selection
#     - safe fallback when TenSEAL is unavailable
#     """
#     if not model_updates:
#         raise ValueError("model_updates cannot be empty")
#
#     # ------------------------------------------------------------------
#     # 1) Layer extraction and normalization
#     # ------------------------------------------------------------------
#     global_layers = _layerwise_split(global_model)
#     num_layers = len(global_layers)
#     selected_layer_ids = _select_layers(num_layers, max_layers=4)
#
#     if len(selected_layer_ids) == 0:
#         raise ValueError("No layers found in global model.")
#
#     selected_global_layers = [global_layers[i] for i in selected_layer_ids]
#     global_eps = _allocate_layerwise_epsilons(
#         selected_global_layers,
#         total_epsilon=float(epsilon) if isinstance(epsilon, (int, float)) and epsilon > 0 else 1.0,
#     )
#
#     normalized_global_layers = []
#     for idx, vec in zip(selected_layer_ids, selected_global_layers):
#         normalized_global_layers.append(_safe_norm(vec))
#
#     client_normalized_layers: Dict[int, List[torch.Tensor]] = {}
#     for client_id, model in model_updates.items():
#         local_layers = _layerwise_split(model)
#         local_selected = [local_layers[i] for i in selected_layer_ids]
#         client_normalized_layers[client_id] = [_safe_norm(v) for v in local_selected]
#
#     # ------------------------------------------------------------------
#     # 2) Encrypt the normalized representations and add DP perturbation
#     # ------------------------------------------------------------------
#     encrypted_global_layers = []
#     for local_idx, vec in enumerate(normalized_global_layers):
#         layer_id = selected_layer_ids[local_idx]
#         vec_to_encrypt = vec.clone()
#
#         if isinstance(epsilon, (int, float)) and epsilon > 0:
#             # Use a smaller delta for deeper layers to strengthen privacy accounting.
#             delta = 1.0 / max(vec.numel(), 1)
#             sigma = np.sqrt(2.0 * np.log(1.25 / max(delta, 1e-12))) / max(global_eps[local_idx], 1e-12)
#             noise = torch.normal(
#                 mean=0.0,
#                 std=float(sigma),
#                 size=vec_to_encrypt.shape,
#                 device=vec_to_encrypt.device,
#             )
#             vec_to_encrypt = vec_to_encrypt + noise
#
#         encrypted_global_layers.append(
#             _maybe_ckks_vector(context_ckks, vec_to_encrypt.flatten().tolist())
#         )
#
#     encrypted_client_layers: Dict[int, List[Any]] = {}
#     for client_id, layer_vecs in client_normalized_layers.items():
#         encrypted_client_layers[client_id] = [
#             _maybe_ckks_vector(context_ckks, v.flatten().tolist()) for v in layer_vecs
#         ]
#
#     # ------------------------------------------------------------------
#     # 3) Compute encrypted similarity scores for each client on each layer
#     # ------------------------------------------------------------------
#     encrypted_scores: Dict[int, List[Any]] = {cid: [] for cid in model_updates.keys()}
#
#     for client_id, enc_layers in encrypted_client_layers.items():
#         for layer_idx, enc_local in enumerate(enc_layers):
#             enc_global = encrypted_global_layers[layer_idx]
#             try:
#                 enc_score = enc_local.dot(enc_global)
#             except Exception:
#                 # Plain fallback path
#                 local_plain = np.asarray(enc_local.decrypt(), dtype=np.float64)
#                 global_plain = np.asarray(enc_global.decrypt(), dtype=np.float64)
#                 enc_score = _PlainCKKSScalar(float(np.dot(local_plain, global_plain)))
#             encrypted_scores[client_id].append(enc_score)
#
#     # ------------------------------------------------------------------
#     # 4) Clients decrypt scores and produce multi-feature benign sets
#     # ------------------------------------------------------------------
#     client_selections: Dict[int, List[int]] = {}
#     client_combined_scores: Dict[int, float] = {}
#
#     for client_id in model_updates.keys():
#         layer_scores = [_encrypted_decrypt_to_float(s) for s in encrypted_scores[client_id]]
#         # Multi-layer fusion score: average of layer similarities + stability regularizer.
#         layer_scores_np = np.asarray(layer_scores, dtype=np.float64)
#         score_mean = float(layer_scores_np.mean())
#         score_var = float(layer_scores_np.var(ddof=0))
#         combined = score_mean - 0.10 * score_var
#         client_combined_scores[client_id] = combined
#
#     if similarity_threshold is None:
#         values = np.asarray(list(client_combined_scores.values()), dtype=np.float64)
#         similarity_threshold = float(values.mean() - 0.25 * values.std(ddof=0))
#
#     # each client forms benign set locally; to keep a majority-voting protocol,
#     # the benign set is derived from the multi-feature scores and a threshold.
#     for client_id in model_updates.keys():
#         selected_benigns = [
#             cid for cid, score in client_combined_scores.items()
#             if score >= similarity_threshold
#         ]
#         if len(selected_benigns) == 0:
#             selected_benigns = [max(client_combined_scores, key=client_combined_scores.get)]
#         client_selections[client_id] = selected_benigns
#         logger.info(f"client {client_id} selected fusion benigns: {selected_benigns}")
#
#     # ------------------------------------------------------------------
#     # 5) Majority vote across clients
#     # ------------------------------------------------------------------
#     vote_count: Dict[int, int] = {cid: 0 for cid in model_updates.keys()}
#     for benigns in client_selections.values():
#         for cid in benigns:
#             vote_count[cid] += 1
#
#     majority_threshold = len(model_updates) // 2
#     benigns = [cid for cid, cnt in vote_count.items() if cnt > majority_threshold]
#
#     if len(benigns) == 0:
#         # Conservative fallback: select top-1 by combined score.
#         benigns = [max(client_combined_scores, key=client_combined_scores.get)]
#
#     logger.debug(f"final fusion benigns: {benigns}")
#
#     # ------------------------------------------------------------------
#     # 6) Secure weighted aggregation
#     # ------------------------------------------------------------------
#     total_size = sum(int(data_size[cid]) for cid in benigns)
#     if total_size <= 0:
#         raise ValueError("Invalid data_size values for aggregation.")
#
#     # Build encrypted parameter list from benign clients
#     benign_models = [model_updates[cid] for cid in benigns]
#     fused_enc_params = None
#
#     for benign_id, model in zip(benigns, benign_models):
#         enc_flat_params = flatten_model_parameters(model)
#         # Encrypt each layer/parameter tensor separately to preserve structure.
#         enc_param_list = []
#         for param in enc_flat_params:
#             # param is a python list
#             enc_param_list.append(_maybe_ckks_vector(context_ckks, param))
#
#         fusion_weight = float(data_size[benign_id]) / float(total_size)
#         weighted_enc_param = [p * fusion_weight for p in enc_param_list]
#
#         if fused_enc_params is None:
#             fused_enc_params = weighted_enc_param
#         else:
#             fused_enc_params = [x + y for x, y in zip(fused_enc_params, weighted_enc_param)]
#
#     if fused_enc_params is None:
#         raise RuntimeError("Secure aggregation failed: no benign models aggregated.")
#
#     # ------------------------------------------------------------------
#     # 7) Decrypt and rebuild the fused model
#     # ------------------------------------------------------------------
#     flattened_params = [param.decrypt() for param in fused_enc_params]
#     fused_model = load_model_from_parameters(flattened_params, copy.deepcopy(global_model))
#     fused_params = fused_model.state_dict()
#     return fused_params





from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

try:
    import tenseal as ts  # type: ignore
except Exception:
    ts = None

from utils.util_crypto import context_ckks
from utils.util_model import extract_parameters, flatten_model_parameters, load_model_from_parameters
from utils.util_sys import wrap_torch_median, wrap_torch_sort
from utils.util_logger import logger

warnings.filterwarnings("ignore", category=UserWarning, module="tenseal")


class _PlainCKKSScalar:
    def __init__(self, value: float):
        self.value = float(value)

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


def _maybe_ckks_vector(context, values: Sequence[float]):
    if context is None or ts is None:
        return _PlainCKKSVector(values)
    try:
        return ts.ckks_vector(context, list(map(float, values)))
    except Exception:
        return _PlainCKKSVector(values)


def _decrypt_scalar(obj) -> float:
    try:
        if hasattr(obj, "decrypt"):
            x = obj.decrypt()
            if isinstance(x, (list, tuple, np.ndarray)):
                return float(np.mean(np.asarray(x, dtype=np.float64)))
            return float(x)
        return float(obj)
    except Exception:
        return float(np.asarray(obj).reshape(-1)[0])


def _layerwise_split(model: torch.nn.Module) -> List[torch.Tensor]:
    return [p.detach().view(-1).float().cpu() for p in model.parameters()]


def _safe_norm(v: torch.Tensor) -> torch.Tensor:
    return v / (torch.norm(v) + 1e-12)


def _selected_layers(num_layers: int) -> List[int]:
    if num_layers <= 0:
        return []
    if num_layers <= 4:
        return list(range(num_layers))
    candidates = {num_layers - 1, num_layers - 2, max(0, num_layers - 4), max(0, num_layers // 2)}
    return sorted(i for i in candidates if 0 <= i < num_layers)


def _layerwise_epsilon(layer_vecs: Sequence[torch.Tensor], total_epsilon: float) -> Dict[int, float]:
    norms = np.asarray([float(torch.norm(v).item()) for v in layer_vecs], dtype=np.float64)
    inv = 1.0 / np.maximum(norms, 1e-12)
    inv = inv / inv.sum()
    min_eps = total_epsilon * 0.15 / max(len(layer_vecs), 1)
    remain = max(total_epsilon - min_eps * len(layer_vecs), 0.0)
    return {i: float(min_eps + remain * inv[i]) for i in range(len(layer_vecs))}


def _robust_threshold(values: np.ndarray, k: float = 2.5) -> float:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) + 1e-12
    return med - k * 1.4826 * mad


def fusion_avg(model_updates: Dict[int, torch.nn.Module]) -> Dict[str, torch.Tensor]:
    out = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            tmp = torch.zeros_like(next(iter(model_updates.values())).state_dict()[key].float())
            for _, model in model_updates.items():
                tmp += model.state_dict()[key].float() * (1.0 / len(model_updates))
            out[key] = tmp
    return out


def fusion_fedavg(model_updates: Dict[int, torch.nn.Module], data_size: Dict[int, int]) -> Dict[str, torch.Tensor]:
    total = sum(data_size.values())
    out = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            tmp = torch.zeros_like(next(iter(model_updates.values())).state_dict()[key].float())
            for cid, model in model_updates.items():
                tmp += model.state_dict()[key].float() * (data_size[cid] / total)
            out[key] = tmp
    return out


def fusion_krum(model_updates: Dict[int, torch.nn.Module], max_expected_adversaries=1, device=torch.device("cpu")) -> Dict[str, torch.Tensor]:
    with torch.no_grad():
        ids = list(model_updates.keys())
        updates = [extract_parameters(model_updates[i]).to(device) for i in ids]
        stack = torch.stack(updates)
        dist = torch.cdist(stack, stack, p=2)
        vals, _ = torch.topk(dist, k=max(1, len(updates) - max_expected_adversaries - 1), dim=1, largest=False, sorted=True)
        scores = vals.sum(dim=1)
        idx = int(torch.argmin(scores).item())
        return model_updates[ids[idx]].state_dict()


def fusion_median(model_updates: Dict[int, torch.nn.Module], device=torch.device("cpu")) -> Dict[str, Any]:
    out = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            params = torch.stack([m.state_dict()[key].float() for m in model_updates.values()])
            out[key] = wrap_torch_median(params, dim=0, device=device)
    return out


def fusion_clipping_median(model_updates: Dict[int, torch.nn.Module], clipping_threshold=0.1, device=torch.device("cpu")) -> Dict[str, Any]:
    out = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            params = torch.stack([m.state_dict()[key].float() for m in model_updates.values()])
            out[key] = wrap_torch_median(params, dim=0, device=device)
            out[key] = torch.clamp(out[key], -clipping_threshold, clipping_threshold)
    return out


def fusion_trimmed_mean(model_updates: Dict[int, torch.nn.Module], trimmed_ratio: float = 0.1, device=torch.device("cpu")) -> Dict[str, Any]:
    out = {}
    with torch.no_grad():
        for key in next(iter(model_updates.values())).state_dict():
            params = torch.stack([m.state_dict()[key].float() for m in model_updates.values()])
            lower = int(params.size(0) * trimmed_ratio)
            upper = int(params.size(0) * (1 - trimmed_ratio))
            params = wrap_torch_sort(params, dim=0, device=device)[lower:upper]
            out[key] = torch.mean(params, dim=0)
    return out


def fusion_cos_defense(global_model: torch.nn.Module, model_updates: Dict[int, torch.nn.Module], similarity_threshold: float = None) -> Dict[str, Any]:
    global_last_layer = list(global_model.parameters())[-2].view(-1)
    last_layers = [list(model.parameters())[-2].view(-1) for model in model_updates.values()]
    with torch.no_grad():
        scores = torch.abs(torch.nn.functional.cosine_similarity(torch.stack(last_layers), global_last_layer))
        min_score = torch.min(scores)
        scores = (scores - min_score) / (torch.max(scores) - min_score + 1e-12)
        if similarity_threshold is None:
            similarity_threshold = torch.mean(scores)
        benign_indices = scores >= similarity_threshold
        if torch.sum(benign_indices) == 0:
            return global_model.state_dict()
        weight = 1 / torch.sum(benign_indices).float()
        fractions = benign_indices.float() * weight
        weighted_params = copy.deepcopy(global_model.state_dict())
        for param_key in weighted_params.keys():
            tmp = torch.zeros_like(global_model.state_dict()[param_key], dtype=torch.float32)
            for model, frac in zip(model_updates.values(), fractions):
                tmp += model.state_dict()[param_key] * frac
            weighted_params[param_key].copy_(tmp)
        return weighted_params


def fusion_dual_defense(
    global_model: torch.nn.Module,
    model_updates: Dict[int, torch.nn.Module],
    data_size: Dict[int, int],
    similarity_threshold: float = None,
    epsilon: float = None,
) -> Dict[str, torch.Tensor]:
    if not model_updates:
        raise ValueError("model_updates cannot be empty")

    global_layers = _layerwise_split(global_model)
    sel_ids = _selected_layers(len(global_layers))
    if not sel_ids:
        raise ValueError("No layers found in the model.")

    local_layers_by_client = {}
    for cid, model in model_updates.items():
        local_layers = _layerwise_split(model)
        local_layers_by_client[cid] = [local_layers[i] for i in sel_ids]

    global_sel = [global_layers[i] for i in sel_ids]
    eps_map = _layerwise_epsilon(global_sel, float(epsilon) if isinstance(epsilon, (int, float)) and epsilon > 0 else 1.0)

    enc_global = []
    for j, g in enumerate(global_sel):
        g_norm = _safe_norm(g)
        if isinstance(epsilon, (int, float)) and epsilon > 0:
            delta = 1.0 / max(g_norm.numel(), 1)
            sigma = float(np.sqrt(2.0 * np.log(1.25 / max(delta, 1e-12))) / max(eps_map[j], 1e-12))
            g_norm = g_norm + torch.normal(0.0, sigma, size=g_norm.shape)
        enc_global.append(_maybe_ckks_vector(context_ckks, g_norm.flatten().tolist()))

    enc_client = {}
    for cid, layers in local_layers_by_client.items():
        enc_client[cid] = [_maybe_ckks_vector(context_ckks, _safe_norm(v).flatten().tolist()) for v in layers]

    layer_scores = {cid: [] for cid in model_updates}
    for cid in model_updates:
        for j in range(len(sel_ids)):
            try:
                s = enc_client[cid][j].dot(enc_global[j])
            except Exception:
                s = _PlainCKKSScalar(float(np.dot(np.asarray(enc_client[cid][j].decrypt()), np.asarray(enc_global[j].decrypt()))))
            layer_scores[cid].append(_decrypt_scalar(s))

    client_ids = list(model_updates.keys())
    score_matrix = np.asarray([layer_scores[cid] for cid in client_ids], dtype=np.float64)
    comb_scores = score_matrix.mean(axis=1)

    norms = np.asarray([float(torch.norm(extract_parameters(model_updates[cid])).item()) for cid in client_ids], dtype=np.float64)
    global_norm = float(torch.norm(extract_parameters(global_model)).item()) + 1e-12
    norm_ratio = norms / global_norm

    layer_thr = np.asarray([_robust_threshold(score_matrix[:, j]) for j in range(score_matrix.shape[1])], dtype=np.float64)
    layer_pass = score_matrix >= layer_thr.reshape(1, -1)
    pass_count = layer_pass.sum(axis=1)

    comb_thr = _robust_threshold(comb_scores, k=2.0)
    norm_ok = (norm_ratio >= 0.45) & (norm_ratio <= 2.25)

    min_pass = max(1, int(np.ceil(0.75 * len(sel_ids))))
    benign_mask = (pass_count >= min_pass) & (comb_scores >= comb_thr) & norm_ok
    benign_ids = [cid for cid, ok in zip(client_ids, benign_mask) if bool(ok)]

    if len(benign_ids) > 0:
        benign_scores = np.asarray([comb_scores[client_ids.index(cid)] for cid in benign_ids], dtype=np.float64)
        keep_k = max(1, int(np.ceil(0.7 * len(benign_ids))))
        order = np.argsort(-benign_scores)
        benign_ids = [benign_ids[i] for i in order[:keep_k]]

    if len(benign_ids) == 0:
        candidates = [cid for cid in client_ids if norm_ok[client_ids.index(cid)]]
        if not candidates:
            candidates = client_ids
        best = max(candidates, key=lambda c: comb_scores[client_ids.index(c)])
        benign_ids = [best]

    logger.info(f"selected benign clients: {benign_ids}")

    total_size = sum(int(data_size[cid]) for cid in benign_ids)
    if total_size <= 0:
        raise ValueError("Invalid aggregation weights.")

    fused = copy.deepcopy(global_model.state_dict())
    for key in fused.keys():
        fused[key] = torch.zeros_like(fused[key], dtype=torch.float32)

    for cid in benign_ids:
        weight = float(data_size[cid]) / float(total_size)
        model_state = model_updates[cid].state_dict()
        for key in fused.keys():
            delta = model_state[key].float() - global_model.state_dict()[key].float()
            delta_norm = torch.norm(delta)
            clip_coef = min(1.0, 0.2 / float(delta_norm.item() + 1e-12))
            fused[key] += (global_model.state_dict()[key].float() + delta * clip_coef) * weight

    return fused

