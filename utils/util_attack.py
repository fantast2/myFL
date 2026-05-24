from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import copy
import random

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PoisonConfig:
    attack_type: str = "sign_flip"     # sign_flip | scale | gaussian | label_flip | random_model | adaptive_mimic | collusive
    attacker_ratio: float = 0.3
    scale_factor: float = 10.0
    gaussian_std: float = 0.5
    label_flip_map: Optional[Dict[int, int]] = None
    mimic_alpha: float = 0.15
    collusive_strength: float = 0.6
    seed: int = 2026


def set_attack_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_malicious_clients(client_ids: Sequence[int], attacker_ratio: float, round_idx: int, seed: int = 2026) -> List[int]:
    """
    Deterministic malicious client sampler.
    """
    set_attack_seed(seed + round_idx)
    client_ids = list(client_ids)
    if len(client_ids) == 0:
        return []
    k = max(1, int(round(len(client_ids) * attacker_ratio)))
    k = min(k, len(client_ids))
    return random.sample(client_ids, k)


def _clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _state_dict_to_model_like(src_state: Dict[str, torch.Tensor], template_model: nn.Module) -> nn.Module:
    """
    Load a state dict-like tensor map into a copied template model.
    """
    model = copy.deepcopy(template_model)
    model.load_state_dict(src_state, strict=True)
    return model


def _add_gaussian_noise_to_state(state: Dict[str, torch.Tensor], std: float) -> Dict[str, torch.Tensor]:
    poisoned = {}
    for k, v in state.items():
        if torch.is_floating_point(v):
            noise = torch.normal(0.0, std, size=v.shape, device=v.device, dtype=v.dtype)
            poisoned[k] = v + noise
        else:
            poisoned[k] = v.clone()
    return poisoned


def _sign_flip_state(state: Dict[str, torch.Tensor], factor: float) -> Dict[str, torch.Tensor]:
    poisoned = {}
    for k, v in state.items():
        if torch.is_floating_point(v):
            poisoned[k] = -factor * v
        else:
            poisoned[k] = v.clone()
    return poisoned


def _scale_state(state: Dict[str, torch.Tensor], factor: float) -> Dict[str, torch.Tensor]:
    poisoned = {}
    for k, v in state.items():
        if torch.is_floating_point(v):
            poisoned[k] = factor * v
        else:
            poisoned[k] = v.clone()
    return poisoned


def _random_model_state(template_model: nn.Module, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """
    Replace a local update with random parameters sampled around zero.
    """
    ref_state = template_model.state_dict()
    poisoned = {}
    for k, v in ref_state.items():
        if torch.is_floating_point(v):
            t = torch.randn_like(v)
            if device is not None:
                t = t.to(device)
            poisoned[k] = t
        else:
            poisoned[k] = v.clone()
    return poisoned


def _adaptive_mimic_state(
    honest_states: List[Dict[str, torch.Tensor]],
    target_state: Dict[str, torch.Tensor],
    alpha: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """
    Malicious client imitates the centroid of honest updates while embedding a small
    adversarial drift. This is useful for testing robust similarity-based defenses.
    """
    if len(honest_states) == 0:
        return _clone_state_dict_from_target(target_state)

    centroid = {}
    for k in target_state.keys():
        if torch.is_floating_point(target_state[k]):
            stacked = torch.stack([s[k].float() for s in honest_states], dim=0)
            centroid[k] = stacked.mean(dim=0)
        else:
            centroid[k] = target_state[k].clone()

    poisoned = {}
    for k in target_state.keys():
        if torch.is_floating_point(target_state[k]):
            drift = torch.randn_like(target_state[k]) * alpha
            poisoned[k] = centroid[k] + drift
        else:
            poisoned[k] = target_state[k].clone()
    return poisoned


def _clone_state_dict_from_target(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state.items()}


def _collusive_poison_states(
    honest_states: List[Dict[str, torch.Tensor]],
    malicious_template: Dict[str, torch.Tensor],
    strength: float = 0.6,
) -> Dict[str, torch.Tensor]:
    """
    Craft a collusive update that is close to honest centroid but systematically shifted.
    """
    if len(honest_states) == 0:
        return _clone_state_dict_from_target(malicious_template)

    poisoned = {}
    for k in malicious_template.keys():
        if torch.is_floating_point(malicious_template[k]):
            stacked = torch.stack([s[k].float() for s in honest_states], dim=0)
            centroid = stacked.mean(dim=0)
            # Shift direction is chosen to remain stealthy but consistent across colluders.
            direction = torch.sign(torch.randn_like(centroid))
            poisoned[k] = centroid + strength * 0.05 * direction
        else:
            poisoned[k] = malicious_template[k].clone()
    return poisoned


def poison_local_update(
    attack_type: str,
    local_model: nn.Module,
    honest_model: Optional[nn.Module] = None,
    honest_peer_states: Optional[List[Dict[str, torch.Tensor]]] = None,
    config: Optional[PoisonConfig] = None,
) -> nn.Module:
    """
    Return a poisoned local model for evaluation.

    Parameters
    ----------
    attack_type:
        One of: sign_flip, scale, gaussian, random_model, adaptive_mimic, collusive
    local_model:
        The trained local model that will be corrupted for attack evaluation.
    honest_model:
        Optional template model for random replacement.
    honest_peer_states:
        Optional list of honest peer states, used by adaptive/collusive attacks.
    config:
        PoisonConfig controlling attack strength.
    """
    cfg = config or PoisonConfig()
    state = _clone_state_dict(local_model)
    attack_type = attack_type.lower().strip()

    if attack_type == "sign_flip":
        poisoned = _sign_flip_state(state, cfg.scale_factor)

    elif attack_type == "scale":
        poisoned = _scale_state(state, cfg.scale_factor)

    elif attack_type == "gaussian":
        poisoned = _add_gaussian_noise_to_state(state, cfg.gaussian_std)

    elif attack_type == "random_model":
        if honest_model is None:
            raise ValueError("random_model attack requires honest_model as template.")
        poisoned = _random_model_state(honest_model, device=next(honest_model.parameters()).device)

    elif attack_type == "adaptive_mimic":
        if honest_peer_states is None:
            raise ValueError("adaptive_mimic attack requires honest_peer_states.")
        poisoned = _adaptive_mimic_state(
            honest_peer_states=honest_peer_states,
            target_state=state,
            alpha=cfg.mimic_alpha,
        )

    elif attack_type == "collusive":
        if honest_peer_states is None:
            raise ValueError("collusive attack requires honest_peer_states.")
        poisoned = _collusive_poison_states(
            honest_states=honest_peer_states,
            malicious_template=state,
            strength=cfg.collusive_strength,
        )

    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")

    local_model.load_state_dict(poisoned, strict=True)
    return local_model


def label_flip_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    label_flip_map: Optional[Dict[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Flip labels for a minibatch. This is useful when you want to simulate
    poisoned local training data rather than post-training model poisoning.
    """
    if label_flip_map is None:
        label_flip_map = {i: (i + 1) % num_classes for i in range(num_classes)}

    y_new = y.clone()
    for src, dst in label_flip_map.items():
        y_new[y == src] = dst
    return x, y_new


def make_malicious_client_plan(
    client_ids: Sequence[int],
    round_idx: int,
    attacker_ratio: float,
    seed: int = 2026,
) -> List[int]:
    """
    A deterministic malicious-client schedule for reproducible experiments.
    """
    return select_malicious_clients(client_ids, attacker_ratio, round_idx, seed=seed)