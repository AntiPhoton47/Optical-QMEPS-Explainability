#!/usr/bin/env python3
"""
Standalone photonic quantum-MEPS animal classification experiment.

- explicit layered memory clips: photonic percept -> functionality -> family -> species
- sequential pairwise deliberation between adjacent layers
- repeated projection/conditioning onto the next active layer after each hop
- trainable inter-layer couplings that play the role of MEPS edge strengths
- multi-photon percept encoding in truncated bosonic Fock space

Two learning schemes are exposed:
- `qfim`: finite-difference gradient with quantum-Fisher natural preconditioning
- `variational`: plain finite-difference descent on the same variational loss
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import itertools
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh, expm

## Basic linear algebra utilities

def dagger(x: np.ndarray) -> np.ndarray:
    # Hermitian adjoint used throughout for quantum states and operators.
    return np.conjugate(x.T)


def normalize_state(psi: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    # All pure-state constructors funnel through here so downstream code can
    # safely assume column-vector, unit-norm kets.
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1, 1)
    nrm = float(np.linalg.norm(psi))
    if nrm < eps:
        raise ValueError("Cannot normalize near-zero state.")
    return psi / nrm


def ket_to_dm(psi: np.ndarray) -> np.ndarray:
    psi = normalize_state(psi)
    return psi @ dagger(psi)


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = x - np.max(x)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(np.asarray(x, dtype=float), eps, None))


def parse_int_list(spec: str) -> Tuple[int, ...]:
    # CLI helper for comma-separated layer/state selections.
    if not spec.strip():
        return ()
    return tuple(int(part.strip()) for part in spec.split(",") if part.strip())


## Core data structures for the photonic quantum-MEPS animal classification experiment


@dataclass
class QMEPSLayerSpec:
    name: str
    size: int


@dataclass
class QMEPSLayerParams:
    onsite: np.ndarray
    hopping: np.ndarray
    density_coupling: np.ndarray


class BaseEnv:
    def reset(self) -> Any:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, Any]]:
        raise NotImplementedError


@dataclass
class Transition:
    percept: Tuple[int, ...]
    action: int
    reward: float
    done: bool
    log_prob: float
    entropy: float
    policy: np.ndarray
    info: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class LearningConfig:
    learning_mode: str = "qfim"
    gamma: float = 0.97
    lr_policy: float = 0.04
    entropy_bonus: float = 0.01
    forgetting: float = 0.02
    return_scale: float = 0.20
    return_scale_decay: float = 0.995
    return_scale_min: float = 0.02
    fd_eps: float = 5e-4
    qfim_eps: float = 1e-4
    qfim_reg: float = 5e-3
    temperature: float = 0.9
    replay_capacity: int = 512
    replay_batch_size: int = 16
    replay_alpha: float = 0.6

    def validate(self) -> None:
        valid = {"variational", "qfim"}
        if self.learning_mode not in valid:
            raise ValueError(f"learning_mode must be one of {sorted(valid)}")


@dataclass(frozen=True)
class DetectorMeasurementConfig:
    layers: Tuple[int, ...] = ()
    states: Tuple[int, ...] = ()
    max_measurements: Optional[int] = None

    def normalized(self, num_layers: int, detector_cutoff: int) -> "DetectorMeasurementConfig":
        layers = tuple(sorted({int(x) for x in self.layers if 0 <= int(x) < num_layers - 1}))
        states = tuple(sorted({int(x) for x in self.states if 0 <= int(x) <= detector_cutoff}))
        max_measurements = None if self.max_measurements is None else max(0, int(self.max_measurements))
        return DetectorMeasurementConfig(layers=layers, states=states, max_measurements=max_measurements)


@dataclass(frozen=True)
class AnimalItem:
    name: str
    properties: Tuple[str, ...]
    functionality: Tuple[str, ...]
    family: str


## Simple on-policy rollout buffer and off-policy experience replay buffer implementations.


class RolloutBuffer:
    def __init__(self) -> None:
        # This short-lived buffer stores the most recent episode before its
        # transitions are promoted into the replay memory.
        self.data: List[Transition] = []

    def append(self, tr: Transition) -> None:
        self.data.append(tr)

    def clear(self) -> None:
        self.data.clear()

    def __len__(self) -> int:
        return len(self.data)

    def discounted_returns(self, gamma: float) -> np.ndarray:
        out = np.zeros(len(self.data), dtype=float)
        G = 0.0
        for i in reversed(range(len(self.data))):
            if self.data[i].done:
                G = 0.0
            G = self.data[i].reward + gamma * G
            out[i] = G
        return out


class ExperienceReplayBuffer:
    def __init__(self, capacity: int = 512, alpha: float = 0.6, seed: int = 0) -> None:
        # Prioritized replay keeps high-signal transitions around longer and
        # samples them more often during finite-difference updates.
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.rng = np.random.default_rng(seed)
        self.data: List[Transition] = []
        self.priorities: List[float] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, transition: Transition, priority: float) -> None:
        priority = float(max(priority, 1e-8))
        if len(self.data) < self.capacity:
            self.data.append(transition)
            self.priorities.append(priority)
        else:
            self.data[self.pos] = transition
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray]:
        if len(self.data) == 0:
            return [], np.array([], dtype=float)
        # Priority^alpha determines the sampling distribution, while the
        # returned weights partially undo that bias inside the loss.
        batch_size = min(int(batch_size), len(self.data))
        probs = np.asarray(self.priorities, dtype=float) ** self.alpha
        probs /= np.sum(probs)
        indices = self.rng.choice(len(self.data), size=batch_size, replace=False, p=probs)
        sampled = [self.data[int(i)] for i in indices]
        weights = 1.0 / np.maximum(probs[indices], 1e-12)
        weights = weights / np.max(weights)
        return sampled, weights.astype(float)


## Dataset and environment for the photonic quantum-MEPS animal classification experiment.


class AnimalDataset:
    def __init__(self) -> None:
        # The task is intentionally small and structured: percept properties,
        # intermediate functionality/family clips, then final species actions.
        self.properties: List[str] = [
            "fur", "feathers", "scales", "gills", "fins", "wings", "tail", "shell",
            "four_legs", "no_legs", "hooves", "stripes",
            "lays_eggs", "mammal", "bird", "reptile", "fish",
            "can_fly", "aquatic", "domestic", "wild", "herbivore", "carnivore", "omnivore",
            "small", "large",
        ]
        self.functionality: List[str] = [
            "pet", "predator", "prey", "flying", "swimming", "terrestrial",
            "domesticated", "wildlife", "grazer", "scavenger",
        ]
        self.families: List[str] = [
            "Felidae", "Canidae", "Accipitridae", "Columbidae", "Selachimorpha",
            "Salmonidae", "Serpentes", "Testudines", "Bovidae", "Equidae",
        ]
        self.animals: List[AnimalItem] = [
            AnimalItem("cat", ("fur", "tail", "four_legs", "mammal", "carnivore", "small", "domestic"),
                       ("pet", "predator", "terrestrial", "domesticated"), "Felidae"),
            AnimalItem("dog", ("fur", "tail", "four_legs", "mammal", "omnivore", "domestic"),
                       ("pet", "terrestrial", "domesticated", "predator"), "Canidae"),
            AnimalItem("wolf", ("fur", "tail", "four_legs", "mammal", "carnivore", "wild", "large"),
                       ("predator", "terrestrial", "wildlife"), "Canidae"),
            AnimalItem("tiger", ("fur", "tail", "four_legs", "mammal", "carnivore", "stripes", "wild", "large"),
                       ("predator", "terrestrial", "wildlife"), "Felidae"),
            AnimalItem("eagle", ("feathers", "wings", "bird", "can_fly", "lays_eggs", "carnivore", "wild"),
                       ("flying", "predator", "wildlife"), "Accipitridae"),
            AnimalItem("pigeon", ("feathers", "wings", "bird", "can_fly", "lays_eggs", "omnivore", "small"),
                       ("flying", "prey", "wildlife"), "Columbidae"),
            AnimalItem("shark", ("gills", "fins", "fish", "aquatic", "carnivore", "wild", "large"),
                       ("swimming", "predator", "wildlife"), "Selachimorpha"),
            AnimalItem("salmon", ("gills", "fins", "fish", "aquatic", "omnivore", "wild"),
                       ("swimming", "prey", "wildlife"), "Salmonidae"),
            AnimalItem("snake", ("scales", "reptile", "lays_eggs", "no_legs", "carnivore", "wild"),
                       ("predator", "terrestrial", "wildlife"), "Serpentes"),
            AnimalItem("turtle", ("scales", "reptile", "lays_eggs", "shell", "aquatic", "omnivore"),
                       ("swimming", "prey", "terrestrial"), "Testudines"),
            AnimalItem("cow", ("fur", "tail", "four_legs", "mammal", "herbivore", "domestic", "large", "hooves"),
                       ("grazer", "prey", "terrestrial", "domesticated"), "Bovidae"),
            AnimalItem("horse", ("fur", "tail", "four_legs", "mammal", "herbivore", "domestic", "large", "hooves"),
                       ("terrestrial", "domesticated"), "Equidae"),
        ]
        self.prop2idx = {name: idx for idx, name in enumerate(self.properties)}
        self.func2idx = {name: idx for idx, name in enumerate(self.functionality)}
        self.family2idx = {name: idx for idx, name in enumerate(self.families)}
        self.species = [item.name for item in self.animals]
        self.species2idx = {name: idx for idx, name in enumerate(self.species)}
        self.photonic_percept_modes: List[str] = [
            "coat_surface",
            "aquatic_body",
            "appendages",
            "skeleton_gait",
            "taxonomy_birth",
            "habitat_behavior",
            "diet",
            "size_pattern",
        ]
        self.property_to_mode: Dict[str, int] = {
            # Raw symbolic properties are compressed into a smaller photonic
            # percept register so the optical memory remains tractable.
            "fur": 0, "feathers": 0, "scales": 0,
            "gills": 1, "fins": 1, "shell": 1,
            "wings": 2, "tail": 2,
            "four_legs": 3, "no_legs": 3, "hooves": 3,
            "lays_eggs": 4, "mammal": 4, "bird": 4, "reptile": 4, "fish": 4,
            "can_fly": 5, "aquatic": 5, "domestic": 5, "wild": 5,
            "herbivore": 6, "carnivore": 6, "omnivore": 6,
            "small": 7, "large": 7, "stripes": 7,
        }

    def encode_photonic_percept(self, item: AnimalItem) -> Tuple[int, ...]:
        active = sorted({self.property_to_mode[name] for name in item.properties if name in self.property_to_mode})
        return tuple(active)

    def ground_truth_intermediate(self, item: AnimalItem) -> Tuple[Tuple[int, ...], int]:
        func = tuple(self.func2idx[name] for name in item.functionality if name in self.func2idx)
        fam = self.family2idx[item.family]
        return func, fam

    def transition_priors(self) -> List[np.ndarray]:
        # These priors seed the hopping matrices with task-aligned structure
        # before learning refines them.
        p_to_f = np.zeros((len(self.photonic_percept_modes), len(self.functionality)), dtype=float)
        f_to_fam = np.zeros((len(self.functionality), len(self.families)), dtype=float)
        fam_to_s = np.zeros((len(self.families), len(self.species)), dtype=float)
        for item in self.animals:
            percept_modes = self.encode_photonic_percept(item)
            func_ids, fam_id = self.ground_truth_intermediate(item)
            species_id = self.species2idx[item.name]
            for pm in percept_modes:
                for func_id in func_ids:
                    p_to_f[pm, func_id] += 1.0
            for func_id in func_ids:
                f_to_fam[func_id, fam_id] += 1.0
            fam_to_s[fam_id, species_id] += 1.0

        priors = [p_to_f, f_to_fam, fam_to_s]
        normed: List[np.ndarray] = []
        for mat in priors:
            row_sums = mat.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums <= 1e-12, 1.0, row_sums)
            normed.append(mat / row_sums)
        return normed


class AnimalClassificationEnv(BaseEnv):
    def __init__(self, dataset: AnimalDataset, dense_reward_weight: float = 0.15, seed: int = 0) -> None:
        self.dataset = dataset
        self.dense_reward_weight = dense_reward_weight
        self.rng = random.Random(seed)
        self.current: Optional[AnimalItem] = None

    def reset(self) -> Tuple[int, ...]:
        self.current = self.rng.choice(self.dataset.animals)
        return self.dataset.encode_photonic_percept(self.current)

    def step(self, action: int) -> Tuple[Tuple[int, ...], float, bool, Dict[str, Any]]:
        if self.current is None:
            raise RuntimeError("reset() must be called before step().")
        item = self.current
        next_obs = self.reset()
        # Episodes are single-step classification trials; the next observation is
        # returned immediately so the caller can keep stepping without an extra reset.
        correct_func, correct_family = self.dataset.ground_truth_intermediate(item)
        info = {
            "target": item.name,
            "target_index": self.dataset.species2idx[item.name],
            "correct_functionality": correct_func,
            "correct_family": correct_family,
            "dense_reward_weight": self.dense_reward_weight,
        }
        reward = 1.0 if int(action) == info["target_index"] else 0.0
        return next_obs, reward, True, info


## Utilities for constructing truncated bosonic Fock spaces and operators for the photonic quantum-MEPS architecture.


def bosonic_basis(num_modes: int, max_total_excitation: int) -> List[Tuple[int, ...]]:
    basis: List[Tuple[int, ...]] = []

    def rec(mode: int, remaining: int, prefix: List[int]) -> None:
        # Recursive composition generation is much cheaper than naively looping
        # over every occupancy tuple in (cutoff + 1) ** num_modes.
        if mode == num_modes:
            basis.append(tuple(prefix))
            return
        for n in range(remaining + 1):
            prefix.append(n)
            rec(mode + 1, remaining - n, prefix)
            prefix.pop()

    rec(0, max_total_excitation, [])
    return basis


class BosonicFockSpace:
    def __init__(self, num_modes: int, max_total_excitation: int) -> None:
        # A basis element is an occupation-number tuple across all modes.
        self.num_modes = num_modes
        self.max_total_excitation = max_total_excitation
        self.basis = bosonic_basis(num_modes, max_total_excitation)
        self.index = {occ: idx for idx, occ in enumerate(self.basis)}
        self.dim = len(self.basis)

    def basis_state(self, occ: Tuple[int, ...]) -> np.ndarray:
        out = np.zeros((self.dim, 1), dtype=np.complex128)
        out[self.index[tuple(occ)], 0] = 1.0
        return out

    def vacuum_state(self) -> np.ndarray:
        return self.basis_state(tuple(0 for _ in range(self.num_modes)))

    def creation(self, mode: int) -> np.ndarray:
        op = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for occ in self.basis:
            n = occ[mode]
            new_occ = list(occ)
            new_occ[mode] += 1
            new_occ_t = tuple(new_occ)
            if sum(new_occ) <= self.max_total_excitation and new_occ_t in self.index:
                op[self.index[new_occ_t], self.index[occ]] = math.sqrt(n + 1)
        return op

    def annihilation(self, mode: int) -> np.ndarray:
        op = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for occ in self.basis:
            n = occ[mode]
            if n > 0:
                new_occ = list(occ)
                new_occ[mode] -= 1
                op[self.index[tuple(new_occ)], self.index[occ]] = math.sqrt(n)
        return op


class SingleModeFockSpace:
    def __init__(self, cutoff: int) -> None:
        # Detector ancillas use a tiny truncated oscillator rather than the full
        # multi-mode basis used for the photonic memory layers.
        self.cutoff = int(cutoff)
        self.dim = self.cutoff + 1

    def basis_state(self, n: int) -> np.ndarray:
        out = np.zeros((self.dim, 1), dtype=np.complex128)
        out[int(n), 0] = 1.0
        return out

    def vacuum_state(self) -> np.ndarray:
        return self.basis_state(0)

    def creation(self) -> np.ndarray:
        op = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for n in range(self.cutoff):
            op[n + 1, n] = math.sqrt(n + 1)
        return op

    def annihilation(self) -> np.ndarray:
        op = np.zeros((self.dim, self.dim), dtype=np.complex128)
        for n in range(1, self.dim):
            op[n - 1, n] = math.sqrt(n)
        return op


## Main photonic quantum-MEPS memory and deliberation architecture


class PhotonicQuantumMEPSMemory:
    """
    Layered photonic MEPS memory with explicit multi-photon clip propagation.

    Each adjacent layer pair has its own truncated bosonic space. Deliberation
    starts from a multi-photon superposition over active percept clips,
    propagates through the corresponding interferometric pair unitary, and is
    then projected onto the next layer before continuing. This is closer to the
    layer-by-layer MEPS picture than a single global unitary.
    """

    def __init__(
        self,
        layer_specs: Optional[Sequence[QMEPSLayerSpec]] = None,
        max_total_excitation: int = 2,
        dt_schedule: Optional[Sequence[float]] = None,
        seed: int = 0,
        coupling_radius: int = 0,
        trainable_components: Sequence[str] = ("hopping",),
        detector_cutoff: int = 1,
        detector_frequency: float = 0.35,
        detector_coupling: float = 0.20,
        detector_target_coupling: float = 0.08,
    ) -> None:
        # Each MEPS transition gets its own pairwise photonic space plus an
        # attached which-way detector ancilla.
        self.layer_specs = list(layer_specs) if layer_specs is not None else [
            QMEPSLayerSpec("percept", 8),
            QMEPSLayerSpec("functionality", 10),
            QMEPSLayerSpec("family", 10),
            QMEPSLayerSpec("species", 12),
        ]
        if len(self.layer_specs) < 2:
            raise ValueError("Need at least two layers.")
        self.num_layers = len(self.layer_specs)
        self.max_total_excitation = int(max_total_excitation)
        self.rng = np.random.default_rng(seed)
        self.coupling_radius = int(coupling_radius)
        self.trainable_components = tuple(trainable_components)
        self.detector_cutoff = int(detector_cutoff)
        self.detector_frequency = float(detector_frequency)
        self.detector_coupling = float(detector_coupling)
        self.detector_target_coupling = float(detector_target_coupling)
        self.dt_schedule = np.array(dt_schedule if dt_schedule is not None else [0.30] * (self.num_layers - 1), dtype=float)
        if len(self.dt_schedule) != self.num_layers - 1:
            raise ValueError("dt_schedule must have length num_layers - 1.")

        self.layer_sizes = [spec.size for spec in self.layer_specs]
        self.layer_spaces = [BosonicFockSpace(size, self.max_total_excitation) for size in self.layer_sizes]
        self.layer_a_ops = [[space.annihilation(i) for i in range(space.num_modes)] for space in self.layer_spaces]
        self.layer_adag_ops = [[space.creation(i) for i in range(space.num_modes)] for space in self.layer_spaces]
        self.layer_n_ops = [[self.layer_adag_ops[k][i] @ self.layer_a_ops[k][i] for i in range(space.num_modes)]
                            for k, space in enumerate(self.layer_spaces)]

        self.pair_spaces: List[BosonicFockSpace] = []
        self.pair_sizes: List[Tuple[int, int]] = []
        self.pair_a_ops: List[List[np.ndarray]] = []
        self.pair_adag_ops: List[List[np.ndarray]] = []
        self.pair_n_ops: List[List[np.ndarray]] = []
        self.detector_spaces: List[SingleModeFockSpace] = []
        self.detector_a_ops: List[np.ndarray] = []
        self.detector_adag_ops: List[np.ndarray] = []
        self.detector_n_ops: List[np.ndarray] = []
        self.layer_params: List[QMEPSLayerParams] = []
        for t in range(self.num_layers - 1):
            left_size = self.layer_specs[t].size
            right_size = self.layer_specs[t + 1].size
            # Deliberation is local to one adjacent layer pair at a time, which
            # keeps the exact bosonic evolution manageable.
            pair_space = BosonicFockSpace(left_size + right_size, self.max_total_excitation)
            self.pair_spaces.append(pair_space)
            self.pair_sizes.append((left_size, right_size))
            a_ops = [pair_space.annihilation(i) for i in range(pair_space.num_modes)]
            adag_ops = [pair_space.creation(i) for i in range(pair_space.num_modes)]
            self.pair_a_ops.append(a_ops)
            self.pair_adag_ops.append(adag_ops)
            self.pair_n_ops.append([adag_ops[i] @ a_ops[i] for i in range(pair_space.num_modes)])
            detector_space = SingleModeFockSpace(self.detector_cutoff)
            self.detector_spaces.append(detector_space)
            self.detector_a_ops.append(detector_space.annihilation())
            self.detector_adag_ops.append(detector_space.creation())
            self.detector_n_ops.append(self.detector_adag_ops[-1] @ self.detector_a_ops[-1])
            self.layer_params.append(self._random_layer_params(t))

        self._pair_projector_cache: Dict[Tuple[int, str, int], np.ndarray] = {}
        self._unitary_cache: Dict[int, np.ndarray] = {}

    def _supported_pairs(self, transition_idx: int) -> List[Tuple[int, int]]:
        # The coupling radius limits trainable edges to a local band, which
        # mimics sparse clip connectivity rather than a dense all-to-all graph.
        left_size, right_size = self.pair_sizes[transition_idx]
        out: List[Tuple[int, int]] = []
        for local_i in range(left_size):
            center = int(round((local_i + 0.5) * right_size / left_size - 0.5))
            lo = max(0, center - self.coupling_radius)
            hi = min(right_size, center + self.coupling_radius + 1)
            for local_j in range(lo, hi):
                out.append((local_i, left_size + local_j))
        return out

    def _random_layer_params(self, transition_idx: int) -> QMEPSLayerParams:
        # Random initialization is later nudged toward the task priors, but we
        # still start from a valid optical Hamiltonian template.
        pair_size = sum(self.pair_sizes[transition_idx])
        onsite = self.rng.normal(0.0, 0.04, size=pair_size)
        hopping = np.zeros((pair_size, pair_size), dtype=float)
        density = np.zeros((pair_size, pair_size), dtype=float)
        for i, j in self._supported_pairs(transition_idx):
            locality_bias = np.exp(-0.25 * abs(i - j))
            coupling = self.rng.normal(0.18 * locality_bias, 0.03)
            hopping[i, j] = hopping[j, i] = coupling
            kerr = self.rng.normal(0.01 * locality_bias, 0.004)
            density[i, j] = density[j, i] = kerr
        return QMEPSLayerParams(onsite=onsite, hopping=hopping, density_coupling=density)

    def inject_task_priors(
        self,
        transition_priors: Sequence[np.ndarray],
        hopping_strength: float = 0.45,
        onsite_strength: float = 0.08,
    ) -> None:
        # The priors act like an inductive bias for MEPS edge strengths: likely
        # percept->functionality->family->species routes start with larger hops.
        for t, prior in enumerate(transition_priors):
            left_size, right_size = self.pair_sizes[t]
            if prior.shape != (left_size, right_size):
                raise ValueError(f"Prior shape mismatch for transition {t}: expected {(left_size, right_size)}, got {prior.shape}")
            params = self.layer_params[t]
            left_bias = prior.sum(axis=1)
            right_bias = prior.sum(axis=0)
            for i in range(left_size):
                params.onsite[i] += onsite_strength * left_bias[i]
            for j in range(right_size):
                params.onsite[left_size + j] += onsite_strength * right_bias[j]
            for i, j in self._supported_pairs(t):
                local_j = j - left_size
                params.hopping[i, j] += hopping_strength * prior[i, local_j]
                params.hopping[j, i] = params.hopping[i, j]
        self._unitary_cache.clear()

    def copy_params(self) -> List[QMEPSLayerParams]:
        return [QMEPSLayerParams(p.onsite.copy(), p.hopping.copy(), p.density_coupling.copy()) for p in self.layer_params]

    def restore_params(self, params: List[QMEPSLayerParams]) -> None:
        self.layer_params = [QMEPSLayerParams(p.onsite.copy(), p.hopping.copy(), p.density_coupling.copy()) for p in params]
        self._unitary_cache.clear()

    def parameter_spec(self) -> List[Tuple[int, str, Tuple[int, ...]]]:
        spec: List[Tuple[int, str, Tuple[int, ...]]] = []
        for t in range(self.num_layers - 1):
            pair_size = sum(self.pair_sizes[t])
            if "onsite" in self.trainable_components:
                spec.extend((t, "onsite", (i,)) for i in range(pair_size))
            if "hopping" in self.trainable_components:
                spec.extend((t, "hopping", (i, j)) for i, j in self._supported_pairs(t))
            if "density_coupling" in self.trainable_components:
                spec.extend((t, "density_coupling", (i, j)) for i, j in self._supported_pairs(t))
        return spec

    def get_parameter_vector(self) -> np.ndarray:
        vals: List[float] = []
        for t, name, idx in self.parameter_spec():
            vals.append(float(getattr(self.layer_params[t], name)[idx]))
        return np.array(vals, dtype=float)

    def set_parameter_vector(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float)
        spec = self.parameter_spec()
        if len(theta) != len(spec):
            raise ValueError("Parameter length mismatch.")
        for val, (t, name, idx) in zip(theta, spec):
            arr = getattr(self.layer_params[t], name)
            arr[idx] = val
            if len(idx) == 2:
                i, j = idx
                arr[j, i] = val
        self._unitary_cache.clear()

    def apply_update(self, delta: np.ndarray) -> None:
        self.set_parameter_vector(self.get_parameter_vector() + np.asarray(delta, dtype=float))

    def _pair_hamiltonian(self, transition_idx: int) -> np.ndarray:
        params = self.layer_params[transition_idx]
        phot_dim = self.pair_spaces[transition_idx].dim
        det_dim = self.detector_spaces[transition_idx].dim
        H_phot = np.zeros((phot_dim, phot_dim), dtype=np.complex128)
        n_ops = self.pair_n_ops[transition_idx]
        a_ops = self.pair_a_ops[transition_idx]
        adag_ops = self.pair_adag_ops[transition_idx]
        for i in range(len(params.onsite)):
            H_phot += params.onsite[i] * n_ops[i]
        for i, j in self._supported_pairs(transition_idx):
            H_phot += params.hopping[i, j] * (adag_ops[i] @ a_ops[j] + adag_ops[j] @ a_ops[i])
            H_phot += params.density_coupling[i, j] * (n_ops[i] @ n_ops[j])

        detector_a = self.detector_a_ops[transition_idx]
        detector_adag = self.detector_adag_ops[transition_idx]
        detector_n = self.detector_n_ops[transition_idx]
        left_size, right_size = self.pair_sizes[transition_idx]
        source_weights = np.linspace(-1.0, 1.0, left_size) if left_size > 1 else np.array([1.0])
        target_weights = np.linspace(-1.0, 1.0, right_size) if right_size > 1 else np.array([1.0])
        source_signature = np.zeros((phot_dim, phot_dim), dtype=np.complex128)
        target_signature = np.zeros((phot_dim, phot_dim), dtype=np.complex128)
        for i, weight in enumerate(source_weights):
            source_signature += weight * n_ops[i]
        for j, weight in enumerate(target_weights):
            target_signature += weight * n_ops[left_size + j]

        identity_phot = np.eye(phot_dim, dtype=np.complex128)
        identity_det = np.eye(det_dim, dtype=np.complex128)
        detector_h = self.detector_frequency * detector_n
        quadrature = detector_a + detector_adag
        # The total Hamiltonian couples the photonic pair dynamics to a
        # detector quadrature so path information can leak into the ancilla.
        H = np.kron(H_phot, identity_det)
        H += np.kron(identity_phot, detector_h)
        H += self.detector_coupling * np.kron(source_signature, quadrature)
        H += self.detector_target_coupling * np.kron(target_signature, detector_n)
        return 0.5 * (H + dagger(H))

    def _pair_unitary(self, transition_idx: int) -> np.ndarray:
        # Exact matrix exponentials are cached because the finite-difference
        # optimizer reuses the same Hamiltonian many times per update.
        if transition_idx not in self._unitary_cache:
            H = self._pair_hamiltonian(transition_idx)
            self._unitary_cache[transition_idx] = expm(-1j * self.dt_schedule[transition_idx] * H)
        return self._unitary_cache[transition_idx]

    def _active_photon_count(self, percept_indices: Sequence[int]) -> int:
        return max(1, min(len(set(int(x) for x in percept_indices)), self.max_total_excitation))

    def initial_layer_state(self, percept_indices: Sequence[int] | int) -> Tuple[np.ndarray, int]:
        if isinstance(percept_indices, int):
            percept_indices = [percept_indices]
        active = sorted({int(x) % self.layer_specs[0].size for x in percept_indices})
        if not active:
            active = [0]
        photon_count = self._active_photon_count(active)
        space = self.layer_spaces[0]
        psi = space.vacuum_state()
        # Active percept clips are encoded as a symmetrized multi-photon
        # excitation over the percept layer.
        collective_creation = sum(self.layer_adag_ops[0][idx] for idx in active) / math.sqrt(len(active))
        for _ in range(photon_count):
            psi = collective_creation @ psi
        return ket_to_dm(normalize_state(psi)), photon_count

    def _embed_layer_state_into_pair(self, layer_state: np.ndarray, transition_idx: int, side: str) -> np.ndarray:
        # When moving from one layer transition to the next, the current reduced
        # layer density matrix is re-embedded into the corresponding pair space.
        left_size, right_size = self.pair_sizes[transition_idx]
        pair_space = self.pair_spaces[transition_idx]
        source_layer_idx = transition_idx if side == "left" else transition_idx + 1
        source_space = self.layer_spaces[source_layer_idx]
        out = np.zeros((pair_space.dim, pair_space.dim), dtype=np.complex128)
        for occ_a, idx_a in source_space.index.items():
            for occ_b, idx_b in source_space.index.items():
                amp = layer_state[idx_a, idx_b]
                if abs(amp) <= 1e-15:
                    continue
                if side == "left":
                    pair_occ_a = tuple(occ_a) + tuple(0 for _ in range(right_size))
                    pair_occ_b = tuple(occ_b) + tuple(0 for _ in range(right_size))
                else:
                    pair_occ_a = tuple(0 for _ in range(left_size)) + tuple(occ_a)
                    pair_occ_b = tuple(0 for _ in range(left_size)) + tuple(occ_b)
                out[pair_space.index[pair_occ_a], pair_space.index[pair_occ_b]] += amp
        return out

    def _joint_with_detector_vacuum(self, pair_dm: np.ndarray, transition_idx: int) -> np.ndarray:
        # Each transition starts with its detector in vacuum before the joint
        # photonic-detector evolution entangles them.
        detector_vacuum = ket_to_dm(self.detector_spaces[transition_idx].vacuum_state())
        return np.kron(pair_dm, detector_vacuum)

    def _project_joint_state(self, joint_state: np.ndarray, transition_idx: int, side: str, photon_count: int) -> Tuple[np.ndarray, float]:
        # This is the MEPS-style conditioning step: after local evolution, keep
        # only the branch where all photons have propagated into the next layer.
        P_pair = self._pair_output_projector(transition_idx, side, photon_count)
        identity_det = np.eye(self.detector_spaces[transition_idx].dim, dtype=np.complex128)
        P = np.kron(P_pair, identity_det)
        proj = P @ joint_state @ P
        prob = float(np.real(np.trace(proj)))
        if prob <= 1e-15:
            return joint_state.copy(), 0.0
        return proj / prob, prob

    def _reduced_detector_state(self, joint_state: np.ndarray, transition_idx: int) -> np.ndarray:
        phot_dim = self.pair_spaces[transition_idx].dim
        det_dim = self.detector_spaces[transition_idx].dim
        reshaped = joint_state.reshape(phot_dim, det_dim, phot_dim, det_dim)
        return np.trace(reshaped, axis1=0, axis2=2)

    def _detector_basis_projector(self, transition_idx: int, detector_state: int) -> np.ndarray:
        phot_dim = self.pair_spaces[transition_idx].dim
        detector_ket = self.detector_spaces[transition_idx].basis_state(detector_state)
        detector_proj = detector_ket @ dagger(detector_ket)
        return np.kron(np.eye(phot_dim, dtype=np.complex128), detector_proj)

    def _mid_circuit_measure_detector(
        self,
        joint_state: np.ndarray,
        transition_idx: int,
        measurement_states: Sequence[int],
        sample_outcome: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Mid-circuit detector measurements can either sample a concrete outcome
        # or apply the non-selective Lüders channel, depending on the caller.
        selected_states = tuple(sorted({int(x) for x in measurement_states if 0 <= int(x) <= self.detector_cutoff}))
        if not selected_states:
            return joint_state, {
                "layer": transition_idx,
                "measured": False,
                "outcome": None,
                "probabilities": {},
            }

        projectors = {state: self._detector_basis_projector(transition_idx, state) for state in selected_states}
        probabilities: Dict[int, float] = {}
        for state, projector in projectors.items():
            probabilities[state] = float(np.real(np.trace(projector @ joint_state)))
        residual_prob = max(0.0, 1.0 - sum(probabilities.values()))

        outcomes: List[int] = list(selected_states)
        probs: List[float] = [probabilities[state] for state in selected_states]
        combined_proj = np.zeros_like(joint_state)
        for projector in projectors.values():
            combined_proj += projector
        complement_proj = np.eye(joint_state.shape[0], dtype=np.complex128) - combined_proj
        if residual_prob > 1e-12:
            outcomes.append(-1)
            probs.append(residual_prob)

        total = sum(probs)
        if total <= 1e-15:
            return joint_state, {
                "layer": transition_idx,
                "measured": True,
                "outcome": None,
                "probabilities": probabilities,
                "residual_probability": residual_prob,
            }

        if sample_outcome:
            # Sampled collapse is used for actual trajectories and recorded outcomes.
            probs = [p / total for p in probs]
            outcome = int(self.rng.choice(outcomes, p=probs))
            collapse_projector = complement_proj if outcome == -1 else projectors[outcome]
            post = collapse_projector @ joint_state @ collapse_projector
            post_prob = float(np.real(np.trace(post)))
            if post_prob > 1e-15:
                post = post / post_prob
            else:
                post = joint_state.copy()
        else:
            # Non-selective Lüders update is used for deterministic quantities
            # such as QFI and finite-difference objectives.
            outcome = None
            post = np.zeros_like(joint_state)
            for projector in projectors.values():
                post += projector @ joint_state @ projector
            post += complement_proj @ joint_state @ complement_proj
            post_prob = float(np.real(np.trace(post)))
            if post_prob > 1e-15:
                post = post / post_prob
        return post, {
            "layer": transition_idx,
            "measured": True,
            "outcome": outcome,
            "probabilities": probabilities,
            "residual_probability": residual_prob,
        }

    def _extract_layer_state_from_joint(self, joint_state: np.ndarray, transition_idx: int, side: str) -> np.ndarray:
        # After conditioning, trace out the detector and the inactive half of
        # the photonic pair to obtain the reduced state for the next layer.
        left_size, _ = self.pair_sizes[transition_idx]
        target_layer_idx = transition_idx if side == "left" else transition_idx + 1
        target_space = self.layer_spaces[target_layer_idx]
        out = np.zeros((target_space.dim, target_space.dim), dtype=np.complex128)
        pair_space = self.pair_spaces[transition_idx]
        det_dm = self._reduced_detector_state(joint_state, transition_idx)
        det_trace = np.real(np.trace(det_dm))
        if det_trace > 1e-15:
            det_dm = det_dm / det_trace

        phot_dim = pair_space.dim
        det_dim = self.detector_spaces[transition_idx].dim
        reshaped = joint_state.reshape(phot_dim, det_dim, phot_dim, det_dim)
        pair_dm = np.trace(reshaped, axis1=1, axis2=3)
        for occ_a, idx_a in pair_space.index.items():
            left_occ_a = occ_a[:left_size]
            right_occ_a = occ_a[left_size:]
            if side == "left":
                if any(right_occ_a):
                    continue
                target_occ_a = left_occ_a
            else:
                if any(left_occ_a):
                    continue
                target_occ_a = right_occ_a
            ta = target_space.index[tuple(target_occ_a)]
            for occ_b, idx_b in pair_space.index.items():
                left_occ_b = occ_b[:left_size]
                right_occ_b = occ_b[left_size:]
                if side == "left":
                    if any(right_occ_b):
                        continue
                    target_occ_b = left_occ_b
                else:
                    if any(left_occ_b):
                        continue
                    target_occ_b = right_occ_b
                tb = target_space.index[tuple(target_occ_b)]
                out[ta, tb] += pair_dm[idx_a, idx_b]

        trace_out = float(np.real(np.trace(out)))
        if trace_out <= 1e-15:
            raise ValueError("Failed to extract non-zero reduced layer state from joint state.")
        return out / trace_out

    def _pair_output_projector(self, transition_idx: int, side: str, photon_count: int) -> np.ndarray:
        key = (transition_idx, side, photon_count)
        if key in self._pair_projector_cache:
            return self._pair_projector_cache[key]
        left_size, _ = self.pair_sizes[transition_idx]
        pair_space = self.pair_spaces[transition_idx]
        P = np.zeros((pair_space.dim, pair_space.dim), dtype=np.complex128)
        for occ, idx in pair_space.index.items():
            total = sum(occ)
            if total != photon_count:
                continue
            left_occ = sum(occ[:left_size])
            right_occ = sum(occ[left_size:])
            keep = (left_occ == photon_count) if side == "left" else (right_occ == photon_count)
            if keep:
                P[idx, idx] = 1.0
        self._pair_projector_cache[key] = P
        return P

    def deliberate_state(
        self,
        percept: Sequence[int] | int,
        max_projection_rounds: int = 1,
        conditional_project: bool = True,
        detector_measurement: Optional[DetectorMeasurementConfig] = None,
        sample_measurements: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # This is the central MEPS deliberation loop: propagate layer by layer,
        # optionally measure detectors mid-circuit, and keep a full record of
        # reduced layer states and detector states.
        layer_state, photon_count = self.initial_layer_state(percept)
        propagation_probs: List[float] = []
        layer_states: List[np.ndarray] = []
        detector_states: List[np.ndarray] = []
        joint_states: List[np.ndarray] = []
        measurement_config = (detector_measurement or DetectorMeasurementConfig()).normalized(
            self.num_layers,
            self.detector_cutoff,
        )
        measurements_done = 0
        measurement_records: List[Dict[str, Any]] = []

        for t in range(self.num_layers - 1):
            pair_dm = self._embed_layer_state_into_pair(layer_state, t, side="left")
            joint_state = self._joint_with_detector_vacuum(pair_dm, t)
            U = self._pair_unitary(t)
            joint_state = U @ joint_state @ dagger(U)

            success_prob = 0.0
            for _ in range(max_projection_rounds):
                pair_proj, success_prob = self._project_joint_state(joint_state, t, side="right", photon_count=photon_count)
                if success_prob > 1e-15:
                    joint_state = pair_proj
                    break

            if conditional_project and success_prob <= 1e-15:
                P_pair = self._pair_output_projector(t, "right", photon_count)
                P = np.kron(P_pair, np.eye(self.detector_spaces[t].dim, dtype=np.complex128))
                proj = P @ joint_state @ P
                norm = float(np.real(np.trace(proj)))
                if norm > 1e-15:
                    joint_state = proj / norm

            should_measure = (
                t in measurement_config.layers
                and len(measurement_config.states) > 0
                and (measurement_config.max_measurements is None or measurements_done < measurement_config.max_measurements)
            )
            if should_measure:
                joint_state, record = self._mid_circuit_measure_detector(
                    joint_state,
                    t,
                    measurement_config.states,
                    sample_outcome=sample_measurements,
                )
                measurement_records.append(record)
                measurements_done += 1

            detector_states.append(self._reduced_detector_state(joint_state, t))
            layer_state = self._extract_layer_state_from_joint(joint_state, t, side="right")
            joint_states.append(joint_state.copy())
            layer_states.append(layer_state.copy())
            propagation_probs.append(success_prob)

        return layer_state, {
            "photon_count": photon_count,
            "propagation_probs": propagation_probs,
            "layer_states": layer_states,
            "detector_states": detector_states,
            "joint_states": joint_states,
            "measurement_records": measurement_records,
            "detector_measurement_config": measurement_config,
            "agent_state": {
                "layer_state": layer_state,
                "detector_states": detector_states,
                "joint_states": joint_states,
                "measurement_records": measurement_records,
            },
        }

    def density(self, percept: Sequence[int] | int) -> np.ndarray:
        # State-based quantities use the deterministic, non-selective
        # measurement channel so QFI is computed on a well-defined density map.
        final_state, _ = self.deliberate_state(percept, sample_measurements=False)
        return final_state

    def layer_mode_expectations_from_state(self, layer_idx: int, state: np.ndarray) -> np.ndarray:
        exps = [
            max(float(np.real(np.trace(state @ self.layer_n_ops[layer_idx][m]))), 0.0)
            for m in range(self.layer_specs[layer_idx].size)
        ]
        return np.asarray(exps, dtype=float)

    def action_expectations(
        self,
        percept: Sequence[int] | int,
        detector_measurement: Optional[DetectorMeasurementConfig] = None,
        sample_measurements: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        final_state, info = self.deliberate_state(
            percept,
            detector_measurement=detector_measurement,
            sample_measurements=sample_measurements,
        )
        exps = self.layer_mode_expectations_from_state(self.num_layers - 1, final_state)
        return exps, {**info, "state": final_state}

    def policy(
        self,
        percept: Sequence[int] | int,
        temperature: float = 1.0,
        detector_measurement: Optional[DetectorMeasurementConfig] = None,
        sample_measurements: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        raw, info = self.action_expectations(
            percept,
            detector_measurement=detector_measurement,
            sample_measurements=sample_measurements,
        )
        if raw.sum() <= 1e-15:
            probs = np.ones(self.layer_specs[-1].size, dtype=float) / self.layer_specs[-1].size
        else:
            probs = softmax(raw / max(float(temperature), 1e-8))
        return probs, info

    def taxonomy_predictions(
        self,
        percept: Sequence[int] | int,
        detector_measurement: Optional[DetectorMeasurementConfig] = None,
        sample_measurements: bool = True,
    ) -> Dict[str, int]:
        _, info = self.deliberate_state(
            percept,
            detector_measurement=detector_measurement,
            sample_measurements=sample_measurements,
        )
        if len(info["layer_states"]) != self.num_layers - 1:
            raise RuntimeError("Unexpected number of intermediate layer states.")
        func_scores = self.layer_mode_expectations_from_state(1, info["layer_states"][0])
        family_scores = self.layer_mode_expectations_from_state(2, info["layer_states"][1])
        species_scores = self.layer_mode_expectations_from_state(3, info["layer_states"][2])
        return {
            "functionality": int(np.argmax(func_scores)),
            "family": int(np.argmax(family_scores)),
            "species": int(np.argmax(species_scores)),
        }


class QuantumFisher:
    @staticmethod
    def density_derivatives(memory: PhotonicQuantumMEPSMemory, percept: Sequence[int] | int, eps: float = 1e-5) -> List[np.ndarray]:
        # Central finite differences are used on the final reduced density
        # matrix, not on sampled trajectories, so the resulting QFI is stable.
        base = memory.copy_params()
        theta0 = memory.get_parameter_vector()
        drhos: List[np.ndarray] = []
        for k in range(len(theta0)):
            tp = theta0.copy()
            tm = theta0.copy()
            tp[k] += eps
            tm[k] -= eps
            memory.set_parameter_vector(tp)
            rho_p = memory.density(percept)
            memory.set_parameter_vector(tm)
            rho_m = memory.density(percept)
            drho = (rho_p - rho_m) / (2.0 * eps)
            drhos.append(0.5 * (drho + dagger(drho)))
        memory.restore_params(base)
        return drhos

    @staticmethod
    def from_density_derivatives(rho: np.ndarray, drhos: Sequence[np.ndarray], eps: float = 1e-12) -> np.ndarray:
        # Symmetric logarithmic derivative formula evaluated in the eigenbasis of rho.
        evals, evecs = eigh(rho)
        evals = np.real(evals)
        denom = evals[:, None] + evals[None, :]
        weight = np.zeros_like(denom, dtype=float)
        mask = denom > eps
        weight[mask] = 2.0 / denom[mask]
        vdag = dagger(evecs)
        mats = [vdag @ drho @ evecs for drho in drhos]
        npar = len(mats)
        F = np.zeros((npar, npar), dtype=float)
        weighted_rev = [weight * M.T for M in mats]
        for i in range(npar):
            for j in range(i, npar):
                fij = float(np.real(np.sum(mats[i] * weighted_rev[j])))
                F[i, j] = fij
                F[j, i] = fij
        return 0.5 * (F + F.T)

    @staticmethod
    def qfim(memory: PhotonicQuantumMEPSMemory, percept: Sequence[int] | int, deriv_eps: float = 1e-5) -> np.ndarray:
        rho = memory.density(percept)
        drhos = QuantumFisher.density_derivatives(memory, percept, eps=deriv_eps)
        return QuantumFisher.from_density_derivatives(rho, drhos)

    @staticmethod
    def natural_gradient(grad: np.ndarray, F: np.ndarray, reg: float = 1e-4) -> np.ndarray:
        return np.linalg.solve(F + reg * np.eye(F.shape[0]), grad)


## Quantum state analysis utilities


def matrix_sqrt_psd(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Positive-semidefinite matrix square root for fidelity/Bures calculations.
    evals, evecs = eigh(0.5 * (x + dagger(x)))
    evals = np.clip(np.real(evals), 0.0, None)
    return evecs @ np.diag(np.sqrt(np.maximum(evals, eps * 0.0))) @ dagger(evecs)


def von_neumann_entropy(rho: np.ndarray, eps: float = 1e-12) -> float:
    evals = np.linalg.eigvalsh(0.5 * (rho + dagger(rho)))
    evals = np.clip(np.real(evals), eps, None)
    return float(-np.sum(evals * np.log(evals)))


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    diff = rho - sigma
    svals = np.linalg.svd(diff, compute_uv=False)
    return float(0.5 * np.sum(np.abs(svals)))


def hilbert_schmidt_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    diff = rho - sigma
    return float(np.real(np.trace(diff @ diff)))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    sqrt_rho = matrix_sqrt_psd(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    sqrt_inner = matrix_sqrt_psd(inner)
    return float(np.real(np.trace(sqrt_inner)) ** 2)


def bures_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    fid = min(max(fidelity(rho, sigma), 0.0), 1.0)
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - np.sqrt(fid)))))


def quantum_jensen_shannon_divergence(rho: np.ndarray, sigma: np.ndarray) -> float:
    mix = 0.5 * (rho + sigma)
    return float(von_neumann_entropy(mix) - 0.5 * von_neumann_entropy(rho) - 0.5 * von_neumann_entropy(sigma))


def l1_coherence(rho: np.ndarray) -> float:
    off_diag = rho - np.diag(np.diag(rho))
    return float(np.sum(np.abs(off_diag)))


def normalized_l1_coherence(rho: np.ndarray) -> float:
    dim = rho.shape[0]
    if dim <= 1:
        return 0.0
    return float(l1_coherence(rho) / (dim - 1))


## Graph and hypergraph analysis utilities


def transition_hopping_matrix(memory: PhotonicQuantumMEPSMemory, transition_idx: int) -> np.ndarray:
    left_size, right_size = memory.pair_sizes[transition_idx]
    out = np.zeros((left_size, right_size), dtype=float)
    for i, j in memory._supported_pairs(transition_idx):
        out[i, j - left_size] = memory.layer_params[transition_idx].hopping[i, j]
    return out


def hypergraph_cardinality_stats(incidence: np.ndarray, axis: int, threshold: float = 1e-12) -> Dict[str, float]:
    mask = np.abs(incidence) > threshold
    card = mask.sum(axis=axis)
    return {
        "count": float(len(card)),
        "avg_cardinality": float(np.mean(card)) if len(card) else 0.0,
        "max_cardinality": float(np.max(card)) if len(card) else 0.0,
        "min_cardinality": float(np.min(card)) if len(card) else 0.0,
    }


def weighted_bipartite_adjacency(h: np.ndarray) -> np.ndarray:
    # The learned hopping matrix is interpreted as a weighted bipartite graph
    # between adjacent MEPS layers.
    left_size, right_size = h.shape
    return np.block([
        [np.zeros((left_size, left_size)), np.abs(h)],
        [np.abs(h.T), np.zeros((right_size, right_size))],
    ])


def support_adjacency(h: np.ndarray, threshold: float = 1e-12) -> np.ndarray:
    return (weighted_bipartite_adjacency(h) > threshold).astype(float)


def graph_components(adjacency: np.ndarray) -> List[List[int]]:
    n = adjacency.shape[0]
    seen = np.zeros(n, dtype=bool)
    components: List[List[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        comp: List[int] = []
        queue: deque[int] = deque([start])
        seen[start] = True
        while queue:
            node = queue.popleft()
            comp.append(node)
            for nxt in np.flatnonzero(adjacency[node] > 0):
                if not seen[nxt]:
                    seen[nxt] = True
                    queue.append(int(nxt))
        components.append(comp)
    return components


def all_pairs_shortest_paths(adjacency: np.ndarray) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[int, ...]]]:
    # Unweighted BFS distances are enough here because structural path analysis
    # is done on the support graph rather than on continuous optical weights.
    n = adjacency.shape[0]
    inf = float("inf")
    dists = np.full((n, n), inf, dtype=float)
    paths: Dict[Tuple[int, int], Tuple[int, ...]] = {}
    for src in range(n):
        dists[src, src] = 0.0
        queue: deque[int] = deque([src])
        parents = {src: None}
        while queue:
            node = queue.popleft()
            for nxt in np.flatnonzero(adjacency[node] > 0):
                nxt = int(nxt)
                if nxt not in parents:
                    parents[nxt] = node
                    dists[src, nxt] = dists[src, node] + 1.0
                    queue.append(nxt)
        for dst in parents:
            path = []
            cur: Optional[int] = dst
            while cur is not None:
                path.append(cur)
                cur = parents[cur]
            paths[(src, dst)] = tuple(reversed(path))
    return dists, paths


def degree_centrality(adjacency: np.ndarray) -> np.ndarray:
    n = adjacency.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=float)
    return np.sum(adjacency > 0, axis=1) / (n - 1)


def closeness_centrality(adjacency: np.ndarray) -> np.ndarray:
    dists, _ = all_pairs_shortest_paths(adjacency)
    n = adjacency.shape[0]
    out = np.zeros(n, dtype=float)
    for i in range(n):
        finite = dists[i][np.isfinite(dists[i]) & (dists[i] > 0)]
        if finite.size:
            out[i] = finite.size / np.sum(finite)
    return out


def betweenness_centrality(adjacency: np.ndarray) -> np.ndarray:
    n = adjacency.shape[0]
    bc = np.zeros(n, dtype=float)
    for s in range(n):
        stack: List[int] = []
        preds: List[List[int]] = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=float)
        sigma[s] = 1.0
        dist = -np.ones(n, dtype=int)
        dist[s] = 0
        queue: deque[int] = deque([s])
        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in np.flatnonzero(adjacency[v] > 0):
                w = int(w)
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    preds[w].append(v)
        delta = np.zeros(n, dtype=float)
        while stack:
            w = stack.pop()
            for v in preds[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bc[w] += delta[w]
    if n > 2:
        bc /= (n - 1) * (n - 2)
    return bc


def eigenvector_centrality(weighted_adjacency: np.ndarray, iters: int = 200, tol: float = 1e-10) -> np.ndarray:
    n = weighted_adjacency.shape[0]
    x = np.ones(n, dtype=float) / max(n, 1)
    for _ in range(iters):
        x_new = np.abs(weighted_adjacency @ x)
        norm = np.linalg.norm(x_new)
        if norm <= tol:
            return np.zeros(n, dtype=float)
        x_new /= norm
        if np.linalg.norm(x_new - x) <= tol:
            return x_new
        x = x_new
    return x


def cheeger_constant_sweep(weighted_adjacency: np.ndarray) -> Dict[str, float]:
    n = weighted_adjacency.shape[0]
    if n <= 1:
        return {"cheeger_estimate": 0.0, "cheeger_lower_bound": 0.0, "cheeger_upper_bound": 0.0}
    degrees = np.sum(weighted_adjacency, axis=1)
    laplacian = np.diag(degrees) - weighted_adjacency
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(degrees > 1e-12, 1.0 / np.sqrt(degrees), 0.0)
    norm_lap = np.diag(inv_sqrt) @ laplacian @ np.diag(inv_sqrt)
    eigvals, eigvecs = eigh(norm_lap)
    lambda2 = float(np.real(eigvals[1])) if eigvals.size > 1 else 0.0
    fiedler = np.real(eigvecs[:, 1]) if eigvecs.shape[1] > 1 else np.zeros(n, dtype=float)
    ordering = np.argsort(fiedler)
    best = float("inf")
    volume_total = float(np.sum(degrees))
    for k in range(1, n):
        S = ordering[:k]
        Sc = ordering[k:]
        vol_S = float(np.sum(degrees[S]))
        vol_Sc = float(np.sum(degrees[Sc]))
        if min(vol_S, vol_Sc) <= 1e-12:
            continue
        cut = float(np.sum(weighted_adjacency[np.ix_(S, Sc)]))
        best = min(best, cut / min(vol_S, vol_Sc))
    if not np.isfinite(best):
        best = 0.0
    return {
        "cheeger_estimate": float(best),
        "cheeger_lower_bound": float(lambda2 / 2.0),
        "cheeger_upper_bound": float(np.sqrt(max(0.0, 2.0 * lambda2))),
    }


def mean_path_jaccard_similarity(paths: Dict[Tuple[int, int], Tuple[int, ...]], n: int) -> float:
    sims: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            path = paths.get((i, j))
            if path is None or len(path) < 2:
                continue
            set_a = set(path[:-1])
            set_b = set(path[1:])
            union = set_a | set_b
            if union:
                sims.append(len(set_a & set_b) / len(union))
    return float(np.mean(sims)) if sims else 0.0


def path_length_similarity(dists: np.ndarray) -> float:
    finite = dists[np.isfinite(dists) & (dists > 0)]
    if finite.size <= 1:
        return 1.0
    return float(1.0 / (1.0 + np.std(finite)))


def simplicial_complex_from_hyperedges(hyperedges: Sequence[Sequence[int]], max_dim: int = 2) -> Dict[int, List[Tuple[int, ...]]]:
    simplices: Dict[int, set[Tuple[int, ...]]] = {k: set() for k in range(max_dim + 1)}
    for edge in hyperedges:
        edge = tuple(sorted(set(int(v) for v in edge)))
        for r in range(1, min(len(edge), max_dim + 1) + 1):
            for face in itertools.combinations(edge, r):
                simplices[r - 1].add(tuple(face))
    return {k: sorted(v) for k, v in simplices.items()}


def boundary_rank_mod2(domain: List[Tuple[int, ...]], codomain: List[Tuple[int, ...]]) -> int:
    if not domain or not codomain:
        return 0
    codomain_index = {simplex: i for i, simplex in enumerate(codomain)}
    mat = np.zeros((len(codomain), len(domain)), dtype=np.uint8)
    for j, simplex in enumerate(domain):
        for face in itertools.combinations(simplex, len(simplex) - 1):
            i = codomain_index[tuple(face)]
            mat[i, j] ^= 1
    return rank_mod2(mat)


def rank_mod2(mat: np.ndarray) -> int:
    mat = np.array(mat, dtype=np.uint8, copy=True)
    rows, cols = mat.shape
    rank = 0
    col = 0
    while rank < rows and col < cols:
        pivot = None
        for r in range(rank, rows):
            if mat[r, col]:
                pivot = r
                break
        if pivot is None:
            col += 1
            continue
        if pivot != rank:
            mat[[rank, pivot]] = mat[[pivot, rank]]
        for r in range(rows):
            if r != rank and mat[r, col]:
                mat[r] ^= mat[rank]
        rank += 1
        col += 1
    return rank


def simplicial_homology_summary(hyperedges: Sequence[Sequence[int]], max_dim: int = 2) -> Dict[str, Any]:
    # Hyperedges induce a simplicial complex via all subsets; Betti numbers are
    # then computed over Z2 using boundary ranks.
    simplices = simplicial_complex_from_hyperedges(hyperedges, max_dim=max_dim)
    counts = {dim: len(simplices[dim]) for dim in simplices}
    boundary_ranks = {
        dim: boundary_rank_mod2(simplices[dim], simplices[dim - 1])
        for dim in range(1, max_dim + 1)
    }
    betti: Dict[int, int] = {}
    for dim in range(max_dim + 1):
        n_k = counts.get(dim, 0)
        rank_k = boundary_ranks.get(dim, 0)
        rank_kp1 = boundary_ranks.get(dim + 1, 0)
        betti[dim] = int(n_k - rank_k - rank_kp1)
    euler = float(sum(((-1) ** dim) * counts.get(dim, 0) for dim in range(max_dim + 1)))
    return {
        "simplex_counts": counts,
        "betti_numbers": betti,
        "euler_characteristic": euler,
    }


def compute_graph_hypergraph_properties(memory: PhotonicQuantumMEPSMemory, threshold: float = 1e-12) -> Dict[str, Any]:
    transitions: List[Dict[str, Any]] = []
    for t in range(memory.num_layers - 1):
        # Every adjacent layer pair contributes one weighted bipartite graph and
        # two induced hypergraphs (source-side and target-side).
        h = transition_hopping_matrix(memory, t)
        mask = np.abs(h) > threshold
        edge_count = int(mask.sum())
        left_size, right_size = h.shape
        total_possible = left_size * right_size
        row_strength = np.sum(np.abs(h), axis=1)
        col_strength = np.sum(np.abs(h), axis=0)
        adjacency = weighted_bipartite_adjacency(h)
        support = support_adjacency(h, threshold=threshold)
        eigvals = np.linalg.eigvals(adjacency)
        spectral_radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        degrees = np.sum(adjacency, axis=1)
        laplacian = np.diag(degrees) - adjacency
        lap_eigs = np.sort(np.real(np.linalg.eigvals(laplacian)))
        algebraic_connectivity = float(lap_eigs[1]) if lap_eigs.size > 1 else 0.0
        weights = np.abs(h[mask])
        weight_probs = weights / np.sum(weights) if weights.size and np.sum(weights) > threshold else np.array([], dtype=float)
        weight_entropy = float(-np.sum(weight_probs * np.log(weight_probs))) if weight_probs.size else 0.0
        dists, paths = all_pairs_shortest_paths(support)
        finite = dists[np.isfinite(dists) & (dists > 0)]
        components = graph_components(support)
        left_hyperedges = [tuple(np.flatnonzero(mask[i])) for i in range(mask.shape[0]) if np.any(mask[i])]
        right_hyperedges = [tuple(np.flatnonzero(mask[:, j])) for j in range(mask.shape[1]) if np.any(mask[:, j])]
        left_homology = simplicial_homology_summary(left_hyperedges, max_dim=2)
        right_homology = simplicial_homology_summary(right_hyperedges, max_dim=2)
        degree_cent = degree_centrality(support)
        closeness_cent = closeness_centrality(support)
        betweenness_cent = betweenness_centrality(support)
        eigen_cent = eigenvector_centrality(adjacency)
        cheeger = cheeger_constant_sweep(adjacency)
        transitions.append({
            "transition": t,
            "shape": h.shape,
            "edge_count": edge_count,
            "density": float(edge_count / max(total_possible, 1)),
            "mean_abs_weight": float(np.mean(weights)) if weights.size else 0.0,
            "max_abs_weight": float(np.max(weights)) if weights.size else 0.0,
            "spectral_radius": spectral_radius,
            "algebraic_connectivity": algebraic_connectivity,
            "row_strength_mean": float(np.mean(row_strength)) if row_strength.size else 0.0,
            "col_strength_mean": float(np.mean(col_strength)) if col_strength.size else 0.0,
            "weight_entropy": weight_entropy,
            "connected_components": int(len(components)),
            "largest_component_size": int(max((len(comp) for comp in components), default=0)),
            "avg_shortest_path": float(np.mean(finite)) if finite.size else 0.0,
            "diameter": float(np.max(finite)) if finite.size else 0.0,
            "path_jaccard_similarity": mean_path_jaccard_similarity(paths, support.shape[0]),
            "path_length_similarity": path_length_similarity(dists),
            "degree_centrality_mean": float(np.mean(degree_cent)) if degree_cent.size else 0.0,
            "closeness_centrality_mean": float(np.mean(closeness_cent)) if closeness_cent.size else 0.0,
            "betweenness_centrality_mean": float(np.mean(betweenness_cent)) if betweenness_cent.size else 0.0,
            "eigenvector_centrality_mean": float(np.mean(eigen_cent)) if eigen_cent.size else 0.0,
            "max_degree_centrality": float(np.max(degree_cent)) if degree_cent.size else 0.0,
            "max_closeness_centrality": float(np.max(closeness_cent)) if closeness_cent.size else 0.0,
            "max_betweenness_centrality": float(np.max(betweenness_cent)) if betweenness_cent.size else 0.0,
            "max_eigenvector_centrality": float(np.max(eigen_cent)) if eigen_cent.size else 0.0,
            **cheeger,
            "source_hypergraph": hypergraph_cardinality_stats(h, axis=1, threshold=threshold),
            "target_hypergraph": hypergraph_cardinality_stats(h, axis=0, threshold=threshold),
            "source_homology": left_homology,
            "target_homology": right_homology,
        })
    return {"transitions": transitions}


def compute_wave_particle_quantities(
    memory: PhotonicQuantumMEPSMemory,
    dataset: AnimalDataset,
    detector_measurement: Optional[DetectorMeasurementConfig] = None,
) -> Dict[str, Any]:
    # Wave-like behaviour is proxied by photonic coherence, while particle-like
    # which-way information is proxied by detector distinguishability.
    per_layer: List[Dict[str, float]] = []
    all_samples: List[List[Dict[str, float]]] = [[] for _ in range(memory.num_layers - 1)]
    for item in dataset.animals:
        percept = dataset.encode_photonic_percept(item)
        _, info = memory.deliberate_state(percept, detector_measurement=detector_measurement)
        for layer in range(memory.num_layers - 1):
            phot_state = info["layer_states"][layer]
            det_state = info["detector_states"][layer]
            det_vacuum = ket_to_dm(memory.detector_spaces[layer].vacuum_state())
            coherence = normalized_l1_coherence(phot_state)
            distinguishability = trace_distance(det_state, det_vacuum)
            hs_dist = hilbert_schmidt_distance(det_state, det_vacuum)
            fid = fidelity(det_state, det_vacuum)
            bures = bures_distance(det_state, det_vacuum)
            qjsd = quantum_jensen_shannon_divergence(det_state, det_vacuum)
            all_samples[layer].append({
                "visibility_coherence": coherence,
                "distinguishability_trace": distinguishability,
                "duality_balance": coherence ** 2 + distinguishability ** 2,
                "detector_hilbert_schmidt": hs_dist,
                "detector_fidelity_to_vacuum": fid,
                "detector_bures_to_vacuum": bures,
                "detector_qjsd_to_vacuum": qjsd,
                "photonic_purity": purity(phot_state),
                "detector_purity": purity(det_state),
                "photonic_entropy": von_neumann_entropy(phot_state),
                "detector_entropy": von_neumann_entropy(det_state),
            })

    for layer, samples in enumerate(all_samples):
        if not samples:
            per_layer.append({"layer": float(layer)})
            continue
        keys = samples[0].keys()
        summary = {"layer": float(layer)}
        for key in keys:
            summary[key] = float(np.mean([sample[key] for sample in samples]))
        per_layer.append(summary)
    return {"per_layer": per_layer}


## Main agent class


class PhotonicAnimalAgent:
    def __init__(self, name: str, memory: PhotonicQuantumMEPSMemory, learning: Optional[LearningConfig] = None, seed: int = 0) -> None:
        self.name = name
        self.memory = memory
        self.learning = learning or LearningConfig()
        self.learning.validate()
        self.rng = random.Random(seed)
        self.episode_buffer = RolloutBuffer()
        self.replay = ExperienceReplayBuffer(
            capacity=self.learning.replay_capacity,
            alpha=self.learning.replay_alpha,
            seed=seed,
        )
        self.detector_measurement = DetectorMeasurementConfig()

    def set_detector_measurement(self, config: Optional[DetectorMeasurementConfig]) -> None:
        # Normalizing once here keeps later policy/objective code simple.
        self.detector_measurement = (config or DetectorMeasurementConfig()).normalized(
            self.memory.num_layers,
            self.memory.detector_cutoff,
        )

    def encode_percept(self, percept: Any) -> Tuple[int, ...]:
        if isinstance(percept, np.ndarray):
            return tuple(int(x) for x in np.flatnonzero(percept > 0.5))
        if isinstance(percept, (list, tuple)):
            return tuple(int(x) for x in percept)
        return (int(percept),)

    def act(self, percept: Any) -> Tuple[int, Dict[str, Any]]:
        key = self.encode_percept(percept)
        # Action selection uses sampled mid-circuit measurements because this is
        # the actual stochastic trajectory experienced by the agent.
        probs, info = self.memory.policy(
            key,
            temperature=self.learning.temperature,
            detector_measurement=self.detector_measurement,
            sample_measurements=True,
        )
        action = self.rng.choices(range(len(probs)), weights=probs, k=1)[0]
        log_prob = float(math.log(max(probs[action], 1e-12)))
        entropy = float(-np.sum(probs * safe_log(probs)))
        preds = self.memory.taxonomy_predictions(key, detector_measurement=self.detector_measurement)
        detector_occupations = [
            float(np.real(np.trace(dm @ self.memory.detector_n_ops[idx])))
            for idx, dm in enumerate(info.get("detector_states", []))
        ]
        info = dict(info)
        info.update({
            "policy": probs.copy(),
            "log_prob": log_prob,
            "entropy": entropy,
            "encoded_percept": key,
            "predicted_functionality": preds["functionality"],
            "predicted_family": preds["family"],
            "detector_occupations": detector_occupations,
            "measurement_records": info.get("measurement_records", []),
        })
        return action, info

    def observe(self, reward: float, done: bool, action: int, info: Dict[str, Any]) -> None:
        self.episode_buffer.append(Transition(
            percept=tuple(info["encoded_percept"]),
            action=int(action),
            reward=float(reward),
            done=bool(done),
            log_prob=float(info["log_prob"]),
            entropy=float(info["entropy"]),
            policy=np.asarray(info["policy"], dtype=float),
            info={k: v for k, v in info.items() if k not in {"policy", "log_prob", "entropy", "encoded_percept"}},
        ))

    def _policy_only(self, percept: Tuple[int, ...]) -> np.ndarray:
        # Learning uses the deterministic non-selective measurement channel so
        # finite differences and QFIM are evaluated on a stable objective.
        probs, _ = self.memory.policy(
            percept,
            temperature=self.learning.temperature,
            detector_measurement=self.detector_measurement,
            sample_measurements=False,
        )
        return probs

    def _variational_objective(
        self,
        transitions: Sequence[Transition],
        returns: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
    ) -> float:
        cfg = self.learning
        if sample_weights is None:
            sample_weights = np.ones(len(transitions), dtype=float)
        # The replay weights re-balance the prioritized sample when forming the objective.
        total = 0.0
        norm = float(np.sum(sample_weights)) if len(sample_weights) else 1.0
        for tr, ret, weight in zip(transitions, returns, sample_weights):
            probs = self._policy_only(tr.percept)
            oldp = tr.policy[tr.action]
            target = oldp - cfg.forgetting * (oldp - 1.0 / len(probs)) + cfg.return_scale * ret
            target = float(np.clip(target, 0.0, 1.0))
            total += weight * 0.5 * (probs[tr.action] - target) ** 2
            total -= weight * cfg.entropy_bonus * float(-np.sum(probs * safe_log(probs)))
        return float(total / max(norm, 1e-12))

    def _finite_difference_gradient(
        self,
        transitions: Sequence[Transition],
        returns: np.ndarray,
        sample_weights: np.ndarray,
    ) -> np.ndarray:
        # Each gradient component is computed against the same replay batch so
        # the only difference between fp and fm is the parameter perturbation.
        cfg = self.learning
        theta0 = self.memory.get_parameter_vector()
        grad = np.zeros_like(theta0)
        base = self.memory.copy_params()
        for k in range(len(theta0)):
            tp = theta0.copy()
            tm = theta0.copy()
            tp[k] += cfg.fd_eps
            tm[k] -= cfg.fd_eps
            self.memory.set_parameter_vector(tp)
            fp = self._variational_objective(transitions, returns, sample_weights)
            self.memory.set_parameter_vector(tm)
            fm = self._variational_objective(transitions, returns, sample_weights)
            grad[k] = (fp - fm) / (2.0 * cfg.fd_eps)
        self.memory.restore_params(base)
        return grad

    def _returns_from_transitions(self, transitions: Sequence[Transition], gamma: float) -> np.ndarray:
        out = np.zeros(len(transitions), dtype=float)
        G = 0.0
        for i in reversed(range(len(transitions))):
            if transitions[i].done:
                G = 0.0
            G = transitions[i].reward + gamma * G
            out[i] = G
        return out

    def update(self) -> Dict[str, float]:
        if len(self.episode_buffer) == 0:
            return {"policy_loss": 0.0, "qfim_trace": 0.0, "avg_return": 0.0}
        cfg = self.learning
        episode_returns = self.episode_buffer.discounted_returns(cfg.gamma)
        for tr, ret in zip(self.episode_buffer.data, episode_returns):
            # Reward and entropy both contribute to replay priority so rare but
            # informative transitions survive longer.
            priority = abs(float(ret)) + float(tr.entropy) + 1e-6
            self.replay.add(tr, priority=priority)

        replay_batch, replay_weights = self.replay.sample(cfg.replay_batch_size)
        if replay_batch:
            transitions = replay_batch
            returns = self._returns_from_transitions(transitions, cfg.gamma)
            sample_weights = replay_weights
        else:
            transitions = list(self.episode_buffer.data)
            returns = episode_returns
            sample_weights = np.ones(len(transitions), dtype=float)

        grad = self._finite_difference_gradient(transitions, returns, sample_weights)
        qfim_trace = 0.0
        if cfg.learning_mode == "qfim":
            # QFIM is averaged over the percepts in the sampled update batch and
            # used as a natural-gradient preconditioner.
            unique_percepts = sorted(set(tr.percept for tr in transitions))
            Fs = [QuantumFisher.qfim(self.memory, percept, deriv_eps=cfg.qfim_eps) for percept in unique_percepts]
            F = sum(Fs) / len(Fs)
            qfim_trace = float(np.trace(F))
            step = QuantumFisher.natural_gradient(grad, F, reg=cfg.qfim_reg)
        else:
            step = grad
        self.memory.apply_update(-cfg.lr_policy * step)
        policy_loss = float(self._variational_objective(transitions, returns, sample_weights))
        cfg.return_scale = max(cfg.return_scale_min, cfg.return_scale * cfg.return_scale_decay)
        avg_return = float(np.mean(episode_returns)) if len(episode_returns) else 0.0
        self.episode_buffer.clear()
        return {
            "policy_loss": policy_loss,
            "qfim_trace": qfim_trace,
            "avg_return": avg_return,
        }


## Visualization utilities


def plot_training_results(
    result: Dict[str, Any],
    agent: PhotonicAnimalAgent,
    dataset: AnimalDataset,
    output_path: Optional[str] = None,
    show_plot: bool = False,
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    history_mean = result.get("history_mean", {})
    history_std = result.get("history_std", {})
    rewards = np.asarray(history_mean.get("reward", []), dtype=float)
    rewards_std = np.asarray(history_std.get("reward", np.zeros_like(rewards)), dtype=float)
    species = np.asarray(history_mean.get("species_correct", []), dtype=float)
    species_std = np.asarray(history_std.get("species_correct", np.zeros_like(species)), dtype=float)
    functionality = np.asarray(history_mean.get("functionality_correct", []), dtype=float)
    functionality_std = np.asarray(history_std.get("functionality_correct", np.zeros_like(functionality)), dtype=float)
    family = np.asarray(history_mean.get("family_correct", []), dtype=float)
    family_std = np.asarray(history_std.get("family_correct", np.zeros_like(family)), dtype=float)
    policy_loss = np.asarray(history_mean.get("policy_loss", []), dtype=float)
    qfim_trace = np.asarray(history_mean.get("qfim_trace", []), dtype=float)
    detector_occupations = np.asarray(history_mean.get("detector_occupations", []), dtype=float)
    measurement_heatmap = np.asarray(result.get("measurement_heatmap_mean", []), dtype=float)
    priors = dataset.transition_priors()

    # Plots use run-averaged trajectories so variability across seeds is visible.
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    ax = axes[0, 0]
    if rewards.size:
        ax.plot(rewards, color="#0f766e", label="reward")
        ax.plot(np.cumsum(rewards) / np.arange(1, len(rewards) + 1), color="#14b8a6", label="running mean")
        ax.fill_between(np.arange(len(rewards)), rewards - rewards_std, rewards + rewards_std, color="#99f6e4", alpha=0.3)
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.legend(loc="best")

    ax = axes[0, 1]
    if species.size:
        ax.plot(species, label="species", color="#1d4ed8")
        ax.plot(functionality, label="functionality", color="#16a34a")
        ax.plot(family, label="family", color="#dc2626")
        ax.fill_between(np.arange(len(species)), species - species_std, species + species_std, color="#93c5fd", alpha=0.15)
        ax.fill_between(np.arange(len(functionality)), functionality - functionality_std, functionality + functionality_std, color="#86efac", alpha=0.15)
        ax.fill_between(np.arange(len(family)), family - family_std, family + family_std, color="#fca5a5", alpha=0.15)
    ax.set_title("Episode Accuracy")
    ax.set_xlabel("Episode")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")

    ax = axes[0, 2]
    if policy_loss.size:
        ax.plot(policy_loss, label="policy loss", color="#7c3aed")
    if qfim_trace.size:
        ax2 = ax.twinx()
        ax2.plot(qfim_trace, label="qfim trace", color="#f59e0b")
        ax2.set_ylabel("QFIM Trace")
    ax.set_title("Optimization Diagnostics")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Policy Loss")

    for idx in range(3):
        ax = axes[1, idx]
        learned = np.asarray(result.get("learned_hopping_mean", [np.zeros_like(priors[idx])])[idx], dtype=float)
        combined = np.concatenate([priors[idx], learned], axis=0)
        im = ax.imshow(combined, aspect="auto", cmap="viridis")
        ax.set_title(f"Transition {idx} Prior/Learned")
        ax.set_xlabel("Target mode")
        ax.set_ylabel("Source mode")
        split = priors[idx].shape[0] - 0.5
        ax.axhline(split, color="white", linewidth=1.0, linestyle="--")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2, 0]
    if detector_occupations.size:
        for layer in range(detector_occupations.shape[1]):
            ax.plot(detector_occupations[:, layer], label=f"layer {layer}", linewidth=1.5)
    ax.set_title("Detector Occupation")
    ax.set_xlabel("Episode")
    ax.set_ylabel("<n_d>")
    if detector_occupations.size:
        ax.legend(loc="best")

    ax = axes[2, 1]
    if measurement_heatmap.size:
        im = ax.imshow(measurement_heatmap, aspect="auto", cmap="magma")
        ax.set_title("Mid-Circuit Detector Measurements")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Detector state")
        ax.set_xticks(range(measurement_heatmap.shape[1]))
        ax.set_yticks(range(measurement_heatmap.shape[0]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.set_title("Mid-Circuit Detector Measurements")
        ax.text(0.5, 0.5, "No measurements", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[2, 2]
    if detector_occupations.size:
        final_detector = detector_occupations[-1]
        ax.bar(np.arange(len(final_detector)), final_detector, color="#0f766e")
    ax.set_title("Final Detector Occupation by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("<n_d>")

    fig.suptitle(
        f"Photonic QMEPS Results | mode={result['learning_mode']} | photons={result['max_total_excitation']} | runs={result.get('num_runs', 1)}",
        fontsize=14,
    )
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return output_path


## Main training loop and evaluation


def build_photonic_animal_agent(
    seed: int = 0,
    learning_mode: str = "qfim",
    max_total_excitation: int = 2,
    coupling_radius: int = 0,
    detector_measurement: Optional[DetectorMeasurementConfig] = None,
) -> Tuple[AnimalDataset, PhotonicAnimalAgent]:
    dataset = AnimalDataset()
    layer_specs = [
        QMEPSLayerSpec("percept", len(dataset.photonic_percept_modes)),
        QMEPSLayerSpec("functionality", len(dataset.functionality)),
        QMEPSLayerSpec("family", len(dataset.families)),
        QMEPSLayerSpec("species", len(dataset.species)),
    ]
    memory = PhotonicQuantumMEPSMemory(
        layer_specs=layer_specs,
        max_total_excitation=max_total_excitation,
        dt_schedule=(0.28, 0.25, 0.22),
        seed=seed,
        coupling_radius=coupling_radius,
        trainable_components=("hopping",),
    )
    # Bias the initial Hamiltonian toward the animal taxonomy before learning begins.
    memory.inject_task_priors(dataset.transition_priors(), hopping_strength=0.55, onsite_strength=0.05)
    learning = LearningConfig(learning_mode=learning_mode)
    agent = PhotonicAnimalAgent("photonic_qmeps_animals", memory=memory, learning=learning, seed=seed)
    agent.set_detector_measurement(detector_measurement)
    return dataset, agent


def evaluate_agent(
    agent: PhotonicAnimalAgent,
    dataset: AnimalDataset,
    episodes: int = 6,
    seed: int = 0,
) -> Dict[str, float]:
    # Evaluation reuses the current policy and detector settings, but does not
    # perform any parameter updates.
    env = AnimalClassificationEnv(dataset, dense_reward_weight=0.0, seed=seed)
    species_correct = 0
    func_correct = 0
    family_correct = 0
    measurement_heatmap = np.zeros((agent.memory.detector_cutoff + 1, agent.memory.num_layers - 1), dtype=float)
    for _ in range(episodes):
        obs = env.reset()
        action, info = agent.act(obs)
        _, reward, _, env_info = env.step(action)
        species_correct += int(reward > 0.5)
        func_correct += int(info["predicted_functionality"] in set(env_info["correct_functionality"]))
        family_correct += int(info["predicted_family"] == env_info["correct_family"])
        for record in info.get("measurement_records", []):
            outcome = record.get("outcome")
            layer = record.get("layer")
            if isinstance(outcome, int) and outcome >= 0 and layer is not None:
                measurement_heatmap[outcome, int(layer)] += 1.0
    return {
        "species_accuracy": species_correct / max(episodes, 1),
        "functionality_accuracy": func_correct / max(episodes, 1),
        "family_accuracy": family_correct / max(episodes, 1),
        "measurement_heatmap": measurement_heatmap,
    }


def run_training(
    episodes: int = 8,
    eval_episodes: int = 6,
    seed: int = 0,
    dense_reward_weight: float = 0.15,
    learning_mode: str = "qfim",
    max_total_excitation: int = 2,
    coupling_radius: int = 0,
    detector_measurement: Optional[DetectorMeasurementConfig] = None,
) -> Dict[str, Any]:
    # Single-run training entry point. Multi-run aggregation is handled by
    # run_experiment_suite further below.
    dataset, agent = build_photonic_animal_agent(
        seed=seed,
        learning_mode=learning_mode,
        max_total_excitation=max_total_excitation,
        coupling_radius=coupling_radius,
        detector_measurement=detector_measurement,
    )
    env = AnimalClassificationEnv(dataset, dense_reward_weight=dense_reward_weight, seed=seed + 17)
    returns: List[float] = []
    diagnostics: List[Dict[str, float]] = []
    history: Dict[str, List[float]] = {
        "reward": [],
        "species_correct": [],
        "functionality_correct": [],
        "family_correct": [],
        "policy_loss": [],
        "qfim_trace": [],
        "avg_return": [],
        "detector_occupations": [],
    }
    measurement_heatmap = np.zeros((agent.memory.detector_cutoff + 1, agent.memory.num_layers - 1), dtype=float)
    species_correct = 0
    func_correct = 0
    family_correct = 0

    for _ in range(episodes):
        obs = env.reset()
        action, act_info = agent.act(obs)
        _, reward, done, env_info = env.step(action)
        func_ok = act_info["predicted_functionality"] in set(env_info["correct_functionality"])
        family_ok = act_info["predicted_family"] == env_info["correct_family"]
        dense_bonus = 0.0
        if func_ok:
            dense_bonus += dense_reward_weight
        if family_ok:
            dense_bonus += dense_reward_weight
        total_reward = reward + dense_bonus
        species_correct += int(reward > 0.5)
        func_correct += int(func_ok)
        family_correct += int(family_ok)
        agent.observe(total_reward, done, action, act_info)
        diag = agent.update()
        diagnostics.append(diag)
        returns.append(total_reward)
        history["reward"].append(total_reward)
        history["species_correct"].append(float(reward > 0.5))
        history["functionality_correct"].append(float(func_ok))
        history["family_correct"].append(float(family_ok))
        history["policy_loss"].append(float(diag["policy_loss"]))
        history["qfim_trace"].append(float(diag["qfim_trace"]))
        history["avg_return"].append(float(diag["avg_return"]))
        history["detector_occupations"].append(list(act_info.get("detector_occupations", [])))
        for record in act_info.get("measurement_records", []):
            outcome = record.get("outcome")
            layer = record.get("layer")
            if isinstance(outcome, int) and outcome >= 0 and layer is not None:
                measurement_heatmap[outcome, int(layer)] += 1.0

    evaluation = evaluate_agent(agent, dataset, episodes=eval_episodes, seed=seed + 31)
    graph_properties = compute_graph_hypergraph_properties(agent.memory)
    duality_quantities = compute_wave_particle_quantities(
        agent.memory,
        dataset,
        detector_measurement=agent.detector_measurement,
    )
    return {
        "agent": agent,
        "dataset": dataset,
        "episodes": episodes,
        "learning_mode": learning_mode,
        "max_total_excitation": max_total_excitation,
        "coupling_radius": coupling_radius,
        "detector_measurement": agent.detector_measurement,
        "mean_training_return": float(np.mean(returns)) if returns else 0.0,
        "training_species_accuracy": species_correct / max(episodes, 1),
        "training_functionality_accuracy": func_correct / max(episodes, 1),
        "training_family_accuracy": family_correct / max(episodes, 1),
        "last_diagnostics": diagnostics[-1] if diagnostics else {},
        "evaluation": evaluation,
        "history": history,
        "measurement_heatmap": measurement_heatmap,
        "graph_properties": graph_properties,
        "duality_quantities": duality_quantities,
    }


def _stack_numeric_dicts(dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not dicts:
        return {}
    out: Dict[str, Any] = {}
    keys = dicts[0].keys()
    for key in keys:
        vals = [d[key] for d in dicts]
        if isinstance(vals[0], dict):
            out[key] = _stack_numeric_dicts(vals)
        elif isinstance(vals[0], np.ndarray):
            arr = np.asarray(vals, dtype=float)
            out[key] = {"mean": np.mean(arr, axis=0), "std": np.std(arr, axis=0)}
        elif isinstance(vals[0], (int, float, np.floating)):
            arr = np.asarray(vals, dtype=float)
            out[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
        else:
            out[key] = vals[0]
    return out


def aggregate_run_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    # Aggregate per-seed histories and derived quantities into mean/std summaries
    # used by the plots and CLI output.
    if not results:
        raise ValueError("Need at least one run result to aggregate.")
    num_runs = len(results)
    dataset = results[0]["dataset"]
    representative_agent = results[-1]["agent"]

    history_keys = results[0]["history"].keys()
    history_mean: Dict[str, Any] = {}
    history_std: Dict[str, Any] = {}
    for key in history_keys:
        arr = np.asarray([res["history"][key] for res in results], dtype=float)
        history_mean[key] = np.mean(arr, axis=0)
        history_std[key] = np.std(arr, axis=0)

    measurement_heatmaps = np.asarray([res["measurement_heatmap"] for res in results], dtype=float)
    learned_hopping_mean: List[np.ndarray] = []
    for t in range(representative_agent.memory.num_layers - 1):
        mats = np.asarray([transition_hopping_matrix(res["agent"].memory, t) for res in results], dtype=float)
        learned_hopping_mean.append(np.mean(mats, axis=0))

    graph_summary = []
    for t in range(representative_agent.memory.num_layers - 1):
        per_transition = [res["graph_properties"]["transitions"][t] for res in results]
        graph_summary.append(_stack_numeric_dicts(per_transition))

    duality_summary = []
    for layer in range(representative_agent.memory.num_layers - 1):
        per_layer = [res["duality_quantities"]["per_layer"][layer] for res in results]
        duality_summary.append(_stack_numeric_dicts(per_layer))

    def metric_mean_std(key: str) -> Tuple[float, float]:
        vals = np.asarray([res[key] for res in results], dtype=float)
        return float(np.mean(vals)), float(np.std(vals))

    evaluation_keys = results[0]["evaluation"].keys()
    evaluation_summary: Dict[str, Dict[str, float]] = {}
    for key in evaluation_keys:
        sample0 = results[0]["evaluation"][key]
        vals = np.asarray([res["evaluation"][key] for res in results], dtype=float)
        if isinstance(sample0, np.ndarray):
            evaluation_summary[key] = {"mean": np.mean(vals, axis=0), "std": np.std(vals, axis=0)}
        else:
            evaluation_summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return {
        "num_runs": num_runs,
        "episodes": results[0]["episodes"],
        "learning_mode": results[0]["learning_mode"],
        "max_total_excitation": results[0]["max_total_excitation"],
        "coupling_radius": results[0]["coupling_radius"],
        "detector_measurement": results[0]["detector_measurement"],
        "dataset": dataset,
        "agent": representative_agent,
        "runs": list(results),
        "history_mean": history_mean,
        "history_std": history_std,
        "measurement_heatmap_mean": np.mean(measurement_heatmaps, axis=0),
        "measurement_heatmap_std": np.std(measurement_heatmaps, axis=0),
        "learned_hopping_mean": learned_hopping_mean,
        "mean_training_return": metric_mean_std("mean_training_return")[0],
        "std_training_return": metric_mean_std("mean_training_return")[1],
        "training_species_accuracy": metric_mean_std("training_species_accuracy")[0],
        "std_training_species_accuracy": metric_mean_std("training_species_accuracy")[1],
        "training_functionality_accuracy": metric_mean_std("training_functionality_accuracy")[0],
        "std_training_functionality_accuracy": metric_mean_std("training_functionality_accuracy")[1],
        "training_family_accuracy": metric_mean_std("training_family_accuracy")[0],
        "std_training_family_accuracy": metric_mean_std("training_family_accuracy")[1],
        "last_diagnostics": _stack_numeric_dicts([res["last_diagnostics"] for res in results]),
        "evaluation": evaluation_summary,
        "graph_properties": {"transitions": graph_summary},
        "duality_quantities": {"per_layer": duality_summary},
    }


def run_experiment_suite(
    num_runs: int = 1,
    seed: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    # Outer loop for repeated experiments with different seeds.
    results = [run_training(seed=seed + run_idx, **kwargs) for run_idx in range(num_runs)]
    return aggregate_run_results(results)


## Entry point


def main() -> None:
    parser = argparse.ArgumentParser(description="Photonic multi-photon quantum-MEPS animal classifier")
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dense-reward-weight", type=float, default=0.15)
    parser.add_argument("--learning-mode", choices=["qfim", "variational"], default="qfim")
    parser.add_argument("--max-photons", type=int, default=2)
    parser.add_argument("--coupling-radius", type=int, default=0)
    parser.add_argument("--measure-layers", type=str, default="")
    parser.add_argument("--measure-detector-states", type=str, default="")
    parser.add_argument("--max-mid-measurements", type=int, default=None)
    parser.add_argument("--plot-path", type=str, default="photonic_qmeps_animals_results.png")
    parser.add_argument("--show-plot", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    measurement_config = DetectorMeasurementConfig(
        layers=parse_int_list(args.measure_layers),
        states=parse_int_list(args.measure_detector_states),
        max_measurements=args.max_mid_measurements,
    )

    result = run_experiment_suite(
        num_runs=args.num_runs,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
        dense_reward_weight=args.dense_reward_weight,
        learning_mode=args.learning_mode,
        max_total_excitation=args.max_photons,
        coupling_radius=args.coupling_radius,
        detector_measurement=measurement_config,
    )
    plot_path = None
    if not args.no_plot:
        plot_path = plot_training_results(
            result=result,
            agent=result["agent"],
            dataset=result["dataset"],
            output_path=args.plot_path,
            show_plot=args.show_plot,
        )
    print("Photonic QMEPS animal classification")
    print("Learning mode:", result["learning_mode"])
    print("Runs:", result["num_runs"])
    print("Episodes:", result["episodes"])
    print("Max photons:", result["max_total_excitation"])
    print("Coupling radius:", result["coupling_radius"])
    print("Detector measurement config:", result["detector_measurement"])
    print("Mean training return:", result["mean_training_return"], "+/-", result["std_training_return"])
    print("Training species accuracy:", result["training_species_accuracy"], "+/-", result["std_training_species_accuracy"])
    print("Training functionality accuracy:", result["training_functionality_accuracy"], "+/-", result["std_training_functionality_accuracy"])
    print("Training family accuracy:", result["training_family_accuracy"], "+/-", result["std_training_family_accuracy"])
    print("Last diagnostics:", result["last_diagnostics"])
    print("Evaluation:", result["evaluation"])
    print("Graph/hypergraph properties:")
    for entry in result["graph_properties"]["transitions"]:
        print(
            f"  layer {int(entry['transition']['mean'])}: edges={entry['edge_count']['mean']:.3f}±{entry['edge_count']['std']:.3f}, "
            f"density={entry['density']['mean']:.3f}±{entry['density']['std']:.3f}, "
            f"spectral_radius={entry['spectral_radius']['mean']:.3f}±{entry['spectral_radius']['std']:.3f}, "
            f"cheeger~={entry['cheeger_estimate']['mean']:.3f}±{entry['cheeger_estimate']['std']:.3f}"
        )
        print(
            f"    centrality: deg={entry['degree_centrality_mean']['mean']:.3f}, "
            f"close={entry['closeness_centrality_mean']['mean']:.3f}, "
            f"between={entry['betweenness_centrality_mean']['mean']:.3f}, "
            f"eigen={entry['eigenvector_centrality_mean']['mean']:.3f}"
        )
        print(
            f"    paths: avg={entry['avg_shortest_path']['mean']:.3f}, diam={entry['diameter']['mean']:.3f}, "
            f"jaccard={entry['path_jaccard_similarity']['mean']:.3f}, len_sim={entry['path_length_similarity']['mean']:.3f}"
        )
        print(
            f"    source homology: betti={entry['source_homology']['betti_numbers']}, "
            f"euler={entry['source_homology']['euler_characteristic']['mean']:.1f}"
        )
        print(
            f"    target homology: betti={entry['target_homology']['betti_numbers']}, "
            f"euler={entry['target_homology']['euler_characteristic']['mean']:.1f}"
        )
    print("Wave-particle quantities:")
    for entry in result["duality_quantities"]["per_layer"]:
        print(
            f"  layer {int(entry['layer']['mean'])}: V={entry['visibility_coherence']['mean']:.3f}±{entry['visibility_coherence']['std']:.3f}, "
            f"D={entry['distinguishability_trace']['mean']:.3f}±{entry['distinguishability_trace']['std']:.3f}, "
            f"V^2+D^2={entry['duality_balance']['mean']:.3f}, "
            f"F(det,vac)={entry['detector_fidelity_to_vacuum']['mean']:.3f}, "
            f"QJSD={entry['detector_qjsd_to_vacuum']['mean']:.3f}"
        )
    if plot_path is not None:
        print("Plot saved to:", plot_path)


if __name__ == "__main__":
    main()
