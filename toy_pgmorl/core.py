from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


GradientMode = Literal["analytic", "reinforce"]


@dataclass(frozen=True)
class ToyConfig:
    seed: int = 7
    population_size: int = 7
    warmup_steps: int = 35
    task_steps: int = 12
    generations: int = 12
    learning_rate: float = 0.035
    policy_std: float = 0.6
    batch_size: int = 128
    weight_resolution: int = 8
    neighborhood_threshold: float = 0.1
    predictor_sigma: float = 0.03
    sparsity_coef: float = 0.18
    reward_offset: float = 50.0
    objective_targets: tuple[float, ...] = (8.0, 2.0, 5.0)
    objective_names: tuple[str, ...] = ("빠른 도착", "배터리 효율", "안전성")
    init_mu_low: float = 0.0
    init_mu_high: float = 10.0
    gradient_mode: GradientMode = "reinforce"

    @property
    def num_objectives(self) -> int:
        return len(self.objective_targets)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyState:
    policy_id: int
    mu: float
    evaluation: np.ndarray
    weight: np.ndarray
    generation: int
    method: str
    source: str
    parent_policy_id: int | None = None


@dataclass
class TransitionRecord:
    before_eval: np.ndarray
    after_eval: np.ndarray
    weight: np.ndarray
    generation: int
    parent_policy_id: int


@dataclass
class TrainOutcome:
    final_mu: float
    before_eval: np.ndarray
    after_eval: np.ndarray
    inner_records: list[dict[str, Any]]


@dataclass
class ExperimentResult:
    config: ToyConfig
    metrics: pd.DataFrame
    population: pd.DataFrame
    archive: pd.DataFrame
    tasks: pd.DataFrame
    predictor_curves: pd.DataFrame
    inner_steps: pd.DataFrame
    true_front: pd.DataFrame
    method_summary: pd.DataFrame
    mapping: pd.DataFrame


def normalize_preferences(values: Sequence[float]) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    arr = np.clip(arr, 1e-8, None)
    return arr / arr.sum()


def one_hot_weight(index: int, num_objectives: int) -> np.ndarray:
    weights = np.zeros(num_objectives, dtype=np.float64)
    weights[index] = 1.0
    return weights


def closed_form_optimum_mu(weights: Sequence[float], config: ToyConfig) -> float:
    normalized = normalize_preferences(weights)
    targets = np.array(config.objective_targets, dtype=np.float64)
    return float(np.sum(normalized * targets))


def dynamic_weight_fields(weight: np.ndarray) -> dict[str, float]:
    return {f"weight_{i + 1}": float(value) for i, value in enumerate(weight)}


def dynamic_eval_fields(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_{i + 1}": float(value) for i, value in enumerate(values)}


def _integer_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    results: list[tuple[int, ...]] = []
    for first in range(total + 1):
        for rest in _integer_compositions(total - first, parts - 1):
            results.append((first, *rest))
    return results


def generate_weight_grid(resolution: int, dimensions: int) -> np.ndarray:
    resolution = max(int(resolution), 1)
    combos = _integer_compositions(resolution, dimensions)
    weights = np.array(combos, dtype=np.float64) / resolution
    # stable sort: first objective descending, then second, etc.
    sort_keys = [weights[:, i] for i in reversed(range(dimensions))]
    order = np.lexsort(sort_keys)
    return weights[order]


def sample_objectives(actions: np.ndarray, config: ToyConfig) -> np.ndarray:
    targets = np.array(config.objective_targets, dtype=np.float64)
    return np.stack([config.reward_offset - np.square(actions - target) for target in targets], axis=-1)


def expected_objectives(mu: float, config: ToyConfig) -> np.ndarray:
    targets = np.array(config.objective_targets, dtype=np.float64)
    return config.reward_offset - (np.square(mu - targets) + config.policy_std**2)


def analytic_gradient(mu: float, weight: np.ndarray, config: ToyConfig) -> float:
    targets = np.array(config.objective_targets, dtype=np.float64)
    return float(np.sum(weight * (-2.0 * (mu - targets))))


def scalarized_expected_return(mu: float, weight: np.ndarray, config: ToyConfig) -> float:
    return float(expected_objectives(mu, config) @ weight)


def pareto_mask(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not mask[i]:
            continue
        for j in range(len(points)):
            if i == j:
                continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                mask[i] = False
                break
    return mask


def pareto_front(points: Iterable[np.ndarray]) -> np.ndarray:
    point_array = np.asarray(list(points), dtype=np.float64)
    if len(point_array) == 0:
        return np.empty((0, 0), dtype=np.float64)
    point_array = np.unique(np.round(point_array, 10), axis=0)
    front = point_array[pareto_mask(point_array)]
    order = np.argsort(front[:, 0])
    return front[order]


def _hypervolume_2d(points: np.ndarray) -> float:
    front = pareto_front(points)
    if len(front) == 0:
        return 0.0
    hv = 0.0
    prev_x = 0.0
    for x, y in front:
        x = max(0.0, float(x))
        y = max(0.0, float(y))
        hv += max(0.0, x - prev_x) * y
        prev_x = max(prev_x, x)
    return float(hv)


def _hypervolume_3d(points: np.ndarray) -> float:
    front = pareto_front(points)
    if len(front) == 0:
        return 0.0
    xs = np.unique(front[:, 0])
    hv = 0.0
    prev_x = 0.0
    for x in xs:
        width = max(0.0, float(x) - prev_x)
        active = front[front[:, 0] >= x][:, 1:]
        hv += width * _hypervolume_2d(active)
        prev_x = float(x)
    return float(hv)


def hypervolume(points: Iterable[np.ndarray]) -> float:
    point_array = np.asarray(list(points), dtype=np.float64)
    if len(point_array) == 0:
        return 0.0
    point_array = np.maximum(point_array, 0.0)
    dimensions = point_array.shape[1]
    if dimensions == 2:
        return _hypervolume_2d(point_array)
    if dimensions == 3:
        return _hypervolume_3d(point_array)
    raise ValueError(f"Unsupported objective dimension: {dimensions}")


def sparsity(points: Iterable[np.ndarray]) -> float:
    front = pareto_front(points)
    if len(front) < 2:
        return 0.0
    total = 0.0
    for dim in range(front.shape[1]):
        sorted_values = np.sort(front[:, dim])
        total += float(np.sum(np.square(np.diff(sorted_values))))
    return total / (len(front) - 1)


def flatten_state(state: PolicyState, snapshot_generation: int | None = None) -> dict[str, Any]:
    row = {
        "method": state.method,
        "generation": state.generation,
        "snapshot_generation": state.generation if snapshot_generation is None else snapshot_generation,
        "policy_id": state.policy_id,
        "parent_policy_id": state.parent_policy_id,
        "mu": float(state.mu),
        "source": state.source,
    }
    row.update(dynamic_weight_fields(state.weight))
    row.update(dynamic_eval_fields("eval", state.evaluation))
    return row


class PerformanceBuffer:
    def __init__(self, max_size: int):
        self.max_size = max(int(max_size), 2)

    def _deduplicate(self, states: list[PolicyState]) -> list[PolicyState]:
        unique: dict[tuple[float, ...], PolicyState] = {}
        for state in states:
            key = tuple(np.round(state.evaluation, 10))
            if key not in unique or state.generation >= unique[key].generation:
                unique[key] = state
        return list(unique.values())

    def update(self, candidates: Iterable[PolicyState]) -> list[PolicyState]:
        states = self._deduplicate(list(candidates))
        if len(states) <= self.max_size:
            return sorted(states, key=lambda item: item.evaluation[0])

        evals = np.asarray([state.evaluation for state in states], dtype=np.float64)
        mins = evals.min(axis=0)
        ranges = np.maximum(evals.max(axis=0) - mins, 1e-8)
        normalized = (evals - mins) / ranges

        selected = [int(np.argmax(normalized.sum(axis=1)))]
        while len(selected) < self.max_size:
            distances = np.linalg.norm(normalized[:, None, :] - normalized[selected][None, :, :], axis=2).min(axis=1)
            scores = 0.6 * distances + 0.4 * normalized.sum(axis=1)
            scores[selected] = -np.inf
            selected.append(int(np.argmax(scores)))

        chosen = [states[index] for index in selected]
        chosen.sort(key=lambda item: item.evaluation[0])
        return chosen


class ParetoArchive:
    def __init__(self):
        self.states: list[PolicyState] = []

    def add(self, new_states: Iterable[PolicyState]) -> None:
        all_states = self.states + list(new_states)
        if not all_states:
            return
        points = np.asarray([state.evaluation for state in all_states], dtype=np.float64)
        mask = pareto_mask(points)
        kept = [state for state, keep in zip(all_states, mask) if keep]
        unique: dict[tuple[float, ...], PolicyState] = {}
        for state in kept:
            key = tuple(np.round(state.evaluation, 10))
            if key not in unique or state.generation >= unique[key].generation:
                unique[key] = state
        self.states = sorted(unique.values(), key=lambda state: state.evaluation[0])

    @property
    def evaluations(self) -> list[np.ndarray]:
        return [state.evaluation for state in self.states]


class PerformancePredictor:
    def __init__(self, config: ToyConfig):
        self.config = config
        self.records: list[TransitionRecord] = []

    def add(self, record: TransitionRecord) -> None:
        self.records.append(record)

    def _collect_neighbors(self, policy_eval: np.ndarray) -> tuple[list[TransitionRecord], float]:
        if not self.records:
            return [], self.config.predictor_sigma

        scale = np.maximum(np.abs(policy_eval), 1.0)
        threshold = self.config.neighborhood_threshold / 2.0
        sigma = self.config.predictor_sigma / 2.0

        for _ in range(12):
            threshold *= 2.0
            sigma *= 2.0
            neighbors = [
                record
                for record in self.records
                if np.all(np.abs(record.before_eval - policy_eval) < threshold * scale)
            ]
            unique_weights = {tuple(np.round(record.weight, 4)) for record in neighbors}
            if len(unique_weights) >= max(4, self.config.num_objectives + 1):
                return neighbors, sigma
        return list(self.records), sigma

    @staticmethod
    def _hyperbolic_fn(x: np.ndarray, A: float, a: float, b: float, c: float) -> np.ndarray:
        z = np.clip(a * (x - b), -60.0, 60.0)
        exp_term = np.exp(z)
        return A * (exp_term - 1.0) / (exp_term + 1.0) + c

    def _fit_dimension(
        self,
        neighbors: list[TransitionRecord],
        sigma: float,
        policy_eval: np.ndarray,
        candidate_x: np.ndarray,
        dim: int,
    ) -> np.ndarray:
        scale = np.maximum(np.abs(policy_eval), 1.0)
        train_x = np.array([record.weight[dim] for record in neighbors], dtype=np.float64)
        deltas = np.array([record.after_eval - record.before_eval for record in neighbors], dtype=np.float64)
        train_y = deltas[:, dim]
        sample_weights = []
        for record in neighbors:
            distance = np.linalg.norm((record.before_eval - policy_eval) / scale)
            sample_weights.append(np.exp(-((distance / max(sigma, 1e-6)) ** 2) / 2.0))
        sample_weights = np.array(sample_weights, dtype=np.float64)

        if np.allclose(train_y, train_y[0]):
            return np.full_like(candidate_x, fill_value=float(train_y[0]))

        def residuals(params: np.ndarray) -> np.ndarray:
            return (self._hyperbolic_fn(train_x, *params) - train_y) * sample_weights

        def jacobian(params: np.ndarray) -> np.ndarray:
            A, a, b, _ = params
            z = np.clip(a * (train_x - b), -60.0, 60.0)
            exp_term = np.exp(z)
            denom = np.square(exp_term + 1.0)
            jac = np.zeros((len(train_x), 4), dtype=np.float64)
            jac[:, 0] = ((exp_term - 1.0) / (exp_term + 1.0)) * sample_weights
            jac[:, 1] = (A * (train_x - b) * (2.0 * exp_term) / denom) * sample_weights
            jac[:, 2] = (A * (-a) * (2.0 * exp_term) / denom) * sample_weights
            jac[:, 3] = sample_weights
            return jac

        A_bound = float(np.clip(np.max(train_y) - np.min(train_y), 1.0, 500.0))
        initial_guess = np.array([max(A_bound / 2.0, 1.0), 1.0, 0.5, float(np.mean(train_y))], dtype=np.float64)
        try:
            result = least_squares(
                residuals,
                initial_guess,
                jac=jacobian,
                loss="soft_l1",
                f_scale=20.0,
                bounds=([0.0, 0.1, -5.0, -500.0], [A_bound, 20.0, 5.0, 500.0]),
            )
            return self._hyperbolic_fn(candidate_x, *result.x)
        except ValueError:
            return np.interp(candidate_x, train_x, train_y, left=float(train_y[0]), right=float(train_y[-1]))

    def predict_grid(self, policy_eval: np.ndarray, candidate_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        neighbors, sigma = self._collect_neighbors(policy_eval)
        if not neighbors:
            deltas = np.zeros_like(candidate_weights)
            return deltas, policy_eval[None, :] + deltas

        dim_predictions = []
        for dim in range(candidate_weights.shape[1]):
            dim_predictions.append(
                self._fit_dimension(neighbors, sigma, policy_eval, candidate_weights[:, dim], dim)
            )
        delta_predictions = np.stack(dim_predictions, axis=1)
        return delta_predictions, policy_eval[None, :] + delta_predictions


def optimize_policy(
    initial_mu: float,
    weight: np.ndarray,
    config: ToyConfig,
    num_steps: int,
    rng: np.random.Generator,
    method: str,
    generation: int,
    slot: int,
) -> TrainOutcome:
    mu = float(initial_mu)
    before_eval = expected_objectives(mu, config)
    inner_records: list[dict[str, Any]] = []

    inner_records.append(
        {
            "method": method,
            "generation": generation,
            "slot": slot,
            "inner_step": 0,
            "mu": mu,
            **dynamic_weight_fields(weight),
            **dynamic_eval_fields("eval", before_eval),
            "scalarized_return": scalarized_expected_return(mu, weight, config),
        }
    )

    for step in range(1, num_steps + 1):
        if config.gradient_mode == "analytic":
            grad = analytic_gradient(mu, weight, config)
        else:
            actions = rng.normal(loc=mu, scale=config.policy_std, size=config.batch_size)
            sampled_rewards = sample_objectives(actions, config)
            scalarized = sampled_rewards @ weight
            baseline = float(np.mean(scalarized))
            score_function = (actions - mu) / (config.policy_std**2)
            grad = float(np.mean((scalarized - baseline) * score_function))

        mu = float(np.clip(mu + config.learning_rate * grad, config.init_mu_low - 1.0, config.init_mu_high + 1.0))
        evals = expected_objectives(mu, config)
        inner_records.append(
            {
                "method": method,
                "generation": generation,
                "slot": slot,
                "inner_step": step,
                "mu": mu,
                **dynamic_weight_fields(weight),
                **dynamic_eval_fields("eval", evals),
                "scalarized_return": scalarized_expected_return(mu, weight, config),
            }
        )

    return TrainOutcome(
        final_mu=mu,
        before_eval=before_eval,
        after_eval=expected_objectives(mu, config),
        inner_records=inner_records,
    )


def record_metrics(method: str, generation: int, population: list[PolicyState], archive: ParetoArchive) -> dict[str, Any]:
    archive_points = archive.evaluations
    return {
        "method": method,
        "generation": generation,
        "hypervolume": hypervolume(archive_points),
        "sparsity": sparsity(archive_points),
        "archive_size": len(archive_points),
        "population_size": len(population),
    }


def _task_score(current_front: list[np.ndarray], predicted_eval: np.ndarray, sparsity_coef: float) -> float:
    augmented = current_front + [predicted_eval]
    return hypervolume(augmented) - sparsity_coef * sparsity(augmented)


def pgmorl_task_selection(
    method: str,
    generation: int,
    population: list[PolicyState],
    archive: ParetoArchive,
    predictor: PerformancePredictor,
    config: ToyConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_weights = generate_weight_grid(config.weight_resolution, config.num_objectives)
    predictor_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    current_front = list(archive.evaluations)

    for state in population:
        delta_predictions, predicted_evals = predictor.predict_grid(state.evaluation, candidate_weights)
        for weight, pred_delta, pred_eval in zip(candidate_weights, delta_predictions, predicted_evals):
            initial_score = _task_score(current_front, pred_eval, config.sparsity_coef)
            predictor_rows.append(
                {
                    "method": method,
                    "generation": generation,
                    "policy_id": state.policy_id,
                    "mu": float(state.mu),
                    "initial_score": float(initial_score),
                    **dynamic_weight_fields(weight),
                    **dynamic_eval_fields("parent_eval", state.evaluation),
                    **dynamic_eval_fields("pred_delta", pred_delta),
                    **dynamic_eval_fields("pred_eval", pred_eval),
                }
            )
            candidate_rows.append(
                {
                    "state": state,
                    "weight": weight,
                    "predicted_eval": pred_eval,
                    "predicted_delta": pred_delta,
                    "candidate_key": (state.policy_id, *tuple(np.round(weight, 6))),
                    "initial_score": initial_score,
                }
            )

    virtual_front = current_front.copy()
    used_pairs: set[tuple[Any, ...]] = set()
    selected_tasks: list[dict[str, Any]] = []

    for _ in range(max(1, len(population))):
        best_row = None
        best_score = float("-inf")
        for row in candidate_rows:
            if row["candidate_key"] in used_pairs:
                continue
            score = _task_score(virtual_front, row["predicted_eval"], config.sparsity_coef)
            if score > best_score:
                best_score = score
                best_row = row | {"score": score}
        if best_row is None:
            break
        used_pairs.add(best_row["candidate_key"])
        virtual_front = pareto_front(virtual_front + [best_row["predicted_eval"]]).tolist()
        selected_tasks.append(best_row)

    return selected_tasks, predictor_rows


def random_task_selection(population: list[PolicyState], config: ToyConfig, rng: np.random.Generator) -> list[dict[str, Any]]:
    tasks = []
    candidate_weights = generate_weight_grid(config.weight_resolution, config.num_objectives)
    for _ in range(max(1, len(population))):
        state = population[int(rng.integers(0, len(population)))]
        weight = candidate_weights[int(rng.integers(0, len(candidate_weights)))]
        tasks.append(
            {
                "state": state,
                "weight": weight,
                "predicted_eval": None,
                "predicted_delta": None,
                "score": np.nan,
                "initial_score": np.nan,
            }
        )
    return tasks


def ra_task_selection(population: list[PolicyState]) -> list[dict[str, Any]]:
    return [
        {
            "state": state,
            "weight": state.weight,
            "predicted_eval": None,
            "predicted_delta": None,
            "score": np.nan,
            "initial_score": np.nan,
        }
        for state in population
    ]


def _greedy_farthest_weights(existing: list[np.ndarray], grid: np.ndarray, count: int) -> list[np.ndarray]:
    selected = [weight.copy() for weight in existing]
    while len(selected) < count:
        if not selected:
            selected.append(grid[np.argmax(np.linalg.norm(grid - (1.0 / grid.shape[1]), axis=1))].copy())
            continue
        existing_array = np.stack(selected, axis=0)
        distances = np.linalg.norm(grid[:, None, :] - existing_array[None, :, :], axis=2).min(axis=1)
        duplicate_mask = np.any(np.all(np.isclose(grid[:, None, :], existing_array[None, :, :], atol=1e-8), axis=2), axis=1)
        distances[duplicate_mask] = -np.inf
        selected.append(grid[int(np.argmax(distances))].copy())
    return selected[:count]


def sample_warmup_weights(config: ToyConfig) -> np.ndarray:
    dims = config.num_objectives
    grid = generate_weight_grid(max(config.weight_resolution, 4), dims)

    base: list[np.ndarray] = [one_hot_weight(i, dims) for i in range(dims)]
    for i, j in combinations(range(dims), 2):
        pair = np.zeros(dims, dtype=np.float64)
        pair[i] = 0.5
        pair[j] = 0.5
        base.append(pair)
    base.append(np.full(dims, 1.0 / dims, dtype=np.float64))

    dedup: list[np.ndarray] = []
    for weight in base:
        if not any(np.allclose(weight, existing) for existing in dedup):
            dedup.append(weight)
    chosen = _greedy_farthest_weights(dedup[: min(len(dedup), config.population_size)], grid, config.population_size)
    return np.stack(chosen, axis=0)


def run_single_method(method: str, config: ToyConfig, initial_mus: np.ndarray, seed_offset: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(config.seed + seed_offset)
    predictor = PerformancePredictor(config)
    population_buffer = PerformanceBuffer(config.population_size)
    archive = ParetoArchive()

    metrics_rows: list[dict[str, Any]] = []
    population_rows: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    predictor_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []

    population: list[PolicyState] = []
    next_policy_id = 0

    warmup_weights = sample_warmup_weights(config)
    warmup_offspring: list[PolicyState] = []
    for slot, (mu, weight) in enumerate(zip(initial_mus, warmup_weights)):
        outcome = optimize_policy(
            initial_mu=float(mu),
            weight=weight,
            config=config,
            num_steps=config.warmup_steps,
            rng=rng,
            method=method,
            generation=0,
            slot=slot,
        )
        inner_rows.extend(outcome.inner_records)
        state = PolicyState(
            policy_id=next_policy_id,
            mu=outcome.final_mu,
            evaluation=outcome.after_eval,
            weight=weight,
            generation=0,
            method=method,
            source="warmup",
        )
        next_policy_id += 1
        warmup_offspring.append(state)
        predictor.add(
            TransitionRecord(
                before_eval=outcome.before_eval,
                after_eval=outcome.after_eval,
                weight=weight,
                generation=0,
                parent_policy_id=state.policy_id,
            )
        )
        task_rows.append(
            {
                "method": method,
                "generation": 0,
                "slot": slot,
                "stage": "warmup",
                "parent_policy_id": None,
                "parent_mu": float(mu),
                **dynamic_weight_fields(weight),
                **dynamic_eval_fields("parent_eval", outcome.before_eval),
                **dynamic_eval_fields("pred_eval", np.full(config.num_objectives, np.nan)),
                **dynamic_eval_fields("actual_eval", outcome.after_eval),
                **dynamic_eval_fields("actual_delta", outcome.after_eval - outcome.before_eval),
                "score": np.nan,
                "initial_score": np.nan,
                "child_policy_id": state.policy_id,
            }
        )

    population = population_buffer.update(warmup_offspring)
    archive.add(warmup_offspring)
    metrics_rows.append(record_metrics(method, 0, population, archive))
    population_rows.extend(flatten_state(state, snapshot_generation=0) for state in population)
    archive_rows.extend({**flatten_state(state, snapshot_generation=0), "source": "archive"} for state in archive.states)

    for generation in range(1, config.generations + 1):
        if method == "PG-MORL Toy":
            selected_tasks, new_predictor_rows = pgmorl_task_selection(method, generation, population, archive, predictor, config)
            predictor_rows.extend(new_predictor_rows)
        elif method == "RA Sweep":
            selected_tasks = ra_task_selection(population)
        else:
            selected_tasks = random_task_selection(population, config, rng)

        offspring: list[PolicyState] = []
        for slot, task in enumerate(selected_tasks):
            parent = task["state"]
            weight = task["weight"]
            outcome = optimize_policy(
                initial_mu=parent.mu,
                weight=weight,
                config=config,
                num_steps=config.task_steps,
                rng=rng,
                method=method,
                generation=generation,
                slot=slot,
            )
            inner_rows.extend(outcome.inner_records)
            child_state = PolicyState(
                policy_id=next_policy_id,
                mu=outcome.final_mu,
                evaluation=outcome.after_eval,
                weight=weight,
                generation=generation,
                method=method,
                source="offspring",
                parent_policy_id=parent.policy_id,
            )
            next_policy_id += 1
            offspring.append(child_state)
            predictor.add(
                TransitionRecord(
                    before_eval=outcome.before_eval,
                    after_eval=outcome.after_eval,
                    weight=weight,
                    generation=generation,
                    parent_policy_id=parent.policy_id,
                )
            )
            pred_eval = task["predicted_eval"]
            pred_delta = task["predicted_delta"]
            task_rows.append(
                {
                    "method": method,
                    "generation": generation,
                    "slot": slot,
                    "stage": "evolution",
                    "parent_policy_id": parent.policy_id,
                    "parent_mu": float(parent.mu),
                    **dynamic_weight_fields(weight),
                    **dynamic_eval_fields("parent_eval", parent.evaluation),
                    **dynamic_eval_fields(
                        "pred_eval",
                        pred_eval if pred_eval is not None else np.full(config.num_objectives, np.nan),
                    ),
                    **dynamic_eval_fields("actual_eval", outcome.after_eval),
                    **dynamic_eval_fields("actual_delta", outcome.after_eval - parent.evaluation),
                    **dynamic_eval_fields(
                        "pred_delta",
                        pred_delta if pred_delta is not None else np.full(config.num_objectives, np.nan),
                    ),
                    "score": float(task["score"]) if task["score"] == task["score"] else np.nan,
                    "initial_score": float(task["initial_score"]) if task["initial_score"] == task["initial_score"] else np.nan,
                    "child_policy_id": child_state.policy_id,
                }
            )

        population = population_buffer.update(population + offspring)
        archive.add(offspring)
        metrics_rows.append(record_metrics(method, generation, population, archive))
        population_rows.extend(flatten_state(state, snapshot_generation=generation) for state in population)
        archive_rows.extend(
            {**flatten_state(state, snapshot_generation=generation), "source": "archive"} for state in archive.states
        )

    task_df = pd.DataFrame(task_rows)
    for index in range(config.num_objectives):
        pred_col = f"pred_eval_{index + 1}"
        actual_col = f"actual_eval_{index + 1}"
        if pred_col in task_df.columns and actual_col in task_df.columns:
            task_df[f"pred_gap_{index + 1}"] = task_df[actual_col] - task_df[pred_col]

    return {
        "metrics": pd.DataFrame(metrics_rows),
        "population": pd.DataFrame(population_rows),
        "archive": pd.DataFrame(archive_rows),
        "tasks": task_df,
        "predictor_curves": pd.DataFrame(predictor_rows),
        "inner_steps": pd.DataFrame(inner_rows),
    }


def build_true_front(config: ToyConfig, num_points: int = 240) -> pd.DataFrame:
    mus = np.linspace(config.init_mu_low, config.init_mu_high, num_points)
    values = np.stack([expected_objectives(mu, config) for mu in mus], axis=0)
    front = pareto_front(values)
    rows = []
    for mu, evaluation in zip(mus, values):
        if any(np.allclose(evaluation, front_point) for front_point in front):
            row = {"mu": mu}
            row.update(dynamic_eval_fields("eval", evaluation))
            rows.append(row)
    return pd.DataFrame(rows)


def paper_mapping() -> pd.DataFrame:
    rows = [
        {
            "Paper Component": "MOMDP continuous control",
            "Toy Version": "직선 통로를 달리는 배송 카트의 순항 속도 제어",
            "Why it helps": "속도 하나만 조절하지만, 빠른 도착 / 배터리 효율 / 안전성 3개 KPI가 서로 충돌하는 상황을 직관적으로 볼 수 있습니다.",
        },
        {
            "Paper Component": "MOPG worker (Algorithm 2)",
            "Toy Version": "속도 평균 μ에 대한 weighted policy gradient",
            "Why it helps": "PPO 세부 구현을 걷어내고 정책 경사의 핵심만 남겨, RL 업데이트 방향을 눈으로 추적할 수 있습니다.",
        },
        {
            "Paper Component": "Hyperbolic improvement predictor",
            "Toy Version": "논문과 같은 4-parameter hyperbolic regression",
            "Why it helps": "각 정책이 어떤 선호도에서 얼마나 개선될지를 예측하는 핵심 역할을 그대로 보여줍니다.",
        },
        {
            "Paper Component": "Prediction-guided task selection",
            "Toy Version": "hypervolume - lambda*sparsity 기준의 greedy selection",
            "Why it helps": "PG-MORL이 왜 아무 작업이나 학습하지 않고, 중요한 작업을 골라서 계산 자원을 쓰는지 확인할 수 있습니다.",
        },
        {
            "Paper Component": "Performance buffer + Pareto archive",
            "Toy Version": "성능과 다양성을 함께 보는 objective-space diversity buffer",
            "Why it helps": "현재 유지하는 정책 집합과, 지금까지 얻은 Pareto archive의 역할 차이를 구분해서 볼 수 있습니다.",
        },
        {
            "Paper Component": "Pareto analysis stage",
            "Toy Version": "대시보드에서는 학습 단계 중심으로 설명",
            "Why it helps": "논문 후반부 family analysis보다, PG-MORL의 핵심 루프와 RL 자원 배분 역할 이해에 집중합니다.",
        },
    ]
    return pd.DataFrame(rows)


def run_experiment(config: ToyConfig) -> ExperimentResult:
    init_rng = np.random.default_rng(config.seed)
    initial_mus = init_rng.uniform(config.init_mu_low, config.init_mu_high, size=config.population_size)

    methods = ["PG-MORL Toy", "RA Sweep", "Random"]
    outputs = [run_single_method(method, config, initial_mus, index * 1000) for index, method in enumerate(methods)]

    metrics = pd.concat([output["metrics"] for output in outputs], ignore_index=True)
    population = pd.concat([output["population"] for output in outputs], ignore_index=True)
    archive = pd.concat([output["archive"] for output in outputs], ignore_index=True)
    tasks = pd.concat([output["tasks"] for output in outputs], ignore_index=True)
    predictor_curves = pd.concat([output["predictor_curves"] for output in outputs], ignore_index=True)
    inner_steps = pd.concat([output["inner_steps"] for output in outputs], ignore_index=True)

    method_summary = (
        metrics.sort_values(["method", "generation"])
        .groupby("method", as_index=False)
        .tail(1)
        .loc[:, ["method", "generation", "hypervolume", "sparsity", "archive_size", "population_size"]]
        .sort_values("hypervolume", ascending=False)
        .reset_index(drop=True)
    )

    return ExperimentResult(
        config=config,
        metrics=metrics,
        population=population,
        archive=archive,
        tasks=tasks,
        predictor_curves=predictor_curves,
        inner_steps=inner_steps,
        true_front=build_true_front(config),
        method_summary=method_summary,
        mapping=paper_mapping(),
    )
