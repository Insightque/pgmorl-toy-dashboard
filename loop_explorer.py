from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from toy_pgmorl.core import hypervolume, pareto_front, sparsity


alt.data_transformers.disable_max_rows()


@dataclass(frozen=True)
class LoopConfig:
    num_objectives: int = 2
    population_size: int = 50
    param_dim: int = 100_000
    weight_candidates: int = 11
    selected_tasks: int = 5
    generations: int = 100
    seed: int = 11
    sparsity_coef: float = 0.18

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fmt_int(value: int) -> str:
    return f"{int(value):,}"


def fmt_shape(*dims: int) -> str:
    inner = " x ".join(fmt_int(dim) for dim in dims)
    return f"({inner})"


def hyperbolic(x: np.ndarray, A: float, a: float, b: float, c: float) -> np.ndarray:
    z = np.clip(a * (x - b), -60.0, 60.0)
    exp_term = np.exp(z)
    return A * (exp_term - 1.0) / (exp_term + 1.0) + c


def build_weight_grid(config: LoopConfig) -> pd.DataFrame:
    w1 = np.linspace(1.0, 0.0, config.weight_candidates)
    w2 = 1.0 - w1
    labels = [f"[{x:.1f}, {y:.1f}]^T" for x, y in zip(w1, w2)]
    return pd.DataFrame(
        {
            "weight_index": np.arange(config.weight_candidates, dtype=int),
            "weight_1": w1,
            "weight_2": w2,
            "weight_label": labels,
        }
    )


def build_population(config: LoopConfig, generation: int) -> pd.DataFrame:
    policy_id = np.arange(config.population_size, dtype=int)
    progress = generation / max(config.generations, 1)
    latent = np.linspace(0.03, 0.97, config.population_size)
    front_1 = 64.0 + 19.0 * latent + 2.0 * np.sin(2.0 * np.pi * latent)
    front_2 = 64.0 + 19.0 * (1.0 - latent) + 2.0 * np.cos(2.0 * np.pi * latent)
    inside_gap = 7.5 * (1.0 - progress) + 1.2 + 1.2 * np.sin(np.pi * latent) ** 2
    eval_1 = front_1 - inside_gap + 0.5 * np.sin((policy_id + 1) * (generation + 1) * 0.17)
    eval_2 = front_2 - inside_gap + 0.5 * np.cos((policy_id + 1) * (generation + 1) * 0.13)
    mu = 1.0 + 9.0 * latent
    warmup_weight_1 = np.clip(latent + 0.08 * np.sin(policy_id * 0.61), 0.0, 1.0)
    warmup_weight_2 = 1.0 - warmup_weight_1
    return pd.DataFrame(
        {
            "policy_id": policy_id,
            "latent": latent,
            "mu": mu,
            "eval_1": eval_1,
            "eval_2": eval_2,
            "warmup_weight_1": warmup_weight_1,
            "warmup_weight_2": warmup_weight_2,
        }
    )


def predictor_params_for_policy(latent: float, progress: float) -> pd.DataFrame:
    params = [
        {
            "objective": "f1",
            "A": 1.15 + 0.65 * (1.0 - latent),
            "a": 4.6,
            "b": 0.18 + 0.60 * latent,
            "c": 1.60 + 0.45 * progress,
        },
        {
            "objective": "f2",
            "A": 1.15 + 0.65 * latent,
            "a": 4.6,
            "b": 0.18 + 0.60 * (1.0 - latent),
            "c": 1.60 + 0.45 * progress,
        },
    ]
    return pd.DataFrame(params)


def build_candidate_predictions(config: LoopConfig, generation: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    population = build_population(config, generation)
    weights = build_weight_grid(config)
    progress = generation / max(config.generations, 1)

    candidate_rows: list[dict[str, Any]] = []
    predictor_rows: list[dict[str, Any]] = []
    for policy in population.itertuples(index=False):
        params_df = predictor_params_for_policy(float(policy.latent), progress)
        w1 = weights["weight_1"].to_numpy(dtype=np.float64)
        w2 = weights["weight_2"].to_numpy(dtype=np.float64)
        pred_delta_1 = hyperbolic(
            w1,
            float(params_df.iloc[0]["A"]),
            float(params_df.iloc[0]["a"]),
            float(params_df.iloc[0]["b"]),
            float(params_df.iloc[0]["c"]),
        )
        pred_delta_2 = hyperbolic(
            w2,
            float(params_df.iloc[1]["A"]),
            float(params_df.iloc[1]["a"]),
            float(params_df.iloc[1]["b"]),
            float(params_df.iloc[1]["c"]),
        )
        pred_eval_1 = float(policy.eval_1) + pred_delta_1
        pred_eval_2 = float(policy.eval_2) + pred_delta_2

        for weight_row, delta_1, delta_2, eval_1, eval_2 in zip(
            weights.itertuples(index=False),
            pred_delta_1,
            pred_delta_2,
            pred_eval_1,
            pred_eval_2,
        ):
            actual_scale = 0.84 + 0.12 * np.sin((policy.policy_id + 1) * (weight_row.weight_index + 1) * 0.37)
            actual_delta_1 = max(0.05, float(delta_1 * actual_scale + 0.08 * np.cos(policy.policy_id + weight_row.weight_index)))
            actual_delta_2 = max(0.05, float(delta_2 * actual_scale + 0.08 * np.sin(policy.policy_id - weight_row.weight_index)))
            candidate_rows.append(
                {
                    "generation": generation,
                    "policy_id": int(policy.policy_id),
                    "mu": float(policy.mu),
                    "latent": float(policy.latent),
                    "parent_eval_1": float(policy.eval_1),
                    "parent_eval_2": float(policy.eval_2),
                    "weight_index": int(weight_row.weight_index),
                    "weight_1": float(weight_row.weight_1),
                    "weight_2": float(weight_row.weight_2),
                    "weight_label": str(weight_row.weight_label),
                    "pred_delta_1": float(delta_1),
                    "pred_delta_2": float(delta_2),
                    "pred_eval_1": float(eval_1),
                    "pred_eval_2": float(eval_2),
                    "actual_delta_1": actual_delta_1,
                    "actual_delta_2": actual_delta_2,
                    "actual_eval_1": float(policy.eval_1 + actual_delta_1),
                    "actual_eval_2": float(policy.eval_2 + actual_delta_2),
                }
            )
        params_df.insert(0, "policy_id", int(policy.policy_id))
        predictor_rows.append(params_df)

    return pd.DataFrame(candidate_rows), pd.concat(predictor_rows, ignore_index=True)


def to_front_df(points: np.ndarray, label: str, x_field: str = "eval_1", y_field: str = "eval_2") -> pd.DataFrame:
    if len(points) == 0:
        return pd.DataFrame(columns=[x_field, y_field, "front_label", "order"])
    front = pareto_front(points)
    if len(front) == 0:
        return pd.DataFrame(columns=[x_field, y_field, "front_label", "order"])
    return pd.DataFrame(
        {
            x_field: front[:, 0],
            y_field: front[:, 1],
            "front_label": label,
            "order": np.arange(len(front), dtype=int),
        }
    )


def select_tasks(
    config: LoopConfig,
    population: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current_front_points = pareto_front(population[["eval_1", "eval_2"]].to_numpy(dtype=np.float64))
    virtual_front = [point for point in current_front_points]
    used_pairs: set[tuple[int, int]] = set()

    round_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    trace_rows.append(
        {
            "round": 0,
            "kind": "Current archive",
            "hypervolume": hypervolume(virtual_front),
            "sparsity": sparsity(virtual_front),
            "front_size": len(virtual_front),
        }
    )

    for round_idx in range(1, config.selected_tasks + 1):
        best_index: int | None = None
        best_score = float("-inf")
        for row_index, row in candidate_df.iterrows():
            key = (int(row["policy_id"]), int(row["weight_index"]))
            if key in used_pairs:
                continue
            augmented = virtual_front + [np.array([row["pred_eval_1"], row["pred_eval_2"]], dtype=np.float64)]
            score = hypervolume(augmented) - config.sparsity_coef * sparsity(augmented)
            round_rows.append(
                {
                    "round": round_idx,
                    "policy_id": int(row["policy_id"]),
                    "weight_index": int(row["weight_index"]),
                    "weight_1": float(row["weight_1"]),
                    "weight_2": float(row["weight_2"]),
                    "weight_label": row["weight_label"],
                    "score": float(score),
                }
            )
            if score > best_score:
                best_score = float(score)
                best_index = int(row_index)

        if best_index is None:
            break

        chosen = candidate_df.loc[best_index].to_dict()
        key = (int(chosen["policy_id"]), int(chosen["weight_index"]))
        used_pairs.add(key)
        virtual_front = pareto_front(
            virtual_front + [np.array([chosen["pred_eval_1"], chosen["pred_eval_2"]], dtype=np.float64)]
        ).tolist()
        chosen["round"] = round_idx
        chosen["selection_score"] = best_score
        selected_rows.append(chosen)
        trace_rows.append(
            {
                "round": round_idx,
                "kind": "Virtual archive after selection",
                "hypervolume": hypervolume(virtual_front),
                "sparsity": sparsity(virtual_front),
                "front_size": len(virtual_front),
            }
        )

    return pd.DataFrame(selected_rows), pd.DataFrame(round_rows), pd.DataFrame(trace_rows)


@st.cache_data(show_spinner=False)
def build_generation_bundle(config: LoopConfig, generation: int) -> dict[str, Any]:
    population = build_population(config, generation)
    candidate_df, predictor_params = build_candidate_predictions(config, generation)
    selected_df, round_scores, trace_df = select_tasks(config, population, candidate_df)

    current_front = to_front_df(population[["eval_1", "eval_2"]].to_numpy(dtype=np.float64), "Current archive front")
    if selected_df.empty:
        predicted_front = pd.DataFrame(columns=["eval_1", "eval_2", "front_label", "order"])
        actual_front = current_front.copy()
    else:
        predicted_points = np.vstack(
            [
                population[["eval_1", "eval_2"]].to_numpy(dtype=np.float64),
                selected_df[["pred_eval_1", "pred_eval_2"]].to_numpy(dtype=np.float64),
            ]
        )
        actual_points = np.vstack(
            [
                population[["eval_1", "eval_2"]].to_numpy(dtype=np.float64),
                selected_df[["actual_eval_1", "actual_eval_2"]].to_numpy(dtype=np.float64),
            ]
        )
        predicted_front = to_front_df(predicted_points, "Predicted virtual front")
        actual_front = to_front_df(actual_points, "Actual updated front")

    return {
        "population": population,
        "candidate_df": candidate_df,
        "predictor_params": predictor_params,
        "selected_df": selected_df,
        "round_scores": round_scores,
        "trace_df": trace_df,
        "current_front": current_front,
        "predicted_front": predicted_front,
        "actual_front": actual_front,
    }


def build_shape_table(config: LoopConfig, generation: int, archive_size: int) -> pd.DataFrame:
    records_rows = config.population_size + generation * config.selected_tasks
    rows = [
        {
            "Phase": "Warm-up",
            "Symbol": r"$\Theta$",
            "Shape": fmt_shape(config.param_dim, config.population_size),
            "Meaning": "현재 유지하는 전체 정책 파라미터 행렬. 각 열이 하나의 policy vector θ_i 입니다.",
            "Cadence": "최초 1회 생성",
        },
        {
            "Phase": "Warm-up",
            "Symbol": r"$W_{warmup}$",
            "Shape": fmt_shape(config.population_size, config.num_objectives),
            "Meaning": "초기 정책마다 하나씩 배정된 warm-up weight 벡터들입니다.",
            "Cadence": "최초 1회 생성",
        },
        {
            "Phase": "Warm-up",
            "Symbol": r"$F_{before}, F_{after}$",
            "Shape": fmt_shape(config.population_size, config.num_objectives),
            "Meaning": "warm-up 전후의 objective 벡터 행렬입니다.",
            "Cadence": "Warm-up 직후 기록",
        },
        {
            "Phase": "Step 2",
            "Symbol": r"$R$",
            "Shape": fmt_shape(records_rows, 2 * config.num_objectives),
            "Meaning": "과거 transition buffer. 각 row가 (사용한 weight, 관측된 ΔF)를 저장합니다.",
            "Cadence": "Warm-up 뒤, 세대마다 n개 행 추가",
        },
        {
            "Phase": "Step 2",
            "Symbol": r"$\Xi$",
            "Shape": fmt_shape(config.population_size, config.num_objectives, 4),
            "Meaning": "각 policy/objective마다 4-파라미터 쌍곡선 predictor 계수 [A, a, b, c] 입니다.",
            "Cadence": "매 세대 시작 시 재피팅",
        },
        {
            "Phase": "Step 3",
            "Symbol": r"$W_{grid}$",
            "Shape": fmt_shape(config.weight_candidates, config.num_objectives),
            "Meaning": "평가할 K개의 weight 후보군입니다.",
            "Cadence": "매 세대 사용",
        },
        {
            "Phase": "Step 3",
            "Symbol": r"$\Delta$",
            "Shape": fmt_shape(config.population_size, config.weight_candidates, config.num_objectives),
            "Meaning": "모든 (policy, weight) 후보에 대한 predicted improvement tensor입니다.",
            "Cadence": "매 세대 새로 계산",
        },
        {
            "Phase": "Step 3",
            "Symbol": r"$F_{pred}$",
            "Shape": fmt_shape(config.population_size, config.weight_candidates, config.num_objectives),
            "Meaning": r"$F(\pi_i) + \Delta^i(\omega_k)$ 로 얻는 예상 미래 성능입니다.",
            "Cadence": "매 세대 새로 계산",
        },
        {
            "Phase": "Step 3",
            "Symbol": r"$Q$",
            "Shape": fmt_shape(config.population_size, config.weight_candidates),
            "Meaning": "각 후보가 virtual archive의 hypervolume을 얼마나 늘릴지 평가한 score matrix입니다.",
            "Cadence": "선택 round마다 재계산",
        },
        {
            "Phase": "Step 3",
            "Symbol": r"$\Theta_{selected}$",
            "Shape": fmt_shape(config.param_dim, config.selected_tasks),
            "Meaning": "이번 세대에 실제 역전파를 수행하는 상위 n개 task의 policy columns입니다.",
            "Cadence": "매 세대 Step 3에서 선택",
        },
        {
            "Phase": "Step 4",
            "Symbol": r"$\Theta_{pareto}$",
            "Shape": fmt_shape(config.param_dim, archive_size),
            "Meaning": "최종 Pareto archive의 파라미터 행렬입니다.",
            "Cadence": "훈련 종료 후 1회 분석",
        },
        {
            "Phase": "Step 4",
            "Symbol": r"$Z$",
            "Shape": fmt_shape(archive_size, 2),
            "Meaning": "t-SNE로 압축한 2D parameter-space embedding입니다.",
            "Cadence": "훈련 종료 후 1회 생성",
        },
        {
            "Phase": "Step 4",
            "Symbol": r"$\theta_{new}$",
            "Shape": fmt_shape(config.param_dim),
            "Meaning": "같은 family 안의 두 anchor를 선형 보간해 만든 새 controller입니다.",
            "Cadence": "사용자가 target KPI를 고를 때마다 즉시 생성",
        },
    ]
    return pd.DataFrame(rows)


def build_step_flow_table(config: LoopConfig, generation: int, archive_size: int) -> pd.DataFrame:
    records_rows = config.population_size + generation * config.selected_tasks
    rows = [
        {
            "Step": "Step 1. Warm-up",
            "Input": r"$\Theta \in \mathbb{R}^{N \times |P|}, W_{warmup} \in \mathbb{R}^{|P| \times m}$",
            "Compute": "모든 |P| policy를 실제 MOPG로 한 번씩 훈련",
            "Output": r"$F_{after}, EP, R$",
            "Cadence": "최초 1회",
            "Cost": f"실제 훈련 {fmt_int(config.population_size)}회",
        },
        {
            "Step": "Step 2. Predictor Fitting",
            "Input": rf"$R$ ({fmt_int(records_rows)} rows), 현재 population $F(\pi)$",
            "Compute": r"policy별 국소 이웃을 모아 predictor tensor $\Xi \in \mathbb{R}^{|P| \times m \times 4}$ 피팅",
            "Output": r"$\Xi$",
            "Cadence": "매 generation 시작",
            "Cost": f"predictor {fmt_int(config.population_size)}개 재피팅",
        },
        {
            "Step": "Step 3. Task Selection",
            "Input": rf"$\Xi, W_{{grid}} \in \mathbb{{R}}^{{{fmt_int(config.weight_candidates)} \times {config.num_objectives}}}$",
            "Compute": r"$\Delta \rightarrow F_{pred} \rightarrow Q$ 계산 후 top-n greedy selection",
            "Output": r"$Q$, selected tasks, predicted / actual offspring, updated $EP$",
            "Cadence": "매 generation",
            "Cost": f"예측 후보 {fmt_int(config.population_size * config.weight_candidates)}개, 실제 훈련 {fmt_int(config.selected_tasks)}개",
        },
        {
            "Step": "Step 4. Pareto Analysis",
            "Input": rf"$\Theta_{{pareto}} \in \mathbb{{R}}^{{N \times {fmt_int(archive_size)}}}$",
            "Compute": "t-SNE -> k-means -> family 내부 선형 보간",
            "Output": r"$Z$, family labels, $\theta_{new}$",
            "Cadence": "훈련 종료 후 1회",
            "Cost": "추가 시뮬레이션 없이 즉시 보간",
        },
    ]
    return pd.DataFrame(rows)


def guide(title: str, text: str, items: list[tuple[str, str]]) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(text)
        st.dataframe(
            pd.DataFrame([{"표식": k, "의미": v} for k, v in items]),
            use_container_width=True,
            hide_index=True,
        )


def fixed_domains(frames: list[pd.DataFrame], x_field: str, y_field: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for frame in frames:
        if frame.empty:
            continue
        xs.extend(frame[x_field].astype(float).tolist())
        ys.extend(frame[y_field].astype(float).tolist())
    if not xs or not ys:
        return [0.0, 1.0], [0.0, 1.0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = max(1.0, 0.08 * (x_max - x_min))
    y_pad = max(1.0, 0.08 * (y_max - y_min))
    return [x_min - x_pad, x_max + x_pad], [y_min - y_pad, y_max + y_pad]


def make_loop_clock_chart(config: LoopConfig) -> alt.Chart:
    rows = [
        {"phase": "Step 1\nWarm-up", "count": 1, "kind": "one-shot"},
        {"phase": "Step 2\nPrediction", "count": config.generations, "kind": "loop"},
        {"phase": "Step 3\nTask Selection", "count": config.generations, "kind": "loop"},
        {"phase": "Step 4\nPareto Analysis", "count": 1, "kind": "one-shot"},
    ]
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("phase:N", title=None),
            y=alt.Y("count:Q", title="실행 횟수"),
            color=alt.Color("kind:N", title=None, scale=alt.Scale(range=["#264653", "#f4a261"])),
            tooltip=["phase", "count"],
        )
        .properties(height=240, title="알고리즘 실행 주기: Warm-up 1회, 루프 M회, 최종 분석 1회")
    )


def make_records_growth_chart(config: LoopConfig) -> alt.Chart:
    generations = np.arange(config.generations + 1, dtype=int)
    row_count = config.population_size + generations * config.selected_tasks
    brute_force = config.population_size + generations * config.population_size * config.weight_candidates
    df = pd.DataFrame(
        {
            "generation": np.concatenate([generations, generations]),
            "rows": np.concatenate([row_count, brute_force]),
            "series": ["PG-MORL buffer rows"] * len(generations) + ["Brute-force training calls"] * len(generations),
        }
    )
    return (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("generation:Q", title="generation"),
            y=alt.Y("rows:Q", title="누적 row / call 수"),
            color=alt.Color("series:N", title=None),
            tooltip=["generation", "series", "rows"],
        )
        .properties(height=250, title="R buffer 성장 vs brute-force 전체 훈련량")
    )


def _front_domains(
    population: pd.DataFrame,
    current_front: pd.DataFrame,
    predicted_front: pd.DataFrame,
    actual_front: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> tuple[list[float], list[float]]:
    return fixed_domains(
        [
            population.rename(columns={"eval_1": "x", "eval_2": "y"}).rename(columns={"x": "eval_1", "y": "eval_2"}),
            selected_df.rename(columns={"pred_eval_1": "eval_1", "pred_eval_2": "eval_2"}),
            selected_df.rename(columns={"actual_eval_1": "eval_1", "actual_eval_2": "eval_2"}),
            current_front,
            predicted_front,
            actual_front,
        ],
        "eval_1",
        "eval_2",
    )


def make_predicted_front_chart(
    population: pd.DataFrame,
    current_front: pd.DataFrame,
    predicted_front: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> alt.Chart:
    x_domain, y_domain = _front_domains(population, current_front, predicted_front, pd.DataFrame(), selected_df)
    x_enc = alt.X("eval_1:Q", title="Performance Space f1", scale=alt.Scale(domain=x_domain, nice=False))
    y_enc = alt.Y("eval_2:Q", title="f2", scale=alt.Scale(domain=y_domain, nice=False))

    base_points = (
        alt.Chart(population)
        .mark_circle(size=90, color="#457b9d", opacity=0.7)
        .encode(x=x_enc, y=y_enc, tooltip=["policy_id", "mu", "eval_1", "eval_2"])
    )
    current_front_line = alt.Chart(current_front).mark_line(strokeWidth=3.5, color="#111111").encode(x=x_enc, y=y_enc)
    predicted_front_line = (
        alt.Chart(predicted_front).mark_line(strokeWidth=3.0, strokeDash=[9, 5], color="#f4a261").encode(x=x_enc, y=y_enc)
    )

    layers: list[alt.Chart] = [base_points, current_front_line]
    if not selected_df.empty:
        predicted_lines = (
            alt.Chart(selected_df)
            .mark_rule(color="#f4a261", strokeWidth=2.8)
            .encode(
                x=alt.X("parent_eval_1:Q", scale=alt.Scale(domain=x_domain, nice=False)),
                y=alt.Y("parent_eval_2:Q", scale=alt.Scale(domain=y_domain, nice=False)),
                x2="pred_eval_1:Q",
                y2="pred_eval_2:Q",
                tooltip=["round", "policy_id", "weight_label", "selection_score"],
            )
        )
        predicted_points = (
            alt.Chart(selected_df)
            .mark_point(shape="diamond", size=120, filled=True, color="#f4a261")
            .encode(
                x=alt.X("pred_eval_1:Q", scale=alt.Scale(domain=x_domain, nice=False)),
                y=alt.Y("pred_eval_2:Q", scale=alt.Scale(domain=y_domain, nice=False)),
                tooltip=["round", "policy_id", "weight_label", "pred_eval_1", "pred_eval_2"],
            )
        )
        layers.extend([predicted_lines, predicted_front_line, predicted_points])

    return alt.layer(*layers).properties(height=280, title="3-3A. Current archive -> predicted virtual front")


def make_actual_front_chart(
    population: pd.DataFrame,
    current_front: pd.DataFrame,
    actual_front: pd.DataFrame,
    selected_df: pd.DataFrame,
) -> alt.Chart:
    x_domain, y_domain = _front_domains(population, current_front, pd.DataFrame(), actual_front, selected_df)
    x_enc = alt.X("eval_1:Q", title="Performance Space f1", scale=alt.Scale(domain=x_domain, nice=False))
    y_enc = alt.Y("eval_2:Q", title="f2", scale=alt.Scale(domain=y_domain, nice=False))

    base_points = (
        alt.Chart(population)
        .mark_circle(size=90, color="#457b9d", opacity=0.7)
        .encode(x=x_enc, y=y_enc, tooltip=["policy_id", "mu", "eval_1", "eval_2"])
    )
    current_front_line = alt.Chart(current_front).mark_line(strokeWidth=3.5, color="#111111").encode(x=x_enc, y=y_enc)
    actual_front_line = alt.Chart(actual_front).mark_line(strokeWidth=3.5, color="#2a9d8f").encode(x=x_enc, y=y_enc)

    layers: list[alt.Chart] = [base_points, current_front_line]
    if not selected_df.empty:
        actual_lines = (
            alt.Chart(selected_df)
            .mark_rule(color="#2a9d8f", strokeWidth=3.0)
            .encode(
                x=alt.X("parent_eval_1:Q", scale=alt.Scale(domain=x_domain, nice=False)),
                y=alt.Y("parent_eval_2:Q", scale=alt.Scale(domain=y_domain, nice=False)),
                x2="actual_eval_1:Q",
                y2="actual_eval_2:Q",
                tooltip=["round", "policy_id", "weight_label"],
            )
        )
        actual_points = (
            alt.Chart(selected_df)
            .mark_point(shape="triangle-up", size=130, filled=True, color="#2a9d8f")
            .encode(
                x=alt.X("actual_eval_1:Q", scale=alt.Scale(domain=x_domain, nice=False)),
                y=alt.Y("actual_eval_2:Q", scale=alt.Scale(domain=y_domain, nice=False)),
                tooltip=["round", "policy_id", "weight_label", "actual_eval_1", "actual_eval_2"],
            )
        )
        layers.extend([actual_lines, actual_front_line, actual_points])

    return alt.layer(*layers).properties(height=280, title="3-3B. Current archive -> actual updated front")


def make_predictor_curve_chart(candidate_df: pd.DataFrame, focus_policy_id: int) -> alt.Chart | None:
    focus = candidate_df[candidate_df["policy_id"] == focus_policy_id].copy()
    if focus.empty:
        return None
    curve = focus.melt(
        id_vars=["weight_1", "weight_2", "weight_label"],
        value_vars=["pred_delta_1", "pred_delta_2"],
        var_name="objective",
        value_name="delta",
    )
    curve["objective"] = curve["objective"].map({"pred_delta_1": "Δf1(ω)", "pred_delta_2": "Δf2(ω)"})
    return (
        alt.Chart(curve)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("weight_1:Q", title="weight ω1"),
            y=alt.Y("delta:Q", title="Predicted improvement ΔF"),
            color=alt.Color("objective:N", title=None),
            tooltip=["weight_label", "objective", "delta"],
        )
        .properties(height=290, title=f"Focus policy #{focus_policy_id}의 predictor curve")
    )


def make_score_heatmap(round_scores: pd.DataFrame, selected_df: pd.DataFrame, round_idx: int) -> alt.Chart | None:
    frame = round_scores[round_scores["round"] == round_idx].copy()
    if frame.empty:
        return None
    frame["policy_label"] = frame["policy_id"].map(lambda value: f"π{value}")
    heat = (
        alt.Chart(frame)
        .mark_rect()
        .encode(
            x=alt.X("weight_label:N", title="weight 후보"),
            y=alt.Y("policy_label:N", sort="descending", title="policy"),
            color=alt.Color("score:Q", title="Q score"),
            tooltip=["policy_id", "weight_label", "score"],
        )
    )
    chosen = selected_df[selected_df["round"] == round_idx].copy()
    if chosen.empty:
        return heat.properties(height=560, title=f"Selection round {round_idx}의 Q(policy, weight) score matrix")
    chosen["policy_label"] = chosen["policy_id"].map(lambda value: f"π{value}")
    marker = (
        alt.Chart(chosen)
        .mark_point(shape="diamond", size=220, filled=False, stroke="#111111", strokeWidth=2.5)
        .encode(x="weight_label:N", y=alt.Y("policy_label:N", sort="descending"))
    )
    return (heat + marker).properties(height=560, title=f"Selection round {round_idx}의 Q(policy, weight) score matrix")


def make_trace_chart(trace_df: pd.DataFrame) -> alt.Chart:
    melted = trace_df.melt(
        id_vars=["round", "kind"],
        value_vars=["hypervolume", "sparsity"],
        var_name="metric",
        value_name="value",
    )
    return (
        alt.Chart(melted)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("round:Q", title="선택 round"),
            y=alt.Y("value:Q", title="metric value"),
            color=alt.Color("metric:N", title=None),
            strokeDash=alt.StrokeDash("kind:N", title=None),
            tooltip=["round", "kind", "metric", "value"],
        )
        .properties(height=280, title="Top-n task를 하나씩 고를 때 virtual archive metric 변화")
    )


def build_step4_analysis(config: LoopConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_population = build_population(config, config.generations)
    rounded_front = {
        tuple(np.round(point, 8))
        for point in pareto_front(final_population[["eval_1", "eval_2"]].to_numpy(dtype=np.float64))
    }
    archive_points = final_population[
        final_population.apply(
            lambda row: tuple(np.round([row["eval_1"], row["eval_2"]], 8)) in rounded_front,
            axis=1,
        )
    ].copy().reset_index(drop=True)
    if archive_points.empty:
        return archive_points, pd.DataFrame()
    archive_points["front_rank"] = np.arange(len(archive_points), dtype=int)
    preview_dim = 16
    indices = np.arange(preview_dim, dtype=float)
    latent = archive_points["latent"].to_numpy(dtype=np.float64)
    coarse_family = np.digitize(latent, bins=[0.34, 0.67])
    centers = np.array(
        [
            np.sin(indices * 0.31) * 0.9,
            np.cos(indices * 0.41) * 0.9,
            np.sin(indices * 0.23 + 1.7) * 0.9,
        ],
        dtype=np.float64,
    )

    vectors = []
    for idx, row in enumerate(archive_points.itertuples(index=False)):
        family = coarse_family[idx]
        noise = 0.08 * np.sin(indices * (0.2 + 0.03 * idx) + row.latent * 5.0)
        performance_term = 0.04 * row.eval_1 - 0.03 * row.eval_2
        vector = centers[family] + noise + performance_term / 10.0
        vectors.append(vector)
    vectors_array = np.stack(vectors, axis=0)

    if len(archive_points) == 1:
        embedding = np.array([[0.0, 0.0]], dtype=np.float64)
        cluster_ids = np.array([0], dtype=int)
    else:
        perplexity = max(2, min(18, len(archive_points) - 1))
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=config.seed,
        ).fit_transform(vectors_array)
        cluster_count = min(3, len(archive_points))
        if cluster_count == 1:
            cluster_ids = np.array([0] * len(archive_points), dtype=int)
        else:
            cluster_ids = KMeans(n_clusters=cluster_count, random_state=config.seed, n_init=10).fit_predict(embedding)
    archive_points = archive_points.copy()
    archive_points["param_x"] = embedding[:, 0]
    archive_points["param_y"] = embedding[:, 1]
    archive_points["cluster_id"] = cluster_ids
    cluster_order = archive_points.groupby("cluster_id", as_index=False)["latent"].mean().sort_values("latent")
    cluster_map = {int(row["cluster_id"]): f"Family {idx + 1}" for idx, row in cluster_order.iterrows()}
    archive_points["family"] = archive_points["cluster_id"].map(cluster_map)

    preview_rows: list[dict[str, Any]] = []
    for idx, vector in enumerate(vectors_array):
        row = archive_points.iloc[idx]
        preview_rows.append(
            {
                "policy_id": int(row["policy_id"]),
                "family": row["family"],
                "mu": float(row["mu"]),
                "eval_1": float(row["eval_1"]),
                "eval_2": float(row["eval_2"]),
                **{f"dim_{dim + 1}": float(value) for dim, value in enumerate(vector)},
            }
        )
    preview_df = pd.DataFrame(preview_rows)
    return archive_points, preview_df


def make_parameter_space_chart(archive_points: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(archive_points)
        .mark_circle(size=120, filled=True, stroke="#102a43", strokeWidth=1.2)
        .encode(
            x=alt.X("param_x:Q", title="Parameter Space t-SNE 1"),
            y=alt.Y("param_y:Q", title="t-SNE 2"),
            color=alt.Color("family:N", title="Family"),
            tooltip=["policy_id", "family", "mu", "eval_1", "eval_2"],
        )
        .properties(height=320, title="Step 4: Θ_pareto를 t-SNE + k-means로 family화")
        .interactive()
    )


def make_interpolation_chart(family_points: pd.DataFrame, left_id: int, right_id: int, beta: float) -> alt.Chart:
    left = family_points[family_points["policy_id"] == left_id].iloc[0]
    right = family_points[family_points["policy_id"] == right_id].iloc[0]
    interp = pd.DataFrame(
        {
            "label": ["θ_X", "θ_Y", "θ_new"],
            "eval_1": [left["eval_1"], right["eval_1"], beta * left["eval_1"] + (1.0 - beta) * right["eval_1"]],
            "eval_2": [left["eval_2"], right["eval_2"], beta * left["eval_2"] + (1.0 - beta) * right["eval_2"]],
        }
    )
    segment = pd.DataFrame(
        {
            "eval_1": [left["eval_1"], right["eval_1"]],
            "eval_2": [left["eval_2"], right["eval_2"]],
        }
    )
    line = alt.Chart(segment).mark_line(strokeWidth=4, color="#7f5539").encode(
        x=alt.X("eval_1:Q", title="Performance Space f1"),
        y=alt.Y("eval_2:Q", title="f2"),
    )
    points = (
        alt.Chart(interp)
        .mark_point(size=170, filled=True)
        .encode(
            x="eval_1:Q",
            y="eval_2:Q",
            shape=alt.Shape("label:N", title=None),
            color=alt.Color("label:N", title=None),
            tooltip=["label", "eval_1", "eval_2"],
        )
    )
    return (line + points).properties(height=320, title="같은 family 내부 선형 보간으로 새 controller 생성")


def main() -> None:
    st.set_page_config(page_title="PG-MORL Loop Explorer", layout="wide")
    st.title("PG-MORL Loop Explorer")
    st.caption(
        "Warm-up 1회, 세대별 Predictor/Task Selection 루프, 마지막 Pareto Analysis까지를 "
        "행렬/벡터의 shape와 실행 주기 관점에서 한 번에 이해하기 위한 교육용 대시보드입니다."
    )

    with st.sidebar:
        st.markdown("**논문 가정값**")
        st.caption("사용자가 정리해 준 설명 흐름에 맞춰 기본값을 맞췄습니다.")
        population_size = st.slider("유지 policy 수 |P|", min_value=10, max_value=80, value=50, step=5)
        param_dim = st.slider("파라미터 차원 N", min_value=10_000, max_value=200_000, value=100_000, step=10_000)
        weight_candidates = st.slider("가중치 후보 수 K", min_value=5, max_value=21, value=11, step=2)
        selected_tasks = st.slider("실제 훈련 task 수 n", min_value=2, max_value=12, value=5, step=1)
        generations = st.slider("총 세대 수 M", min_value=10, max_value=200, value=100, step=10)
        generation = st.slider("현재 보고 싶은 generation g", min_value=1, max_value=generations, value=min(20, generations), step=1)
        round_idx = st.slider("현재 보고 싶은 selection round r", min_value=1, max_value=max(1, selected_tasks), value=1, step=1)

        config = LoopConfig(
            population_size=population_size,
            param_dim=param_dim,
            weight_candidates=weight_candidates,
            selected_tasks=selected_tasks,
            generations=generations,
        )

        st.markdown("**핵심 절감량**")
        predicted_candidates = config.population_size * config.weight_candidates
        brute_force = predicted_candidates
        saved_ratio = 1.0 - (config.selected_tasks / max(brute_force, 1))
        st.metric("세대당 예측 후보 수 |P|×K", fmt_int(predicted_candidates))
        st.metric("세대당 실제 MOPG 실행 수 n", fmt_int(config.selected_tasks))
        st.metric("시뮬레이터 호출 절감률", f"{saved_ratio * 100:.1f}%")

    bundle = build_generation_bundle(config, generation)
    population = bundle["population"]
    candidate_df = bundle["candidate_df"]
    predictor_params = bundle["predictor_params"]
    selected_df = bundle["selected_df"]
    round_scores = bundle["round_scores"]
    trace_df = bundle["trace_df"]
    current_front = bundle["current_front"]
    predicted_front = bundle["predicted_front"]
    actual_front = bundle["actual_front"]

    focus_policy_id = st.select_slider(
        "Focus policy 선택",
        options=population["policy_id"].astype(int).tolist(),
        value=int(population.iloc[len(population) // 2]["policy_id"]),
    )
    st.caption(
        "이 선택기는 Step 2 predictor를 들여다보기 위한 것입니다. "
        "아래 Step 3의 실제 selected task는 이 focus policy와 다를 수 있습니다."
    )
    focus_candidates = candidate_df[candidate_df["policy_id"] == focus_policy_id].copy()
    focus_policy = population[population["policy_id"] == focus_policy_id].iloc[0]
    current_archive_size = len(actual_front) if not actual_front.empty else len(current_front)
    focus_selected_rows = selected_df[selected_df["policy_id"] == focus_policy_id].copy()

    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("Warm-up 실제 훈련 수", fmt_int(config.population_size))
    with metric_cols[1]:
        st.metric(f"g={generation}일 때 R row 수", fmt_int(config.population_size + generation * config.selected_tasks))
    with metric_cols[2]:
        st.metric("매 세대 predictor 피팅 수", fmt_int(config.population_size))
    with metric_cols[3]:
        st.metric("매 세대 score matrix 크기", fmt_shape(config.population_size, config.weight_candidates))
    with metric_cols[4]:
        st.metric("현재 예시 archive 크기", fmt_int(current_archive_size))

    loop_cols = st.columns(5)
    with loop_cols[0]:
        with st.container(border=True):
            st.markdown("**Step 1**")
            st.caption("Warm-up")
            st.write("한 번만 실행")
    with loop_cols[1]:
        with st.container(border=True):
            st.markdown("**Step 2**")
            st.caption("Predictor fitting")
            st.write(f"현재 generation `g={generation}`")
    with loop_cols[2]:
        with st.container(border=True):
            st.markdown("**Step 3-1**")
            st.caption("Q matrix selection")
            st.write(f"현재 round `r={round_idx}`")
    with loop_cols[3]:
        with st.container(border=True):
            st.markdown("**Step 3-2**")
            st.caption("Actual MOPG update")
            st.write(f"Top-{config.selected_tasks} task 훈련")
    with loop_cols[4]:
        with st.container(border=True):
            st.markdown("**Step 4**")
            st.caption("Pareto interpolation")
            st.write("훈련 종료 후")

    with st.container(border=True):
        st.markdown("**먼저 이 순서로 읽으면 됩니다**")
        st.write("1. `Step별 입력 -> 계산 -> 출력` 표에서 각 단계가 무엇을 받아 무엇을 내보내는지 먼저 봅니다.")
        st.write("2. `행렬 / 벡터 보드`에서 각 심볼의 shape와 갱신 주기를 확인합니다.")
        st.write("3. `Generation 해부`에서 predictor -> Q matrix -> top-n selection -> front 업데이트가 실제로 어떻게 연결되는지 봅니다.")
        st.write("4. 마지막으로 `Step 4`에서 왜 학습이 끝난 뒤에도 새 controller를 즉시 만들 수 있는지 확인합니다.")

    if focus_selected_rows.empty:
        st.info(f"focus policy π{focus_policy_id}는 Step 2 predictor를 보는 용도이며, 이 generation에서는 실제 selected task로 뽑히지 않았습니다.")
    else:
        selected_rounds = ", ".join(str(int(value)) for value in focus_selected_rows["round"].tolist())
        st.success(f"focus policy π{focus_policy_id}는 이 generation에서 실제로 선택되었고, round {selected_rounds}에서 뽑혔습니다.")

    with st.expander("초보자가 가장 많이 헷갈리는 질문 먼저 보기"):
        st.markdown(
            "- `Q matrix 한 칸은 무엇인가?`: 하나의 `(policy i, weight k)` 조합을 실제로 훈련했을 때 virtual archive가 얼마나 좋아질지에 대한 점수입니다.\n"
            "- `Δ와 F_pred의 차이는 무엇인가?`: `Δ`는 개선량 벡터이고, `F_pred = 현재 성능 + 개선량`은 예상되는 절대 성능 벡터입니다.\n"
            "- `왜 550개를 다 훈련하지 않고 5개만 훈련하나?`: predictor로 먼저 순위를 매겨서, 실제 시뮬레이터는 top-n task에만 씁니다.\n"
            "- `R buffer에는 무엇이 쌓이나?`: 실제로 훈련한 task에서 관측된 `(사용한 weight, before/after 성능 차이 ΔF)`가 쌓입니다.\n"
            "- `Step 4의 보간은 왜 가능한가?`: 비슷한 Pareto policy들이 parameter-space에서 family를 이루기 때문에, 같은 family 안에서는 중간 타협점을 선형 보간으로 근사할 수 있다고 보는 것입니다."
        )

    top_left, top_right = st.columns([1.05, 1.1], gap="large")
    with top_left:
        st.altair_chart(make_loop_clock_chart(config), use_container_width=True)
        guide(
            "전체 루프 Legend + 설명",
            "Warm-up은 최초 1회만 실행되고, Step 2와 Step 3는 generation마다 반복됩니다. Step 4는 훈련이 끝난 뒤 한 번만 수행됩니다.",
            [
                ("청록 막대", "한 번만 수행되는 단계"),
                ("주황 막대", "generation마다 반복 수행되는 단계"),
            ],
        )
    with top_right:
        st.altair_chart(make_records_growth_chart(config), use_container_width=True)
        guide(
            "R buffer 성장 설명",
            "PG-MORL은 모든 (policy, weight)를 실제로 훈련하지 않고, 소수의 selected task만 실제 MOPG를 돌립니다. 그래서 R buffer는 P + g·n 행으로 증가합니다.",
            [
                ("파란 선", "실제로 R에 쌓이는 transition row 수"),
                ("주황 선", "brute-force로 모든 후보를 실제 훈련했을 때의 누적 호출량"),
            ],
        )

    st.divider()
    st.markdown("**Step별 입력 -> 계산 -> 출력**")
    st.caption(
        "이 표는 초보자 기준으로 가장 중요한 요약입니다. 각 단계에서 '무엇이 들어오고', "
        "'무엇을 계산하고', '무엇이 다음 단계로 넘어가는지'만 먼저 잡으면 뒤의 그래프가 훨씬 쉽게 읽힙니다."
    )
    st.dataframe(build_step_flow_table(config, generation, current_archive_size), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**행렬 / 벡터 보드**")
    st.caption(
        f"generation {generation} 기준으로 지금 메모리 안에서 굴러가는 핵심 텐서를 표로 정리했습니다. "
        "shape는 논문-scale 추상화이고, 각 row의 설명은 실제 loop 안에서 그 텐서가 언제 만들어지고 언제 갱신되는지 뜻합니다."
    )
    shape_table = build_shape_table(config, generation, current_archive_size)
    st.dataframe(shape_table, use_container_width=True, hide_index=True)

    st.latex(
        r"""
        F_{\text{pred}}[i, k, :] = F(\pi_i) + \Delta^i(\omega_k), \quad
        Q[i, k] = \mathcal{H}(EP^\* \cup F_{\text{pred}}[i,k]) - \lambda \mathcal{S}(EP^\* \cup F_{\text{pred}}[i,k])
        """
    )
    st.caption(
        "즉 Step 2는 `policy별 predictor tensor Ξ`를 만들고, Step 3는 그 predictor를 이용해 "
        "`Δ → F_pred → Q → Top-n task` 순서로 흘러갑니다."
    )
    st.caption(
        "여기서 핵심은 `Δ`가 개선량 tensor, `F_pred`가 예상 절대성능 tensor, `Q`가 top-n selection용 점수 행렬이라는 점입니다."
    )
    st.caption("Q는 '클수록 좋은' composite score이며, heatmap의 진한 칸일수록 실제로 뽑힐 가능성이 높습니다.")

    st.divider()
    st.markdown(f"**Generation {generation} 해부**")
    row1_left, row1_right = st.columns([1.25, 1.0], gap="large")
    with row1_left:
        st.altair_chart(
            make_predicted_front_chart(population, current_front, predicted_front, selected_df),
            use_container_width=True,
        )
        st.altair_chart(
            make_actual_front_chart(population, current_front, actual_front, selected_df),
            use_container_width=True,
        )
        guide(
            "Front 변화 Legend + 설명",
            "초보자에게는 predicted와 actual을 분리해서 읽는 편이 더 쉽습니다. 첫 번째 그래프는 predictor가 상상한 virtual front, 두 번째 그래프는 실제 MOPG 후 만들어진 updated front입니다.",
            [
                ("파란 원", "현재 population의 policy"),
                ("검은 실선", "generation 시작 시점의 current archive front"),
                ("주황 다이아 + 점선", "선택된 task의 predicted offspring과 predicted virtual front"),
                ("초록 삼각형 + 실선", "실제 학습 후 offspring과 actual updated front"),
            ],
        )
    with row1_right:
        predictor_chart = make_predictor_curve_chart(candidate_df, focus_policy_id)
        if predictor_chart is not None:
            st.altair_chart(predictor_chart, use_container_width=True)
        guide(
            "Predictor Curve Legend + 설명",
            f"focus policy π{focus_policy_id}의 현재 성능은 F(π)=[{focus_policy['eval_1']:.2f}, {focus_policy['eval_2']:.2f}] 입니다. "
            "이 그래프는 weight를 바꿨을 때 각 objective의 개선량 Δf가 어떻게 달라지는지 보여줍니다.",
            [
                ("파란/주황 곡선", "objective별 predicted improvement Δf"),
                ("점 마커", "discretized weight 후보 K개에서 실제 평가한 위치"),
            ],
        )

        focus_params = predictor_params[predictor_params["policy_id"] == focus_policy_id].copy()
        st.markdown("**focus policy의 predictor 계수 Ξ_i ∈ R^{m×4}**")
        st.dataframe(focus_params, use_container_width=True, hide_index=True)
        st.caption(
            "논문식으로 보면 각 objective마다 [A, a, b, c] 4개 계수를 피팅해서 "
            "단조 쌍곡선 predictor를 만듭니다."
        )
        st.caption("해석 팁: `A`는 최대 변화 폭, `b`는 곡선의 중심 weight, `c`는 기본 개선량 offset처럼 읽으면 됩니다.")

    row2_left, row2_right = st.columns([1.0, 1.0], gap="large")
    with row2_left:
        score_heatmap = make_score_heatmap(round_scores, selected_df, round_idx)
        if score_heatmap is not None:
            st.altair_chart(score_heatmap, use_container_width=True)
        guide(
            "Q score matrix Legend + 설명",
            "한 round 안에서는 현재 virtual archive를 기준으로 모든 (policy, weight) 후보를 다시 채점합니다. "
            "검은 다이아는 그 round에서 실제로 선택된 top-1 task입니다.",
            [
                ("heat color", "score Q가 높을수록 이번 round에 선택될 가능성이 큼"),
                ("검은 다이아", "해당 round에서 실제로 선택된 task"),
            ],
        )
        st.caption("이 heatmap은 현재 round의 순위표이고, 바로 아래 벡터 예시는 이 round에서 실제로 선택된 task를 보여줍니다.")
    with row2_right:
        st.altair_chart(make_trace_chart(trace_df), use_container_width=True)
        guide(
            "Virtual Archive Metric Legend + 설명",
            "task를 하나씩 고를 때마다 virtual archive를 업데이트하므로, 다음 round의 Q score matrix는 이전 round와 달라집니다.",
            [
                ("HV 선", "virtual archive hypervolume"),
                ("Sparsity 선", "virtual archive sparsity"),
                ("round 0", "선택 전 current archive 상태"),
            ],
        )
        active_task = selected_df[selected_df["round"] == round_idx].copy()
        if active_task.empty:
            active_task = selected_df.head(1).copy()
        if active_task.empty:
            active_task = focus_candidates.sort_values("pred_delta_1", ascending=False).head(1).copy()
        focus_task = active_task.iloc[0]
        with st.container(border=True):
            st.markdown("**선택된 셀 상세: Q matrix 한 칸이 의미하는 것**")
            st.write(
                {
                    "round": int(focus_task["round"]) if "round" in focus_task else round_idx,
                    "policy": f"π{int(focus_task['policy_id'])}",
                    "weight": str(focus_task["weight_label"]),
                    "Q[i,k]": f"{float(focus_task['selection_score'] if 'selection_score' in focus_task else np.nan):.2f}",
                }
            )
            st.latex(
                rf"""
                F(\pi_{{{int(focus_task["policy_id"])}}}) =
                \begin{{bmatrix}}
                {focus_task["parent_eval_1"]:.2f} \\
                {focus_task["parent_eval_2"]:.2f}
                \end{{bmatrix}},
                \quad
                \Delta =
                \begin{{bmatrix}}
                {focus_task["pred_delta_1"]:.2f} \\
                {focus_task["pred_delta_2"]:.2f}
                \end{{bmatrix}},
                \quad
                F_{{pred}} =
                \begin{{bmatrix}}
                {focus_task["pred_eval_1"]:.2f} \\
                {focus_task["pred_eval_2"]:.2f}
                \end{{bmatrix}},
                \quad
                F'_{{\text{{actual}}}} =
                \begin{{bmatrix}}
                {focus_task["actual_eval_1"]:.2f} \\
                {focus_task["actual_eval_2"]:.2f}
                \end{{bmatrix}}
                """
            )
            st.caption(
                "즉 Q matrix의 한 칸은 `(policy i, weight k)` 후보 하나를 뜻합니다. "
                "그 칸을 누가 이겼는지 보고, predictor가 만든 `Δ`와 `F_pred`, 실제 훈련 후 `F'_actual`을 비교하면 "
                "selection과 update가 한 줄로 연결됩니다."
            )

    st.markdown("**이번 generation의 Top-n selected tasks**")
    if selected_df.empty:
        st.info("선택된 task가 없습니다.")
    else:
        selected_view = selected_df[
            [
                "round",
                "policy_id",
                "weight_label",
                "selection_score",
                "pred_eval_1",
                "pred_eval_2",
                "actual_eval_1",
                "actual_eval_2",
            ]
        ].copy()
        selected_view.columns = [
            "Round",
            "Policy",
            "Selected weight",
            "Q score",
            "Predicted f1",
            "Predicted f2",
            "Actual f1",
            "Actual f2",
        ]
        st.dataframe(selected_view, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.markdown("**현재 보고 있는 selection round의 벡터 흐름 예시**")
        st.latex(
            rf"""
            F(\pi_{{{int(focus_task["policy_id"])}}}) =
            \begin{{bmatrix}}
            {focus_task["parent_eval_1"]:.2f} \\
            {focus_task["parent_eval_2"]:.2f}
            \end{{bmatrix}},
            \quad
            \Delta^{{{int(focus_task["policy_id"])}}}(\omega_{{{int(focus_task["weight_index"])}}}) =
            \begin{{bmatrix}}
            {focus_task["pred_delta_1"]:.2f} \\
            {focus_task["pred_delta_2"]:.2f}
            \end{{bmatrix}},
            \quad
            F_{{pred}} =
            \begin{{bmatrix}}
            {focus_task["pred_eval_1"]:.2f} \\
            {focus_task["pred_eval_2"]:.2f}
            \end{{bmatrix}}
            """
        )
        st.caption(
            f"지금은 round {int(focus_task['round'])}에서 실제로 선택된 task 예시를 보여주고 있습니다. "
            "Step 3에서는 모든 policy i와 모든 weight k에 대해 이런 벡터 덧셈을 만들어 보고, "
            "그중 Q가 가장 큰 (i, k)를 뽑아 실제 역전파를 돌립니다."
        )

    st.divider()
    st.markdown("**Step 4: 훈련 종료 후 Pareto Analysis / 보간**")
    st.warning(
        "이 Step 4 화면의 θ preview는 100,000차원 전체 파라미터를 직접 그린 것이 아니라, "
        "최종 Pareto archive의 구조를 이해시키기 위한 sampled/embedded view 입니다."
    )
    archive_points, preview_df = build_step4_analysis(config)
    step4_left, step4_right = st.columns([1.0, 1.0], gap="large")
    with step4_left:
        st.altair_chart(make_parameter_space_chart(archive_points), use_container_width=True)
        guide(
            "Parameter Space Legend + 설명",
            "최종 archive의 policy vectors를 압축한 뒤 family로 묶습니다. 실제 논문은 10만 차원 θ를 다루고, 이 앱은 그 구조를 설명하기 위한 preview vector를 사용합니다.",
            [
                ("색 점", "Pareto archive의 policy"),
                ("색깔", "k-means로 묶인 family"),
            ],
        )
    with step4_right:
        family_options = archive_points["family"].drop_duplicates().tolist()
        selected_family = st.selectbox("보간을 볼 family", family_options)
        family_points = archive_points[archive_points["family"] == selected_family].sort_values("mu").copy()
        if len(family_points) < 2:
            st.info("이 family에는 보간할 anchor가 충분하지 않습니다.")
        else:
            anchor_ids = family_points["policy_id"].astype(int).tolist()
            left_default = anchor_ids[0]
            right_default = anchor_ids[min(1, len(anchor_ids) - 1)]
            left_id = st.selectbox("좌 anchor θ_X", anchor_ids, index=anchor_ids.index(left_default), key="left_anchor")
            right_choices = [value for value in anchor_ids if value != left_id]
            right_id = st.selectbox(
                "우 anchor θ_Y",
                right_choices,
                index=min(0, len(right_choices) - 1),
                key="right_anchor",
            )
            beta = st.slider("보간 계수 β", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
            st.altair_chart(make_interpolation_chart(family_points, left_id, right_id, beta), use_container_width=True)
            guide(
                "Interpolation Legend + 설명",
                "같은 family 안의 두 anchor를 섞어 새로운 controller를 만듭니다. "
                "논문에서는 이 과정을 통해 학습 때 직접 보지 못한 KPI 타협점도 즉시 생성합니다.",
                [
                    ("θ_X / θ_Y", "보간에 사용하는 두 anchor controller"),
                    ("θ_new", "βθ_X + (1-β)θ_Y 로 만든 새 controller"),
                ],
            )

            left_vec = preview_df[preview_df["policy_id"] == left_id].iloc[0]
            right_vec = preview_df[preview_df["policy_id"] == right_id].iloc[0]
            dim_cols = [column for column in preview_df.columns if column.startswith("dim_")][:8]
            preview_rows = []
            for dim in dim_cols:
                preview_rows.append(
                    {
                        "coordinate": dim,
                        "θ_X": float(left_vec[dim]),
                        "θ_Y": float(right_vec[dim]),
                        "θ_new": float(beta * left_vec[dim] + (1.0 - beta) * right_vec[dim]),
                    }
                )
            st.markdown("**θ_new preview (앞 8개 샘플 좌표)**")
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
            st.caption(
                f"표에는 설명용으로 일부 좌표만 보여주지만, 실제 생성되는 controller의 shape는 "
                f"θ_new ∈ R^{fmt_int(config.param_dim)} 입니다."
            )

    st.divider()
    with st.expander("수식 / 루프 의사코드 / 논문 서술 대응"):
        st.markdown(
            "- `Warm-up 1회`: |P|개 policy를 실제 훈련해 첫 population / archive / R buffer를 만든다.\n"
            "- `매 generation`: predictor 피팅 -> 모든 (policy, weight) 후보 평가 -> top-n task 실제 훈련 -> archive 갱신.\n"
            "- `훈련 종료 후`: Θ_pareto를 family로 묶고, 같은 family 안에서 선형 보간해 θ_new를 만든다."
        )
        st.latex(r"\Theta = [\theta_1, \dots, \theta_{|\mathcal{P}|}] \in \mathbb{R}^{N \times |\mathcal{P}|}")
        st.latex(r"F_{\text{pred}}[i,k,:] = F(\pi_i) + \Delta^i(\omega_k)")
        st.latex(r"Q[i,k] = \mathcal{H}(EP^\* \cup F_{\text{pred}}[i,k]) - \lambda \mathcal{S}(EP^\* \cup F_{\text{pred}}[i,k])")
        st.latex(r"\theta_{new} = \beta \theta_X + (1-\beta)\theta_Y")
        st.code(
            """Warm-up (1 time):
  initialize Θ in R^{N x |P|}
  assign warm-up weights
  run MOPG for all |P| policies
  store transitions into R

For generation g = 1 ... M:
  fit predictor tensor Ξ for all policies
  build weight grid W_grid with K candidates
  compute Δ in R^{|P| x K x m}
  compute F_pred = F + Δ
  score all pairs into Q in R^{|P| x K}
  greedily choose top-n tasks
  run real MOPG only for those n tasks
  append new rows to R and update EP / population

After training:
  collect Θ_pareto from final archive
  embed to Z in R^{|EP| x 2}
  cluster into families
  interpolate inside one family to get θ_new
""",
            language="text",
        )


if __name__ == "__main__":
    main()
