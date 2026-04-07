from __future__ import annotations

import json
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from toy_pgmorl.core import (
    ToyConfig,
    closed_form_optimum_mu,
    expected_objectives,
    hypervolume,
    pareto_front,
    run_experiment,
    sample_objectives,
    sparsity,
)


alt.data_transformers.disable_max_rows()

PGMORL_METHOD = "PG-MORL Toy"


def objective_labels(config: ToyConfig) -> list[str]:
    return [f"f{i + 1}: {name}" for i, name in enumerate(config.objective_names)]


def config_signature(config: ToyConfig) -> str:
    return json.dumps(config.to_dict(), sort_keys=True, ensure_ascii=False)


@st.cache_data(show_spinner=False)
def load_result(config: ToyConfig):
    return run_experiment(config)


@st.cache_data(show_spinner=False)
def build_family_analysis(archive_snapshot: pd.DataFrame, config: ToyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshot = archive_snapshot.drop_duplicates(subset=["policy_id"]).copy()
    if snapshot.empty:
        return pd.DataFrame(), pd.DataFrame()

    snapshot = snapshot.sort_values("mu").reset_index(drop=True)
    features = snapshot[["mu", "eval_1", "eval_2"]].to_numpy(dtype=np.float64)

    if len(snapshot) >= 4:
        perplexity = max(2, min(8, len(snapshot) - 1))
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=config.seed,
        ).fit_transform(features)
    else:
        embedding = np.column_stack(
            [
                features[:, 0],
                features[:, 1] - features[:, 2],
            ]
        )

    cluster_count = min(3, max(1, len(snapshot) // 8 + 1))
    if len(snapshot) < 3 or cluster_count == 1:
        raw_labels = np.zeros(len(snapshot), dtype=int)
    else:
        raw_labels = KMeans(n_clusters=cluster_count, random_state=config.seed, n_init=10).fit_predict(embedding)

    snapshot["param_x"] = embedding[:, 0]
    snapshot["param_y"] = embedding[:, 1]
    snapshot["raw_family"] = raw_labels

    family_order = (
        snapshot.groupby("raw_family", as_index=False)["mu"]
        .mean()
        .sort_values("mu")
        .reset_index(drop=True)
    )
    family_map = {int(row["raw_family"]): f"Family {idx}" for idx, row in family_order.iterrows()}
    snapshot["family"] = snapshot["raw_family"].map(family_map)

    interpolated_rows: list[dict[str, Any]] = []
    interp_id = 0
    for family_name, group in snapshot.groupby("family", sort=False):
        ordered = group.sort_values("mu").reset_index(drop=True)
        if len(ordered) == 1:
            row = ordered.iloc[0]
            evaluation = expected_objectives(float(row["mu"]), config)
            interpolated_rows.append(
                {
                    "interp_id": interp_id,
                    "family": family_name,
                    "alpha": 0.0,
                    "mu": float(row["mu"]),
                    "eval_1": float(evaluation[0]),
                    "eval_2": float(evaluation[1]),
                    "param_x": float(row["param_x"]),
                    "param_y": float(row["param_y"]),
                }
            )
            interp_id += 1
            continue

        for left_idx in range(len(ordered) - 1):
            left = ordered.iloc[left_idx]
            right = ordered.iloc[left_idx + 1]
            for alpha in np.linspace(0.0, 1.0, 28):
                mu = float((1.0 - alpha) * left["mu"] + alpha * right["mu"])
                evaluation = expected_objectives(mu, config)
                interpolated_rows.append(
                    {
                        "interp_id": interp_id,
                        "family": family_name,
                        "alpha": float(alpha),
                        "mu": mu,
                        "eval_1": float(evaluation[0]),
                        "eval_2": float(evaluation[1]),
                        "param_x": float((1.0 - alpha) * left["param_x"] + alpha * right["param_x"]),
                        "param_y": float((1.0 - alpha) * left["param_y"] + alpha * right["param_y"]),
                    }
                )
                interp_id += 1

    interpolation = pd.DataFrame(interpolated_rows)
    return snapshot, interpolation


def init_walkthrough_state(config: ToyConfig) -> None:
    signature = config_signature(config)
    if st.session_state.get("config_signature") == signature:
        return

    st.session_state.config_signature = signature
    st.session_state.warmup_done = False
    st.session_state.visible_generation = 0
    st.session_state.analysis_done = False
    st.session_state.focus_policy_id = None
    st.session_state.curve_interp_id = None
    st.session_state.status_message = "Warm-up 이전 상태입니다. Step 1을 눌러 첫 세대를 만드세요."


def reset_walkthrough(config: ToyConfig) -> None:
    st.session_state.config_signature = config_signature(config)
    st.session_state.warmup_done = False
    st.session_state.visible_generation = 0
    st.session_state.analysis_done = False
    st.session_state.focus_policy_id = None
    st.session_state.curve_interp_id = None
    st.session_state.status_message = "Walkthrough가 초기화되었습니다. Step 1부터 다시 진행하세요."


def extract_selection_value(event: Any, selection_name: str, field: str) -> Any | None:
    if event is None:
        return None
    selection = getattr(event, "selection", None)
    if selection is None:
        return None

    if hasattr(selection, "get"):
        payload = selection.get(selection_name, {})
    else:
        payload = getattr(selection, selection_name, {})

    if payload in ({}, None):
        return None

    value = payload.get(field) if hasattr(payload, "get") else getattr(payload, field, None)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def gauge_progress(current: float, minimum: float, maximum: float, invert: bool = False) -> float:
    if maximum <= minimum:
        return 1.0
    normalized = (current - minimum) / (maximum - minimum)
    normalized = float(np.clip(normalized, 0.0, 1.0))
    return 1.0 - normalized if invert else normalized


def render_gauge(title: str, current: float, delta: float, progress: float, help_text: str) -> None:
    with st.container(border=True):
        st.caption(title)
        st.metric("Current", f"{current:.2f}", f"{delta:+.2f}")
        st.progress(progress)
        st.caption(help_text)


def reward_curve_df(config: ToyConfig) -> pd.DataFrame:
    actions = np.linspace(config.init_mu_low, config.init_mu_high, 280)
    rewards = sample_objectives(actions, config)
    rows: list[dict[str, Any]] = []
    labels = objective_labels(config)
    for idx, label in enumerate(labels):
        for action, reward in zip(actions, rewards[:, idx]):
            rows.append({"action": action, "reward": reward, "objective": label})
    return pd.DataFrame(rows)


def predictor_generation_for_view(visible_generation: int) -> int:
    return 1 if visible_generation == 0 else visible_generation


def parent_population_snapshot(population_pg: pd.DataFrame, visible_generation: int) -> pd.DataFrame:
    snapshot_generation = 0 if visible_generation == 0 else visible_generation - 1
    snapshot = population_pg[population_pg["snapshot_generation"] == snapshot_generation].copy()
    return snapshot.drop_duplicates(subset=["policy_id"]).sort_values("policy_id").reset_index(drop=True)


def current_archive_snapshot(archive_pg: pd.DataFrame, visible_generation: int) -> pd.DataFrame:
    snapshot = archive_pg[archive_pg["snapshot_generation"] == visible_generation].copy()
    return snapshot.drop_duplicates(subset=["policy_id"]).sort_values("eval_1").reset_index(drop=True)


def previous_archive_snapshot(archive_pg: pd.DataFrame, visible_generation: int) -> pd.DataFrame:
    if visible_generation <= 0:
        return pd.DataFrame(columns=archive_pg.columns)
    return current_archive_snapshot(archive_pg, visible_generation - 1)


def current_task_rows(tasks_pg: pd.DataFrame, visible_generation: int) -> pd.DataFrame:
    if visible_generation <= 0:
        return pd.DataFrame()
    rows = tasks_pg[tasks_pg["generation"] == visible_generation].copy()
    return rows.sort_values(["slot", "score"], ascending=[True, False]).reset_index(drop=True)


def current_predictor_rows(predictor_pg: pd.DataFrame, visible_generation: int) -> pd.DataFrame:
    target_generation = predictor_generation_for_view(visible_generation)
    rows = predictor_pg[predictor_pg["generation"] == target_generation].copy()
    return rows.sort_values(["policy_id", "weight_1"]).reset_index(drop=True)


def initial_policy_rows(tasks_pg: pd.DataFrame) -> pd.DataFrame:
    warmup = tasks_pg[tasks_pg["generation"] == 0].copy()
    if warmup.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "policy_id": warmup["slot"].astype(int),
            "mu": warmup["parent_mu"],
            "eval_1": warmup["parent_eval_1"],
            "eval_2": warmup["parent_eval_2"],
        }
    )


def front_df_from_points(points: np.ndarray, label: str) -> pd.DataFrame:
    if points.size == 0:
        return pd.DataFrame(columns=["eval_1", "eval_2", "front_label", "order"])
    front = pareto_front(points)
    if front.size == 0:
        return pd.DataFrame(columns=["eval_1", "eval_2", "front_label", "order"])
    return pd.DataFrame(
        {
            "eval_1": front[:, 0],
            "eval_2": front[:, 1],
            "front_label": label,
            "order": np.arange(len(front)),
        }
    )


def snapshot_front_df(snapshot: pd.DataFrame, label: str) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame(columns=["eval_1", "eval_2", "front_label", "order"])
    points = snapshot[["eval_1", "eval_2"]].to_numpy(dtype=np.float64)
    return front_df_from_points(points, label)


def predicted_virtual_front_df(prev_archive: pd.DataFrame, task_rows: pd.DataFrame) -> pd.DataFrame:
    predicted = task_rows.dropna(subset=["pred_eval_1", "pred_eval_2"]).copy()
    if prev_archive.empty and predicted.empty:
        return pd.DataFrame(columns=["eval_1", "eval_2", "front_label", "order"])
    all_points: list[np.ndarray] = []
    if not prev_archive.empty:
        all_points.extend(prev_archive[["eval_1", "eval_2"]].to_numpy(dtype=np.float64))
    if not predicted.empty:
        all_points.extend(predicted[["pred_eval_1", "pred_eval_2"]].to_numpy(dtype=np.float64))
    return front_df_from_points(np.asarray(all_points, dtype=np.float64), "Predicted virtual front")


def update_focus_policy(candidate_points: pd.DataFrame) -> int | None:
    if candidate_points.empty:
        st.session_state.focus_policy_id = None
        return None

    valid_ids = candidate_points["policy_id"].astype(int).tolist()
    current_id = st.session_state.get("focus_policy_id")
    if current_id not in valid_ids:
        st.session_state.focus_policy_id = valid_ids[0]
    return st.session_state.focus_policy_id


def update_interp_selection(interpolation: pd.DataFrame) -> int | None:
    if interpolation.empty:
        st.session_state.curve_interp_id = None
        return None

    valid_ids = interpolation["interp_id"].astype(int).tolist()
    current_id = st.session_state.get("curve_interp_id")
    if current_id not in valid_ids:
        st.session_state.curve_interp_id = valid_ids[0]
    return st.session_state.curve_interp_id


def render_chart_guide(title: str, purpose: str, items: list[tuple[str, str]]) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(purpose)
        st.dataframe(
            pd.DataFrame([{"표식": label, "의미": meaning} for label, meaning in items]),
            use_container_width=True,
            hide_index=True,
        )


def front_summary_table(prev_archive: pd.DataFrame, predicted_front: pd.DataFrame, current_archive: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not prev_archive.empty:
        points = prev_archive[["eval_1", "eval_2"]].to_numpy(dtype=np.float64)
        rows.append(
            {
                "Front": "Previous archive front",
                "Meaning": "진화 전 기준 front",
                "Points": len(points),
                "Hypervolume": round(float(hypervolume(points)), 2),
                "Sparsity": round(float(sparsity(points)), 2),
            }
        )
    if not predicted_front.empty:
        points = predicted_front[["eval_1", "eval_2"]].to_numpy(dtype=np.float64)
        rows.append(
            {
                "Front": "Predicted virtual front",
                "Meaning": "선택된 predicted offspring을 넣었을 때",
                "Points": len(points),
                "Hypervolume": round(float(hypervolume(points)), 2),
                "Sparsity": round(float(sparsity(points)), 2),
            }
        )
    if not current_archive.empty:
        points = current_archive[["eval_1", "eval_2"]].to_numpy(dtype=np.float64)
        rows.append(
            {
                "Front": "Updated archive front",
                "Meaning": "실제 offspring 학습 후 현재 front",
                "Points": len(points),
                "Hypervolume": round(float(hypervolume(points)), 2),
                "Sparsity": round(float(sparsity(points)), 2),
            }
        )
    return pd.DataFrame(rows)


def padded_domain(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0.0, 1.0]
    minimum = float(arr.min())
    maximum = float(arr.max())
    span = max(maximum - minimum, 1.0)
    pad = 0.08 * span
    return [minimum - pad, maximum + pad]


def compute_performance_domains(
    true_front: pd.DataFrame,
    population_pg: pd.DataFrame,
    archive_pg: pd.DataFrame,
    tasks_pg: pd.DataFrame,
) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []

    for frame, cols in [
        (true_front, ["eval_1", "eval_2"]),
        (population_pg, ["eval_1", "eval_2"]),
        (archive_pg, ["eval_1", "eval_2"]),
        (tasks_pg, ["parent_eval_1", "parent_eval_2"]),
        (tasks_pg, ["pred_eval_1", "pred_eval_2"]),
        (tasks_pg, ["actual_eval_1", "actual_eval_2"]),
    ]:
        if frame.empty:
            continue
        if cols[0] in frame.columns:
            x_values.extend(frame[cols[0]].dropna().astype(float).tolist())
        if cols[1] in frame.columns:
            y_values.extend(frame[cols[1]].dropna().astype(float).tolist())

    return padded_domain(x_values), padded_domain(y_values)


def perf_x(field: str, x_domain: list[float], title: str = "Performance Space: f1"):
    return alt.X(f"{field}:Q", title=title, scale=alt.Scale(domain=x_domain, nice=False))


def perf_y(field: str, y_domain: list[float], title: str = "f2"):
    return alt.Y(f"{field}:Q", title=title, scale=alt.Scale(domain=y_domain, nice=False))


def make_warmup_chart(
    initial_points: pd.DataFrame,
    warmup_population: pd.DataFrame,
    current_front: pd.DataFrame,
    oracle_front: pd.DataFrame,
    show_oracle: bool,
    x_domain: list[float],
    y_domain: list[float],
) -> alt.Chart:
    selector = alt.selection_point(name="policy_focus", fields=["policy_id"], on="click", clear=False)

    move_segments = warmup_population.rename(
        columns={
            "eval_1": "x2",
            "eval_2": "y2",
        }
    ).merge(
        initial_points.rename(columns={"eval_1": "x", "eval_2": "y"}),
        on="policy_id",
        how="left",
        suffixes=("", "_init"),
    )

    layers: list[alt.Chart] = []

    if show_oracle and not oracle_front.empty:
        oracle_layer = (
            alt.Chart(oracle_front)
            .mark_line(strokeDash=[8, 5], strokeWidth=1.8, color="#9aa1a9")
            .encode(x=perf_x("eval_1", x_domain), y=perf_y("eval_2", y_domain))
        )
        layers.append(oracle_layer)

    initial_layer = (
        alt.Chart(initial_points)
        .mark_circle(size=70, color="#adb5bd", opacity=0.85)
        .encode(
            x=perf_x("eval_1", x_domain),
            y=perf_y("eval_2", y_domain),
            tooltip=["policy_id", "mu", "eval_1", "eval_2"],
        )
    )

    move_layer = (
        alt.Chart(move_segments)
        .mark_rule(color="#adb5bd", strokeWidth=2)
        .encode(x=perf_x("x", x_domain), y=perf_y("y", y_domain), x2="x2:Q", y2="y2:Q")
    )

    warmup_layer = (
        alt.Chart(warmup_population)
        .mark_circle(size=150, filled=True, stroke="#111111", strokeWidth=1.8)
        .encode(
            x=perf_x("eval_1", x_domain),
            y=perf_y("eval_2", y_domain),
            color=alt.condition(selector, alt.value("#e76f51"), alt.value("#2a9d8f")),
            tooltip=["policy_id", "mu", "eval_1", "eval_2", "weight_1", "weight_2"],
        )
        .add_params(selector)
    )

    current_front_layer = (
        alt.Chart(current_front)
        .mark_line(strokeWidth=3.8, color="#111111")
        .encode(x=perf_x("eval_1", x_domain), y=perf_y("eval_2", y_domain))
    )

    layers.extend([move_layer, initial_layer, current_front_layer, warmup_layer])

    return alt.layer(*layers).properties(
        height=470,
        title="Step 1. Warm-up Performance Space",
    ).interactive()


def make_front_overview_chart(
    parent_population: pd.DataFrame,
    prev_archive_snapshot: pd.DataFrame,
    current_archive_snapshot_df: pd.DataFrame,
    visible_generation: int,
    oracle_front: pd.DataFrame,
    show_oracle: bool,
    x_domain: list[float],
    y_domain: list[float],
) -> alt.Chart:
    selector = alt.selection_point(name="policy_focus", fields=["policy_id"], on="click", clear=False)

    prev_front = snapshot_front_df(prev_archive_snapshot, "Previous archive front")
    current_front = snapshot_front_df(current_archive_snapshot_df, "Updated archive front")

    layers: list[alt.Chart] = []

    if show_oracle and not oracle_front.empty:
        oracle_layer = (
            alt.Chart(oracle_front)
            .mark_line(strokeDash=[8, 5], strokeWidth=1.8, color="#9aa1a9")
            .encode(x=perf_x("eval_1", x_domain), y=perf_y("eval_2", y_domain))
        )
        layers.append(oracle_layer)

    prev_front_layer = (
        alt.Chart(prev_front)
        .mark_line(strokeWidth=3.8, color="#111111")
        .encode(x=perf_x("eval_1", x_domain), y=perf_y("eval_2", y_domain))
    )

    current_front_layer = (
        alt.Chart(current_front)
        .mark_line(strokeWidth=4.2, color="#2a9d8f")
        .encode(x=perf_x("eval_1", x_domain), y=perf_y("eval_2", y_domain))
    )

    parent_points = (
        alt.Chart(parent_population)
        .mark_circle(size=150, filled=True, stroke="#0d1b2a", strokeWidth=1.8)
        .encode(
            x=perf_x("eval_1", x_domain),
            y=perf_y("eval_2", y_domain),
            color=alt.condition(selector, alt.value("#e63946"), alt.value("#457b9d")),
            tooltip=["policy_id", "mu", "eval_1", "eval_2", "weight_1", "weight_2"],
        )
        .add_params(selector)
    )

    prev_archive_points = (
        alt.Chart(prev_archive_snapshot)
        .mark_point(size=85, filled=True, color="#111111")
        .encode(
            x=perf_x("eval_1", x_domain),
            y=perf_y("eval_2", y_domain),
            tooltip=["policy_id", "mu", "eval_1", "eval_2"],
        )
    )

    current_archive_points = (
        alt.Chart(current_archive_snapshot_df)
        .mark_square(size=95, filled=True, color="#2a9d8f")
        .encode(
            x=perf_x("eval_1", x_domain),
            y=perf_y("eval_2", y_domain),
            tooltip=["policy_id", "mu", "eval_1", "eval_2"],
        )
    )

    layers.extend(
        [
            prev_front_layer,
            current_front_layer,
            prev_archive_points,
            current_archive_points,
            parent_points,
        ]
    )

    return (
        alt.layer(*layers)
        .properties(
            height=300,
            title=f"Performance Space A. 기준 front와 업데이트된 front 비교 (generation {visible_generation})",
        )
    )


def make_prediction_space_chart(
    parent_population: pd.DataFrame,
    predictor_rows: pd.DataFrame,
    task_rows: pd.DataFrame,
    focus_policy_id: int,
    visible_generation: int,
    x_domain: list[float],
    y_domain: list[float],
) -> alt.Chart | None:
    focus_parent = parent_population[parent_population["policy_id"] == focus_policy_id].copy()
    focus_predictions = predictor_rows[predictor_rows["policy_id"] == focus_policy_id].copy()
    focus_selected = task_rows[task_rows["parent_policy_id"] == focus_policy_id].copy()
    if focus_parent.empty or focus_predictions.empty:
        return None

    parent_point = (
        alt.Chart(focus_parent)
        .mark_circle(size=170, filled=True, color="#e63946", stroke="#111111", strokeWidth=1.8)
        .encode(
            x=perf_x("eval_1", x_domain),
            y=perf_y("eval_2", y_domain),
            tooltip=["policy_id", "mu", "eval_1", "eval_2"],
        )
    )

    candidate_lines = (
        alt.Chart(focus_predictions)
        .mark_rule(color="#c0c7d1", strokeWidth=1.4, opacity=0.38)
        .encode(
            x=perf_x("parent_eval_1", x_domain),
            y=perf_y("parent_eval_2", y_domain),
            x2="pred_eval_1:Q",
            y2="pred_eval_2:Q",
            tooltip=["weight_1", "weight_2", "pred_eval_1", "pred_eval_2", "initial_score"],
        )
    )

    candidate_points = (
        alt.Chart(focus_predictions)
        .mark_point(size=70, filled=True, color="#adb5bd", opacity=0.7)
        .encode(
            x=perf_x("pred_eval_1", x_domain),
            y=perf_y("pred_eval_2", y_domain),
            tooltip=["weight_1", "weight_2", "pred_eval_1", "pred_eval_2", "initial_score"],
        )
    )

    layers: list[alt.Chart] = [candidate_lines, candidate_points, parent_point]

    if not focus_selected.empty:
        selected_lines = (
            alt.Chart(focus_selected.dropna(subset=["pred_eval_1", "pred_eval_2"]))
            .mark_rule(color="#f4a261", strokeWidth=3.2, opacity=0.95)
            .encode(
                x=perf_x("parent_eval_1", x_domain),
                y=perf_y("parent_eval_2", y_domain),
                x2="pred_eval_1:Q",
                y2="pred_eval_2:Q",
                tooltip=["slot", "weight_1", "weight_2", "score"],
            )
        )
        selected_points = (
            alt.Chart(focus_selected.dropna(subset=["pred_eval_1", "pred_eval_2"]))
            .mark_point(shape="diamond", size=150, filled=True, color="#f4a261")
            .encode(
                x=perf_x("pred_eval_1", x_domain),
                y=perf_y("pred_eval_2", y_domain),
                tooltip=["slot", "weight_1", "weight_2", "score"],
            )
        )
        layers.extend([selected_lines, selected_points])

    return alt.layer(*layers).properties(
        height=320,
        title=f"Performance Space B. Focus policy #{focus_policy_id}의 predictor fan-out (generation {visible_generation})",
    )


def make_update_space_chart(
    prev_archive_snapshot: pd.DataFrame,
    current_archive_snapshot_df: pd.DataFrame,
    task_rows: pd.DataFrame,
    visible_generation: int,
    x_domain: list[float],
    y_domain: list[float],
) -> alt.Chart | None:
    if task_rows.empty:
        return None

    prev_front = snapshot_front_df(prev_archive_snapshot, "Previous archive front")
    current_front = snapshot_front_df(current_archive_snapshot_df, "Updated archive front")
    predicted_front = predicted_virtual_front_df(prev_archive_snapshot, task_rows)

    prev_front_layer = (
        alt.Chart(prev_front)
        .mark_line(strokeWidth=3.0, color="#111111")
        .encode(x=perf_x("eval_1", x_domain), y=perf_y("eval_2", y_domain))
    )

    predicted_front_layer = (
        alt.Chart(predicted_front)
        .mark_line(strokeWidth=3.0, strokeDash=[10, 6], color="#f4a261")
        .encode(x=perf_x("eval_1", x_domain), y=perf_y("eval_2", y_domain))
    )

    current_front_layer = (
        alt.Chart(current_front)
        .mark_line(strokeWidth=3.6, color="#2a9d8f")
        .encode(x=perf_x("eval_1", x_domain), y=perf_y("eval_2", y_domain))
    )

    predicted_lines = (
        alt.Chart(task_rows.dropna(subset=["pred_eval_1", "pred_eval_2"]))
        .mark_rule(color="#f4a261", strokeWidth=3.0, opacity=0.95)
        .encode(
            x=perf_x("parent_eval_1", x_domain),
            y=perf_y("parent_eval_2", y_domain),
            x2="pred_eval_1:Q",
            y2="pred_eval_2:Q",
            tooltip=["slot", "weight_1", "weight_2", "score"],
        )
    )

    actual_lines = (
        alt.Chart(task_rows)
        .mark_rule(color="#2a9d8f", strokeWidth=3.2, opacity=0.95)
        .encode(
            x=perf_x("parent_eval_1", x_domain),
            y=perf_y("parent_eval_2", y_domain),
            x2="actual_eval_1:Q",
            y2="actual_eval_2:Q",
            tooltip=["slot", "actual_eval_1", "actual_eval_2"],
        )
    )

    parent_points = (
        alt.Chart(task_rows)
        .mark_circle(size=110, filled=True, color="#457b9d")
        .encode(
            x=perf_x("parent_eval_1", x_domain),
            y=perf_y("parent_eval_2", y_domain),
            tooltip=["slot", "parent_policy_id", "parent_eval_1", "parent_eval_2"],
        )
    )

    predicted_points = (
        alt.Chart(task_rows.dropna(subset=["pred_eval_1", "pred_eval_2"]))
        .mark_point(shape="diamond", size=135, filled=True, color="#f4a261")
        .encode(
            x=perf_x("pred_eval_1", x_domain),
            y=perf_y("pred_eval_2", y_domain),
            tooltip=["slot", "pred_eval_1", "pred_eval_2", "score"],
        )
    )

    actual_points = (
        alt.Chart(task_rows)
        .mark_point(shape="triangle-up", size=140, filled=True, color="#2a9d8f")
        .encode(
            x=perf_x("actual_eval_1", x_domain),
            y=perf_y("actual_eval_2", y_domain),
            tooltip=["slot", "actual_eval_1", "actual_eval_2"],
        )
    )

    return alt.layer(
        prev_front_layer,
        predicted_front_layer,
        current_front_layer,
        predicted_lines,
        actual_lines,
        parent_points,
        predicted_points,
        actual_points,
    ).properties(
        height=320,
        title=f"Performance Space C. 선택된 task의 예측 결과와 실제 offspring (generation {visible_generation})",
    )


def make_predictor_chart(
    predictor_rows: pd.DataFrame,
    selected_task_rows: pd.DataFrame,
    focus_policy_id: int,
    predictor_generation: int,
) -> alt.Chart | None:
    focus_rows = predictor_rows[predictor_rows["policy_id"] == focus_policy_id].copy()
    if focus_rows.empty:
        return None

    curve_df = focus_rows.melt(
        id_vars=["weight_1"],
        value_vars=["pred_delta_1", "pred_delta_2"],
        var_name="objective",
        value_name="delta",
    )
    curve_df["objective"] = curve_df["objective"].map(
        {
            "pred_delta_1": "Δf1(w)",
            "pred_delta_2": "Δf2(w)",
        }
    )

    markers = selected_task_rows[selected_task_rows["parent_policy_id"] == focus_policy_id].copy()
    marker_df = markers.melt(
        id_vars=["weight_1"],
        value_vars=["pred_delta_1", "pred_delta_2"],
        var_name="objective",
        value_name="delta",
    )
    marker_df["objective"] = marker_df["objective"].map(
        {
            "pred_delta_1": "Δf1(w)",
            "pred_delta_2": "Δf2(w)",
        }
    )

    lines = (
        alt.Chart(curve_df)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("weight_1:Q", title="weight ω1"),
            y=alt.Y("delta:Q", title="Predicted improvement ΔF"),
            color=alt.Color("objective:N", title=None),
            tooltip=["weight_1", "objective", "delta"],
        )
    )

    points = (
        alt.Chart(curve_df)
        .mark_point(size=55, filled=True)
        .encode(
            x="weight_1:Q",
            y="delta:Q",
            color="objective:N",
        )
    )

    layers = [lines, points]
    if not marker_df.empty:
        chosen_markers = (
            alt.Chart(marker_df)
            .mark_point(shape="diamond", size=180, filled=True, color="#e76f51")
            .encode(
                x="weight_1:Q",
                y="delta:Q",
                tooltip=["weight_1", "objective", "delta"],
            )
        )
        layers.append(chosen_markers)

    return alt.layer(*layers).properties(
        height=280,
        title=f"Step 2. Predictor Curve: policy #{focus_policy_id}의 weight별 예상 개선량 (generation {predictor_generation})",
    )


def make_task_detail_charts(task_rows: pd.DataFrame, focus_policy_id: int, inner_steps: pd.DataFrame) -> tuple[alt.Chart | None, alt.Chart | None]:
    focus_tasks = task_rows[task_rows["parent_policy_id"] == focus_policy_id].copy()
    if focus_tasks.empty:
        return None, None

    top_task = focus_tasks.sort_values("score", ascending=False).iloc[0]
    delta_df = pd.DataFrame(
        [
            {"objective": "f1", "kind": "Predicted Δ", "delta": top_task["pred_delta_1"]},
            {"objective": "f1", "kind": "Actual Δ", "delta": top_task["actual_delta_1"]},
            {"objective": "f2", "kind": "Predicted Δ", "delta": top_task["pred_delta_2"]},
            {"objective": "f2", "kind": "Actual Δ", "delta": top_task["actual_delta_2"]},
        ]
    )
    delta_chart = (
        alt.Chart(delta_df)
        .mark_bar()
        .encode(
            x=alt.X("objective:N", title=None),
            y=alt.Y("delta:Q", title="Δ value"),
            color=alt.Color("kind:N", title=None),
            xOffset="kind:N",
            tooltip=["objective", "kind", "delta"],
        )
        .properties(height=250, title=f"선택된 best task slot {int(top_task['slot'])}: 예측 개선량 vs 실제 개선량")
    )

    if inner_steps.empty:
        return delta_chart, None

    trace_df = inner_steps.melt(
        id_vars=["inner_step"],
        value_vars=["eval_1", "eval_2"],
        var_name="objective",
        value_name="value",
    )
    trace_df["objective"] = trace_df["objective"].map({"eval_1": "f1", "eval_2": "f2"})

    trace_chart = (
        alt.Chart(trace_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("inner_step:Q", title="MOPG inner step"),
            y=alt.Y("value:Q", title="Objective value"),
            color=alt.Color("objective:N", title=None),
            tooltip=["inner_step", "objective", "value"],
        )
        .properties(height=250, title="선택된 task의 실제 학습 trajectory")
    )
    return delta_chart, trace_chart


def make_parameter_space_chart(family_points: pd.DataFrame) -> alt.Chart:
    selector = alt.selection_point(name="param_pick", fields=["policy_id"], on="click", clear=False)
    return (
        alt.Chart(family_points)
        .mark_circle(size=140, filled=True, stroke="#102a43", strokeWidth=1.2)
        .encode(
            x=alt.X("param_x:Q", title="Parameter Space t-SNE 1"),
            y=alt.Y("param_y:Q", title="t-SNE 2"),
            color=alt.Color("family:N", title="Family"),
            tooltip=["policy_id", "mu", "family", "eval_1", "eval_2"],
            opacity=alt.condition(selector, alt.value(1.0), alt.value(0.85)),
        )
        .add_params(selector)
        .properties(height=330, title="Step 4. Parameter Space: t-SNE + k-means family 구조")
        .interactive()
    )


def make_continuous_front_chart(interpolation: pd.DataFrame, family_points: pd.DataFrame) -> alt.Chart:
    selector = alt.selection_point(name="curve_pick", fields=["interp_id"], on="click", clear=False)

    family_lines = (
        alt.Chart(interpolation)
        .mark_line(strokeWidth=4)
        .encode(
            x=alt.X("eval_1:Q", title="Performance Space f1"),
            y=alt.Y("eval_2:Q", title="f2"),
            color=alt.Color("family:N", title="Family"),
        )
    )

    selectable_points = (
        alt.Chart(interpolation)
        .mark_point(size=80, filled=True)
        .encode(
            x="eval_1:Q",
            y="eval_2:Q",
            color=alt.Color("family:N", legend=None),
            opacity=alt.condition(selector, alt.value(1.0), alt.value(0.22)),
            tooltip=["family", "mu", "eval_1", "eval_2", "alpha"],
        )
        .add_params(selector)
    )

    anchor_points = (
        alt.Chart(family_points)
        .mark_point(shape="diamond", size=120, filled=True, color="#111111")
        .encode(
            x="eval_1:Q",
            y="eval_2:Q",
            tooltip=["policy_id", "family", "mu", "eval_1", "eval_2"],
        )
    )

    return (
        alt.layer(family_lines, selectable_points, anchor_points)
        .properties(height=330, title="Step 4. Continuous Pareto Front with interpolation")
        .interactive()
    )


st.set_page_config(page_title="PGMORL Toy Example", layout="wide")

st.title("PGMORL Toy Example")
st.caption("논문 흐름을 4-step으로 따라가며, 각 차트 아래에서 그 그래프가 무엇을 보여주는지와 표식 의미를 바로 확인할 수 있도록 구성한 dashboard입니다.")

config = ToyConfig()
init_walkthrough_state(config)

with st.spinner("Toy experiment를 준비하는 중입니다..."):
    result = load_result(config)

metrics_pg = result.metrics[result.metrics["method"] == PGMORL_METHOD].copy().reset_index(drop=True)
population_pg = result.population[result.population["method"] == PGMORL_METHOD].copy().reset_index(drop=True)
archive_pg = result.archive[result.archive["method"] == PGMORL_METHOD].copy().reset_index(drop=True)
tasks_pg = result.tasks[result.tasks["method"] == PGMORL_METHOD].copy().reset_index(drop=True)
predictor_pg = result.predictor_curves[result.predictor_curves["method"] == PGMORL_METHOD].copy().reset_index(drop=True)
inner_pg = result.inner_steps[result.inner_steps["method"] == PGMORL_METHOD].copy().reset_index(drop=True)
x_domain, y_domain = compute_performance_domains(result.true_front, population_pg, archive_pg, tasks_pg)

control_col, visual_col = st.columns([0.95, 1.8], gap="large")

with control_col:
    st.subheader("컨트롤 패널")

    with st.container(border=True):
        st.markdown("**알고리즘 설정**")
        st.caption("이 toy는 학습 결과를 미리 계산해 둔 상태에서 step-by-step으로 보여줍니다.")
        st.write(
            {
                "population_size": config.population_size,
                "warmup_steps": config.warmup_steps,
                "task_steps": config.task_steps,
                "generations": config.generations,
                "weight_resolution": config.weight_resolution,
                "optimizer": config.gradient_mode,
            }
        )

    with st.container(border=True):
        st.markdown("**Step 1. Warm-up**")
        st.caption("무작위 초기 정책을 여러 preference 방향으로 짧게 업데이트해, 첫 population과 첫 Pareto archive front를 형성합니다.")
        warmup_clicked = st.button("Warm-up 시작", use_container_width=True)
        if warmup_clicked:
            st.session_state.warmup_done = True
            st.session_state.visible_generation = 0
            st.session_state.analysis_done = False
            st.session_state.curve_interp_id = None
            st.session_state.status_message = "첫 번째 세대의 정책 인구가 형성되었으며, 외부 파레토 아카이브가 초기화되었습니다."

        st.markdown("**Step 2. Improvement Prediction**")
        st.caption("정책 점 하나를 클릭하면, 그 policy를 여러 weight로 더 학습했을 때 어디로 이동할지와 objective 개선량 ΔF를 predictor로 보여줍니다.")

        st.markdown("**Step 3. Task Selection + Evolution**")
        st.caption("예측 결과를 바탕으로 hypervolume을 가장 잘 넓힐 policy-weight 쌍을 고르고, 실제 MOPG 업데이트 후 offspring과 새 front를 확인합니다.")
        evolve_clicked = st.button(
            "작업 선택 및 진화 (1 Generation)",
            use_container_width=True,
            disabled=not st.session_state.warmup_done or st.session_state.visible_generation >= config.generations,
        )
        if evolve_clicked:
            st.session_state.visible_generation += 1
            generation = st.session_state.visible_generation
            st.session_state.status_message = (
                f"Generation {generation}에서 hypervolume - λ·sparsity 기준으로 상위 task를 선택하고 offspring을 생성했습니다."
            )

        st.markdown("**Step 4. Pareto Analysis**")
        st.caption("최종 archive를 parameter-space family로 묶고, 같은 family 내부를 보간해 continuous Pareto front를 시각화합니다.")
        analysis_clicked = st.button(
            "파레토 분석 및 보간 (Interpolation)",
            use_container_width=True,
            disabled=not st.session_state.warmup_done or st.session_state.visible_generation < 1,
        )
        if analysis_clicked:
            st.session_state.analysis_done = True
            st.session_state.status_message = (
                "t-SNE와 k-means로 policy family를 시각화하고, family 내부 선형 보간으로 continuous Pareto front를 구성했습니다."
            )

        if st.button("Walkthrough 초기화", use_container_width=True):
            reset_walkthrough(config)

    with st.container(border=True):
        st.markdown("**현재 상태**")
        st.write(
            {
                "warmup_done": st.session_state.warmup_done,
                "visible_generation": st.session_state.visible_generation,
                "analysis_done": st.session_state.analysis_done,
            }
        )
        st.info(st.session_state.status_message)
        st.caption(
            "`weight`는 action을 직접 바꾸는 값이 아니라, 현재 policy를 어떤 objective trade-off 방향으로 업데이트할지 정하는 preference입니다."
        )
        st.caption(
            "이 toy의 `action a`는 literal speed가 아니라 generic scalar control input이고, `policy μ`는 그 action distribution의 평균입니다."
        )

with visual_col:
    st.subheader("시각화 패널")
    with st.container(border=True):
        st.markdown("**대시보드 읽는 순서**")
        st.caption("각 그래프 아래에는 그 차트만을 위한 설명과 legend가 따로 붙습니다. 먼저 A에서 front의 전체 변화를 보고, B에서 한 policy의 예측을 확인한 뒤, C에서 실제 선택 결과가 front를 어떻게 바꾸는지 읽으면 됩니다.")
        st.write("1. `Warm-up`: 초기 정책이 첫 archive front를 만드는 과정")
        st.write("2. `Performance Space A`: 세대 시작 전 front와 업데이트 후 front 비교")
        st.write("3. `Performance Space B` + `Predictor Curve`: 한 policy의 예측 fan-out과 weight별 예상 개선량")
        st.write("4. `Performance Space C`: 선택된 task의 predicted virtual front와 actual offspring 비교")
        st.write("5. `Step 4 분석`: parameter-space family와 continuous Pareto front 해석")
    show_oracle = st.checkbox(
        "이론적 최적 Pareto front 참고선 보기",
        value=False,
        help="회색 점선은 이 toy 환경에서만 알 수 있는 oracle reference입니다. 논문 실전 상황에서는 보통 알 수 없는 '정답선'입니다.",
    )

    if not st.session_state.warmup_done:
        initial_points = initial_policy_rows(tasks_pg)
        warmup_population = population_pg[population_pg["snapshot_generation"] == 0].drop_duplicates(subset=["policy_id"])
        warmup_archive_snapshot = current_archive_snapshot(archive_pg, 0)
        warmup_front = snapshot_front_df(warmup_archive_snapshot, "Current archive front")
        focus_points = warmup_population.copy()
        focus_id = update_focus_policy(focus_points)

        gauge_row = st.columns(3)
        with gauge_row[0]:
            with st.container(border=True):
                st.caption("Hypervolume Gauge")
                st.metric("Current", "0.00", "Warm-up 전")
                st.progress(0)
        with gauge_row[1]:
            with st.container(border=True):
                st.caption("Sparsity Gauge")
                st.metric("Current", "0.00", "Warm-up 전")
                st.progress(0)
        with gauge_row[2]:
            with st.container(border=True):
                st.caption("조작 가이드")
                st.write("1. 좌측에서 `Warm-up 시작`")
                st.write("2. 우측 Performance Space에서 정책 클릭")
                st.write("3. `작업 선택 및 진화` 반복")

        warmup_chart = make_warmup_chart(
            initial_points,
            warmup_population,
            warmup_front,
            result.true_front,
            show_oracle,
            x_domain,
            y_domain,
        )
        warmup_event = st.altair_chart(
            warmup_chart,
            use_container_width=True,
            on_select="rerun",
            selection_mode=["policy_focus"],
            key="warmup_performance_chart",
        )
        selected_policy = extract_selection_value(warmup_event, "policy_focus", "policy_id")
        if selected_policy is not None:
            st.session_state.focus_policy_id = int(selected_policy)

        render_chart_guide(
            "Warm-up 차트 Legend + 설명",
            "이 그래프는 초기 랜덤 정책이 Warm-up을 거친 뒤 첫 Pareto archive를 어떻게 만드는지 보여줍니다. 이 단계에서는 검은 실선이 현재 알고리즘이 실제로 가진 유일한 기준 front입니다.",
            [
                ("회색 점", "Warm-up 전 랜덤 초기 정책"),
                ("회색 선", "각 초기 정책이 Warm-up 후 이동한 경로"),
                ("청록 점", "Warm-up 후 첫 세대 population"),
                ("검은 실선", "Warm-up 직후의 현재 Pareto archive front"),
                ("회색 점선", "oracle reference front. 체크박스를 켰을 때만 보이는 이론적 최적 front"),
            ] if show_oracle else [
                ("회색 점", "Warm-up 전 랜덤 초기 정책"),
                ("회색 선", "각 초기 정책이 Warm-up 후 이동한 경로"),
                ("청록 점", "Warm-up 후 첫 세대 population"),
                ("검은 실선", "Warm-up 직후의 현재 Pareto archive front"),
            ],
        )
        st.caption("이 단계에서의 기준 front는 `Warm-up이 끝난 직후 archive` 입니다. 회색 점선 oracle은 비교 참고선일 뿐이고, 실제 학습 기준은 항상 검은 실선입니다.")
        st.dataframe(
            front_summary_table(pd.DataFrame(), pd.DataFrame(), warmup_archive_snapshot),
            use_container_width=True,
            hide_index=True,
        )

    else:
        visible_generation = st.session_state.visible_generation
        current_metrics = metrics_pg[metrics_pg["generation"] == visible_generation].iloc[0]
        previous_metrics = metrics_pg[metrics_pg["generation"] == max(0, visible_generation - 1)].iloc[0]
        hv_delta = float(current_metrics["hypervolume"] - previous_metrics["hypervolume"]) if visible_generation > 0 else float(current_metrics["hypervolume"])
        sparsity_delta = float(previous_metrics["sparsity"] - current_metrics["sparsity"]) if visible_generation > 0 else 0.0

        gauge_row = st.columns(3)
        with gauge_row[0]:
            render_gauge(
                "Hypervolume Gauge",
                float(current_metrics["hypervolume"]),
                hv_delta,
                gauge_progress(
                    float(current_metrics["hypervolume"]),
                    float(metrics_pg["hypervolume"].min()),
                    float(metrics_pg["hypervolume"].max()),
                ),
                "면적이 커질수록 Pareto front가 더 넓고 좋습니다.",
            )
        with gauge_row[1]:
            render_gauge(
                "Sparsity Gauge",
                float(current_metrics["sparsity"]),
                sparsity_delta,
                gauge_progress(
                    float(current_metrics["sparsity"]),
                    float(metrics_pg["sparsity"].min()),
                    float(metrics_pg["sparsity"].max()),
                    invert=True,
                ),
                "값이 낮을수록 Pareto 점들이 더 촘촘합니다.",
            )
        with gauge_row[2]:
            with st.container(border=True):
                st.caption("현재 walkthrough")
                st.metric("Generation", str(visible_generation), f"/ {config.generations}")
                st.metric("Archive Size", str(int(current_metrics["archive_size"])), f"+{int(current_metrics['archive_size'] - previous_metrics['archive_size']) if visible_generation > 0 else int(current_metrics['archive_size'])}")

        parent_population = parent_population_snapshot(population_pg, visible_generation)
        prev_archive = previous_archive_snapshot(archive_pg, visible_generation)
        archive_snapshot = current_archive_snapshot(archive_pg, visible_generation)
        task_rows = current_task_rows(tasks_pg, visible_generation)
        predictor_rows = current_predictor_rows(predictor_pg, visible_generation)
        focus_id = update_focus_policy(parent_population)

        family_points = pd.DataFrame()
        interpolation = pd.DataFrame()
        if st.session_state.analysis_done:
            family_points, interpolation = build_family_analysis(archive_snapshot, config)
            update_interp_selection(interpolation)

        if visible_generation == 0:
            initial_points = initial_policy_rows(tasks_pg)
            warmup_front = snapshot_front_df(archive_snapshot, "Current archive front")
            performance_chart = make_warmup_chart(
                initial_points,
                parent_population,
                warmup_front,
                result.true_front,
                show_oracle,
                x_domain,
                y_domain,
            )
            performance_event = st.altair_chart(
                performance_chart,
                use_container_width=True,
                on_select="rerun",
                selection_mode=["policy_focus"],
                key="performance_space_chart",
            )
        else:
            performance_chart = make_front_overview_chart(
                parent_population,
                prev_archive,
                archive_snapshot,
                visible_generation,
                result.true_front,
                show_oracle,
                x_domain,
                y_domain,
            )
            performance_event = st.altair_chart(
                performance_chart,
                use_container_width=True,
                on_select="rerun",
                selection_mode=["policy_focus"],
                key="performance_space_chart",
            )

        selected_policy = extract_selection_value(performance_event, "policy_focus", "policy_id")
        if selected_policy is not None:
            st.session_state.focus_policy_id = int(selected_policy)
            focus_id = int(selected_policy)

        render_chart_guide(
            "Performance Space A Legend + 설명",
            "이 overview 차트는 '현재 세대에 진입하기 전의 기준 front'와 '실제 offspring 학습 후의 업데이트된 front'만 비교합니다. 축 스케일은 모든 세대에서 고정됩니다.",
            [
                ("검은 실선", "이전 generation까지의 archive front. 이번 세대의 기준선"),
                ("초록 실선", "실제 offspring을 archive에 반영한 뒤의 현재 front"),
                ("검은 점", "이전 archive에 포함된 Pareto policies"),
                ("초록 사각형", "업데이트 후 archive에 포함된 Pareto policies"),
                ("파란 원", "이번 세대 task selection 전에 존재하던 population"),
                ("회색 점선", "oracle reference front. 체크박스를 켰을 때만 보이는 이론적 최적 front"),
            ] if show_oracle else [
                ("검은 실선", "이전 generation까지의 archive front. 이번 세대의 기준선"),
                ("초록 실선", "실제 offspring을 archive에 반영한 뒤의 현재 front"),
                ("검은 점", "이전 archive에 포함된 Pareto policies"),
                ("초록 사각형", "업데이트 후 archive에 포함된 Pareto policies"),
                ("파란 원", "이번 세대 task selection 전에 존재하던 population"),
            ],
        )
        if visible_generation > 0:
            predicted_front = predicted_virtual_front_df(prev_archive, task_rows)
            st.info(
                "이 overview는 front 비교에만 집중합니다. "
                "예측 fan-out과 predictor curve는 바로 아래의 `Performance Space B`, 실제 선택 결과와 front 갱신은 `Performance Space C`에서 따로 보여줍니다."
            )
            st.dataframe(
                front_summary_table(prev_archive, predicted_front, archive_snapshot),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Generation 0에서는 Warm-up 직후 archive만 존재합니다. 검은 실선이 현재 기준 Pareto front입니다.")
            st.dataframe(
                front_summary_table(pd.DataFrame(), pd.DataFrame(), archive_snapshot),
                use_container_width=True,
                hide_index=True,
            )

        predictor_generation = predictor_generation_for_view(visible_generation)
        predictor_chart = make_predictor_chart(
            predictor_pg[predictor_pg["generation"] == predictor_generation].copy(),
            tasks_pg[tasks_pg["generation"] == predictor_generation].copy(),
            focus_id,
            predictor_generation,
        )

        lower_left, lower_right = st.columns(2)
        with lower_left:
            if visible_generation > 0:
                prediction_space_chart = make_prediction_space_chart(
                    parent_population,
                    predictor_rows,
                    task_rows,
                    focus_id,
                    visible_generation,
                    x_domain,
                    y_domain,
                )
                if prediction_space_chart is not None:
                    st.altair_chart(prediction_space_chart, use_container_width=True)
                    render_chart_guide(
                        "Performance Space B Legend + 설명",
                        "이 그래프는 현재 클릭한 policy 하나만 고정하고, 그 policy를 여러 weight로 학습했을 때 predictor가 예상하는 이동 방향을 보여줍니다. 여기서는 아직 실제 학습은 일어나지 않았고, 오직 예측만 봅니다.",
                        [
                            ("빨간 점", "현재 focus policy의 위치"),
                            ("연한 회색 선", "가능한 모든 weight 후보에 대한 예측 궤적"),
                            ("연한 회색 점", "각 weight 후보의 predicted offspring 위치"),
                            ("주황 굵은 선", "실제로 선택된 weight의 predicted trajectory"),
                            ("주황 다이아", "선택된 predicted offspring 위치"),
                        ],
                    )
                else:
                    st.info("이 policy에 대한 predictor fan-out을 표시할 데이터가 아직 없습니다.")
            if predictor_chart is not None:
                st.altair_chart(predictor_chart, use_container_width=True)
                focus_parent = parent_population[parent_population["policy_id"] == focus_id]
                if not focus_parent.empty:
                    mu_value = float(focus_parent.iloc[0]["mu"])
                    render_chart_guide(
                        "Step 2 Predictor Curve Legend + 설명",
                        f"현재 focus policy #{focus_id}의 평균 action μ는 {mu_value:.2f}입니다. 이 그래프는 weight에 따라 predictor가 예상하는 objective improvement ΔF를 보여줍니다. Performance Space B가 '어디로 이동할지'를 보여준다면, 이 차트는 '각 objective가 얼마나 늘어날지'를 보여줍니다.",
                        [
                            ("파란/주황 곡선", "각 objective에 대한 predicted Δf(w)"),
                            ("작은 점", "weight 샘플 grid에서 계산된 predictor 값"),
                            ("마름모 마커", "실제로 선택된 weight에서의 predicted Δf"),
                        ],
                    )
            else:
                st.info("이 정책에 대한 predictor curve가 아직 없습니다. Warm-up 이후 첫 evolution generation에서 생성됩니다.")

        with lower_right:
            if visible_generation > 0:
                update_space_chart = make_update_space_chart(
                    prev_archive,
                    archive_snapshot,
                    task_rows,
                    visible_generation,
                    x_domain,
                    y_domain,
                )
                if update_space_chart is not None:
                    st.altair_chart(update_space_chart, use_container_width=True)
                    render_chart_guide(
                        "Performance Space C Legend + 설명",
                        "이 그래프는 실제로 선택된 task들만 모아서, 예측 기반 선택이 어떻게 virtual front를 만들고 실제 학습 후 front를 어떻게 바꾸는지 보여줍니다. 검은 실선에서 출발해 주황 점선은 '예상', 초록 실선은 '실제 결과'를 뜻합니다.",
                        [
                            ("검은 실선", "이전 archive front"),
                            ("주황 점선", "선택된 predicted offspring을 넣어 계산한 virtual front"),
                            ("초록 실선", "실제 offspring 학습 후 업데이트된 archive front"),
                            ("파란 원", "각 선택된 task의 parent policy 위치"),
                            ("주황 굵은 선 + 다이아", "선택된 task의 predicted trajectory와 predicted offspring"),
                            ("초록 굵은 선 + 삼각형", "실제 학습 후 trajectory와 actual offspring"),
                        ],
                    )
                task_inner = inner_pg[
                    (inner_pg["generation"] == visible_generation)
                    & (inner_pg["slot"].isin(task_rows[task_rows["parent_policy_id"] == focus_id]["slot"]))
                ].copy()
                detail_left_chart, detail_right_chart = make_task_detail_charts(task_rows, focus_id, task_inner)
                if detail_left_chart is not None:
                    st.altair_chart(detail_left_chart, use_container_width=True)
                    render_chart_guide(
                        "Predicted vs Actual Improvement Legend + 설명",
                        "선택된 best task 하나에 대해 predictor가 예상한 개선량과 실제 학습 후 개선량을 objective별로 비교합니다.",
                        [
                            ("Predicted Δ 막대", "하이퍼볼릭 predictor가 예측한 개선량"),
                            ("Actual Δ 막대", "실제 MOPG 후 측정된 개선량"),
                        ],
                    )
                if detail_right_chart is not None:
                    st.altair_chart(detail_right_chart, use_container_width=True)
                    render_chart_guide(
                        "Objective Trajectory Legend + 설명",
                        "선택된 best task를 실제로 학습하는 동안, 각 inner step에서 objective 값이 어떻게 변하는지 보여줍니다.",
                        [
                            ("f1 선", "첫 번째 objective의 학습 trajectory"),
                            ("f2 선", "두 번째 objective의 학습 trajectory"),
                        ],
                    )
            else:
                curve_df = reward_curve_df(config)
                focus_row = parent_population[parent_population["policy_id"] == focus_id].iloc[0]
                optimum_mu = closed_form_optimum_mu(
                    [focus_row["weight_1"], focus_row["weight_2"]],
                    config,
                )
                action_inspector = st.slider(
                    "Action inspector a",
                    min_value=float(config.init_mu_low),
                    max_value=float(config.init_mu_high),
                    value=float(focus_row["mu"]),
                    step=0.1,
                )
                reward_vector = sample_objectives(np.array([action_inspector]), config)[0]
                reward_chart = (
                    alt.Chart(curve_df)
                    .mark_line(strokeWidth=3)
                    .encode(
                        x=alt.X("action:Q", title="Action a"),
                        y=alt.Y("reward:Q", title="Reward"),
                        color=alt.Color("objective:N", title=None),
                    )
                    .properties(height=280, title="Action이 objective reward를 어떻게 바꾸는가")
                )
                marker_df = pd.DataFrame(
                    {
                        "action": [action_inspector, action_inspector],
                        "reward": [float(reward_vector[0]), float(reward_vector[1])],
                        "objective": objective_labels(config),
                    }
                )
                marker_chart = (
                    alt.Chart(marker_df)
                    .mark_point(size=120, filled=True)
                    .encode(x="action:Q", y="reward:Q", color="objective:N", tooltip=["objective", "reward"])
                )
                st.altair_chart(reward_chart + marker_chart, use_container_width=True)
                render_chart_guide(
                    "Action vs Reward Curve Legend + 설명",
                    f"Warm-up 직후 focus policy의 현재 μ={float(focus_row['mu']):.2f}, 이론적 optimum μ*(w)={optimum_mu:.2f} 입니다. 이 그래프는 action 자체가 reward를 어떻게 바꾸는지 보여줍니다.",
                    [
                        ("색깔별 곡선", "각 objective reward 함수"),
                        ("점 마커", "현재 inspector action a에서의 reward 값"),
                    ],
                )

        if st.session_state.analysis_done and not family_points.empty and not interpolation.empty:
            st.divider()
            analysis_left, analysis_right = st.columns(2)
            with analysis_left:
                st.altair_chart(make_parameter_space_chart(family_points), use_container_width=True)
                render_chart_guide(
                    "Parameter Space View Legend + 설명",
                    "실제 논문은 고차원 신경망 파라미터를 t-SNE로 내립니다. 이 toy는 scalar policy μ와 performance descriptor를 함께 사용해 family 구조를 시각화합니다.",
                    [
                        ("색깔별 점", "같은 family로 군집화된 Pareto policies"),
                        ("점 간 거리", "parameter-space 상의 상대적 유사성"),
                    ],
                )

            with analysis_right:
                curve_chart = make_continuous_front_chart(interpolation, family_points)
                curve_event = st.altair_chart(
                    curve_chart,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode=["curve_pick"],
                    key="continuous_curve_chart",
                )
                selected_interp = extract_selection_value(curve_event, "curve_pick", "interp_id")
                if selected_interp is not None:
                    st.session_state.curve_interp_id = int(selected_interp)

                interp_id = update_interp_selection(interpolation)
                selected_controller = interpolation[interpolation["interp_id"] == interp_id]
                render_chart_guide(
                    "Continuous Pareto Front Legend + 설명",
                    "같은 family 안에서 인접 policy를 선형 보간해, 점으로만 있던 Pareto set을 연속 곡선처럼 보여줍니다. 곡선 위 반투명 점을 클릭하면 그 타협점의 보간 controller를 읽을 수 있습니다.",
                    [
                        ("색깔 선", "family 내부 보간으로 얻은 continuous Pareto segment"),
                        ("검은 다이아", "실제로 archive에 있던 anchor Pareto policies"),
                        ("반투명 점", "클릭 가능한 보간 controller 후보"),
                    ],
                )
                if not selected_controller.empty:
                    row = selected_controller.iloc[0]
                    st.success(
                        f"현재 보간된 컨트롤러가 생성되었습니다. "
                        f"μ={row['mu']:.2f}, f1={row['eval_1']:.2f}, f2={row['eval_2']:.2f}, {row['family']}"
                    )

st.divider()
with st.expander("수식 / 의사코드 / 논문 대응 보기"):
    st.markdown(
        "- `policy πθ`: 평균 action `μ=θ`를 갖는 Gaussian policy\n"
        "- `action a`: policy가 실제로 샘플하는 scalar control input\n"
        "- `weight w`: action이 아니라, policy gradient의 선호도 벡터\n"
        "- `task = (policy, weight)`: 현재 policy를 특정 선호도로 더 학습시키는 작업"
    )
    st.latex(r"\pi_\theta(a) = \mathcal{N}(a; \mu=\theta, \sigma^2)")
    st.latex(r"r_i(a) = C - (a - t_i)^2")
    st.latex(r"J(\theta, w) = \sum_i w_i F_i(\theta)")
    st.code(
        """Warm-up:
  random policies -> evenly spread weights -> MOPG -> first population + archive

Prediction:
  click one policy -> fit monotonic hyperbolic model Δ_j^i(ω_j)
  read both predictor curve and predictor fan-out in performance space

Task Selection:
  select policy-weight pairs that maximize
      hypervolume(predicted archive) - λ * sparsity(predicted archive)
  compare predicted virtual front vs actual updated front

Pareto Analysis:
  cluster parameter-space points into families
  linearly interpolate within a family to get a continuous Pareto front
""",
        language="text",
    )
    st.dataframe(result.mapping, use_container_width=True, hide_index=True)
