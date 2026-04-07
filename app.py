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
    run_experiment,
    sample_objectives,
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


def make_warmup_chart(
    true_front: pd.DataFrame,
    initial_points: pd.DataFrame,
    warmup_population: pd.DataFrame,
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

    true_front_layer = (
        alt.Chart(true_front)
        .mark_line(strokeDash=[8, 5], strokeWidth=2.5, color="#6c757d")
        .encode(x=alt.X("eval_1:Q", title="Performance Space: f1"), y=alt.Y("eval_2:Q", title="f2"))
    )

    initial_layer = (
        alt.Chart(initial_points)
        .mark_circle(size=70, color="#adb5bd", opacity=0.85)
        .encode(
            x="eval_1:Q",
            y="eval_2:Q",
            tooltip=["policy_id", "mu", "eval_1", "eval_2"],
        )
    )

    move_layer = (
        alt.Chart(move_segments)
        .mark_rule(color="#adb5bd", strokeWidth=2)
        .encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    )

    warmup_layer = (
        alt.Chart(warmup_population)
        .mark_circle(size=150, filled=True, stroke="#111111", strokeWidth=1.8)
        .encode(
            x="eval_1:Q",
            y="eval_2:Q",
            color=alt.condition(selector, alt.value("#e76f51"), alt.value("#2a9d8f")),
            tooltip=["policy_id", "mu", "eval_1", "eval_2", "weight_1", "weight_2"],
        )
        .add_params(selector)
    )

    return (
        alt.layer(true_front_layer, move_layer, initial_layer, warmup_layer)
        .properties(height=470, title="Step 1. Warm-up: 초기 정책이 고르게 퍼지며 첫 세대를 형성")
        .interactive()
    )


def make_evolution_chart(
    true_front: pd.DataFrame,
    parent_population: pd.DataFrame,
    archive_snapshot: pd.DataFrame,
    predictor_rows: pd.DataFrame,
    task_rows: pd.DataFrame,
    analysis_interpolation: pd.DataFrame,
    analysis_done: bool,
    visible_generation: int,
) -> alt.Chart:
    selector = alt.selection_point(name="policy_focus", fields=["policy_id"], on="click", clear=False)

    true_front_layer = (
        alt.Chart(true_front)
        .mark_line(strokeDash=[8, 5], strokeWidth=2.0, color="#6c757d")
        .encode(x=alt.X("eval_1:Q", title="Performance Space: f1"), y=alt.Y("eval_2:Q", title="f2"))
    )

    candidate_lines = (
        alt.Chart(predictor_rows)
        .mark_rule(color="#c0c7d1", strokeWidth=1.2, opacity=0.22)
        .encode(
            x="parent_eval_1:Q",
            y="parent_eval_2:Q",
            x2="pred_eval_1:Q",
            y2="pred_eval_2:Q",
            tooltip=["policy_id", "weight_1", "pred_eval_1", "pred_eval_2", "initial_score"],
        )
    )

    selected_prediction_lines = (
        alt.Chart(task_rows.dropna(subset=["pred_eval_1", "pred_eval_2"]))
        .mark_rule(color="#f4a261", strokeWidth=3.0, opacity=0.95)
        .encode(
            x="parent_eval_1:Q",
            y="parent_eval_2:Q",
            x2="pred_eval_1:Q",
            y2="pred_eval_2:Q",
            tooltip=["slot", "weight_1", "weight_2", "score"],
        )
    )

    actual_lines = (
        alt.Chart(task_rows)
        .mark_rule(color="#2a9d8f", strokeWidth=3.0, opacity=0.92)
        .encode(
            x="parent_eval_1:Q",
            y="parent_eval_2:Q",
            x2="actual_eval_1:Q",
            y2="actual_eval_2:Q",
            tooltip=["slot", "actual_eval_1", "actual_eval_2"],
        )
    )

    parent_points = (
        alt.Chart(parent_population)
        .mark_circle(size=150, filled=True, stroke="#0d1b2a", strokeWidth=1.8)
        .encode(
            x="eval_1:Q",
            y="eval_2:Q",
            color=alt.condition(selector, alt.value("#e63946"), alt.value("#457b9d")),
            tooltip=["policy_id", "mu", "eval_1", "eval_2", "weight_1", "weight_2"],
        )
        .add_params(selector)
    )

    offspring_points = (
        alt.Chart(task_rows)
        .mark_point(shape="triangle-up", size=150, filled=True, color="#2a9d8f")
        .encode(
            x="actual_eval_1:Q",
            y="actual_eval_2:Q",
            tooltip=["slot", "child_policy_id", "actual_eval_1", "actual_eval_2"],
        )
    )

    predicted_points = (
        alt.Chart(task_rows.dropna(subset=["pred_eval_1", "pred_eval_2"]))
        .mark_point(shape="diamond", size=130, filled=True, color="#f4a261")
        .encode(
            x="pred_eval_1:Q",
            y="pred_eval_2:Q",
            tooltip=["slot", "pred_eval_1", "pred_eval_2", "score"],
        )
    )

    archive_points = (
        alt.Chart(archive_snapshot)
        .mark_point(size=95, filled=True, color="#111111")
        .encode(
            x="eval_1:Q",
            y="eval_2:Q",
            tooltip=["policy_id", "mu", "eval_1", "eval_2"],
        )
    )

    layers = [
        true_front_layer,
        candidate_lines,
        selected_prediction_lines,
        actual_lines,
        archive_points,
        predicted_points,
        offspring_points,
        parent_points,
    ]

    if analysis_done and not analysis_interpolation.empty:
        family_lines = (
            alt.Chart(analysis_interpolation)
            .mark_line(strokeWidth=3.0)
            .encode(
                x="eval_1:Q",
                y="eval_2:Q",
                color=alt.Color("family:N", title="Pareto family"),
            )
        )
        layers.insert(1, family_lines)

    return (
        alt.layer(*layers)
        .properties(
            height=470,
            title=f"Step 3. Generation {visible_generation}: 예측 기반 task selection과 실제 offspring",
        )
        .interactive()
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
        title=f"Step 2. Improvement Predictor for policy #{focus_policy_id} (generation {predictor_generation})",
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
        .properties(height=250, title=f"선택된 best task slot {int(top_task['slot'])}: predicted vs actual")
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
        .properties(height=250, title="실제 학습 중 objective trajectory")
    )
    return delta_chart, trace_chart


def make_parameter_space_chart(family_points: pd.DataFrame) -> alt.Chart:
    selector = alt.selection_point(name="param_pick", fields=["policy_id"], on="click", clear=False)
    return (
        alt.Chart(family_points)
        .mark_circle(size=140, filled=True, stroke="#102a43", strokeWidth=1.2)
        .encode(
            x=alt.X("param_x:Q", title="Parameter Space View: t-SNE 1"),
            y=alt.Y("param_y:Q", title="t-SNE 2"),
            color=alt.Color("family:N", title="Family"),
            tooltip=["policy_id", "mu", "family", "eval_1", "eval_2"],
            opacity=alt.condition(selector, alt.value(1.0), alt.value(0.85)),
        )
        .add_params(selector)
        .properties(height=330, title="Step 4. Parameter Space: t-SNE + k-means family view")
        .interactive()
    )


def make_continuous_front_chart(interpolation: pd.DataFrame, family_points: pd.DataFrame) -> alt.Chart:
    selector = alt.selection_point(name="curve_pick", fields=["interp_id"], on="click", clear=False)

    family_lines = (
        alt.Chart(interpolation)
        .mark_line(strokeWidth=4)
        .encode(
            x=alt.X("eval_1:Q", title="Performance Space: f1"),
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
st.caption("논문 흐름을 그대로 따라가는 4-step interactive dashboard: Warm-up → Predictor → Task Selection → Pareto Analysis")

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
        warmup_clicked = st.button("Warm-up 시작", use_container_width=True)
        if warmup_clicked:
            st.session_state.warmup_done = True
            st.session_state.visible_generation = 0
            st.session_state.analysis_done = False
            st.session_state.curve_interp_id = None
            st.session_state.status_message = "첫 번째 세대의 정책 인구가 형성되었으며, 외부 파레토 아카이브가 초기화되었습니다."

        st.markdown("**Step 2. Improvement Prediction**")
        st.caption("성능 공간에서 정책 점을 클릭하면, 그 정책의 하이퍼볼릭 예측 모델을 아래에 표시합니다.")

        st.markdown("**Step 3. Task Selection + Evolution**")
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

    if not st.session_state.warmup_done:
        initial_points = initial_policy_rows(tasks_pg)
        warmup_population = population_pg[population_pg["snapshot_generation"] == 0].drop_duplicates(subset=["policy_id"])
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

        warmup_chart = make_warmup_chart(result.true_front, initial_points, warmup_population)
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

        st.caption("회색 점은 랜덤 초기 정책, 청록 점은 Warm-up 후 첫 세대 population입니다. 버튼을 누르면 이 상태가 walkthrough의 시작점이 됩니다.")

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
            performance_chart = make_warmup_chart(result.true_front, initial_points, parent_population)
            performance_event = st.altair_chart(
                performance_chart,
                use_container_width=True,
                on_select="rerun",
                selection_mode=["policy_focus"],
                key="performance_space_chart",
            )
        else:
            performance_chart = make_evolution_chart(
                result.true_front,
                parent_population,
                archive_snapshot,
                predictor_rows,
                task_rows,
                interpolation,
                st.session_state.analysis_done,
                visible_generation,
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

        predictor_generation = predictor_generation_for_view(visible_generation)
        predictor_chart = make_predictor_chart(
            predictor_pg[predictor_pg["generation"] == predictor_generation].copy(),
            tasks_pg[tasks_pg["generation"] == predictor_generation].copy(),
            focus_id,
            predictor_generation,
        )

        lower_left, lower_right = st.columns(2)
        with lower_left:
            if predictor_chart is not None:
                st.altair_chart(predictor_chart, use_container_width=True)
                focus_parent = parent_population[parent_population["policy_id"] == focus_id]
                if not focus_parent.empty:
                    mu_value = float(focus_parent.iloc[0]["mu"])
                    st.caption(
                        f"현재 focus policy #{focus_id}의 평균 action μ는 {mu_value:.2f}입니다. "
                        f"이 위의 곡선은 각 weight ω에서 예상되는 Δf를 나타냅니다."
                    )
            else:
                st.info("이 정책에 대한 predictor curve가 아직 없습니다. Warm-up 이후 첫 evolution generation에서 생성됩니다.")

        with lower_right:
            if visible_generation > 0:
                task_inner = inner_pg[
                    (inner_pg["generation"] == visible_generation)
                    & (inner_pg["slot"].isin(task_rows[task_rows["parent_policy_id"] == focus_id]["slot"]))
                ].copy()
                detail_left_chart, detail_right_chart = make_task_detail_charts(task_rows, focus_id, task_inner)
                if detail_left_chart is not None:
                    st.altair_chart(detail_left_chart, use_container_width=True)
                if detail_right_chart is not None:
                    st.altair_chart(detail_right_chart, use_container_width=True)
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
                st.caption(
                    f"이 focus policy의 현재 μ={float(focus_row['mu']):.2f}, 이론적 optimum μ*(w)={optimum_mu:.2f} 입니다."
                )

        if st.session_state.analysis_done and not family_points.empty and not interpolation.empty:
            st.divider()
            analysis_left, analysis_right = st.columns(2)
            with analysis_left:
                st.altair_chart(make_parameter_space_chart(family_points), use_container_width=True)
                st.caption(
                    "Toy note: 실제 논문은 고차원 NN parameter를 t-SNE로 내립니다. 여기서는 scalar policy μ와 performance descriptor를 함께 사용해 family 구조를 시각화합니다."
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
                if not selected_controller.empty:
                    row = selected_controller.iloc[0]
                    st.success(
                        f"현재 보간된 컨트롤러가 생성되었습니다. "
                        f"μ={row['mu']:.2f}, f1={row['eval_1']:.2f}, f2={row['eval_2']:.2f}, {row['family']}"
                    )
                    st.caption("논문에서 말하는 continuous Pareto front를 toy 수준에서 직접 만져보는 단계입니다. 곡선 위 다른 점을 클릭하면 즉시 다른 controller를 볼 수 있습니다.")

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
  fit monotonic hyperbolic model Δ_j^i(ω_j) for each focused policy

Task Selection:
  select policy-weight pairs that maximize
      hypervolume(predicted archive) - λ * sparsity(predicted archive)

Pareto Analysis:
  cluster parameter-space points into families
  linearly interpolate within a family to get a continuous Pareto front
""",
        language="text",
    )
    st.dataframe(result.mapping, use_container_width=True, hide_index=True)
