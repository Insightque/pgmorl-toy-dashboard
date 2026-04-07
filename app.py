from __future__ import annotations

from itertools import combinations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from toy_pgmorl.core import (
    ToyConfig,
    closed_form_optimum_mu,
    run_experiment,
    sample_objectives,
)


alt.data_transformers.disable_max_rows()


def objective_labels(config: ToyConfig) -> list[str]:
    return [f"{idx + 1}. {name}" for idx, name in enumerate(config.objective_names)]


@st.cache_data(show_spinner=False)
def load_result(config: ToyConfig):
    return run_experiment(config)


def metric_chart(metrics: pd.DataFrame, field: str, title: str, y_title: str) -> alt.Chart:
    return (
        alt.Chart(metrics)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("generation:Q", title="Generation"),
            y=alt.Y(f"{field}:Q", title=y_title),
            color=alt.Color("method:N", title="Method"),
            tooltip=["method", "generation", field],
        )
        .properties(title=title, height=280)
    )


def build_reward_curve_df(config: ToyConfig) -> pd.DataFrame:
    actions = np.linspace(config.init_mu_low, config.init_mu_high, 300)
    rewards = sample_objectives(actions, config)
    rows: list[dict[str, float | str]] = []
    for obj_index, label in enumerate(objective_labels(config)):
        for action, reward in zip(actions, rewards[:, obj_index]):
            rows.append({"action": action, "reward": reward, "objective": label})
    return pd.DataFrame(rows)


def reward_curve_chart(
    curve_df: pd.DataFrame,
    config: ToyConfig,
    manual_action: float,
    parent_mu: float,
    child_mu: float,
    optimum_mu: float,
) -> alt.Chart:
    labels = objective_labels(config)
    action_reward = sample_objectives(np.array([manual_action]), config)[0]
    marker_rows = []
    for label, reward in zip(labels, action_reward):
        marker_rows.append({"action": manual_action, "reward": reward, "objective": label})
    marker_df = pd.DataFrame(marker_rows)

    base = (
        alt.Chart(curve_df)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("action:Q", title="Action a"),
            y=alt.Y("reward:Q", title="Reward"),
            color=alt.Color("objective:N", title="Reward component"),
            tooltip=["objective", "action", "reward"],
        )
        .properties(height=340, title="Reward Curves: action a가 reward를 어떻게 바꾸는가")
    )

    manual_points = (
        alt.Chart(marker_df)
        .mark_point(size=90, filled=True)
        .encode(x="action:Q", y="reward:Q", color="objective:N", tooltip=["objective", "action", "reward"])
    )

    rule_df = pd.DataFrame(
        {
            "x": [parent_mu, child_mu, optimum_mu],
            "label": ["부모 policy의 평균 action μ", "학습 후 child μ'", "해당 weight의 이론적 optimum μ*(w)"],
            "color": ["#1f77b4", "#2ca02c", "#d62728"],
        }
    )
    rules = (
        alt.Chart(rule_df)
        .mark_rule(strokeDash=[8, 4], strokeWidth=2)
        .encode(x="x:Q", color=alt.Color("label:N", scale=None), tooltip=["label", "x"])
    )

    return base + manual_points + rules


def pair_choices(config: ToyConfig) -> list[tuple[int, int]]:
    return list(combinations(range(config.num_objectives), 2))


def pareto_projection_chart(
    config: ToyConfig,
    true_front: pd.DataFrame,
    archive: pd.DataFrame,
    population: pd.DataFrame,
    tasks: pd.DataFrame,
    axis_pair: tuple[int, int],
) -> alt.Chart:
    x_idx, y_idx = axis_pair
    x_field = f"eval_{x_idx + 1}"
    y_field = f"eval_{y_idx + 1}"
    x_title = objective_labels(config)[x_idx]
    y_title = objective_labels(config)[y_idx]

    layers = []
    if not true_front.empty:
        layers.append(
            alt.Chart(true_front)
            .mark_line(strokeDash=[6, 4], color="#555555", strokeWidth=2.5)
            .encode(x=alt.X(f"{x_field}:Q", title=x_title), y=alt.Y(f"{y_field}:Q", title=y_title))
        )

    if not archive.empty:
        layers.append(
            alt.Chart(archive)
            .mark_point(size=85, filled=True, color="#111111")
            .encode(
                x=f"{x_field}:Q",
                y=f"{y_field}:Q",
                tooltip=["policy_id", "mu", x_field, y_field],
            )
        )

    if not population.empty:
        layers.append(
            alt.Chart(population)
            .mark_circle(size=90, opacity=0.45, color="#1f77b4")
            .encode(
                x=f"{x_field}:Q",
                y=f"{y_field}:Q",
                tooltip=["policy_id", "mu", x_field, y_field],
            )
        )

    if not tasks.empty and tasks[f"pred_eval_{x_idx + 1}"].notna().any():
        layers.append(
            alt.Chart(tasks.dropna(subset=[f"pred_eval_{x_idx + 1}", f"pred_eval_{y_idx + 1}"]))
            .mark_point(shape="diamond", size=120, color="#ff7f0e")
            .encode(
                x=alt.X(f"pred_eval_{x_idx + 1}:Q", title=x_title),
                y=alt.Y(f"pred_eval_{y_idx + 1}:Q", title=y_title),
                tooltip=["slot", "score", f"pred_eval_{x_idx + 1}", f"pred_eval_{y_idx + 1}"],
            )
        )

    if not tasks.empty:
        layers.append(
            alt.Chart(tasks)
            .mark_triangle(size=120, color="#2ca02c")
            .encode(
                x=alt.X(f"actual_eval_{x_idx + 1}:Q", title=x_title),
                y=alt.Y(f"actual_eval_{y_idx + 1}:Q", title=y_title),
                tooltip=["slot", f"actual_eval_{x_idx + 1}", f"actual_eval_{y_idx + 1}"],
            )
        )

    return alt.layer(*layers).properties(height=420, title="Pareto Projection").interactive()


def delta_bar_chart(config: ToyConfig, task_row: pd.Series) -> alt.Chart:
    rows = []
    for idx, label in enumerate(objective_labels(config), start=1):
        predicted = task_row.get(f"pred_delta_{idx}")
        actual = task_row.get(f"actual_delta_{idx}")
        if pd.notna(predicted):
            rows.append({"objective": label, "kind": "Predicted Δ", "delta": predicted})
        rows.append({"objective": label, "kind": "Actual Δ", "delta": actual})
    delta_df = pd.DataFrame(rows)
    return (
        alt.Chart(delta_df)
        .mark_bar()
        .encode(
            x=alt.X("objective:N", title=None),
            y=alt.Y("delta:Q", title="Objective improvement"),
            color=alt.Color("kind:N", title=None),
            xOffset="kind:N",
            tooltip=["objective", "kind", "delta"],
        )
        .properties(height=280, title="Predicted vs Actual Improvement")
    )


def inner_trace_chart(inner_steps: pd.DataFrame) -> alt.Chart:
    long_df = inner_steps.melt(
        id_vars=["inner_step"],
        value_vars=[column for column in inner_steps.columns if column.startswith("eval_")],
        var_name="objective",
        value_name="value",
    )
    long_df["objective"] = long_df["objective"].str.replace("eval_", "Objective ", regex=False)
    return (
        alt.Chart(long_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("inner_step:Q", title="Inner optimization step"),
            y=alt.Y("value:Q", title="Expected objective value"),
            color=alt.Color("objective:N", title=None),
            tooltip=["inner_step", "objective", "value"],
        )
        .properties(height=280, title="Inside One MOPG Update")
    )


st.set_page_config(page_title="PG-MORL Toy Dashboard", layout="wide")

st.title("PG-MORL Toy Dashboard")
st.caption(
    "Prediction-Guided MORL 논문의 핵심을 수식, policy/action semantics, task selection, Pareto 결과로 바로 연결해서 보는 toy demo"
)

with st.sidebar:
    st.header("학습 설정")
    seed = st.number_input("Seed", min_value=0, max_value=9999, value=7, step=1)
    population_size = st.slider("Warm-up policy 수", min_value=4, max_value=12, value=7, step=1)
    warmup_steps = st.slider("Warm-up gradient steps", min_value=5, max_value=60, value=25, step=5)
    task_steps = st.slider("세대별 gradient steps", min_value=2, max_value=24, value=8, step=2)
    generations = st.slider("Generation 수", min_value=2, max_value=18, value=8, step=1)
    learning_rate = st.slider("Learning rate", min_value=0.005, max_value=0.12, value=0.035, step=0.005)
    policy_std = st.slider("Policy std σ", min_value=0.1, max_value=1.2, value=0.6, step=0.05)
    batch_size = st.slider("REINFORCE batch size", min_value=32, max_value=512, value=128, step=32)
    weight_resolution = st.slider("Weight grid 해상도", min_value=4, max_value=32, value=14, step=1)
    gradient_mode = st.radio(
        "Inner optimizer",
        options=["analytic", "reinforce"],
        format_func=lambda mode: "Analytic (설명용, 더 안정적)" if mode == "analytic" else "REINFORCE (논문 감각에 더 가까움)",
    )

    st.divider()
    st.subheader("이 UI에서 바가 의미하는 것")
    st.markdown(
        "- `학습 설정` 바는 실험을 다시 돌립니다.\n"
        "- `generation / task / action inspector` 바는 이미 돌린 로그를 들여다보는 용도입니다.\n"
        "- 이 toy는 `state`를 고정한 1-step 환경이라 state 바를 아예 제거했습니다."
    )

config = ToyConfig(
    seed=int(seed),
    population_size=int(population_size),
    warmup_steps=int(warmup_steps),
    task_steps=int(task_steps),
    generations=int(generations),
    learning_rate=float(learning_rate),
    policy_std=float(policy_std),
    batch_size=int(batch_size),
    weight_resolution=int(weight_resolution),
    gradient_mode=gradient_mode,
)

with st.spinner("Toy PG-MORL 실험을 실행하고 있습니다..."):
    result = load_result(config)

st.markdown(
    "[PG-MORL 논문](https://people.csail.mit.edu/jiex/papers/PGMORL/paper.pdf) · "
    "[원본 구현](https://github.com/mit-gfx/PGMORL) · "
    "[morl-baselines 구현](https://github.com/LucasAlegre/morl-baselines)"
)

top_left, top_mid, top_right = st.columns([1.25, 1.25, 1.5])
with top_left:
    st.metric("Best Final Hypervolume", f"{result.method_summary.iloc[0]['hypervolume']:.1f}", result.method_summary.iloc[0]["method"])
with top_mid:
    st.metric("Best Final Sparsity", f"{result.method_summary['sparsity'].min():.2f}", "lower is better")
with top_right:
    st.dataframe(result.method_summary, use_container_width=True, hide_index=True)

tabs = st.tabs(["개념 · Inspector", "학습 곡선", "Pareto Front", "Task Selection", "수식 · 의사코드"])

with tabs[0]:
    st.subheader("먼저 semantics를 고정합니다")
    st.markdown(
        "- `state s`: 고정된 단일 상태입니다. 논문 핵심은 state 조작이 아니라 `weight가 policy를 어떻게 이동시키는가`입니다.\n"
        "- `policy πθ`: 평균 action `μ=θ`를 갖는 Gaussian policy입니다.\n"
        "- `action a`: 현재 policy에서 한 번 샘플된 실제 제어 입력입니다.\n"
        "- `weight w`: reward preference이며, action을 즉시 정하지 않고 `policy update 방향`을 정합니다."
    )

    st.info("즉, `weight -> 바로 action`이 아니라 `weight + 현재 policy -> gradient update -> 새로운 policy -> action 샘플` 순서입니다.")

    inspect_method = st.selectbox("Inspector용 method", result.metrics["method"].unique().tolist(), index=0, key="inspect_method")
    inspect_generation = st.slider("Inspector generation", min_value=0, max_value=config.generations, value=min(2, config.generations), key="inspect_generation")
    inspect_tasks = result.tasks[(result.tasks["method"] == inspect_method) & (result.tasks["generation"] == inspect_generation)]
    if inspect_tasks.empty:
        st.warning("이 generation에 inspect할 task가 없습니다.")
    else:
        inspect_slot = st.slider(
            "Inspector task slot",
            min_value=int(inspect_tasks["slot"].min()),
            max_value=int(inspect_tasks["slot"].max()),
            value=int(inspect_tasks["slot"].min()),
            key="inspect_slot",
        )
        inspect_task = inspect_tasks[inspect_tasks["slot"] == inspect_slot].iloc[0]
        inner_steps = result.inner_steps[
            (result.inner_steps["method"] == inspect_method)
            & (result.inner_steps["generation"] == inspect_generation)
            & (result.inner_steps["slot"] == inspect_slot)
        ].sort_values("inner_step")

        if inner_steps.empty:
            st.warning("선택된 task의 inner-step 로그를 찾지 못했습니다.")
        else:
            parent_mu = float(inner_steps.iloc[0]["mu"])
            child_mu = float(inner_steps.iloc[-1]["mu"])
            selected_weights = np.array([inspect_task["weight_1"], inspect_task["weight_2"]], dtype=np.float64)
            optimum_mu = closed_form_optimum_mu(selected_weights, config)

            st.markdown(
                f"`선택된 task의 weight`는 `{selected_weights[0]:.2f}, {selected_weights[1]:.2f}` 이고, "
                f"그 weight에 대한 이 toy의 이론적 optimum은 `μ*(w) = {optimum_mu:.2f}` 입니다."
            )

            concept_left, concept_right = st.columns([1.1, 1.4])
            with concept_left:
                st.metric("부모 policy 평균 action μ", f"{parent_mu:.2f}")
                st.metric("학습 후 child μ'", f"{child_mu:.2f}")
                st.metric("이론적 optimum μ*(w)", f"{optimum_mu:.2f}")
                manual_action = st.slider(
                    "Action inspector a  (관찰용, 학습 결과는 안 바뀜)",
                    min_value=float(config.init_mu_low),
                    max_value=float(config.init_mu_high),
                    value=float(child_mu),
                    step=0.1,
                )
                reward_vector = sample_objectives(np.array([manual_action]), config)[0]
                reward_df = pd.DataFrame(
                    {
                        "objective": objective_labels(config),
                        "reward": reward_vector,
                    }
                )
                st.dataframe(reward_df, use_container_width=True, hide_index=True)
            with concept_right:
                curve_df = build_reward_curve_df(config)
                st.altair_chart(
                    reward_curve_chart(curve_df, config, manual_action, parent_mu, child_mu, optimum_mu),
                    use_container_width=True,
                )

with tabs[1]:
    st.subheader("Learning Curves")
    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        st.altair_chart(metric_chart(result.metrics, "hypervolume", "Hypervolume over generations", "Hypervolume"), use_container_width=True)
    with curve_col2:
        st.altair_chart(metric_chart(result.metrics, "sparsity", "Sparsity over generations", "Sparsity"), use_container_width=True)
    st.altair_chart(
        metric_chart(result.metrics, "archive_size", "Archive size over generations", "Archive size"),
        use_container_width=True,
    )

with tabs[2]:
    st.subheader("Pareto Front Snapshot")
    pair_options = pair_choices(config)
    pair_labels = [f"{objective_labels(config)[x]} vs {objective_labels(config)[y]}" for x, y in pair_options]
    chosen_pair_label = st.selectbox("Projection axes", pair_labels, index=0)
    chosen_pair = pair_options[pair_labels.index(chosen_pair_label)]
    pareto_method = st.selectbox("Pareto view method", result.metrics["method"].unique().tolist(), index=0, key="pareto_method")
    pareto_generation = st.slider("Pareto view generation", min_value=0, max_value=config.generations, value=config.generations, key="pareto_generation")

    archive_snapshot = result.archive[
        (result.archive["method"] == pareto_method) & (result.archive["snapshot_generation"] == pareto_generation)
    ]
    population_snapshot = result.population[
        (result.population["method"] == pareto_method) & (result.population["snapshot_generation"] == pareto_generation)
    ]
    task_snapshot = result.tasks[(result.tasks["method"] == pareto_method) & (result.tasks["generation"] == pareto_generation)]

    st.altair_chart(
        pareto_projection_chart(config, result.true_front, archive_snapshot, population_snapshot, task_snapshot, chosen_pair),
        use_container_width=True,
    )

    st.markdown(
        "- 검은 점: 현재까지의 external Pareto archive\n"
        "- 파란 점: 현재 세대의 population buffer\n"
        "- 주황 다이아: predictor가 예상한 offspring\n"
        "- 초록 삼각형: 실제 MOPG 후 offspring"
    )

with tabs[3]:
    st.subheader("Task Selection and One Update")
    task_method = st.selectbox("Task table method", result.metrics["method"].unique().tolist(), index=0, key="task_method")
    task_generation = st.slider("Task table generation", min_value=0, max_value=config.generations, value=min(1, config.generations), key="task_generation")
    current_tasks = result.tasks[(result.tasks["method"] == task_method) & (result.tasks["generation"] == task_generation)].copy()
    st.dataframe(current_tasks, use_container_width=True, hide_index=True)

    if not current_tasks.empty:
        selected_slot = st.slider(
            "Detailed task slot",
            min_value=int(current_tasks["slot"].min()),
            max_value=int(current_tasks["slot"].max()),
            value=int(current_tasks["slot"].min()),
            key="detail_slot",
        )
        task_row = current_tasks[current_tasks["slot"] == selected_slot].iloc[0]
        inner_steps = result.inner_steps[
            (result.inner_steps["method"] == task_method)
            & (result.inner_steps["generation"] == task_generation)
            & (result.inner_steps["slot"] == selected_slot)
        ].sort_values("inner_step")

        detail_left, detail_right = st.columns(2)
        with detail_left:
            st.altair_chart(delta_bar_chart(config, task_row), use_container_width=True)
        with detail_right:
            st.altair_chart(inner_trace_chart(inner_steps), use_container_width=True)

        mu_trace = (
            alt.Chart(inner_steps)
            .mark_line(point=True, strokeWidth=3, color="#1f77b4")
            .encode(
                x=alt.X("inner_step:Q", title="Inner step"),
                y=alt.Y("mu:Q", title="Policy mean action μ"),
                tooltip=["inner_step", "mu", "scalarized_return"],
            )
            .properties(height=260, title="How the policy itself moves")
        )
        st.altair_chart(mu_trace, use_container_width=True)

with tabs[4]:
    st.subheader("수식으로 다시 정리")
    st.latex(r"\pi_\theta(a) = \mathcal{N}(a; \mu=\theta, \sigma^2)")
    st.latex(r"r_i(a) = C - (a - t_i)^2")
    st.latex(r"F_i(\theta) = \mathbb{E}_{a \sim \pi_\theta}[r_i(a)] = C - ((\theta - t_i)^2 + \sigma^2)")
    st.latex(r"J(\theta, w) = \sum_i w_i F_i(\theta)")
    st.latex(r"\nabla_\theta J(\theta, w) = \mathbb{E}\left[(w^\top r(a)) \nabla_\theta \log \pi_\theta(a)\right]")
    st.latex(r"\theta^\*(w) = \sum_i w_i t_i")

    st.subheader("Toy pseudocode")
    st.code(
        """Warm-up:
  initialize N policies with random mu values
  assign evenly spaced weights
  run MOPG for m_w steps
  update population buffer + Pareto archive

Evolution:
  fit hyperbolic predictor from RL history
  greedily choose (policy, weight) pairs that maximize
      hypervolume(predicted archive) - lambda * sparsity(predicted archive)
  run MOPG on selected tasks for m_t steps
  update population buffer + Pareto archive
""",
        language="text",
    )

    st.subheader("논문과 toy의 대응")
    st.dataframe(result.mapping, use_container_width=True, hide_index=True)
