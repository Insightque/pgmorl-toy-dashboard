from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from toy_pgmorl.core import (
    ToyConfig,
    build_true_front,
    closed_form_optimum_mu,
    expected_objectives,
    normalize_preferences,
    one_hot_weight,
    paper_mapping,
    optimize_policy,
    run_experiment,
)


st.set_page_config(page_title="PG-MORL 한글 Toy Dashboard", layout="wide")

METHOD_COLORS = {
    "PG-MORL Toy": "#1f77b4",
    "RA Sweep": "#ff7f0e",
    "Random": "#2ca02c",
}

POINT_COLORS = {
    "현재 정책": "#d62728",
    "선호도 기준 최적 속도": "#1f77b4",
    "Warm-up/Offspring": "#ff7f0e",
    "현재 Population": "#2ca02c",
    "Pareto Archive": "#9467bd",
    "Predictor 후보": "#8c564b",
    "실제 선택된 작업": "#d62728",
}


def objective_labels(config: ToyConfig) -> list[str]:
    return list(config.objective_names)


def eval_cols(config: ToyConfig) -> list[str]:
    return [f"eval_{index + 1}" for index in range(config.num_objectives)]


def weight_cols(config: ToyConfig) -> list[str]:
    return [f"weight_{index + 1}" for index in range(config.num_objectives)]


def point_frame(mu: float, evaluation: np.ndarray, label: str, config: ToyConfig) -> pd.DataFrame:
    row = {"mu": float(mu), "label": label}
    for index, value in enumerate(evaluation):
        row[f"eval_{index + 1}"] = float(value)
    return pd.DataFrame([row])


def normalized_weight_caption(weight: np.ndarray, config: ToyConfig) -> str:
    return " / ".join(f"{name} {value:.2f}" for name, value in zip(objective_labels(config), weight))


def scenario_table(config: ToyConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "KPI": objective_labels(config),
            "물리적 의미": [
                "고객에게 빨리 도착하도록 평균 속도를 높게 유지하고 싶습니다.",
                "배터리 소모를 줄이려면 너무 빠르게 달리면 안 됩니다.",
                "복도에서 사람과 부딪히지 않으려면 중간 속도 근처가 안전합니다.",
            ],
            "가장 좋아하는 속도": list(config.objective_targets),
        }
    )


def build_speed_landscape(config: ToyConfig, weight: np.ndarray, points: int = 241) -> tuple[pd.DataFrame, pd.DataFrame]:
    mus = np.linspace(config.init_mu_low, config.init_mu_high, points)
    rows = []
    for mu in mus:
        evaluation = expected_objectives(float(mu), config)
        row = {"mu": float(mu), "가중합 점수": float(evaluation @ weight)}
        for index, name in enumerate(objective_labels(config)):
            row[name] = float(evaluation[index])
        rows.append(row)
    wide = pd.DataFrame(rows)
    return wide, wide.melt(id_vars=["mu"], var_name="series", value_name="reward")


def speed_ruler_chart(config: ToyConfig, current_mu: float, optimum_mu: float) -> alt.Chart:
    track = pd.DataFrame({"speed": [config.init_mu_low, config.init_mu_high], "lane": [0, 0]})
    points = []
    for target, name in zip(config.objective_targets, objective_labels(config)):
        points.append({"speed": float(target), "lane": 0.0, "series": f"{name} 선호 속도"})
    points.append({"speed": float(current_mu), "lane": 0.08, "series": "현재 정책"})
    points.append({"speed": float(optimum_mu), "lane": -0.08, "series": "선호도 기준 최적 속도"})
    points_df = pd.DataFrame(points)
    color_domain = list(POINT_COLORS.keys()) + [f"{name} 선호 속도" for name in objective_labels(config)]
    color_range = list(POINT_COLORS.values()) + ["#17becf", "#bcbd22", "#7f7f7f"]
    base = alt.Chart(track).mark_line(color="#4c566a", strokeWidth=4).encode(
        x=alt.X("speed:Q", title="카트의 순항 속도 μ", scale=alt.Scale(domain=[config.init_mu_low, config.init_mu_high])),
        y=alt.Y("lane:Q", axis=None),
    )
    marks = alt.Chart(points_df).mark_point(filled=True, size=180).encode(
        x="speed:Q",
        y="lane:Q",
        color=alt.Color("series:N", title="", scale=alt.Scale(domain=color_domain, range=color_range)),
        tooltip=["series:N", alt.Tooltip("speed:Q", format=".2f")],
    )
    labels = alt.Chart(points_df).mark_text(dy=-12, fontSize=11).encode(x="speed:Q", y="lane:Q", text="series:N")
    return (base + marks + labels).properties(height=180)


def landscape_chart(landscape_long: pd.DataFrame) -> alt.Chart:
    color_scale = alt.Scale(
        domain=["빠른 도착", "배터리 효율", "안전성", "가중합 점수"],
        range=["#d62728", "#2ca02c", "#1f77b4", "#111111"],
    )
    return (
        alt.Chart(landscape_long)
        .mark_line(strokeWidth=3)
        .encode(
            x=alt.X("mu:Q", title="카트의 순항 속도 μ"),
            y=alt.Y("reward:Q", title="기대 보상"),
            color=alt.Color("series:N", title="", scale=color_scale),
            tooltip=["series:N", alt.Tooltip("mu:Q", format=".2f"), alt.Tooltip("reward:Q", format=".2f")],
        )
        .properties(height=320)
    )


def build_projection_chart(
    config: ToyConfig,
    true_front: pd.DataFrame,
    overlays: list[tuple[str, pd.DataFrame]],
    first: int,
    second: int,
    title: str,
) -> alt.Chart:
    x_col = f"eval_{first + 1}"
    y_col = f"eval_{second + 1}"
    front = alt.Chart(true_front).mark_line(color="#9aa5b1", strokeWidth=3).encode(
        x=alt.X(f"{x_col}:Q", title=objective_labels(config)[first]),
        y=alt.Y(f"{y_col}:Q", title=objective_labels(config)[second]),
        tooltip=[alt.Tooltip("mu:Q", format=".2f"), alt.Tooltip(f"{x_col}:Q", format=".2f"), alt.Tooltip(f"{y_col}:Q", format=".2f")],
    )
    layers = [front]
    frames = []
    for label, df in overlays:
        if df is None or df.empty:
            continue
        local = df.copy()
        local["series"] = label
        frames.append(local)
    if frames:
        points = pd.concat(frames, ignore_index=True)
        layers.append(
            alt.Chart(points)
            .mark_circle(size=90, opacity=0.85)
            .encode(
                x=f"{x_col}:Q",
                y=f"{y_col}:Q",
                color=alt.Color(
                    "series:N",
                    title="",
                    scale=alt.Scale(domain=list(POINT_COLORS.keys()), range=list(POINT_COLORS.values())),
                ),
                tooltip=[
                    "series:N",
                    alt.Tooltip("mu:Q", format=".2f"),
                    alt.Tooltip(f"{x_col}:Q", format=".2f"),
                    alt.Tooltip(f"{y_col}:Q", format=".2f"),
                ],
            )
        )
    return alt.layer(*layers).properties(title=title, height=260)


def metric_curve(metrics: pd.DataFrame, value_col: str, title: str) -> alt.Chart:
    return (
        alt.Chart(metrics)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("generation:Q", title="세대"),
            y=alt.Y(f"{value_col}:Q", title=title),
            color=alt.Color("method:N", title="", scale=alt.Scale(domain=list(METHOD_COLORS), range=list(METHOD_COLORS.values()))),
            tooltip=["method:N", "generation:Q", alt.Tooltip(f"{value_col}:Q", format=".3f")],
        )
        .properties(height=280)
    )


def weight_history_chart(tasks: pd.DataFrame, config: ToyConfig) -> alt.Chart:
    weight_frame = tasks.loc[tasks["generation"] > 0, ["generation", *weight_cols(config)]].copy()
    if weight_frame.empty:
        return alt.Chart(pd.DataFrame({"generation": [], "weight": [], "objective": []})).mark_line()
    long = weight_frame.melt(id_vars=["generation"], var_name="objective", value_name="weight")
    rename = {f"weight_{index + 1}": name for index, name in enumerate(objective_labels(config))}
    long["objective"] = long["objective"].map(rename)
    summary = long.groupby(["generation", "objective"], as_index=False)["weight"].mean()
    return (
        alt.Chart(summary)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("generation:Q", title="세대"),
            y=alt.Y("weight:Q", title="선택된 작업의 평균 비중"),
            color=alt.Color("objective:N", title="", scale=alt.Scale(range=["#d62728", "#2ca02c", "#1f77b4"])),
            tooltip=["generation:Q", "objective:N", alt.Tooltip("weight:Q", format=".3f")],
        )
        .properties(height=280)
    )


def policy_trace_chart(trace: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(trace)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("inner_step:Q", title="gradient step"),
            y=alt.Y("mu:Q", title="카트의 순항 속도 μ"),
            tooltip=["inner_step:Q", alt.Tooltip("mu:Q", format=".3f"), alt.Tooltip("scalarized_return:Q", format=".3f")],
        )
        .properties(height=280)
    )


def objective_trace_chart(trace: pd.DataFrame, config: ToyConfig) -> alt.Chart:
    frame = trace.loc[:, ["inner_step", *eval_cols(config)]].melt(id_vars=["inner_step"], var_name="objective", value_name="reward")
    rename = {f"eval_{index + 1}": name for index, name in enumerate(objective_labels(config))}
    frame["objective"] = frame["objective"].map(rename)
    return (
        alt.Chart(frame)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("inner_step:Q", title="gradient step"),
            y=alt.Y("reward:Q", title="각 KPI의 기대 보상"),
            color=alt.Color("objective:N", title="", scale=alt.Scale(range=["#d62728", "#2ca02c", "#1f77b4"])),
            tooltip=["inner_step:Q", "objective:N", alt.Tooltip("reward:Q", format=".3f")],
        )
        .properties(height=280)
    )


@st.cache_data(show_spinner=False)
def cached_experiment(config_dict: dict) -> object:
    return run_experiment(ToyConfig(**config_dict))


base_config = ToyConfig()

st.title("PG-MORL을 한국어로 이해하는 3-KPI Toy Dashboard")
st.markdown(
    """
이 대시보드는 논문 속 복잡한 로봇 제어를 아주 단순한 물리 비유로 바꿉니다.
여기서는 **직선 복도를 달리는 배송 카트의 평균 순항 속도 `μ`** 하나만 조절합니다.
하지만 평가 기준은 1개가 아니라 3개입니다.
"""
)

st.dataframe(scenario_table(base_config), use_container_width=True, hide_index=True)
st.info(
    "핵심 직관: RL은 보통 KPI 하나만 크게 만들지만, MORL은 여러 KPI가 충돌할 때 어떤 균형점을 찾을지 함께 학습합니다. "
    "PG-MORL은 그중에서도 '지금 어떤 선호도(weight)로 학습하는 것이 Pareto front를 가장 잘 넓힐지'를 예측해서 작업을 고릅니다."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1. 문제 이해", "2. 직접 한 번 학습", "3. PG-MORL 비교", "4. 한 세대 해부", "5. 논문 대응"])

if "experiment_result" not in st.session_state:
    st.session_state["experiment_result"] = None
if "experiment_config_dict" not in st.session_state:
    st.session_state["experiment_config_dict"] = None

experiment_result = st.session_state["experiment_result"]
experiment_config = ToyConfig(**st.session_state["experiment_config_dict"]) if st.session_state["experiment_config_dict"] else base_config

with tab1:
    st.subheader("RL과 MORL의 차이를 속도 하나로 보기")
    left, right = st.columns([1.1, 1.4], gap="large")
    with left:
        st.markdown("**조절값**")
        intuition_mu = st.slider("현재 정책 속도 μ", 0.0, 10.0, 5.0, 0.1, key="intuition_mu")
        intuition_sigma = st.slider("정책의 흔들림 σ", 0.1, 1.5, 0.6, 0.05, key="intuition_sigma")
        raw_fast = st.slider("중요도: 빠른 도착", 0.1, 5.0, 1.0, 0.1, key="intuition_fast")
        raw_battery = st.slider("중요도: 배터리 효율", 0.1, 5.0, 1.0, 0.1, key="intuition_battery")
        raw_safe = st.slider("중요도: 안전성", 0.1, 5.0, 1.0, 0.1, key="intuition_safe")
        st.caption(
            "위 3개 중요도는 MORL의 선호도 벡터입니다. 값이 클수록 그 KPI를 더 강하게 신경 쓰도록 scalarization 가중치가 바뀝니다."
        )

    config_intuition = ToyConfig(policy_std=float(intuition_sigma))
    weight = normalize_preferences([raw_fast, raw_battery, raw_safe])
    current_eval = expected_objectives(float(intuition_mu), config_intuition)
    optimum_mu = closed_form_optimum_mu(weight, config_intuition)
    optimum_eval = expected_objectives(float(optimum_mu), config_intuition)
    _, landscape_long = build_speed_landscape(config_intuition, weight)
    true_front = build_true_front(config_intuition)

    with right:
        metric_cols = st.columns(4)
        metric_cols[0].metric("현재 속도 μ", f"{intuition_mu:.2f}")
        metric_cols[1].metric("MORL 선호도 기준 최적 속도", f"{optimum_mu:.2f}")
        metric_cols[2].metric("현재 가중합 점수", f"{current_eval @ weight:.2f}")
        metric_cols[3].metric("정규화된 선호도", normalized_weight_caption(weight, config_intuition))
        st.caption("가중합 점수는 3개 KPI를 하나의 scalar objective로 합친 값입니다. MORL은 이 비중을 바꿔가며 여러 균형점을 찾습니다.")

    st.altair_chart(speed_ruler_chart(config_intuition, intuition_mu, optimum_mu), use_container_width=True)
    st.caption("속도 눈금: 각 KPI가 좋아하는 대표 속도와, 현재 정책 및 선택한 선호도 기준 최적 속도를 함께 표시합니다.")

    chart_left, chart_right = st.columns([1.15, 1.0], gap="large")
    with chart_left:
        st.altair_chart(landscape_chart(landscape_long), use_container_width=True)
        st.caption(
            "곡선 해석: 빨리 달리면 도착 KPI는 좋아지지만, 배터리 효율과 안전성은 나빠질 수 있습니다. "
            "검은 선은 사용자가 정한 선호도로 3개 KPI를 합친 결과입니다."
        )

    overlays = [
        ("현재 정책", point_frame(intuition_mu, current_eval, "현재 정책", config_intuition)),
        ("선호도 기준 최적 속도", point_frame(optimum_mu, optimum_eval, "선호도 기준 최적 속도", config_intuition)),
    ]
    with chart_right:
        proj_cols = st.columns(3)
        proj_cols[0].altair_chart(
            build_projection_chart(config_intuition, true_front, overlays, 0, 1, "빠른 도착 vs 배터리 효율"),
            use_container_width=True,
        )
        proj_cols[1].altair_chart(
            build_projection_chart(config_intuition, true_front, overlays, 0, 2, "빠른 도착 vs 안전성"),
            use_container_width=True,
        )
        proj_cols[2].altair_chart(
            build_projection_chart(config_intuition, true_front, overlays, 1, 2, "배터리 효율 vs 안전성"),
            use_container_width=True,
        )
        st.caption("회색 선은 가능한 Pareto front, 점은 현재 정책과 선호도 기준 균형점입니다. 한쪽을 좋게 만들면 다른 쪽이 나빠지는 trade-off를 눈으로 확인할 수 있습니다.")

    st.markdown("**일반 RL과 MORL 차이**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "구분": "일반 RL",
                    "무엇을 학습하나": "보통 하나의 scalar reward만 최대화",
                    "이 예제에서": "예: 빠른 도착만 최대로 하도록 속도를 점점 8 근처로 올림",
                },
                {
                    "구분": "MORL",
                    "무엇을 학습하나": "여러 reward를 함께 보고 Pareto trade-off를 탐색",
                    "이 예제에서": "빠른 도착, 배터리, 안전성의 균형점들을 여러 개 학습",
                },
                {
                    "구분": "PG-MORL",
                    "무엇을 학습하나": "어떤 weight로 다음 학습을 해야 Pareto front를 가장 잘 넓힐지 예측",
                    "이 예제에서": "이미 충분한 영역은 덜 보고, 빈 영역을 채우는 방향을 고름",
                },
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.subheader("하나의 정책이 실제로 어떻게 움직이는지 직접 보기")
    control_left, control_right = st.columns([1.05, 1.15], gap="large")
    with control_left:
        mode = st.radio("학습 모드", ["단일 KPI RL", "가중합 MORL"], horizontal=True, key="manual_mode")
        manual_mu = st.slider("시작 속도 μ", 0.0, 10.0, 1.5, 0.1, key="manual_mu")
        manual_steps = st.slider("학습 step 수", 5, 80, 20, 1, key="manual_steps")
        manual_lr = st.slider("학습률", 0.005, 0.15, 0.04, 0.005, key="manual_lr")
        manual_std = st.slider("정책의 흔들림 σ", 0.1, 1.5, 0.6, 0.05, key="manual_std")
        manual_batch = st.slider("샘플 수", 32, 512, 128, 32, key="manual_batch_size")
        manual_grad = st.selectbox("기울기 계산 방식", ["reinforce", "analytic"], index=0, key="manual_grad")
        st.caption(
            "REINFORCE는 샘플로 gradient를 추정하고, analytic은 이 toy 환경에서 닫힌형 gradient를 사용합니다. "
            "둘 다 같은 방향성을 보지만, REINFORCE는 약간 더 흔들립니다."
        )

        if mode == "단일 KPI RL":
            target_name = st.selectbox("집중할 KPI", objective_labels(base_config), index=0, key="manual_single_objective")
            objective_index = objective_labels(base_config).index(target_name)
            manual_weight = one_hot_weight(objective_index, base_config.num_objectives)
            st.info(f"이 설정은 일반 RL처럼 `{target_name}` 하나만 최대로 만드는 상황입니다.")
        else:
            manual_fast = st.slider("중요도: 빠른 도착", 0.1, 5.0, 1.0, 0.1, key="manual_fast")
            manual_battery = st.slider("중요도: 배터리 효율", 0.1, 5.0, 1.0, 0.1, key="manual_battery")
            manual_safe = st.slider("중요도: 안전성", 0.1, 5.0, 1.0, 0.1, key="manual_safe")
            manual_weight = normalize_preferences([manual_fast, manual_battery, manual_safe])
            st.info("이 설정은 MORL의 weighted-sum worker입니다. 선택한 선호도에 따라 하나의 policy gradient run이 다른 균형점으로 이동합니다.")

    manual_config = ToyConfig(
        policy_std=float(manual_std),
        learning_rate=float(manual_lr),
        batch_size=int(manual_batch),
        gradient_mode=manual_grad,
    )
    manual_rng = np.random.default_rng(manual_config.seed + 999)
    manual_outcome = optimize_policy(
        initial_mu=float(manual_mu),
        weight=manual_weight,
        config=manual_config,
        num_steps=int(manual_steps),
        rng=manual_rng,
        method="Manual",
        generation=0,
        slot=0,
    )
    manual_trace = pd.DataFrame(manual_outcome.inner_records)
    before_eval = manual_outcome.before_eval
    after_eval = manual_outcome.after_eval
    before_mu = float(manual_trace.iloc[0]["mu"])
    after_mu = float(manual_trace.iloc[-1]["mu"])
    manual_true_front = build_true_front(manual_config)
    manual_optimum_mu = closed_form_optimum_mu(manual_weight, manual_config)
    manual_optimum_eval = expected_objectives(float(manual_optimum_mu), manual_config)

    with control_right:
        metric_cols = st.columns(4)
        metric_cols[0].metric("시작 속도", f"{before_mu:.2f}")
        metric_cols[1].metric("학습 후 속도", f"{after_mu:.2f}")
        metric_cols[2].metric("이론상 균형 속도", f"{manual_optimum_mu:.2f}")
        metric_cols[3].metric("선호도", normalized_weight_caption(manual_weight, manual_config))
        st.caption("학습 후 속도가 이론상 균형 속도에 가까워질수록, 이 policy gradient가 올바른 방향으로 움직인다고 볼 수 있습니다.")

        before_after_cols = st.columns(2)
        before_after_cols[0].dataframe(
            pd.DataFrame({"KPI": objective_labels(manual_config), "시작 보상": np.round(before_eval, 2)}),
            hide_index=True,
            use_container_width=True,
        )
        before_after_cols[1].dataframe(
            pd.DataFrame({"KPI": objective_labels(manual_config), "학습 후 보상": np.round(after_eval, 2)}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("단일 KPI RL이라면 한 열만 크게 좋아지고, MORL이라면 여러 KPI가 함께 타협점으로 이동하는 모습을 볼 수 있습니다.")

    st.altair_chart(speed_ruler_chart(manual_config, before_mu, after_mu), use_container_width=True)
    st.caption("여기서는 '현재 정책' 자리에 시작 속도, '선호도 기준 최적 속도' 자리에 학습 후 속도를 표시합니다. 두 점의 이동만 봐도 학습 방향을 이해하기 쉽습니다.")

    trace_cols = st.columns(2)
    trace_cols[0].altair_chart(policy_trace_chart(manual_trace), use_container_width=True)
    trace_cols[0].caption("속도 trajectory: gradient step이 진행되면서 카트가 어느 속도로 이동하는지 보여줍니다.")
    trace_cols[1].altair_chart(objective_trace_chart(manual_trace, manual_config), use_container_width=True)
    trace_cols[1].caption("KPI trajectory: 한 KPI를 올리면 다른 KPI가 깎일 수 있는 trade-off가 step마다 드러납니다.")

    manual_overlays = [
        ("현재 정책", point_frame(before_mu, before_eval, "현재 정책", manual_config)),
        ("Warm-up/Offspring", point_frame(after_mu, after_eval, "Warm-up/Offspring", manual_config)),
        ("선호도 기준 최적 속도", point_frame(manual_optimum_mu, manual_optimum_eval, "선호도 기준 최적 속도", manual_config)),
    ]
    proj_cols = st.columns(3)
    proj_cols[0].altair_chart(
        build_projection_chart(manual_config, manual_true_front, manual_overlays, 0, 1, "빠른 도착 vs 배터리 효율"),
        use_container_width=True,
    )
    proj_cols[1].altair_chart(
        build_projection_chart(manual_config, manual_true_front, manual_overlays, 0, 2, "빠른 도착 vs 안전성"),
        use_container_width=True,
    )
    proj_cols[2].altair_chart(
        build_projection_chart(manual_config, manual_true_front, manual_overlays, 1, 2, "배터리 효율 vs 안전성"),
        use_container_width=True,
    )
    st.caption("빨간 점은 시작 위치, 주황 점은 학습 결과, 파란 점은 선택한 선호도에서의 목표 균형점입니다.")

with tab3:
    st.subheader("PG-MORL이 왜 필요한지 baseline과 비교")
    st.markdown("**조절값**")
    top = st.columns(5)
    exp_population = top[0].slider("Population 크기", 4, 12, 5, 1, key="exp_population")
    exp_warmup = top[1].slider("Warm-up step", 10, 80, 12, 1, key="exp_warmup")
    exp_task = top[2].slider("후속 학습 step", 4, 30, 6, 1, key="exp_task")
    exp_generations = top[3].slider("세대 수", 4, 20, 4, 1, key="exp_generations")
    exp_resolution = top[4].slider("weight 격자 해상도", 4, 12, 4, 1, key="exp_resolution")

    bottom = st.columns(5)
    exp_lr = bottom[0].slider("학습률", 0.01, 0.08, 0.035, 0.005, key="exp_learning_rate")
    exp_std = bottom[1].slider("정책의 흔들림 σ", 0.2, 1.2, 0.6, 0.05, key="exp_policy_std")
    exp_sparsity = bottom[2].slider("다양성 패널티 λ", 0.0, 0.5, 0.18, 0.02, key="exp_sparsity")
    exp_batch = bottom[3].slider("샘플 수", 32, 512, 64, 32, key="exp_batch_size")
    exp_grad = bottom[4].selectbox("gradient 계산", ["reinforce", "analytic"], index=1, key="exp_grad")

    st.caption(
        "Population은 동시에 유지할 정책 수, Warm-up은 첫 Pareto seed를 만드는 학습 길이, weight 해상도는 PG-MORL이 검토할 선호도 후보의 촘촘함입니다. "
        "기본 gradient는 데모 반응성을 위해 analytic으로 두었습니다."
    )

    experiment_config = ToyConfig(
        population_size=int(exp_population),
        warmup_steps=int(exp_warmup),
        task_steps=int(exp_task),
        generations=int(exp_generations),
        learning_rate=float(exp_lr),
        policy_std=float(exp_std),
        batch_size=int(exp_batch),
        weight_resolution=int(exp_resolution),
        sparsity_coef=float(exp_sparsity),
        gradient_mode=exp_grad,
    )

    run_clicked = st.button("현재 설정으로 실험 실행", key="exp_run_button", type="primary")
    if run_clicked:
        with st.spinner("실험을 실행하고 있습니다..."):
            experiment_result = cached_experiment(experiment_config.to_dict())
            st.session_state["experiment_result"] = experiment_result
            st.session_state["experiment_config_dict"] = experiment_config.to_dict()
    else:
        experiment_result = st.session_state["experiment_result"]
        if st.session_state["experiment_config_dict"]:
            experiment_config = ToyConfig(**st.session_state["experiment_config_dict"])

    if experiment_result is None:
        st.info("탭 3과 4의 비교 실험은 버튼을 눌렀을 때만 실행됩니다. 먼저 위 버튼으로 현재 설정을 한 번 실행해 주세요.")
    else:
        st.caption("아래 결과는 마지막으로 실행한 설정 기준입니다. 슬라이더를 바꾼 뒤 다시 보고 싶다면 버튼을 다시 눌러 주세요.")
        summary = experiment_result.method_summary.rename(
            columns={
                "method": "방법",
                "generation": "마지막 세대",
                "hypervolume": "Hypervolume",
                "sparsity": "Sparsity",
                "archive_size": "Archive 크기",
                "population_size": "Population 크기",
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.caption("Hypervolume이 클수록 Pareto front를 넓고 좋게 덮었다는 뜻이고, Sparsity는 front 위 점 간격이 얼마나 벌어져 있는지 보여줍니다.")

        metric_cols = st.columns(2)
        metric_cols[0].altair_chart(metric_curve(experiment_result.metrics, "hypervolume", "Hypervolume"), use_container_width=True)
        metric_cols[0].caption("세대별 Hypervolume: PG-MORL이 빈 영역을 잘 메우면 더 빠르게 커집니다.")
        metric_cols[1].altair_chart(metric_curve(experiment_result.metrics, "sparsity", "Sparsity"), use_container_width=True)
        metric_cols[1].caption("세대별 Sparsity: front를 고르게 채우는지 볼 수 있습니다.")

        lower_cols = st.columns([1.0, 1.1], gap="large")
        pg_tasks = experiment_result.tasks.loc[experiment_result.tasks["method"] == "PG-MORL Toy"].copy()
        lower_cols[0].altair_chart(weight_history_chart(pg_tasks, experiment_config), use_container_width=True)
        lower_cols[0].caption("선택된 작업의 평균 weight: 세대가 지나며 PG-MORL이 어느 KPI 비중을 자주 선택하는지 보여줍니다.")

        final_method = lower_cols[1].selectbox(
            "최종 Pareto 스냅샷을 볼 방법",
            ["PG-MORL Toy", "RA Sweep", "Random"],
            index=0,
            key="exp_final_method",
        )
        final_archive = experiment_result.archive.loc[
            (experiment_result.archive["method"] == final_method)
            & (experiment_result.archive["snapshot_generation"] == experiment_config.generations)
        ].copy()
        final_overlays = [("Pareto Archive", final_archive)]
        panels = lower_cols[1].columns(3)
        panels[0].altair_chart(
            build_projection_chart(experiment_config, experiment_result.true_front, final_overlays, 0, 1, "빠른 도착 vs 배터리"),
            use_container_width=True,
        )
        panels[1].altair_chart(
            build_projection_chart(experiment_config, experiment_result.true_front, final_overlays, 0, 2, "빠른 도착 vs 안전"),
            use_container_width=True,
        )
        panels[2].altair_chart(
            build_projection_chart(experiment_config, experiment_result.true_front, final_overlays, 1, 2, "배터리 vs 안전"),
            use_container_width=True,
        )
        lower_cols[1].caption("회색 선은 이 toy 문제의 참 Pareto front, 보라 점은 각 방법이 실제로 확보한 archive입니다.")

with tab4:
    st.subheader("한 세대 안에서 predictor와 task selection이 하는 일")
    if experiment_result is None:
        st.warning("먼저 3번 탭에서 실험을 실행해 주세요.")
    else:
        inspect_cols = st.columns(3)
        inspect_method = inspect_cols[0].selectbox("방법 선택", ["PG-MORL Toy", "RA Sweep", "Random"], index=0, key="inspect_method")
        inspect_generation = inspect_cols[1].slider(
            "볼 세대",
            0,
            int(experiment_config.generations),
            min(1, int(experiment_config.generations)),
            1,
            key="inspect_generation",
        )
        inspect_limit = inspect_cols[2].slider("후보 표 최대 행 수", 5, 25, 10, 1, key="inspect_limit")

        generation_archive = experiment_result.archive.loc[
            (experiment_result.archive["method"] == inspect_method)
            & (experiment_result.archive["snapshot_generation"] == inspect_generation)
        ].copy()
        generation_population = experiment_result.population.loc[
            (experiment_result.population["method"] == inspect_method)
            & (experiment_result.population["snapshot_generation"] == inspect_generation)
        ].copy()
        generation_tasks = experiment_result.tasks.loc[
            (experiment_result.tasks["method"] == inspect_method)
            & (experiment_result.tasks["generation"] == inspect_generation)
        ].copy()

        selected_points = generation_tasks.loc[:, ["parent_mu", *[f"actual_eval_{i + 1}" for i in range(experiment_config.num_objectives)]]].copy()
        if not selected_points.empty:
            selected_points = selected_points.rename(
                columns={"parent_mu": "mu", **{f"actual_eval_{i + 1}": f"eval_{i + 1}" for i in range(experiment_config.num_objectives)}}
            )

        overlays = [("Pareto Archive", generation_archive), ("현재 Population", generation_population), ("실제 선택된 작업", selected_points)]
        panel_cols = st.columns(3)
        panel_cols[0].altair_chart(
            build_projection_chart(experiment_config, experiment_result.true_front, overlays, 0, 1, "빠른 도착 vs 배터리"),
            use_container_width=True,
        )
        panel_cols[1].altair_chart(
            build_projection_chart(experiment_config, experiment_result.true_front, overlays, 0, 2, "빠른 도착 vs 안전"),
            use_container_width=True,
        )
        panel_cols[2].altair_chart(
            build_projection_chart(experiment_config, experiment_result.true_front, overlays, 1, 2, "배터리 vs 안전"),
            use_container_width=True,
        )
        st.caption("초록 점은 현재 population, 보라 점은 지금까지 누적 Pareto archive, 빨간 점은 이 세대에서 실제로 학습해서 얻은 offspring입니다.")

        if not generation_tasks.empty:
            task_table = generation_tasks.copy()
            task_table["weight"] = task_table.apply(
                lambda row: normalized_weight_caption(np.array([row[col] for col in weight_cols(experiment_config)]), experiment_config),
                axis=1,
            )
            show_cols = ["slot", "stage", "parent_policy_id", "parent_mu", "weight", "score", "initial_score"]
            st.dataframe(
                task_table.loc[:, [col for col in show_cols if col in task_table.columns]].rename(
                    columns={
                        "slot": "슬롯",
                        "stage": "단계",
                        "parent_policy_id": "부모 정책",
                        "parent_mu": "부모 속도 μ",
                        "weight": "선호도",
                        "score": "선택 후 점수",
                        "initial_score": "단독 후보 점수",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("표 해석: `단독 후보 점수`는 그 후보만 추가했을 때의 가치, `선택 후 점수`는 greedy selection 과정에서 실제로 채택될 때의 점수입니다.")

        if inspect_method == "PG-MORL Toy" and inspect_generation > 0:
            candidates = experiment_result.predictor_curves.loc[
                (experiment_result.predictor_curves["method"] == inspect_method)
                & (experiment_result.predictor_curves["generation"] == inspect_generation)
            ].copy()
            if not candidates.empty:
                selected_keys = {
                    (
                        int(row["parent_policy_id"]),
                        *tuple(round(float(row[f"weight_{index + 1}"]), 6) for index in range(experiment_config.num_objectives)),
                    )
                    for _, row in generation_tasks.iterrows()
                }
                candidates["selected"] = candidates.apply(
                    lambda row: (
                        int(row["policy_id"]),
                        *tuple(round(float(row[f"weight_{index + 1}"]), 6) for index in range(experiment_config.num_objectives)),
                    )
                    in selected_keys,
                    axis=1,
                )
                candidates["weight"] = candidates.apply(
                    lambda row: normalized_weight_caption(np.array([row[f"weight_{index + 1}"] for index in range(experiment_config.num_objectives)]), experiment_config),
                    axis=1,
                )
                candidate_table = candidates.sort_values("initial_score", ascending=False).head(int(inspect_limit)).loc[:, ["policy_id", "mu", "weight", "initial_score", "selected"]]
                st.dataframe(
                    candidate_table.rename(
                        columns={
                            "policy_id": "부모 정책",
                            "mu": "부모 속도 μ",
                            "weight": "검토한 선호도",
                            "initial_score": "예측 점수",
                            "selected": "실제 선택 여부",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                bar_data = candidate_table.rename(
                    columns={
                        "policy_id": "부모 정책",
                        "weight": "검토한 선호도",
                        "initial_score": "예측 점수",
                        "selected": "실제 선택 여부",
                    }
                ).assign(rank=np.arange(len(candidate_table)))
                bar = (
                    alt.Chart(bar_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("rank:O", title="상위 후보 순서"),
                        y=alt.Y("예측 점수:Q", title="예측 점수"),
                        color=alt.Color("실제 선택 여부:N", title="", scale=alt.Scale(domain=[True, False], range=["#d62728", "#9aa5b1"])),
                        tooltip=["부모 정책:N", "검토한 선호도:N", alt.Tooltip("예측 점수:Q", format=".3f"), "실제 선택 여부:N"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(bar, use_container_width=True)
                st.caption("빨간 막대는 predictor가 좋다고 본 후보 중 실제로 선택된 작업입니다. 이것이 PG-MORL의 핵심인 prediction-guided task selection입니다.")
            else:
                st.info("이 세대에는 predictor 후보가 없습니다.")
        elif inspect_generation == 0:
            st.info("0세대는 warm-up 단계입니다. predictor가 task selection에 개입하기 전, 초기 Pareto seed를 만드는 단계로 보면 됩니다.")
        else:
            st.info("이 방법은 predictor를 쓰지 않는 baseline이라 후보 점수 표가 없습니다.")

with tab5:
    st.subheader("논문과 toy 예제의 대응")
    mapping_source = experiment_result.mapping if experiment_result is not None else paper_mapping()
    st.dataframe(mapping_source, use_container_width=True, hide_index=True)
    st.caption("논문에서 복잡해 보이는 요소들을 속도 하나의 toy 문제로 줄였지만, 핵심 루프는 그대로 유지했습니다.")

    st.markdown("**수식 대응**")
    st.latex(r"\pi_{\theta}(a) = \mathcal{N}(a \mid \mu=\theta, \sigma^2)")
    st.latex(r"r_1(a) = 50 - (a - 8)^2,\quad r_2(a) = 50 - (a - 2)^2,\quad r_3(a) = 50 - (a - 5)^2")
    st.latex(r"J_w(\mu) = w_1 \mathbb{E}[r_1] + w_2 \mathbb{E}[r_2] + w_3 \mathbb{E}[r_3], \quad \sum_i w_i = 1")
    st.latex(r"\Delta_j(w_j) \approx A_j \frac{\exp(a_j(w_j-b_j)) - 1}{\exp(a_j(w_j-b_j)) + 1} + c_j")
    st.caption("마지막 식은 논문과 같은 형태의 hyperbolic predictor입니다. 각 objective 축에서 weight가 바뀔 때 얼마나 개선될지 곡선으로 근사합니다.")

    st.markdown("**쉬운 의사코드**")
    st.code(
        """1. 서로 다른 선호도(weight)로 warm-up 학습을 해서 초기 정책들을 만든다.
2. 얻어진 정책들을 Pareto archive와 performance buffer에 저장한다.
3. 각 현재 정책에 대해:
   - 여러 weight 후보를 넣어 본다.
   - predictor가 "이 weight로 더 학습하면 어느 KPI가 얼마나 좋아질지" 예측한다.
4. 예측 결과를 이용해 hypervolume이 잘 늘고 front가 고르게 채워질 작업을 고른다.
5. 선택된 작업만 실제 policy gradient로 학습한다.
6. 새 정책을 archive에 추가하고, 다음 세대로 넘어간다.""",
        language="text",
    )
    st.info(
        "논문을 읽을 때는 이렇게 연결해서 보시면 됩니다. "
        "`worker`는 2번 탭의 단일 policy gradient, `predictor`는 4번 탭의 후보 점수표, "
        "`task selection`은 predictor 후보 중 실제로 뽑힌 빨간 막대, `archive`는 3번과 4번 탭의 보라 점입니다."
    )
