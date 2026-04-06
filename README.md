# PG-MORL Toy Dashboard

이 프로젝트는 ICML 2020 논문 "Prediction-Guided Multi-Objective Reinforcement Learning for Continuous Robot Control"의 핵심 루프를 아주 작은 toy 문제로 다시 만든 설명용 데모입니다.

핵심 목표는 아래 3가지를 한 번에 보는 것입니다.
- 논문의 warm-up -> predictor -> task selection -> MOPG -> Pareto archive 흐름
- weighted-sum policy gradient와 Pareto front의 관계
- prediction-guided selection이 RA-like sweep / random selection과 어떻게 다른지

## Toy problem

- 물리적 비유: 직선 복도를 달리는 배송 카트의 평균 순항 속도 `μ`를 조절
- 정책: `pi_theta(a) = N(mu=theta, sigma^2)`
- 1-step continuous action bandit
- 보상 벡터:
  - `r_1(a) = 50 - (a - 8)^2`  : 빠른 도착
  - `r_2(a) = 50 - (a - 2)^2`  : 배터리 효율
  - `r_3(a) = 50 - (a - 5)^2`  : 안전성
- 선호도 벡터 `w = (w_1, w_2, w_3)`가 주어지면 weighted-sum worker는 `J_w = w_1 E[r_1] + w_2 E[r_2] + w_3 E[r_3]`를 최적화

MuJoCo/PPO 전체를 쓰는 대신, 논문의 핵심 구조만 남기고 환경을 단순화해서 수식과 그래프를 바로 연결했습니다.

## Files

- `app.py`: Streamlit dashboard
- `toy_pgmorl/core.py`: toy environment, metrics, predictor, PG-MORL loop, baselines
- `run_demo.py`: 대시보드 없이 기본 설정으로 한 번 실행해서 최종 지표를 출력하는 스크립트

## Run

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

브라우저에서 열리면 아래를 순서대로 보면 됩니다.
- `1. 문제 이해`
  배송 카트 속도 예제로 RL과 MORL의 차이를 이해
- `2. 직접 한 번 학습`
  정책 평균 `μ`와 선호도 `w`를 직접 바꿔가며 policy gradient 업데이트 확인
- `3. PG-MORL 역할 보기`
  `실험 실행` 버튼으로 PG-MORL, 고정 가중치 스윕, 무작위 선택 비교
- `4. 한 세대 해부`
  predictor가 본 후보와 실제 선택된 작업, Pareto snapshot 비교
- `5. 논문 대응`
  토이 예제와 원 논문 알고리즘의 대응표와 수식 정리

## Headless check

```bash
python3 run_demo.py
```

기본 비교 설정은 데모가 너무 무겁지 않도록 조금 가볍게 잡았습니다.
- `population_size = 5`
- `warmup_steps = 12`
- `task_steps = 6`
- `generations = 4`
- `weight_resolution = 4`
- `batch_size = 64`
- `gradient_mode = analytic`

더 정교한 곡선을 보고 싶으면 대시보드에서 세대 수, 샘플 수, weight 해상도를 올리면 됩니다.

## References

- Paper: https://people.csail.mit.edu/jiex/papers/PGMORL/paper.pdf
- Original code: https://github.com/mit-gfx/PGMORL
- morl-baselines implementation: https://github.com/LucasAlegre/morl-baselines
