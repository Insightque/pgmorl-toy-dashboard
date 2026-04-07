# PG-MORL Toy Dashboard

PG-MORL 논문의 핵심 루프를 가장 작은 예제로 이해하기 위한 Streamlit 대시보드입니다.

이 toy는 복잡한 continuous-control 환경 대신, `1-step scalar action` 환경을 사용합니다. 덕분에 다음 관계를 수식과 그래프에서 바로 볼 수 있습니다.

- `policy`: 평균 action `μ`를 만드는 함수
- `action`: policy에서 실제로 샘플된 값 `a`
- `weight`: reward preference. action을 바로 정하는 값이 아니라 policy update 방향을 정하는 값
- `offspring policy`: 현재 policy를 선택한 weight로 몇 step 학습한 뒤 얻는 새 policy

## 왜 이렇게 단순화했나

논문과 원본 구현의 핵심은 다음입니다.

1. 여러 objective 사이의 trade-off를 가진 policy 집합을 찾는다.
2. 모든 `(policy, weight)`를 다 학습하지 않고, predictor로 “어디를 학습하면 Pareto가 가장 좋아질지”를 고른다.
3. hypervolume / sparsity 기준으로 다음 task를 선택한다.

이 toy는 그 구조를 유지하되, PPO와 Mujoco를 제거하고 `policy mean μ` 하나만 학습합니다.

## 수식

정책:

```math
\pi_\theta(a) = \mathcal{N}(a; \mu=\theta, \sigma^2)
```

각 objective reward:

```math
r_i(a) = C - (a - t_i)^2
```

정책의 기대 objective:

```math
F_i(\theta) = \mathbb{E}[r_i(a)] = C - ((\theta - t_i)^2 + \sigma^2)
```

가중합 objective:

```math
J(\theta, w) = \sum_i w_i F_i(\theta)
```

이 toy에서 해당 weight의 이론적 최적 policy mean은:

```math
\theta^\*(w) = \sum_i w_i t_i
```

즉, `weight -> 즉시 action`이 아니라:

```text
weight + current policy -> gradient update -> new policy -> sampled action
```

## UI 해석

- Sidebar의 학습 설정:
  실험을 다시 실행합니다.
- `method / generation / slot` selector:
  이미 실행된 로그를 어떤 관점으로 볼지 정합니다.
- `Action inspector a` slider:
  reward curve를 이해하기 위한 관찰용 슬라이더입니다. 학습 결과는 바꾸지 않습니다.
- `state` slider가 없는 이유:
  이 toy는 state를 고정해서 policy / action / weight 관계만 집중해서 보도록 만들었습니다.

## 실행

```bash
python3 run_demo.py
streamlit run app.py
```

대시보드는 기본적으로 `http://localhost:8501` 에서 열립니다.

## 파일 구성

- [app.py](/Users/insightque/Documents/Harness Agent/app.py): Streamlit dashboard
- [toy_pgmorl/core.py](/Users/insightque/Documents/Harness Agent/toy_pgmorl/core.py): toy environment, predictor, PG-MORL loop, baselines
- [run_demo.py](/Users/insightque/Documents/Harness Agent/run_demo.py): headless 실행용 요약 스크립트

## 참고 자료

- [PG-MORL Paper](https://people.csail.mit.edu/jiex/papers/PGMORL/paper.pdf)
- [Original PGMORL Repo](https://github.com/mit-gfx/PGMORL)
- [morl-baselines PGMORL](https://github.com/LucasAlegre/morl-baselines)
