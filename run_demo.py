from toy_pgmorl.core import ToyConfig, run_experiment


def main():
    config = ToyConfig(
        gradient_mode="reinforce",
        warmup_steps=35,
        task_steps=12,
        generations=12,
        learning_rate=0.035,
        batch_size=128,
        sparsity_coef=0.18,
    )
    result = run_experiment(config)
    print(result.method_summary.to_string(index=False))


if __name__ == "__main__":
    main()
