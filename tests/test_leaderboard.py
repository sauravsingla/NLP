from nlp_reference.leaderboard import benchmark_model, model_registry, render_markdown


def test_model_registry_contains_reference_baselines() -> None:
    assert set(model_registry()) == {
        "logistic_regression",
        "linear_svm",
        "complement_naive_bayes",
    }


def test_benchmark_and_markdown_rendering() -> None:
    train_texts = [
        "excellent film and acting",
        "wonderful story and cast",
        "terrible movie and boring plot",
        "awful acting and weak script",
        "great direction and music",
        "bad pacing and poor dialogue",
    ]
    train_labels = [1, 1, 0, 0, 1, 0]
    test_texts = ["excellent story", "awful boring movie"]
    test_labels = [1, 0]
    model = model_registry()["logistic_regression"](42)

    result = benchmark_model(
        dataset_name="fixture",
        model_name="logistic_regression",
        model=model,
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
    )

    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.examples_per_second > 0
    markdown = render_markdown([result])
    assert "# NLP Benchmark Leaderboard" in markdown
    assert "| fixture | logistic_regression | 1.0000 |" in markdown
