from __future__ import annotations

import pytest

from nlp_reference.datasets import available_datasets, load_text_dataset


def test_registry_contains_multiple_problem_datasets() -> None:
    assert available_datasets() == ("ag_news", "dbpedia_14", "imdb", "yelp_polarity")


def test_unknown_dataset_is_rejected_before_download() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        load_text_dataset("not-a-dataset")


@pytest.mark.parametrize("argument", [0, -1])
def test_invalid_sample_limits_are_rejected(argument: int) -> None:
    with pytest.raises(ValueError, match="max_train must be positive"):
        load_text_dataset("ag_news", max_train=argument)

    with pytest.raises(ValueError, match="max_test must be positive"):
        load_text_dataset("ag_news", max_test=argument)
