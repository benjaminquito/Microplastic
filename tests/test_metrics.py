from microscan_ai.metrics import classification_metrics, confusion_matrix


def test_confusion_matrix_orientation() -> None:
    assert confusion_matrix([0, 0, 1], [0, 1, 1], 2) == [[1, 1], [0, 1]]


def test_classification_metrics() -> None:
    result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"])
    assert result["accuracy"] == 0.75
    assert result["per_class"]["a"]["support"] == 2
