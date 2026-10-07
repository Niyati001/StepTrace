import pytest

from diagnose.evaluate import score

TRUTH = ["HEALTHY", "HEALTHY", "COMMUNICATION", "STRAGGLER", "DATA_STALL", "COMMUNICATION", "STRAGGLER"]
PRED = ["HEALTHY", "COMMUNICATION", "COMMUNICATION", "STRAGGLER", "HEALTHY", "COMMUNICATION", "COMMUNICATION"]


def test_score_by_hand():
    s = score(TRUTH, PRED)
    cm = s["confusion_matrix"]   # labels: HEALTHY, COMMUNICATION, STRAGGLER, DATA_STALL
    assert cm == [[1, 1, 0, 0], [0, 2, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0]]
    assert s["accuracy"] == pytest.approx(4 / 7)
    c = s["per_class"]["COMMUNICATION"]
    assert c["precision"] == pytest.approx(2 / 4) and c["recall"] == 1.0 and c["support"] == 2
    assert s["per_class"]["DATA_STALL"]["recall"] == 0.0 and s["per_class"]["DATA_STALL"]["precision"] == 0.0


def test_matches_sklearn_when_available():
    try:
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    except Exception:  # not installed, or blocked by local policy (e.g. Windows Application Control)
        pytest.skip("scikit-learn not importable here")
    s = score(TRUTH, PRED)
    labels = s["labels"]
    assert confusion_matrix(TRUTH, PRED, labels=labels).tolist() == s["confusion_matrix"]
    p, r, _, sup = precision_recall_fscore_support(TRUTH, PRED, labels=labels, zero_division=0)
    for i, lab in enumerate(labels):
        assert s["per_class"][lab]["precision"] == pytest.approx(p[i])
        assert s["per_class"][lab]["recall"] == pytest.approx(r[i])
        assert s["per_class"][lab]["support"] == sup[i]
