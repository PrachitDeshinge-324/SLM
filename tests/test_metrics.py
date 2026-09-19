import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.metrics import compute_metrics

def test_compute_metrics():
    # 1. Standard correct case
    extracted = ["72", "72", "72", "73", "72"]
    truth = "72"
    res = compute_metrics(extracted, truth)
    assert res["majority_answer"] == "72"
    assert res["confidence"] == 0.8
    assert res["correctness"] == True
    assert res["oracle_correctness"] == True
    assert res["failure_rate"] == 0.0
    
    # 2. Case with extraction failures
    extracted = ["A", None, "A", "B", None]
    truth = "A"
    res = compute_metrics(extracted, truth)
    assert res["majority_answer"] == "A"
    assert res["confidence"] == 2 / 3 # 2 out of 3 valid samples
    assert res["correctness"] == True
    assert res["oracle_correctness"] == True
    assert res["failure_rate"] == 0.4 # 2 out of 5 failed
    
    # 3. Incorrect case but oracle correct
    extracted = ["10", "10", "12"]
    truth = "12"
    res = compute_metrics(extracted, truth)
    assert res["majority_answer"] == "10"
    assert res["confidence"] == 2 / 3
    assert res["correctness"] == False
    assert res["oracle_correctness"] == True
    assert res["failure_rate"] == 0.0
    
    # 4. Total failure case
    extracted = [None, None]
    truth = "C"
    res = compute_metrics(extracted, truth)
    assert res["majority_answer"] == None
    assert res["confidence"] == 0.0
    assert res["correctness"] == False
    assert res["oracle_correctness"] == False
    assert res["failure_rate"] == 1.0

if __name__ == "__main__":
    test_compute_metrics()
    print("All metrics tests passed!")
