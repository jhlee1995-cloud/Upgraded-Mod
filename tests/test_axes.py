"""
tests/test_axes.py -- per-axis unit tests (bug isolation).

Each axis: clean baseline, fires on its target, quiet on non-targets. Known
input -> known output, so a regression points at the exact broken axis. No GPU.

Run: python -m pytest tests/test_axes.py -v   (or: python tests/test_axes.py)
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extract.synth_activations import generate
from extract.axis_registry import AxisRef, SINGLE_BATCH_AXES
from sklearn.metrics import roc_auc_score


def _auc(fn, ref, clean, dist):
    sc, sd = fn(clean, ref), fn(dist, ref)
    y = np.r_[np.zeros(len(sc)), np.ones(len(sd))]
    s = np.r_[sc, sd]
    return roc_auc_score(y, s) if len(np.unique(s)) > 1 else 0.5


def _setup():
    data, centers, lab = generate(n=1500, seed=0)
    ref = AxisRef(data["clean"], lab, n_classes=10)
    return data, ref


def test_deviation_fires_on_energy():
    data, ref = _setup()
    assert _auc(SINGLE_BATCH_AXES["DEVIATION"], ref, data["clean"], data["energy"]) > 0.95

def test_deviation_quiet_on_typeb():
    data, ref = _setup()
    # one-sided deviation must NOT fire on energy-normal type-b (cluster_b)
    auc = _auc(SINGLE_BATCH_AXES["DEVIATION"], ref, data["clean"], data["cluster_b"])
    assert auc < 0.65, f"DEVIATION leaked onto type-b (auc={auc:.2f})"

def test_consensus_fires_on_structure():
    data, ref = _setup()
    assert _auc(SINGLE_BATCH_AXES["CONSENSUS"], ref, data["clean"], data["structure"]) > 0.8

def test_consensus_fires_on_typea():
    data, ref = _setup()
    assert _auc(SINGLE_BATCH_AXES["CONSENSUS"], ref, data["clean"], data["cluster_a"]) > 0.8

def test_consensus_blind_to_typeb():
    data, ref = _setup()
    # the KNOWN structural blind spot — consensus must NOT catch type-b
    auc = _auc(SINGLE_BATCH_AXES["CONSENSUS"], ref, data["clean"], data["cluster_b"])
    assert auc < 0.65, f"CONSENSUS unexpectedly caught type-b (auc={auc:.2f})"

def test_cluster_distance_is_sole_typeb_signal():
    data, ref = _setup()
    # distance is the only axis with type-b signal (even if partial)
    d_auc = _auc(SINGLE_BATCH_AXES["CLUSTER_DISTANCE"], ref, data["clean"], data["cluster_b"])
    c_auc = _auc(SINGLE_BATCH_AXES["CONSENSUS"], ref, data["clean"], data["cluster_b"])
    assert d_auc > c_auc, "CLUSTER_DISTANCE should beat CONSENSUS on type-b"

def test_subnet_consensus_complementary():
    data, ref = _setup()
    # soft consensus should respond to confidence wobble (here: structure as proxy)
    assert _auc(SINGLE_BATCH_AXES["SUBNET_CONSENSUS"], ref, data["clean"], data["structure"]) > 0.8


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {repr(e)[:80]}")
    print(f"\n{passed}/{len(tests)} axis tests passed")
