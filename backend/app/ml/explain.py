"""Stage 6: SHAP explainable AI for the highest-risk transactions."""
import numpy as np
import shap

def explain(f, X, idx, baseline):
    if len(idx) == 0:
        return []
    ex = shap.KernelExplainer(f, baseline.reshape(1, -1))
    sv = np.asarray(ex.shap_values(X[idx], nsamples=512, silent=True)).reshape(len(idx), -1)
    base = float(np.ravel(ex.expected_value)[0])
    out = f(X[idx])
    return [{"phi": sv[k].tolist(), "base": base, "out": float(out[k])} for k in range(len(idx))]
