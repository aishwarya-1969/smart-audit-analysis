"""End-to-end audit pipeline: clean -> features -> BiLSTM/Attention/IBPNN -> SHAP -> scores -> recommendations."""
import json
import numpy as np
import pandas as pd
from . import scoring as sc
from .explain import explain
from .historical import compare, historical_recs
from .features import FEATURES, W, build
from .model import predict, train
from .preprocess import clean

def _records(df):
    return json.loads(df.to_json(orient="records"))

def run(raw, history=None):
    df, log, has_approver = clean(raw)
    if len(df) < 10:
        raise ValueError("Need at least 10 valid rows.")
    X = build(df)
    rule = X @ W
    Q = float(np.quantile(rule, .995))
    Q = Q if Q > 0 else 1.0
    target = np.clip(rule / Q, 0, 1)                                 # weak labels from the expert rule score
    if df.label.sum() > 0:                                           # use real labels when the CSV has them
        target = np.where(df.label.to_numpy() > 0, 1.0, target)
    model = train(X, target)

    def f(Z):                                                        # blended risk function (also explained by SHAP)
        Z = np.asarray(Z, dtype=float)
        return .5 * np.clip(Z @ W / Q, 0, 1) + .5 * predict(model, Z)[0]

    risk = f(X)
    attention = predict(model, X)[1].mean(0)
    df["risk"] = risk
    thr = float(max(np.quantile(risk, .95), .4))
    flagged = np.where(risk >= thr)[0]
    top = flagged[np.argsort(-risk[flagged])][:1000]
    df["ty"] = np.array(FEATURES)[(X * W).argmax(1)]
    shap_rows = explain(f, X, top[:40], X.mean(0))

    vend, emp = sc.vendors(df, thr), sc.employees(df, thr)
    comp, cs = sc.compliance(df, X, has_approver)
    total, exposure = float(df.amount.sum()), float(df.amount[risk >= thr].sum())
    hc, hs = sc.health(log, cs, vend, emp, exposure, total)
    top5 = float(df.amount.iloc[top[:5]].sum())
    rows = [{"id": str(df.id.iloc[i]), "d": None if pd.isna(df.date.iloc[i]) else df.date.iloc[i].strftime("%Y-%m-%d"),
             "v": df.vendor.iloc[i], "e": df.employee.iloc[i], "a": float(df.amount.iloc[i]),
             "r": float(risk[i]), "ty": df.ty.iloc[i]} for i in top]
    hist = compare(df, history)
    return {"n": len(df), "L": log, "comp": comp, "cs": cs, "tot": total, "exp": exposure, "hc": hc, "hs": hs,
            "recs": sc.recommendations(len(flagged), exposure, top5, comp, vend, emp) + historical_recs(hist), "hist": hist,
            "vend": _records(vend), "emp": _records(emp), "A": attention.tolist(),
            "hist": np.histogram(risk, bins=10, range=(0, 1))[0].tolist(),
            "rows": rows, "X": X[top].tolist(), "fl": list(range(len(rows))), "shap": shap_rows,
            "nfl": int(len(flagged)), "thr": thr}
