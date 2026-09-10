"""Métricas e utilitários de avaliação."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def avaliar(y_true, y_prob, limiar: float = 0.5) -> dict:
    """Conjunto padrão de métricas de classificação binária."""
    y_pred = (np.asarray(y_prob) >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "acuracia": accuracy_score(y_true, y_pred),
        "acuracia_balanceada": balanced_accuracy_score(y_true, y_pred),
        "precisao": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "limiar": limiar,
        "vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp),
    }


def tabela_metricas(resultados: dict[str, dict]) -> pd.DataFrame:
    """Comparação lado a lado de vários modelos."""
    return (
        pd.DataFrame(resultados)
        .T.loc[:, ["roc_auc", "pr_auc", "acuracia", "acuracia_balanceada", "precisao", "recall", "f1", "brier"]]
        .sort_values("roc_auc", ascending=False)
        .round(4)
    )


def melhor_limiar(y_true, y_prob, criterio: str = "f1") -> tuple[float, float]:
    """
    Busca o limiar de decisão que maximiza o critério escolhido.

    Em política pública o custo de errar não é simétrico: deixar de identificar
    uma criança em risco (falso negativo) costuma custar mais caro do que
    incluir uma criança a mais em um programa de reforço (falso positivo).
    """
    grade = np.linspace(0.05, 0.95, 91)
    melhor, valor = 0.5, -1.0
    for t in grade:
        y_pred = (np.asarray(y_prob) >= t).astype(int)
        if criterio == "f1":
            v = f1_score(y_true, y_pred, zero_division=0)
        elif criterio == "balanceada":
            v = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(criterio)
        if v > valor:
            melhor, valor = float(t), float(v)
    return melhor, valor
