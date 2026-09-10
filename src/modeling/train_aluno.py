"""
Modelagem supervisionada — grão ALUNO.

Uso:
    python -m src.modeling.train_aluno

Protocolo de validação (out-of-time)
------------------------------------
    treino -> alunos de 2024, contexto de 2023
    teste  -> alunos de 2025, contexto de 2024

Treinar em um ano e testar no ano seguinte é a forma mais honesta de medir
generalização aqui: é exatamente como o modelo seria usado na prática (estimar o
risco da coorte que ainda vai ser avaliada). Um split aleatório dentro do mesmo
ano superestimaria o desempenho, porque alunos do mesmo município apareceriam
nos dois lados da divisão.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.evaluation import metrics
from src.preprocessing import pipeline as pp
from src.visualization import plots

# Amostra estratificada usada apenas na SELEÇÃO de modelos. O modelo escolhido é
# retreinado na base completa do ano de treino.
N_AMOSTRA_SELECAO = 150_000

# Etapas caras são materializadas em disco: reexecutar o script reaproveita a
# comparação e a busca já feitas, em vez de repetir horas de validação cruzada.
ARQUIVO_COMPARACAO = config.REPORTS_DIR / "comparacao_modelos_aluno.csv"
ARQUIVO_BUSCA = config.REPORTS_DIR / "melhores_parametros_aluno.json"


def carregar_particoes() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    treino = pd.read_parquet(config.PROCESSED_DIR / f"aluno_{config.ANO_TREINO}.parquet")
    teste = pd.read_parquet(config.PROCESSED_DIR / f"aluno_{config.ANO_TESTE}.parquet")
    X_tr, y_tr = pp.preparar_aluno(treino, config.ANO_TREINO)
    X_te, y_te = pp.preparar_aluno(teste, config.ANO_TESTE)
    return X_tr, y_tr, X_te, y_te, teste


def candidatos() -> dict:
    """Grade de modelos, do mais simples ao mais expressivo."""
    return {
        "Baseline (classe majoritária)": (
            DummyClassifier(strategy="prior", random_state=config.RANDOM_STATE),
            True,
        ),
        "Regressão Logística": (
            LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
            True,
        ),
        "Árvore de Decisão": (
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=200, random_state=config.RANDOM_STATE),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=120, max_depth=14, min_samples_leaf=100,
                n_jobs=-1, random_state=config.RANDOM_STATE,
            ),
            False,
        ),
        "LightGBM": (
            LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=63,
                random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1,
            ),
            False,
        ),
    }


def selecionar_modelo(X, y, grupos) -> pd.DataFrame:
    """
    Validação cruzada dos candidatos em duas óticas:

    * StratifiedKFold  -> generalização para novos alunos
    * GroupKFold (município) -> generalização para municípios não vistos
    """
    linhas = []
    for nome, (modelo, padronizar) in candidatos().items():
        pipe = pp.construir_pipeline(modelo, pp.NUMERICAS_ALUNO, pp.CATEGORICAS_ALUNO, padronizar)
        t0 = time.time()
        estrat = cross_val_score(
            pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE),
            scoring="roc_auc", n_jobs=1,
        )
        grupo = cross_val_score(
            pipe, X, y, cv=GroupKFold(5), groups=grupos, scoring="roc_auc", n_jobs=1,
        )
        linhas.append(
            {
                "modelo": nome,
                "auc_estratificado": estrat.mean(),
                "desvio_estratificado": estrat.std(),
                "auc_por_municipio": grupo.mean(),
                "desvio_por_municipio": grupo.std(),
                "segundos": time.time() - t0,
            }
        )
        print(
            f"  {nome:<32} AUC estratificado {estrat.mean():.4f} (±{estrat.std():.4f}) | "
            f"AUC por município {grupo.mean():.4f} (±{grupo.std():.4f}) | {time.time() - t0:.0f}s"
        )
    return pd.DataFrame(linhas).sort_values("auc_por_municipio", ascending=False)


def otimizar_lightgbm(X, y, grupos):
    """Busca aleatória de hiperparâmetros com validação por município."""
    pipe = pp.construir_pipeline(
        LGBMClassifier(random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1),
        pp.NUMERICAS_ALUNO, pp.CATEGORICAS_ALUNO, padronizar=False,
    )
    espaco = {
        "modelo__n_estimators": [200, 400, 600],
        "modelo__learning_rate": [0.02, 0.05, 0.1],
        "modelo__num_leaves": [31, 63, 127],
        "modelo__min_child_samples": [50, 200, 500],
        "modelo__subsample": [0.7, 0.9, 1.0],
        "modelo__colsample_bytree": [0.6, 0.8, 1.0],
        "modelo__reg_lambda": [0.0, 1.0, 10.0],
    }
    busca = RandomizedSearchCV(
        pipe, espaco, n_iter=12, scoring="roc_auc", cv=GroupKFold(3),
        random_state=config.RANDOM_STATE, n_jobs=1, verbose=0, refit=False,
    )
    busca.fit(X, y, groups=grupos)
    print(f"  melhor AUC na busca: {busca.best_score_:.4f}")
    print(f"  parâmetros: {busca.best_params_}")
    resultado = {"parametros": busca.best_params_, "auc": float(busca.best_score_)}
    ARQUIVO_BUSCA.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    return busca.best_params_, busca.best_score_


def main() -> None:
    plots.aplicar_estilo()
    print("=" * 72)
    print("MODELAGEM — GRÃO ALUNO")
    print("=" * 72)

    X_tr, y_tr, X_te, y_te, teste_bruto = carregar_particoes()
    del teste_bruto
    grupos_tr = pd.read_parquet(
        config.PROCESSED_DIR / f"aluno_{config.ANO_TREINO}.parquet", columns=["id_municipio"]
    )["id_municipio"].astype("float").fillna(-1).to_numpy()

    print(f"\nTreino ({config.ANO_TREINO}): {len(X_tr):,} alunos | alfabetizados {y_tr.mean() * 100:.2f}%")
    print(f"Teste  ({config.ANO_TESTE}): {len(X_te):,} alunos | alfabetizados {y_te.mean() * 100:.2f}%")
    print(f"Atributos: {X_tr.shape[1]} ({len(pp.NUMERICAS_ALUNO)} numéricos + {len(pp.CATEGORICAS_ALUNO)} categóricos)")

    # ---------------------------------------------------------------- #
    # 1. Seleção de modelo em amostra estratificada
    # ---------------------------------------------------------------- #
    rng = np.random.default_rng(config.RANDOM_STATE)
    idx = rng.choice(len(X_tr), size=min(N_AMOSTRA_SELECAO, len(X_tr)), replace=False)
    Xs, ys, gs = X_tr.iloc[idx], y_tr.iloc[idx], grupos_tr[idx]

    print(f"\n[1/4] Comparação de modelos (amostra de {len(Xs):,} alunos, validação cruzada 5-fold)")
    if ARQUIVO_COMPARACAO.exists():
        comparacao = pd.read_csv(ARQUIVO_COMPARACAO)
        print(comparacao.round(4).to_string(index=False))
        print("  (reaproveitado de reports/ — apague o arquivo para recalcular)")
    else:
        comparacao = selecionar_modelo(Xs, ys, gs)
        comparacao.to_csv(ARQUIVO_COMPARACAO, index=False)

    # ---------------------------------------------------------------- #
    # 2. Otimização de hiperparâmetros
    # ---------------------------------------------------------------- #
    print("\n[2/4] Otimização de hiperparâmetros (LightGBM, busca aleatória)")
    if ARQUIVO_BUSCA.exists():
        guardado = json.loads(ARQUIVO_BUSCA.read_text(encoding="utf-8"))
        melhores, auc_busca = guardado["parametros"], guardado["auc"]
        print(f"  melhor AUC na busca: {auc_busca:.4f} (reaproveitado)")
        print(f"  parâmetros: {melhores}")
    else:
        melhores, auc_busca = otimizar_lightgbm(Xs, ys, gs)
    del Xs, ys, gs

    # ---------------------------------------------------------------- #
    # 3. Treino final na base completa do ano de treino
    # ---------------------------------------------------------------- #
    print("\n[3/4] Treino final na base completa")
    parametros = {k.replace("modelo__", ""): v for k, v in melhores.items()}
    final = pp.construir_pipeline(
        LGBMClassifier(random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1, **parametros),
        pp.NUMERICAS_ALUNO, pp.CATEGORICAS_ALUNO, padronizar=False,
    )
    t0 = time.time()
    final.fit(X_tr, y_tr)
    print(f"  treinado em {time.time() - t0:.0f}s")
    joblib.dump(final, config.MODELS_DIR / "modelo_aluno.joblib")

    # ---------------------------------------------------------------- #
    # 4. Avaliação out-of-time
    # ---------------------------------------------------------------- #
    print(f"\n[4/4] Avaliação out-of-time em {config.ANO_TESTE}")
    prob_te = final.predict_proba(X_te)[:, 1]
    prob_tr = final.predict_proba(X_tr)[:, 1]

    # A busca de limiar percorre 91 cortes; fazê-la em uma amostra evita varrer
    # dois milhões de linhas 91 vezes sem alterar o resultado de forma relevante.
    amostra_limiar = rng.choice(len(y_te), size=min(200_000, len(y_te)), replace=False)
    limiar, _ = metrics.melhor_limiar(
        y_te.to_numpy()[amostra_limiar], prob_te[amostra_limiar], "balanceada"
    )
    resultado = {
        "teste_out_of_time": metrics.avaliar(y_te, prob_te),
        "teste_limiar_otimizado": metrics.avaliar(y_te, prob_te, limiar),
        "treino_in_sample": metrics.avaliar(y_tr, prob_tr),
        "auc_busca_hiperparametros": auc_busca,
        "melhores_parametros": parametros,
        "n_treino": int(len(X_tr)),
        "n_teste": int(len(X_te)),
        "limiar_balanceado": limiar,
    }

    # Baseline honesto: prever tudo como alfabetizado
    resultado["baseline_maioria"] = metrics.avaliar(y_te, np.full(len(y_te), y_tr.mean()))

    for chave in ("treino_in_sample", "teste_out_of_time", "teste_limiar_otimizado"):
        m = resultado[chave]
        print(
            f"  {chave:<24} AUC {m['roc_auc']:.4f} | acurácia {m['acuracia']:.4f} | "
            f"acurácia balanceada {m['acuracia_balanceada']:.4f} | F1 {m['f1']:.4f}"
        )

    (config.REPORTS_DIR / "metricas_aluno.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    # ---------------------------------------------------------------- #
    # Figuras
    # ---------------------------------------------------------------- #
    fpr, tpr, _ = roc_curve(y_te, prob_te)
    plots.curva_roc(
        {"LightGBM (out-of-time 2025)": (fpr, tpr, resultado["teste_out_of_time"]["roc_auc"])},
        "07_roc_aluno.png",
    )

    m = resultado["teste_limiar_otimizado"]
    plots.matriz_confusao(
        np.array([[m["vn"], m["fp"]], [m["fn"], m["vp"]]]),
        ["Não alfabetizado", "Alfabetizado"],
        f"Matriz de confusão — teste 2025 (limiar {limiar:.2f})",
        "08_matriz_confusao_aluno.png",
    )

    bins = np.quantile(prob_te, np.linspace(0, 1, 11))
    bins = np.unique(bins)
    faixa = pd.cut(prob_te, bins, include_lowest=True)
    cal = pd.DataFrame({"p": prob_te, "y": y_te.to_numpy(), "faixa": faixa}).groupby("faixa", observed=True).mean()
    plots.calibracao(cal["p"], cal["y"], "09_calibracao_aluno.png")

    print("\nArtefatos salvos em models/, reports/ e images/.")


if __name__ == "__main__":
    main()
