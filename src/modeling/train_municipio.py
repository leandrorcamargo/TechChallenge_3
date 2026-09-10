"""
Modelagem supervisionada — grão MUNICÍPIO.

Uso:
    python -m src.modeling.train_municipio

Responde à pergunta de negócio "como antecipar municípios que podem não atingir
as metas futuras?". O alvo é binário: o município ficou ABAIXO da meta de
alfabetização do ano.

POR QUE O ANO-BASE É 2025, E NÃO 2024
-------------------------------------
As metas municipais do Compromisso Nacional foram calculadas a partir da linha
de base de 2023 (correlação de 0,98 entre a meta de 2024 e a taxa de 2023).
Consequência: para o ano de 2024, a variável "esforço necessário"
(meta do ano − taxa do ano anterior) é quase uma constante — desvio-padrão de
6 p.p. e correlação de apenas 0,06 com o alvo. O exercício de 2024 é
estruturalmente degenerado: a meta e a feature saem da mesma medição.

Em 2025 a defasagem já é real (a meta vem de 2023, o contexto vem de 2024) e o
esforço necessário passa a discriminar (desvio-padrão de 14,8 p.p., correlação
de 0,37). Por isso a modelagem usa a safra de 2025, validada por:

* StratifiedKFold — generalização para novos municípios;
* GroupKFold por UF — generalização para estados inteiros não vistos no treino,
  o teste mais duro disponível com uma única safra utilizável.

O ano de 2024 é mantido como diagnóstico, e não como treino.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.evaluation import metrics
from src.preprocessing import features as feat
from src.preprocessing import pipeline as pp
from src.visualization import plots

ANO_BASE = 2025
ANO_PROJECAO = 2026


def candidatos() -> dict:
    return {
        "Baseline (classe majoritária)": (DummyClassifier(strategy="prior"), True),
        "Regressão Logística": (
            LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE), True,
        ),
        "Árvore de Decisão": (
            DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=config.RANDOM_STATE), False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=400, max_depth=10, min_samples_leaf=10,
                class_weight="balanced", n_jobs=-1, random_state=config.RANDOM_STATE,
            ), False,
        ),
        "LightGBM": (
            LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30,
                reg_lambda=10.0, random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1,
            ), False,
        ),
    }


def diagnostico_2024() -> dict:
    """
    Mostra por que a safra de 2024 não serve como treino: a meta do ano foi
    derivada da mesma medição usada como atributo.
    """
    d24 = pd.read_csv(config.PROCESSED_DIR / "municipio_2024.csv")
    d25 = pd.read_csv(config.PROCESSED_DIR / f"municipio_{ANO_BASE}.csv")
    return {
        "2024": {
            "corr_meta_com_taxa_anterior": float(d24["meta_ano"].corr(d24["mun_taxa_alfabetizacao_lag"])),
            "dp_esforco_necessario": float(d24["esforco_necessario"].std()),
            "auc_do_esforco_necessario": float(roc_auc_score(d24["nao_atingiu_meta"], d24["esforco_necessario"])),
        },
        "2025": {
            "corr_meta_com_taxa_anterior": float(d25["meta_ano"].corr(d25["mun_taxa_alfabetizacao_lag"])),
            "dp_esforco_necessario": float(d25["esforco_necessario"].std()),
            "auc_do_esforco_necessario": float(roc_auc_score(d25["nao_atingiu_meta"], d25["esforco_necessario"])),
        },
    }


def construir_base_projecao() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Base para projetar o risco de 2026: contexto observado em 2025 combinado com
    a meta que o município precisa cumprir em 2026.
    """
    base = feat.construir_dataset_municipio(ANO_BASE)
    contexto = feat.agregar_municipios(
        feat.loaders.carregar_alunos(ANO_BASE), apenas_rede_municipal=True
    ).add_suffix("_lag").rename(columns={"id_municipio_lag": "id_municipio"})

    proj = base[
        ["id_municipio", "regiao", "porte_municipio", "log_populacao", "pib_per_capita",
         "pct_vab_adm_publica", "nivel_alfabetizacao", "percentual_participacao",
         "meta_alfabetizacao_2030", f"meta_alfabetizacao_{ANO_PROJECAO}"]
    ].copy()
    proj = proj.merge(contexto, on="id_municipio", how="inner")
    proj["meta_ano"] = proj[f"meta_alfabetizacao_{ANO_PROJECAO}"]
    proj["gap_lag_meta_ano"] = proj["mun_taxa_alfabetizacao_lag"] - proj["meta_ano"]
    proj["gap_lag_meta_2030"] = proj["mun_taxa_alfabetizacao_lag"] - proj["meta_alfabetizacao_2030"]
    proj["esforco_necessario"] = proj["meta_ano"] - proj["mun_taxa_alfabetizacao_lag"]
    proj = proj[proj["meta_ano"].notna()]
    X = proj[pp.NUMERICAS_MUNICIPIO + pp.CATEGORICAS_MUNICIPIO].astype(
        {c: "float64" for c in pp.NUMERICAS_MUNICIPIO}
    )
    return proj, X


def main() -> None:
    plots.aplicar_estilo()
    print("=" * 72)
    print("MODELAGEM — GRÃO MUNICÍPIO (risco de não atingir a meta)")
    print("=" * 72)

    diag = diagnostico_2024()
    print("\n[0/5] Diagnóstico das safras disponíveis")
    for ano, d in diag.items():
        print(
            f"  {ano}: corr(meta, taxa anterior) {d['corr_meta_com_taxa_anterior']:.3f} | "
            f"dp do esforço {d['dp_esforco_necessario']:.2f} p.p. | "
            f"AUC do esforço isolado {d['auc_do_esforco_necessario']:.3f}"
        )
    print("  -> 2024 é degenerada (meta ancorada na mesma medição); modelagem usa 2025.")

    base = pd.read_csv(config.PROCESSED_DIR / f"municipio_{ANO_BASE}.csv")
    X, y = pp.preparar_municipio(base)
    grupos = base["regiao"].fillna("Desconhecida") + "/" + base["id_municipio"].astype(str).str[:2]
    print(f"\nBase {ANO_BASE}: {len(X):,} municípios | abaixo da meta {y.mean() * 100:.1f}%")
    print(f"Atributos: {X.shape[1]}")

    # ---------------------------------------------------------------- #
    # 1. Baseline analítico — a regra que o modelo precisa superar
    # ---------------------------------------------------------------- #
    estrat = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    por_uf = GroupKFold(5)
    esforco = base["esforco_necessario"].to_numpy()
    auc_regra_estrat = float(np.mean([roc_auc_score(y.iloc[te], esforco[te]) for _, te in estrat.split(base, y)]))
    auc_regra_uf = float(np.mean([roc_auc_score(y.iloc[te], esforco[te]) for _, te in por_uf.split(base, y, groups=grupos)]))
    auc_regra = float(roc_auc_score(y, esforco))
    print(f"\n[1/5] Baseline analítico — ordenar os municípios pelo esforço necessário")
    print(f"      AUC agregado {auc_regra:.4f} | por fold estratificado {auc_regra_estrat:.4f} | por UF {auc_regra_uf:.4f}")
    print("      É a régua: um modelo que não superar a conta de subtração não se justifica.")

    # ---------------------------------------------------------------- #
    # 2. Comparação de modelos
    # ---------------------------------------------------------------- #
    print("\n[2/5] Comparação de modelos (5-fold estratificado e 5-fold por UF)")
    linhas = [{"modelo": "Baseline analítico (esforço necessário)",
               "auc_estratificado": auc_regra_estrat, "desvio_estratificado": 0.0,
               "auc_por_uf": auc_regra_uf, "desvio_por_uf": 0.0}]
    for nome, (modelo, padronizar) in candidatos().items():
        pipe = pp.construir_pipeline(modelo, pp.NUMERICAS_MUNICIPIO, pp.CATEGORICAS_MUNICIPIO, padronizar)
        a = cross_val_score(pipe, X, y, cv=estrat, scoring="roc_auc")
        b = cross_val_score(pipe, X, y, cv=por_uf, groups=grupos, scoring="roc_auc")
        linhas.append({"modelo": nome, "auc_estratificado": a.mean(), "desvio_estratificado": a.std(),
                       "auc_por_uf": b.mean(), "desvio_por_uf": b.std()})
        print(f"  {nome:<32} AUC {a.mean():.4f} (±{a.std():.4f}) | por UF {b.mean():.4f} (±{b.std():.4f})")
    comparacao = pd.DataFrame(linhas).sort_values("auc_por_uf", ascending=False)
    comparacao.to_csv(config.REPORTS_DIR / "comparacao_modelos_municipio.csv", index=False)

    # ---------------------------------------------------------------- #
    # 3. Otimização de hiperparâmetros
    # ---------------------------------------------------------------- #
    print("\n[3/5] Otimização de hiperparâmetros (LightGBM)")
    espaco = {
        "modelo__n_estimators": [150, 300, 500],
        "modelo__learning_rate": [0.02, 0.05, 0.1],
        "modelo__num_leaves": [7, 15, 31],
        "modelo__min_child_samples": [20, 40, 80],
        "modelo__subsample": [0.7, 0.9, 1.0],
        "modelo__colsample_bytree": [0.6, 0.8, 1.0],
        "modelo__reg_lambda": [1.0, 10.0, 30.0],
    }
    # A busca usa a validação estratificada porque ela reproduz o cenário real de
    # uso: prever um novo ano em municípios de estados já observados. A validação
    # por UF entra depois, como teste de estresse.
    busca = RandomizedSearchCV(
        pp.construir_pipeline(
            LGBMClassifier(random_state=config.RANDOM_STATE, n_jobs=1, verbose=-1),
            pp.NUMERICAS_MUNICIPIO, pp.CATEGORICAS_MUNICIPIO, padronizar=False,
        ),
        espaco, n_iter=25, scoring="roc_auc", cv=estrat,
        random_state=config.RANDOM_STATE, n_jobs=-1, refit=False,
    )
    busca.fit(X, y)
    parametros = {k.replace("modelo__", ""): v for k, v in busca.best_params_.items()}
    print(f"  melhor AUC (validação estratificada): {busca.best_score_:.4f}")
    print(f"  parâmetros: {parametros}")

    # ---------------------------------------------------------------- #
    # 4. Predições fora da amostra e avaliação honesta
    # ---------------------------------------------------------------- #
    print("\n[4/5] Avaliação com predições fora da amostra")
    final = pp.construir_pipeline(
        LGBMClassifier(random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1, **parametros),
        pp.NUMERICAS_MUNICIPIO, pp.CATEGORICAS_MUNICIPIO, padronizar=False,
    )
    prob_oof = cross_val_predict(final, X, y, cv=estrat, method="predict_proba", n_jobs=1)[:, 1]
    prob_oof_uf = cross_val_predict(
        final, X, y, cv=por_uf, groups=grupos, method="predict_proba", n_jobs=1
    )[:, 1]
    limiar, _ = metrics.melhor_limiar(y, prob_oof, "balanceada")

    final.fit(X, y)
    joblib.dump(final, config.MODELS_DIR / "modelo_municipio.joblib")

    resultado = {
        "ano_base": ANO_BASE,
        "diagnostico_safras": diag,
        "baseline_analitico_auc": auc_regra,
        "baseline_analitico_auc_estratificado": auc_regra_estrat,
        "baseline_analitico_auc_por_uf": auc_regra_uf,
        "fora_da_amostra": metrics.avaliar(y, prob_oof),
        "fora_da_amostra_por_uf": metrics.avaliar(y, prob_oof_uf),
        "fora_da_amostra_limiar_otimizado": metrics.avaliar(y, prob_oof, limiar),
        "treino_in_sample": metrics.avaliar(y, final.predict_proba(X)[:, 1]),
        "auc_busca_hiperparametros": float(busca.best_score_),
        "melhores_parametros": parametros,
        "limiar_balanceado": limiar,
        "n_municipios": int(len(X)),
    }
    for chave in ("treino_in_sample", "fora_da_amostra", "fora_da_amostra_por_uf",
                  "fora_da_amostra_limiar_otimizado"):
        m = resultado[chave]
        print(
            f"  {chave:<34} AUC {m['roc_auc']:.4f} | acurácia {m['acuracia']:.4f} | "
            f"recall {m['recall']:.4f} | F1 {m['f1']:.4f}"
        )
    print(
        f"  ganho sobre o baseline analítico — municípios novos: "
        f"{resultado['fora_da_amostra']['roc_auc'] - auc_regra_estrat:+.4f} AUC | "
        f"estados novos: {resultado['fora_da_amostra_por_uf']['roc_auc'] - auc_regra_uf:+.4f} AUC"
    )

    fpr, tpr, _ = roc_curve(y, prob_oof)
    fpr_b, tpr_b, _ = roc_curve(y, esforco)
    plots.curva_roc(
        {
            f"LightGBM (fora da amostra, {ANO_BASE})": (fpr, tpr, resultado["fora_da_amostra"]["roc_auc"]),
            "Baseline analítico (esforço necessário)": (fpr_b, tpr_b, auc_regra),
        },
        "10_roc_municipio.png",
        titulo="Curva ROC — risco municipal de não atingir a meta",
    )
    m = resultado["fora_da_amostra_limiar_otimizado"]
    plots.matriz_confusao(
        np.array([[m["vn"], m["fp"]], [m["fn"], m["vp"]]]),
        ["Atingiu a meta", "Não atingiu"],
        f"Matriz de confusão — municípios em {ANO_BASE} (limiar {limiar:.2f})",
        "11_matriz_confusao_municipio.png",
    )

    # ---------------------------------------------------------------- #
    # 5. Projeção de risco para 2026
    # ---------------------------------------------------------------- #
    print(f"\n[5/5] Projeção de risco para {ANO_PROJECAO}")
    proj, X_proj = construir_base_projecao()
    proj["risco_nao_atingir_meta"] = final.predict_proba(X_proj)[:, 1]
    ibge = feat.loaders.carregar_ibge()[["id_municipio", "nome_municipio_ibge", "sigla_uf"]]
    ranking = (
        proj.merge(ibge, on="id_municipio", how="left")
        .sort_values("risco_nao_atingir_meta", ascending=False)
        .loc[:, ["id_municipio", "nome_municipio_ibge", "sigla_uf", "regiao",
                 "mun_taxa_alfabetizacao_lag", "meta_ano", "esforco_necessario",
                 "mun_n_avaliados_lag", "risco_nao_atingir_meta"]]
        .rename(columns={"nome_municipio_ibge": "municipio",
                         "mun_taxa_alfabetizacao_lag": "taxa_2025",
                         "meta_ano": f"meta_{ANO_PROJECAO}",
                         "mun_n_avaliados_lag": "alunos_avaliados_2025"})
    )
    ranking.to_csv(config.REPORTS_DIR / f"ranking_risco_municipios_{ANO_PROJECAO}.csv", index=False)
    alto_risco = ranking[ranking["risco_nao_atingir_meta"] >= limiar]
    resultado["projecao_2026"] = {
        "municipios_avaliados": int(len(ranking)),
        "municipios_em_alto_risco": int(len(alto_risco)),
        "pct_em_alto_risco": float(len(alto_risco) / len(ranking) * 100),
        "criancas_em_municipios_de_alto_risco": int(alto_risco["alunos_avaliados_2025"].sum()),
        "esforco_medio_alto_risco_pp": float(alto_risco["esforco_necessario"].mean()),
    }
    print(
        f"  {len(alto_risco):,} municípios acima do limiar de risco "
        f"({len(alto_risco) / len(ranking) * 100:.1f}%), cobrindo "
        f"{int(alto_risco['alunos_avaliados_2025'].sum()):,} crianças avaliadas em 2025"
    )

    top = ranking.head(15)
    plots.barras_horizontais(
        (top["municipio"].fillna("?") + " / " + top["sigla_uf"].fillna("?")).tolist(),
        (top["risco_nao_atingir_meta"] * 100).tolist(),
        f"Maior risco de não atingir a meta de {ANO_PROJECAO} (%)",
        "12_top_municipios_risco.png",
        cor=config.STATUS["critico"],
    )

    (config.REPORTS_DIR / "metricas_municipio.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    print("\nArtefatos salvos em models/, reports/ e images/.")


if __name__ == "__main__":
    main()
