"""
Interpretabilidade dos modelos — Feature Importance e SHAP Values.

Uso:
    python -m src.evaluation.interpretabilidade

O objetivo aqui não é ranquear variáveis por curiosidade: é responder às
perguntas de negócio do desafio. Quais fatores mais pesam na alfabetização?
Quais municípios concentram risco? Que regiões se parecem entre si?
"""
from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config
from src.preprocessing import pipeline as pp
from src.visualization import plots

N_SHAP_ALUNO = 30_000

TRADUCAO = {
    "mun_taxa_alfabetizacao_lag": "Taxa de alfabetização do município (ano anterior)",
    "mun_media_proficiencia_lag": "Proficiência média do município (ano anterior)",
    "mun_dp_proficiencia_lag": "Dispersão da proficiência no município",
    "mun_pct_defasagem_severa_lag": "% de alunos em defasagem severa no município",
    "mun_dp_entre_escolas_lag": "Desigualdade entre escolas do município",
    "mun_pior_escola_lag": "Desempenho da pior escola do município",
    "mun_taxa_presenca_lag": "Taxa de presença na avaliação (município)",
    "mun_pct_rede_municipal_lag": "% de alunos na rede municipal",
    "mun_n_avaliados_lag": "Alunos avaliados no município (ano anterior)",
    "mun_n_escolas_lag": "Escolas avaliadas no município (ano anterior)",
    "uf_taxa_alfabetizacao_lag": "Taxa de alfabetização da UF (ano anterior)",
    "uf_media_proficiencia_lag": "Proficiência média da UF (ano anterior)",
    "uf_dp_proficiencia_lag": "Dispersão da proficiência na UF",
    "uf_taxa_presenca_lag": "Taxa de presença na UF",
    "esc_porte_atual": "Porte da escola (alunos avaliados)",
    "mun_porte_atual": "Porte do município (alunos avaliados)",
    "mun_n_escolas_atual": "Nº de escolas do município",
    "meta_ano": "Meta de alfabetização do ano",
    "meta_alfabetizacao_2030": "Meta de 2030",
    "nivel_alfabetizacao": "Nível de alfabetização do município (INEP)",
    "percentual_participacao": "% de participação na avaliação",
    "gap_meta_ano": "Distância para a meta do ano",
    "gap_meta_2030": "Distância para a meta de 2030",
    "gap_lag_meta_ano": "Distância para a meta do ano",
    "gap_lag_meta_2030": "Distância para a meta de 2030",
    "esforco_necessario": "Esforço necessário para bater a meta (p.p.)",
    "dist_municipio_uf": "Diferença entre município e sua UF",
    "posicao_relativa_uf": "Posição do município dentro da UF",
    "log_populacao": "População do município (log)",
    "pib_per_capita": "PIB per capita",
    "pct_vab_adm_publica": "% do PIB em administração pública",
    "municipio_sem_historico": "Município sem histórico no ano anterior",
}


def _traduzir(nome: str) -> str:
    base = nome.replace("dependencia_nome_", "Rede: ").replace("regiao_", "Região: ")
    base = base.replace("porte_municipio_", "Porte: ").replace("sigla_uf_", "UF: ")
    base = base.replace("missingindicator_", "Ausente: ")
    return TRADUCAO.get(nome, base)


def _nomes_features(modelo) -> list[str]:
    return list(modelo.named_steps["preprocessamento"].get_feature_names_out())


def importancia_ganho(modelo, prefixo: str, titulo: str, n: int = 18) -> pd.DataFrame:
    nomes = _nomes_features(modelo)
    ganho = modelo.named_steps["modelo"].booster_.feature_importance(importance_type="gain")
    imp = (
        pd.DataFrame({"feature": nomes, "ganho": ganho})
        .assign(ganho_pct=lambda d: d["ganho"] / d["ganho"].sum() * 100)
        .sort_values("ganho_pct", ascending=False)
    )
    top = imp.head(n)
    plots.importancias(
        [_traduzir(f) for f in top["feature"]],
        top["ganho_pct"].tolist(),
        titulo,
        f"{prefixo}_importancia.png",
        xlabel="Contribuição para o ganho do modelo (%)",
    )
    return imp


def analise_shap(modelo, X: pd.DataFrame, prefixo: str, titulo: str) -> pd.DataFrame:
    """SHAP no espaço transformado (pós-pré-processamento), com beeswarm."""
    pre = modelo.named_steps["preprocessamento"]
    Xt = pd.DataFrame(pre.transform(X), columns=_nomes_features(modelo))
    explicador = shap.TreeExplainer(modelo.named_steps["modelo"])
    valores = explicador.shap_values(Xt)
    if isinstance(valores, list):
        valores = valores[1]

    Xt_legivel = Xt.rename(columns={c: _traduzir(c) for c in Xt.columns})

    plots.aplicar_estilo()
    fig = plt.figure(figsize=(9.0, 6.4))
    shap.summary_plot(valores, Xt_legivel, max_display=15, show=False, plot_size=None,
                      color_bar_label="Valor da variável")
    ax = plt.gca()
    fig.patch.set_facecolor(config.COR_FUNDO)
    ax.set_facecolor(config.COR_FUNDO)
    ax.tick_params(colors=config.COR_TEXTO_FRACO)
    ax.set_xlabel("Impacto na predição (valor SHAP)", color=config.COR_TEXTO_FRACO)
    for t in ax.get_yticklabels():
        t.set_color(config.COR_TEXTO)
    ax.set_title(titulo, color=config.COR_TEXTO, fontsize=13, pad=14)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    plots.salvar(fig, f"{prefixo}_shap_beeswarm.png")

    medio = (
        pd.DataFrame({"feature": Xt.columns, "shap_medio_abs": np.abs(valores).mean(axis=0)})
        .sort_values("shap_medio_abs", ascending=False)
    )
    top = medio.head(15)
    plots.importancias(
        [_traduzir(f) for f in top["feature"]],
        top["shap_medio_abs"].tolist(),
        f"{titulo} — média absoluta",
        f"{prefixo}_shap_barras.png",
        xlabel="|SHAP| médio",
    )
    return medio


def main() -> None:
    plots.aplicar_estilo()
    insights: dict = {}

    # ------------------------------------------------------------------ #
    # Modelo de aluno
    # ------------------------------------------------------------------ #
    print("[1/3] Modelo de aluno")
    modelo_aluno = joblib.load(config.MODELS_DIR / "modelo_aluno.joblib")
    teste = pd.read_parquet(config.PROCESSED_DIR / f"aluno_{config.ANO_TESTE}.parquet")
    X_te, y_te = pp.preparar_aluno(teste, config.ANO_TESTE)

    imp_aluno = importancia_ganho(
        modelo_aluno, "13_aluno", "O que mais pesa na predição de alfabetização do aluno"
    )
    amostra = X_te.sample(min(N_SHAP_ALUNO, len(X_te)), random_state=config.RANDOM_STATE)
    shap_aluno = analise_shap(
        modelo_aluno, amostra, "14_aluno", "SHAP — modelo de alfabetização do aluno"
    )
    insights["aluno_top_ganho"] = {
        _traduzir(r.feature): round(r.ganho_pct, 2) for r in imp_aluno.head(10).itertuples()
    }
    insights["aluno_top_shap"] = {
        _traduzir(r.feature): round(r.shap_medio_abs, 4) for r in shap_aluno.head(10).itertuples()
    }

    # ------------------------------------------------------------------ #
    # Modelo de município
    # ------------------------------------------------------------------ #
    print("[2/3] Modelo de município")
    modelo_mun = joblib.load(config.MODELS_DIR / "modelo_municipio.joblib")
    mun_te = pd.read_csv(config.PROCESSED_DIR / f"municipio_{config.ANO_TESTE}.csv")
    X_mun, _ = pp.preparar_municipio(mun_te)

    imp_mun = importancia_ganho(
        modelo_mun, "15_municipio", "O que mais pesa no risco de não atingir a meta"
    )
    shap_mun = analise_shap(
        modelo_mun, X_mun, "16_municipio", "SHAP — risco municipal de não atingir a meta"
    )
    insights["municipio_top_ganho"] = {
        _traduzir(r.feature): round(r.ganho_pct, 2) for r in imp_mun.head(10).itertuples()
    }
    insights["municipio_top_shap"] = {
        _traduzir(r.feature): round(r.shap_medio_abs, 4) for r in shap_mun.head(10).itertuples()
    }

    # ------------------------------------------------------------------ #
    # Perguntas de negócio
    # ------------------------------------------------------------------ #
    print("[3/3] Perguntas de negócio")
    prob = modelo_aluno.predict_proba(X_te)[:, 1]
    teste = teste.assign(prob_alfabetizado=prob, risco=1 - prob)

    por_regiao = (
        teste.groupby("regiao")
        .agg(risco_medio=("risco", "mean"), alunos=("risco", "size"), taxa_real=("alfabetizado", "mean"))
        .sort_values("risco_medio", ascending=False)
    )
    por_regiao["risco_medio"] *= 100
    por_regiao["taxa_real"] *= 100
    insights["risco_por_regiao"] = por_regiao.round(2).to_dict("index")

    # Municípios com maior massa de crianças em risco previsto
    por_mun = (
        teste.groupby(["id_municipio", "sigla_uf"])
        .agg(
            risco_medio=("risco", "mean"),
            alunos=("risco", "size"),
            criancas_em_risco=("risco", "sum"),
        )
        .reset_index()
        .sort_values("criancas_em_risco", ascending=False)
    )
    ibge = pd.read_csv(config.CSV_IBGE_MUNICIPIOS)[["id_municipio", "nome_municipio_ibge"]]
    por_mun = por_mun.merge(ibge, on="id_municipio", how="left")
    por_mun.to_csv(config.REPORTS_DIR / "municipios_criancas_em_risco.csv", index=False)
    insights["top_municipios_massa_de_risco"] = (
        por_mun.head(10)[["nome_municipio_ibge", "sigla_uf", "criancas_em_risco", "risco_medio"]]
        .round(2)
        .to_dict("records")
    )

    # Concentração: quantos municípios respondem por metade das crianças em risco
    ordenado = por_mun.sort_values("criancas_em_risco", ascending=False)
    acumulado = ordenado["criancas_em_risco"].cumsum() / ordenado["criancas_em_risco"].sum()
    insights["concentracao_do_risco"] = {
        "municipios_para_50pct_do_risco": int((acumulado < 0.5).sum() + 1),
        "total_municipios": int(len(ordenado)),
        "total_criancas_em_risco_estimadas": float(ordenado["criancas_em_risco"].sum()),
    }

    destino = config.REPORTS_DIR / "insights.json"
    destino.write_text(json.dumps(insights, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(insights, indent=2, ensure_ascii=False, default=float)[:2000])
    print(f"\n[interpretabilidade] insights -> {destino}")


if __name__ == "__main__":
    main()
