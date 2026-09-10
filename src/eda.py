"""
Análise exploratória — Tech Challenge Fase 3.

Uso:
    python -m src.eda

Gera as figuras em images/ e um resumo numérico em reports/eda_estatisticas.json,
que alimenta o README e a apresentação (nenhum número é digitado à mão).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import features, loaders, pipeline
from src.visualization import plots

REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]


def main() -> None:
    plots.aplicar_estilo()
    resumo: dict = {}

    # ------------------------------------------------------------------ #
    # 1. Evolução do indicador e distribuição da proficiência
    # ------------------------------------------------------------------ #
    proficiencias, serie_nacional = {}, {}
    for ano in config.ANOS:
        alunos = loaders.carregar_alunos(ano)
        presentes = alunos[(alunos["presente"] == 1) & alunos["proficiencia"].notna()]
        proficiencias[ano] = presentes["proficiencia"]
        publica = presentes[presentes["dependencia"].isin(config.REDE_PUBLICA)]
        serie_nacional[ano] = {
            "avaliados": int(len(presentes)),
            "taxa_geral": float(presentes["alfabetizado"].mean() * 100),
            "taxa_rede_publica": float(publica["alfabetizado"].mean() * 100),
            "media_proficiencia": float(presentes["proficiencia"].mean()),
            "taxa_ausencia": float((alunos["presente"] == 0).mean() * 100),
        }
    resumo["serie_nacional"] = serie_nacional
    plots.distribuicao_proficiencia(proficiencias, "01_distribuicao_proficiencia.png")

    # Comparação com o indicador oficial publicado (validação cruzada de fontes)
    oficial = loaders.carregar_metas_brasil().set_index("ano")["taxa_alfabetizacao"].to_dict()
    resumo["validacao_indicador_nacional"] = {
        str(a): {
            "calculado_microdados": round(serie_nacional[a]["taxa_rede_publica"], 2),
            "publicado_base_dos_dados": float(oficial.get(a, np.nan)),
        }
        for a in config.ANOS
    }

    # ------------------------------------------------------------------ #
    # 2. Recortes territoriais e de rede
    # ------------------------------------------------------------------ #
    alunos25 = loaders.carregar_alunos(2025)
    presentes25 = alunos25[(alunos25["presente"] == 1) & alunos25["proficiencia"].notna()]

    por_uf = (presentes25.groupby("sigla_uf")["alfabetizado"].mean() * 100).sort_values()
    plots.barras_horizontais(
        por_uf.index.tolist(),
        por_uf.values.tolist(),
        "Taxa de alfabetização por UF — 2025 (% dos avaliados)",
        "02_taxa_por_uf.png",
        destaque=list(range(5)),  # as cinco UFs em situação mais crítica
    )
    resumo["uf_extremos"] = {
        "piores": por_uf.head(5).round(1).to_dict(),
        "melhores": por_uf.tail(5).round(1).to_dict(),
        "amplitude_pp": float(por_uf.max() - por_uf.min()),
    }

    por_dep = (
        presentes25.assign(rede=presentes25["dependencia"].map(config.DEPENDENCIA))
        .groupby("rede")["alfabetizado"]
        .agg(["mean", "size"])
    )
    por_dep["mean"] *= 100
    resumo["por_dependencia_2025"] = por_dep.round(2).to_dict("index")

    # ------------------------------------------------------------------ #
    # 2b. Teste de estabilidade do identificador de escola entre edições
    # ------------------------------------------------------------------ #
    a24 = loaders.carregar_alunos(2024)
    mapa24 = a24.dropna(subset=["id_municipio"]).groupby("id_escola")["id_municipio"].first()
    mapa25 = presentes25.dropna(subset=["id_municipio"]).groupby("id_escola")["id_municipio"].first()
    juncao = mapa24.to_frame("mun_2024").join(mapa25.to_frame("mun_2025"), how="inner").dropna()
    resumo["estabilidade_id_escola"] = {
        "ids_presentes_nos_dois_anos": int(len(juncao)),
        "pct_mesmo_municipio": float((juncao["mun_2024"] == juncao["mun_2025"]).mean() * 100),
    }

    # ------------------------------------------------------------------ #
    # 3. Base de modelagem: correlações e hipóteses
    # ------------------------------------------------------------------ #
    df = pd.read_parquet(config.PROCESSED_DIR / "aluno_2025.parquet")
    amostra = df.sample(300_000, random_state=config.RANDOM_STATE)

    cols_corr = [
        "alfabetizado",
        "mun_taxa_alfabetizacao_lag",
        "mun_media_proficiencia_lag",
        "mun_pct_defasagem_severa_lag",
        "mun_dp_entre_escolas_lag",
        "mun_pior_escola_lag",
        "mun_taxa_presenca_lag",
        "uf_taxa_alfabetizacao_lag",
        "posicao_relativa_uf",
        "gap_meta_2030",
        "log_populacao",
        "pib_per_capita",
        "pct_vab_adm_publica",
    ]
    plots.correlacoes(
        amostra.astype({c: "float64" for c in cols_corr}),
        cols_corr,
        "Correlação de Spearman — alvo e principais atributos",
        "03_correlacoes.png",
    )
    corr_alvo = (
        amostra[cols_corr].astype("float64").corr(method="spearman")["alfabetizado"].drop("alfabetizado")
    )
    resumo["correlacao_com_alvo"] = corr_alvo.round(3).to_dict()

    # Persistência do desempenho: município em 2024 x 2025
    mun24 = features.agregar_municipios(loaders.carregar_alunos(2024))[
        ["id_municipio", "mun_taxa_alfabetizacao", "mun_n_avaliados"]
    ].rename(columns={"mun_taxa_alfabetizacao": "taxa_2024"})
    mun25 = features.agregar_municipios(alunos25)[["id_municipio", "mun_taxa_alfabetizacao"]].rename(
        columns={"mun_taxa_alfabetizacao": "taxa_2025"}
    )
    comp = mun24.merge(mun25, on="id_municipio").dropna()
    plots.dispersao(
        comp["taxa_2024"],
        comp["taxa_2025"],
        "Persistência do desempenho municipal entre anos",
        "Taxa de alfabetização 2024 (%)",
        "Taxa de alfabetização 2025 (%)",
        "04_persistencia_municipal.png",
        cor_valores=np.log1p(comp["mun_n_avaliados"]),
        rotulo_cor="log(alunos avaliados)",
    )
    resumo["persistencia_municipal"] = {
        "spearman_2024_2025": float(comp["taxa_2024"].corr(comp["taxa_2025"], method="spearman")),
        "municipios": int(len(comp)),
    }

    # Contexto socioeconômico
    ibge = loaders.carregar_ibge()
    socio = comp.merge(ibge, on="id_municipio", how="left").dropna(subset=["pib_per_capita"])
    plots.dispersao(
        np.log10(socio["pib_per_capita"]),
        socio["taxa_2025"],
        "Renda municipal e alfabetização",
        "log10(PIB per capita, R$)",
        "Taxa de alfabetização 2025 (%)",
        "05_pib_x_alfabetizacao.png",
        cor_valores=np.log1p(socio["populacao"]),
        rotulo_cor="log(população)",
    )
    resumo["correlacao_pib_taxa"] = float(
        socio["pib_per_capita"].corr(socio["taxa_2025"], method="spearman")
    )

    por_regiao = (
        socio.groupby("regiao")["taxa_2025"].mean().reindex(REGIOES).dropna()
    )
    plots.barras_horizontais(
        por_regiao.index.tolist(),
        por_regiao.values.tolist(),
        "Taxa média de alfabetização por região — 2025",
        "06_taxa_por_regiao.png",
    )
    resumo["por_regiao_2025"] = por_regiao.round(2).to_dict()

    # Desigualdade dentro do município
    resumo["heterogeneidade_interna"] = {
        "dp_media_entre_escolas": float(df["mun_dp_entre_escolas_lag"].mean()),
        "municipios_com_escola_abaixo_de_30pct": float(
            (df.groupby("id_municipio")["mun_pior_escola_lag"].first() < 30).mean() * 100
        ),
    }

    # Distância para a meta de 2030
    munmeta = pd.read_csv(config.PROCESSED_DIR / "municipio_2025.csv")
    resumo["meta_2030"] = {
        "municipios": int(len(munmeta)),
        "ja_atingiram_2030": float((munmeta["taxa_observada"] >= munmeta["meta_alfabetizacao_2030"]).mean() * 100),
        "abaixo_da_meta_do_ano": float(munmeta["nao_atingiu_meta"].mean() * 100),
        "gap_mediano_para_2030_pp": float((munmeta["meta_alfabetizacao_2030"] - munmeta["taxa_observada"]).median()),
    }

    destino = config.REPORTS_DIR / "eda_estatisticas.json"
    destino.write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(resumo, indent=2, ensure_ascii=False, default=float))
    print(f"\n[eda] resumo -> {destino}")


if __name__ == "__main__":
    main()
