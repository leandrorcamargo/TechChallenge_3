"""
Engenharia de atributos e construção das bases analíticas.

Duas bases são produzidas:

* `construir_dataset_aluno(ano)`     -> um registro por aluno avaliado no ano.
  Alvo: o aluno foi considerado alfabetizado (proficiência >= 743).

* `construir_dataset_municipio(ano)` -> um registro por município (rede
  Municipal). Alvo: o município atingiu a meta de alfabetização daquele ano.

REGRA DE OURO CONTRA DATA LEAKAGE
---------------------------------
Todo atributo de contexto (escola, município, UF) é calculado com dados do ano
ANTERIOR ao ano da predição. Nenhuma informação do ano alvo — nem do próprio
aluno, nem agregada dos seus colegas — entra como feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.preprocessing import loaders

# Faixa de proficiência bem abaixo do corte: sinaliza defasagem severa.
CORTE_DEFASAGEM_SEVERA = 700.0


# --------------------------------------------------------------------------- #
# Agregações de contexto (calculadas por ano)
# --------------------------------------------------------------------------- #
def _base_presentes(alunos: pd.DataFrame) -> pd.DataFrame:
    """Somente alunos presentes na aplicação e com proficiência calculada."""
    return alunos[(alunos["presente"] == 1) & alunos["proficiencia"].notna()].copy()


def agregar_escolas(alunos: pd.DataFrame) -> pd.DataFrame:
    """Perfil de desempenho de cada escola em um ano."""
    presentes = _base_presentes(alunos)
    presentes["_defasagem_severa"] = (presentes["proficiencia"] < CORTE_DEFASAGEM_SEVERA).astype(float)

    agg = presentes.groupby("id_escola").agg(
        esc_n_avaliados=("proficiencia", "size"),
        esc_taxa_alfabetizacao=("alfabetizado", "mean"),
        esc_media_proficiencia=("proficiencia", "mean"),
        esc_dp_proficiencia=("proficiencia", "std"),
        esc_p25_proficiencia=("proficiencia", lambda s: s.quantile(0.25)),
        esc_pct_defasagem_severa=("_defasagem_severa", "mean"),
    )
    presenca = alunos.groupby("id_escola")["presente"].mean().rename("esc_taxa_presenca")
    agg = agg.join(presenca)
    agg["esc_taxa_alfabetizacao"] *= 100
    agg["esc_pct_defasagem_severa"] *= 100
    agg["esc_taxa_presenca"] *= 100
    return agg.reset_index()


def agregar_municipios(alunos: pd.DataFrame, apenas_rede_municipal: bool = False) -> pd.DataFrame:
    """
    Perfil de desempenho de cada município em um ano.

    `apenas_rede_municipal=True` reproduz o recorte usado nas metas do
    Compromisso Nacional Criança Alfabetizada (rede Municipal).
    """
    base = alunos
    if apenas_rede_municipal:
        base = base[base["dependencia"] == config.REDE_MUNICIPAL]

    presentes = _base_presentes(base)
    presentes["_defasagem_severa"] = (presentes["proficiencia"] < CORTE_DEFASAGEM_SEVERA).astype(float)

    agg = presentes.groupby("id_municipio").agg(
        mun_n_avaliados=("proficiencia", "size"),
        mun_n_escolas=("id_escola", "nunique"),
        mun_taxa_alfabetizacao=("alfabetizado", "mean"),
        mun_media_proficiencia=("proficiencia", "mean"),
        mun_dp_proficiencia=("proficiencia", "std"),
        mun_pct_defasagem_severa=("_defasagem_severa", "mean"),
    )

    # Heterogeneidade interna: quanto as escolas do município diferem entre si.
    por_escola = presentes.groupby(["id_municipio", "id_escola"])["alfabetizado"].mean() * 100
    hetero = por_escola.groupby("id_municipio").agg(
        mun_dp_entre_escolas="std",
        mun_pior_escola="min",
    )
    presenca = base.groupby("id_municipio")["presente"].mean().rename("mun_taxa_presenca")
    pct_municipal = (
        base.assign(_m=(base["dependencia"] == config.REDE_MUNICIPAL).astype(float))
        .groupby("id_municipio")["_m"]
        .mean()
        .rename("mun_pct_rede_municipal")
    )

    agg = agg.join(hetero).join(presenca).join(pct_municipal)
    agg["mun_taxa_alfabetizacao"] *= 100
    agg["mun_pct_defasagem_severa"] *= 100
    agg["mun_taxa_presenca"] *= 100
    agg["mun_pct_rede_municipal"] *= 100
    return agg.reset_index()


def agregar_ufs(alunos: pd.DataFrame) -> pd.DataFrame:
    """Perfil de desempenho de cada UF em um ano."""
    presentes = _base_presentes(alunos)
    agg = presentes.groupby("sigla_uf").agg(
        uf_taxa_alfabetizacao=("alfabetizado", "mean"),
        uf_media_proficiencia=("proficiencia", "mean"),
        uf_dp_proficiencia=("proficiencia", "std"),
    )
    presenca = alunos.groupby("sigla_uf")["presente"].mean().rename("uf_taxa_presenca")
    agg = agg.join(presenca)
    agg["uf_taxa_alfabetizacao"] *= 100
    agg["uf_taxa_presenca"] *= 100
    return agg.reset_index()


# --------------------------------------------------------------------------- #
# Metas e enriquecimento externo
# --------------------------------------------------------------------------- #
def _metas_estaticas() -> pd.DataFrame:
    """Trajetória de metas por município (constante ao longo dos anos)."""
    metas = loaders.carregar_metas_municipio()
    cols = ["id_municipio"] + [f"meta_alfabetizacao_{a}" for a in range(2024, 2031)]
    return metas[cols].drop_duplicates("id_municipio")


def _metas_ano(ano: int) -> pd.DataFrame:
    """Situação do município no ano informado (taxa observada e nível INEP)."""
    metas = loaders.carregar_metas_municipio()
    ano_disp = metas[metas["ano"] == ano]
    if ano_disp.empty:  # anos fora da janela do arquivo da Fase 2
        return pd.DataFrame(columns=["id_municipio", "nivel_alfabetizacao", "percentual_participacao"])
    return ano_disp[["id_municipio", "nivel_alfabetizacao", "percentual_participacao"]].drop_duplicates(
        "id_municipio"
    )


def _enriquecer_ibge(df: pd.DataFrame) -> pd.DataFrame:
    ibge = loaders.carregar_ibge()
    ibge = ibge[
        ["id_municipio", "regiao", "mesorregiao", "populacao", "pib_per_capita", "pct_vab_adm_publica"]
    ]
    df = df.merge(ibge, on="id_municipio", how="left")
    df["log_populacao"] = np.log1p(df["populacao"])
    df["porte_municipio"] = pd.cut(
        df["populacao"],
        bins=[-np.inf, 10_000, 25_000, 100_000, 500_000, np.inf],
        labels=["Muito pequeno", "Pequeno", "Médio", "Grande", "Metrópole"],
    ).astype(str)
    return df


# --------------------------------------------------------------------------- #
# Base analítica — grão ALUNO
# --------------------------------------------------------------------------- #
def _estrutura_atual(alunos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Atributos ESTRUTURAIS do ano corrente — porte da escola e do município.

    São contagens de matrícula/aplicação, conhecidas antes da prova acontecer
    (vêm do Censo Escolar) e independentes de qualquer resultado. Por isso podem
    ser usadas sem violar a regra de não-vazamento.
    """
    presentes = _base_presentes(alunos)
    escola = presentes.groupby("id_escola").size().rename("esc_porte_atual").reset_index()
    municipio = (
        presentes.groupby("id_municipio")
        .agg(mun_porte_atual=("id_aluno", "size"), mun_n_escolas_atual=("id_escola", "nunique"))
        .reset_index()
    )
    return escola, municipio


def construir_dataset_aluno(ano: int, ano_contexto: int | None = None) -> pd.DataFrame:
    """
    Base de modelagem no grão aluno para o ano informado.

    O contexto de desempenho (município e UF) vem sempre do ano ANTERIOR — daí o
    sufixo `_lag` — o que garante que o modelo só use informação disponível
    antes de a avaliação acontecer.

    Atributos de escola do ano anterior NÃO são usados: o INEP reanonimiza o
    ID_ESCOLA a cada edição dos microdados (apenas ~2% dos ids se mantêm no
    mesmo município entre 2024 e 2025), de modo que qualquer histórico por
    escola seria ruído. Ver notebooks/01_analise_exploratoria.ipynb.
    """
    ano_contexto = ano_contexto or (ano - 1)

    alunos = loaders.carregar_alunos(ano)
    contexto = loaders.carregar_alunos(ano_contexto)

    # Universo de modelagem: alunos que efetivamente fizeram a prova.
    # Ausentes recebem alfabetizado=0 por definição operacional do INEP, o que
    # tornaria o alvo trivialmente previsível pela ausência.
    df = _base_presentes(alunos)
    df = df[["ano", "id_aluno", "sigla_uf", "id_escola", "id_municipio", "dependencia", "alfabetizado"]].copy()
    df["dependencia_nome"] = df["dependencia"].map(config.DEPENDENCIA).astype(str)

    # --- contexto defasado (t-1) -------------------------------------------
    mun = agregar_municipios(contexto).add_suffix("_lag").rename(columns={"id_municipio_lag": "id_municipio"})
    uf = agregar_ufs(contexto).add_suffix("_lag").rename(columns={"sigla_uf_lag": "sigla_uf"})
    df = df.merge(mun, on="id_municipio", how="left").merge(uf, on="sigla_uf", how="left")

    # --- estrutura do ano corrente (não depende de resultado) --------------
    esc_atual, mun_atual = _estrutura_atual(alunos)
    df = df.merge(esc_atual, on="id_escola", how="left").merge(mun_atual, on="id_municipio", how="left")

    df["municipio_sem_historico"] = df["mun_n_avaliados_lag"].isna().astype(int)

    # --- metas e enriquecimento externo ------------------------------------
    df = df.merge(_metas_estaticas(), on="id_municipio", how="left")
    df = df.merge(_metas_ano(ano_contexto), on="id_municipio", how="left")
    df = _enriquecer_ibge(df)

    # --- atributos derivados ------------------------------------------------
    meta_do_ano = f"meta_alfabetizacao_{ano}"
    if meta_do_ano in df.columns:
        df["gap_meta_ano"] = df["mun_taxa_alfabetizacao_lag"] - df[meta_do_ano]
    df["gap_meta_2030"] = df["mun_taxa_alfabetizacao_lag"] - df["meta_alfabetizacao_2030"]
    df["dist_municipio_uf"] = df["mun_taxa_alfabetizacao_lag"] - df["uf_taxa_alfabetizacao_lag"]

    # Posição relativa do município dentro da própria UF (0 = pior, 1 = melhor)
    por_mun = df.drop_duplicates("id_municipio")[["id_municipio", "sigla_uf", "mun_taxa_alfabetizacao_lag"]]
    por_mun["posicao_relativa_uf"] = por_mun.groupby("sigla_uf")["mun_taxa_alfabetizacao_lag"].rank(pct=True)
    df = df.merge(por_mun[["id_municipio", "posicao_relativa_uf"]], on="id_municipio", how="left")

    df["alfabetizado"] = df["alfabetizado"].astype(int)
    return df


# --------------------------------------------------------------------------- #
# Base analítica — grão MUNICÍPIO
# --------------------------------------------------------------------------- #
def construir_dataset_municipio(ano: int, ano_contexto: int | None = None) -> pd.DataFrame:
    """
    Base de modelagem no grão município (rede Municipal).

    Alvo: `nao_atingiu_meta` — o município ficou abaixo da meta de alfabetização
    definida para o ano. Features vêm exclusivamente do ano anterior.
    """
    ano_contexto = ano_contexto or (ano - 1)

    alunos = loaders.carregar_alunos(ano)
    contexto = loaders.carregar_alunos(ano_contexto)

    # Resultado observado no ano alvo (rede Municipal) -> define o alvo.
    resultado = agregar_municipios(alunos, apenas_rede_municipal=True)
    resultado = resultado[["id_municipio", "mun_taxa_alfabetizacao", "mun_n_avaliados"]].rename(
        columns={
            "mun_taxa_alfabetizacao": "taxa_observada",
            "mun_n_avaliados": "n_avaliados_ano",
        }
    )

    features = agregar_municipios(contexto, apenas_rede_municipal=True)
    features = features.add_suffix("_lag").rename(columns={"id_municipio_lag": "id_municipio"})

    df = resultado.merge(features, on="id_municipio", how="inner")
    df = df.merge(_metas_estaticas(), on="id_municipio", how="inner")
    df = df.merge(_metas_ano(ano_contexto), on="id_municipio", how="left")
    df = _enriquecer_ibge(df)

    meta_do_ano = f"meta_alfabetizacao_{ano}"
    df = df[df[meta_do_ano].notna()].copy()
    df["meta_ano"] = df[meta_do_ano]
    df["gap_lag_meta_ano"] = df["mun_taxa_alfabetizacao_lag"] - df["meta_ano"]
    df["gap_lag_meta_2030"] = df["mun_taxa_alfabetizacao_lag"] - df["meta_alfabetizacao_2030"]
    df["esforco_necessario"] = df["meta_ano"] - df["mun_taxa_alfabetizacao_lag"]

    df["ano"] = ano
    df["nao_atingiu_meta"] = (df["taxa_observada"] < df["meta_ano"]).astype(int)
    return df
