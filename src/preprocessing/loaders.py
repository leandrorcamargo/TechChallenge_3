"""
Leitura das fontes brutas.

Camada de I/O do projeto: lê os microdados do INEP direto dos .zip, os arquivos
de metas/indicadores herdados da camada Gold da Fase 2 e as bases externas do
IBGE. Nada aqui transforma dado de negócio — só padroniza tipos e nomes.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from src import config

# Colunas dos microdados que interessam ao projeto. As demais (respostas item a
# item, gabaritos, pesos) ficam de fora — ver config.COLUNAS_LEAKAGE.
COLS_ALUNO = [
    "NU_ANO_AVALIACAO",
    "CO_UF",
    "SG_UF",
    "ID_ALUNO",
    "TP_SERIE",
    "ID_ESCOLA",
    "TP_DEPENDENCIA",
    "CO_MUNICIPIO",
    "NO_MUNICIPIO",
    "IN_PRESENCA_LP",
    "VL_PROFICIENCIA_LP",
    "IN_ALFABETIZADO",
]


def _ler_csv_do_zip(zip_path: Path, nome_interno: str, **kwargs) -> pd.DataFrame:
    """Lê um CSV de dentro de um .zip sem descompactar em disco."""
    with zipfile.ZipFile(zip_path) as z:
        alvo = next(n for n in z.namelist() if n.endswith(nome_interno))
        with z.open(alvo) as fh:
            return pd.read_csv(fh, sep=";", encoding="latin-1", low_memory=False, **kwargs)


def carregar_alunos(ano: int, usar_cache: bool = True) -> pd.DataFrame:
    """
    Microdados de aluno (TS_ALUNO) de um ano da Avaliação da Alfabetização.

    Retorna uma linha por aluno avaliado, já com nomes de coluna em snake_case.
    Alunos ausentes são mantidos aqui — o filtro acontece na etapa de features,
    de forma explícita e documentada.
    """
    cache = config.INTERIM_DIR / f"alunos_{ano}.parquet"
    if usar_cache and cache.exists():
        return pd.read_parquet(cache)

    df = _ler_csv_do_zip(config.ZIP_MICRODADOS[ano], "TS_ALUNO.csv", usecols=COLS_ALUNO)
    df = df.rename(
        columns={
            "NU_ANO_AVALIACAO": "ano",
            "CO_UF": "codigo_uf",
            "SG_UF": "sigla_uf",
            "ID_ALUNO": "id_aluno",
            "TP_SERIE": "serie",
            "ID_ESCOLA": "id_escola",
            "TP_DEPENDENCIA": "dependencia",
            "CO_MUNICIPIO": "id_municipio",
            "NO_MUNICIPIO": "nome_municipio",
            "IN_PRESENCA_LP": "presente",
            "VL_PROFICIENCIA_LP": "proficiencia",
            "IN_ALFABETIZADO": "alfabetizado",
        }
    )
    df["ano"] = df["ano"].astype("int16")
    for c in ("codigo_uf", "serie", "dependencia", "presente", "alfabetizado"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int16")
    for c in ("id_escola", "id_municipio"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["proficiencia"] = pd.to_numeric(df["proficiencia"], errors="coerce").astype("float32")

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def carregar_metas_municipio() -> pd.DataFrame:
    """Trajetória de metas 2024–2030 por município (rede Municipal)."""
    df = pd.read_csv(config.CSV_METAS_MUNICIPIO)
    df["rede"] = df["rede"].astype(str)
    return df


def carregar_metas_uf() -> pd.DataFrame:
    return pd.read_csv(config.CSV_METAS_UF)


def carregar_metas_brasil() -> pd.DataFrame:
    return pd.read_csv(config.CSV_METAS_BRASIL)


def carregar_indicador_municipio() -> pd.DataFrame:
    """Indicador Criança Alfabetizada agregado por município (Base dos Dados)."""
    return pd.read_csv(config.CSV_INDICADOR_MUNICIPIO)


def carregar_indicador_uf() -> pd.DataFrame:
    return pd.read_csv(config.CSV_INDICADOR_UF)


def carregar_ibge() -> pd.DataFrame:
    """
    Base externa do IBGE por município: região, UF, população residente,
    PIB per capita e participação da administração pública no valor adicionado.

    Materializada por src/preprocessing/fetch_externo.py.
    """
    if not config.CSV_IBGE_MUNICIPIOS.exists():
        raise FileNotFoundError(
            "Base do IBGE não encontrada. Rode: python -m src.preprocessing.fetch_externo"
        )
    geo = pd.read_csv(config.CSV_IBGE_MUNICIPIOS)
    socio = pd.read_csv(config.CSV_IBGE_SOCIOECONOMICO)
    return geo.merge(socio, on="id_municipio", how="left")
