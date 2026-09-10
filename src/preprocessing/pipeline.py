"""
Pipeline de pré-processamento (scikit-learn).

O pré-processamento é integrado ao modelo dentro de um `Pipeline`, de modo que
imputação, padronização e encoding sejam ajustados SOMENTE no fold/partição de
treino. É essa integração que impede o vazamento de estatísticas do conjunto de
teste para o de treino.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Dicionário de features — grão ALUNO
# --------------------------------------------------------------------------- #
NUMERICAS_ALUNO = [
    # --- Contexto de desempenho do município no ano anterior (t-1) ---------
    "mun_n_avaliados_lag",
    "mun_n_escolas_lag",
    "mun_taxa_alfabetizacao_lag",
    "mun_media_proficiencia_lag",
    "mun_dp_proficiencia_lag",
    "mun_pct_defasagem_severa_lag",
    "mun_dp_entre_escolas_lag",
    "mun_pior_escola_lag",
    "mun_taxa_presenca_lag",
    "mun_pct_rede_municipal_lag",
    # --- Contexto de desempenho da UF no ano anterior (t-1) ----------------
    "uf_taxa_alfabetizacao_lag",
    "uf_media_proficiencia_lag",
    "uf_dp_proficiencia_lag",
    "uf_taxa_presenca_lag",
    # --- Estrutura do ano corrente (matrícula, não resultado) --------------
    "esc_porte_atual",
    "mun_porte_atual",
    "mun_n_escolas_atual",
    # --- Metas do Compromisso Nacional Criança Alfabetizada ---------------
    "meta_ano",
    "meta_alfabetizacao_2030",
    "nivel_alfabetizacao",
    "percentual_participacao",
    "gap_meta_ano",
    "gap_meta_2030",
    # --- Posição relativa --------------------------------------------------
    "dist_municipio_uf",
    "posicao_relativa_uf",
    # --- Socioeconômico e territorial (IBGE) -------------------------------
    "log_populacao",
    "pib_per_capita",
    "pct_vab_adm_publica",
    # --- Sinalizador de ausência de histórico ------------------------------
    "municipio_sem_historico",
]

CATEGORICAS_ALUNO = [
    "dependencia_nome",
    "sigla_uf",
    "regiao",
    "porte_municipio",
]

ALVO_ALUNO = "alfabetizado"

# --------------------------------------------------------------------------- #
# Dicionário de features — grão MUNICÍPIO
# --------------------------------------------------------------------------- #
NUMERICAS_MUNICIPIO = [
    "mun_n_avaliados_lag",
    "mun_n_escolas_lag",
    "mun_taxa_alfabetizacao_lag",
    "mun_media_proficiencia_lag",
    "mun_dp_proficiencia_lag",
    "mun_pct_defasagem_severa_lag",
    "mun_dp_entre_escolas_lag",
    "mun_pior_escola_lag",
    "mun_taxa_presenca_lag",
    "meta_ano",
    "meta_alfabetizacao_2030",
    "nivel_alfabetizacao",
    "percentual_participacao",
    "gap_lag_meta_ano",
    "gap_lag_meta_2030",
    "esforco_necessario",
    "log_populacao",
    "pib_per_capita",
    "pct_vab_adm_publica",
]

CATEGORICAS_MUNICIPIO = ["regiao", "porte_municipio"]

ALVO_MUNICIPIO = "nao_atingiu_meta"


# --------------------------------------------------------------------------- #
# Preparação da matriz
# --------------------------------------------------------------------------- #
def preparar_aluno(df: pd.DataFrame, ano: int) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa X e y do dataset de aluno e harmoniza o nome da meta do ano.

    A coluna `meta_alfabetizacao_<ano>` muda de nome conforme o ano; renomear
    para `meta_ano` mantém o mesmo esquema de features entre treino e teste.
    """
    df = df.copy()
    df["meta_ano"] = df[f"meta_alfabetizacao_{ano}"]
    X = df[NUMERICAS_ALUNO + CATEGORICAS_ALUNO].astype(
        {c: "float64" for c in NUMERICAS_ALUNO}
    )
    y = df[ALVO_ALUNO].astype(int)
    return X, y


def preparar_municipio(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    X = df[NUMERICAS_MUNICIPIO + CATEGORICAS_MUNICIPIO].astype(
        {c: "float64" for c in NUMERICAS_MUNICIPIO}
    )
    y = df[ALVO_MUNICIPIO].astype(int)
    return X, y


# --------------------------------------------------------------------------- #
# Pré-processador
# --------------------------------------------------------------------------- #
def construir_preprocessador(
    numericas: list[str],
    categoricas: list[str],
    padronizar: bool = True,
) -> ColumnTransformer:
    """
    ColumnTransformer com dois ramos:

    * numéricas  -> imputação pela mediana (robusta a assimetria) + padronização
    * categóricas-> imputação pela moda + One-Hot (ignorando categorias inéditas)

    `padronizar=False` para modelos baseados em árvore, que não se beneficiam da
    escala e ficam mais rápidos sem ela.
    """
    passos_num: list = [("imputacao", SimpleImputer(strategy="median", add_indicator=True))]
    if padronizar:
        passos_num.append(("escala", StandardScaler()))

    ramo_numerico = Pipeline(passos_num)
    ramo_categorico = Pipeline(
        [
            ("imputacao", SimpleImputer(strategy="most_frequent")),
            ("encoding", OneHotEncoder(handle_unknown="ignore", min_frequency=0.001, sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        [
            ("numericas", ramo_numerico, numericas),
            ("categoricas", ramo_categorico, categoricas),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _para_float32(X):
    """Reduz a matriz a 32 bits — com ~2 milhões de linhas isso corta o consumo
    de memória pela metade sem qualquer perda prática de precisão."""
    return X.astype(np.float32, copy=False)


def construir_pipeline(modelo, numericas, categoricas, padronizar: bool = True) -> Pipeline:
    """Pré-processamento + estimador em um único objeto treinável e serializável."""
    return Pipeline(
        [
            ("preprocessamento", construir_preprocessador(numericas, categoricas, padronizar)),
            ("compactacao", FunctionTransformer(_para_float32, accept_sparse=True)),
            ("modelo", modelo),
        ]
    )
