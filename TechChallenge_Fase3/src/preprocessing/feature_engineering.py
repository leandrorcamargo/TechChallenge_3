"""Módulo para engenharia de features."""
from pyspark.sql import functions as F
import pandas as pd
import numpy as np


def select_features_modelo_agregado():
    """Retorna lista de features para modelo agregado."""
    return [
        'media_portugues', 'soma_niveis_basicos', 'soma_niveis_avancados', 'ano',
        'nivel_0', 'nivel_1', 'nivel_2', 'nivel_3', 'nivel_4',
        'nivel_5', 'nivel_6', 'nivel_7', 'nivel_8', 'codigo_uf', 'rede'
    ]


def select_features_modelo_individual_sem_leakage():
    """Retorna lista de features sem leakage para modelo individual."""
    return ['presenca', 'serie', 'caderno', 'rede', 'ano']


def enriquecer_com_indicadores_municipais(df_alunos, df_indicador):
    """Enriquece dados dos alunos com indicadores municipais."""
    features_mun = ['taxa_alfabetizacao', 'media_portugues'] + [f'nivel_{i}' for i in range(9)]
    df_ind_join = df_indicador.select(['id_municipio', 'ano'] + features_mun).distinct()
    
    df_enriquecido = df_alunos.join(df_ind_join, on=['id_municipio', 'ano'], how='left')
    df_enriquecido = df_enriquecido.withColumn('sigla_uf', F.substring(F.col('id_municipio'), 1, 2))
    
    return df_enriquecido


def enriquecer_com_metas_uf(df, df_meta_uf):
    """Adiciona metas estaduais aos dados."""
    df_meta_2024 = df_meta_uf.filter(F.col('ano_meta') == 2024).select(
        'sigla_uf', F.col('meta').alias('meta_uf_2024')
    ).distinct()
    
    df_meta_2030 = df_meta_uf.filter(F.col('ano_meta') == 2030).select(
        'sigla_uf', F.col('meta').alias('meta_uf_2030')
    ).distinct()
    
    df_meta_join = df_meta_2024.join(df_meta_2030, on='sigla_uf', how='outer')
    return df.join(df_meta_join, on='sigla_uf', how='left')


def criar_features_escola(df):
    """Cria features agregadas por escola."""
    escola_stats = df.groupBy('id_escola', 'ano').agg(
        F.avg('presenca').alias('escola_presenca_media'),
        F.stddev('presenca').alias('escola_presenca_std'),
        F.count('*').alias('escola_num_alunos'),
        F.avg('alfabetizado').alias('escola_taxa_alfabetizacao')
    ).fillna(0, subset=['escola_presenca_std'])
    
    return df.join(escola_stats, on=['id_escola', 'ano'], how='left')


def criar_features_municipio_rede(df):
    """Cria features agregadas por município/rede."""
    mun_rede_stats = df.groupBy('id_municipio', 'rede', 'ano').agg(
        F.avg('presenca').alias('mun_rede_presenca_media'),
        F.count('*').alias('mun_rede_num_alunos')
    )
    
    return df.join(mun_rede_stats, on=['id_municipio', 'rede', 'ano'], how='left')


def criar_features_interacao(df_pd):
    """Cria features de interação e transformação."""
    # Transformações não-lineares
    df_pd['presenca_quadratica'] = df_pd['presenca'] ** 2
    df_pd['presenca_cubica'] = df_pd['presenca'] ** 3
    
    # Interações com contexto
    if 'escola_presenca_media' in df_pd.columns:
        df_pd['presenca_vs_escola'] = df_pd['presenca'] - df_pd['escola_presenca_media']
    
    if 'mun_rede_presenca_media' in df_pd.columns:
        df_pd['presenca_vs_mun'] = df_pd['presenca'] - df_pd['mun_rede_presenca_media']
    
    # Bins de presença
    df_pd['presenca_categoria'] = pd.cut(
        df_pd['presenca'], 
        bins=[0, 0.5, 0.7, 0.85, 0.95, 1.0],
        labels=['muito_baixa', 'baixa', 'media', 'boa', 'excelente']
    ).astype(str)
    
    # Tamanho da escola
    if 'escola_num_alunos' in df_pd.columns:
        df_pd['escola_tamanho'] = pd.cut(
            df_pd['escola_num_alunos'],
            bins=[0, 50, 200, 500, 10000],
            labels=['pequena', 'media', 'grande', 'muito_grande']
        ).astype(str)
    
    return df_pd


def filtrar_alunos_sem_repeticao(df):
    """Filtra alunos que aparecem em apenas 1 ano (evita vazamento temporal)."""
    alunos_unicos = df.groupBy('id_aluno').agg(
        F.countDistinct('ano').alias('n_anos')
    ).filter(F.col('n_anos') == 1).select('id_aluno')
    
    return df.join(alunos_unicos, on='id_aluno', how='inner')


def adicionar_features_historicas(df, df_indicador):
    """Adiciona performance histórica (ano anterior) do município/rede."""
    # Para 2024, usar indicadores de 2023
    df_hist_2023 = df_indicador.filter(F.col('ano') == 2023).select(
        'id_municipio', 'rede',
        F.col('taxa_alfabetizacao').alias('taxa_hist'),
        F.col('media_portugues').alias('media_hist')
    ).distinct()
    
    # Média estadual de 2023
    df_hist_uf = df_indicador.filter(F.col('ano') == 2023).withColumn(
        'sigla_uf', F.substring(F.col('id_municipio'), 1, 2)
    ).groupBy('sigla_uf', 'rede').agg(
        F.mean('taxa_alfabetizacao').alias('taxa_uf_hist'),
        F.mean('media_portugues').alias('media_uf_hist')
    )
    
    df = df.withColumn('sigla_uf', F.substring(F.col('id_municipio'), 1, 2))
    df = df.join(df_hist_2023, on=['id_municipio', 'rede'], how='left')
    df = df.join(df_hist_uf, on=['sigla_uf', 'rede'], how='left')
    
    return df
