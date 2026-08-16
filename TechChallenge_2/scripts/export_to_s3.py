"""Exportacao das camadas Bronze/Silver/Gold para S3.

Le as Delta Tables do Unity Catalog (workspace.bronze/silver/gold) e exporta
copias em Parquet para o bucket S3, organizadas por camada. Util para backup,
auditoria e consumo fora do Databricks.

Saida (S3):
    s3://bucket/processed/bronze/*.parquet
    s3://bucket/processed/silver/*.parquet
    s3://bucket/processed/gold/*.parquet

Uso:
    python scripts/export_to_s3.py
"""

from __future__ import annotations

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "boto3", "python-dotenv", "pyarrow"])

import os
from io import BytesIO
from pathlib import Path

import boto3
from dotenv import load_dotenv
from pyspark.sql import SparkSession

# Configuracao S3
S3_BUCKET = "amzn-s3-fiap-tech2"
S3_PREFIX = "processed/"

# Configuracao Unity Catalog
CATALOG = "workspace"
CAMADAS = ["bronze", "silver", "gold"]

# Credenciais
ENV_PATH = Path("/Workspace/Users/justinofilipe03@gmail.com/TechChallenge_2/.env")


def exportar_camada(spark, s3, schema: str, s3_key_prefix: str) -> dict[str, int]:
    """Exporta todas as tabelas de um schema Unity Catalog para S3 em Parquet.

    Converte Spark DataFrame -> Pandas -> Parquet em memoria -> upload S3.
    Evita dependencia de permissoes Spark-S3 (usa boto3 com credenciais .env).

    Args:
        spark: SparkSession ativa.
        s3: Cliente boto3.
        schema: Nome do schema (bronze/silver/gold).
        s3_key_prefix: Prefixo S3 (ex: processed/bronze/).

    Returns:
        Dicionario {nome_tabela: num_registros}.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    full_schema = f"{CATALOG}.{schema}"
    tabelas = spark.sql(f"SHOW TABLES IN {full_schema}").collect()

    resultado = {}
    for t in tabelas:
        tabela = t.tableName
        df_spark = spark.table(f"{full_schema}.{tabela}")
        count = df_spark.count()

        # Converte Spark -> Pandas -> Arrow -> Parquet em memoria
        df_pandas = df_spark.toPandas()
        arrow_table = pa.Table.from_pandas(df_pandas)

        buffer = BytesIO()
        pq.write_table(arrow_table, buffer)
        buffer.seek(0)

        # Upload para S3
        s3_key = f"{s3_key_prefix}{tabela}.parquet"
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buffer.getvalue())

        resultado[tabela] = count
        print(f"  {tabela}: {count:,} registros")

    return resultado


def exportar() -> None:
    load_dotenv(ENV_PATH)
    s3 = boto3.client("s3")
    spark = SparkSession.builder.appName("export-to-s3").getOrCreate()

    print(f"Exportando para s3://{S3_BUCKET}/{S3_PREFIX}\n")

    totais = {}
    for camada in CAMADAS:
        print(f"[{camada.upper()}]")
        s3_key_prefix = f"{S3_PREFIX}{camada}/"
        resultado = exportar_camada(spark, s3, camada, s3_key_prefix)
        totais[camada] = sum(resultado.values())
        print()

    print("Exportacao concluida:")
    for camada, total in totais.items():
        print(f"  {camada}: {total:,} registros")

    # Verifica
    print(f"\nArquivos em s3://{S3_BUCKET}/{S3_PREFIX}:")
    for camada in CAMADAS:
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}{camada}/")
        if "Contents" in response:
            print(f"  {camada}: {len(response['Contents'])} arquivo(s)")


if __name__ == "__main__":
    exportar()