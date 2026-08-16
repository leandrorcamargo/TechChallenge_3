"""Preparacao da fonte (E1) - extrai data/ do S3 para source/ no workspace.

Nao faz transformacao de dados: apenas descompacta/organiza os arquivos que ja
temos no S3 numa pasta source/ no workspace com nomes logicos, para a camada Bronze ler.

Saida (source/):
    indicador_alfabetizacao_uf.csv.gz
    indicador_alfabetizacao_municipio.csv.gz
    meta_alfabetizacao_brasil.csv.gz
    meta_alfabetizacao_uf.csv.gz
    meta_alfabetizacao_municipio.csv.gz
    ts_aluno/2023.csv, 2024.csv, 2025.csv
    ts_municipio/2023.csv, 2024.csv, 2025.csv
    ts_estado/2023.csv, 2024.csv, 2025.csv

Uso:
    python scripts/prep_source.py
"""

from __future__ import annotations

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "boto3", "python-dotenv"])

import os
import re
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import boto3
from dotenv import load_dotenv

# Configuracao S3
S3_BUCKET = "amzn-s3-fiap-tech2"
S3_PREFIX = "data/"

# Configuracao local (workspace)
SOURCE = Path("/Workspace/Users/justinofilipe03@gmail.com/TechChallenge_2/source")

# Base dos Dados (.csv.gz): arquivo original -> nome logico na source/
BASE_DOS_DADOS = {
    "br_inep_avaliacao_alfabetizacao_uf.csv.gz": "indicador_alfabetizacao_uf.csv.gz",
    "br_inep_avaliacao_alfabetizacao_municipio.csv.gz": "indicador_alfabetizacao_municipio.csv.gz",
    "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_brasil.csv.gz": "meta_alfabetizacao_brasil.csv.gz",
    "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_uf.csv.gz": "meta_alfabetizacao_uf.csv.gz",
    "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_municipio.csv.gz": "meta_alfabetizacao_municipio.csv.gz",
}

# Microdados
MICRODADOS_ZIPS = [
    "microdados_avaliacao_da_alfabetizacao_2023.zip",
    "microdados_avaliacao_da_alfabetizacao_2024.zip",
    "microdados_AEEB_2025.zip",
]
TS_TABLES = {"TS_ALUNO": "ts_aluno", "TS_MUNICIPIO": "ts_municipio", "TS_ESTADO": "ts_estado"}
_YEAR = re.compile(r"(20\d{2})")


def preparar() -> None:
    env_path = Path("/Workspace/Users/justinofilipe03@gmail.com/TechChallenge_2/.env")
    load_dotenv(env_path)
    
    s3 = boto3.client("s3")
    SOURCE.mkdir(parents=True, exist_ok=True)

    # Base dos Dados
    for origem, destino in BASE_DOS_DADOS.items():
        s3_key = f"{S3_PREFIX}{origem}"
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        destino_path = SOURCE / destino
        
        with open(destino_path, "wb") as fout:
            fout.write(obj["Body"].read())
        print(f"copiado do S3: {destino}")

    # Microdados
    for zip_name in MICRODADOS_ZIPS:
        year = _YEAR.search(zip_name).group(1)
        s3_key = f"{S3_PREFIX}{zip_name}"
        
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        zip_bytes = BytesIO(obj["Body"].read())
        
        with zipfile.ZipFile(zip_bytes) as zf:
            for member in zf.namelist():
                stem = Path(member).stem.upper()
                if member.startswith("DADOS/") and stem in TS_TABLES:
                    dest_dir = SOURCE / TS_TABLES[stem]
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    target = dest_dir / f"{year}.csv"
                    with zf.open(member) as fin, open(target, "wb") as fout:
                        shutil.copyfileobj(fin, fout, length=1024 * 1024)
                    print(f"extraido: {TS_TABLES[stem]}/{year}.csv")

    print(f"\nOK -> {SOURCE}")


if __name__ == "__main__":
    preparar()