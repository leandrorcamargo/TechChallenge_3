"""Upload dos dados da pasta data/ para o S3.

Faz upload dos arquivos que ja temos localmente para o bucket S3,
mantendo a estrutura esperada pelo prep_source.py.

Uso:
    python scripts/upload_to_s3.py
"""

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "boto3", "python-dotenv"])

import os
from pathlib import Path
import boto3
from dotenv import load_dotenv

# Configuracao
REPO = Path("/Workspace/Users/justinofilipe03@gmail.com/TechChallenge_2")
DATA_DIR = REPO / "data"
ENV_PATH = REPO / ".env"
S3_BUCKET = "amzn-s3-fiap-tech2"
S3_PREFIX = "data/"

# Arquivos para upload
ARQUIVOS = [
    "br_inep_avaliacao_alfabetizacao_uf.csv.gz",
    "br_inep_avaliacao_alfabetizacao_municipio.csv.gz",
    "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_brasil.csv.gz",
    "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_uf.csv.gz",
    "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_municipio.csv.gz",
    "microdados_avaliacao_da_alfabetizacao_2023.zip",
    "microdados_avaliacao_da_alfabetizacao_2024.zip",
    "microdados_AEEB_2025.zip",
]


def upload_para_s3():
    load_dotenv(ENV_PATH)
    s3 = boto3.client("s3")
    
    print(f"Iniciando upload para s3://{S3_BUCKET}/{S3_PREFIX}\n")
    
    total_bytes = 0
    sucesso = 0
    
    for arquivo in ARQUIVOS:
        arquivo_path = DATA_DIR / arquivo
        
        if not arquivo_path.exists():
            print(f"IGNORADO: {arquivo} (nao encontrado)")
            continue
        
        s3_key = f"{S3_PREFIX}{arquivo}"
        tamanho_mb = arquivo_path.stat().st_size / (1024 * 1024)
        
        try:
            print(f"Uploading {arquivo} ({tamanho_mb:.2f} MB)...", end=" ")
            s3.upload_file(str(arquivo_path), S3_BUCKET, s3_key)
            print("OK")
            total_bytes += arquivo_path.stat().st_size
            sucesso += 1
        except Exception as e:
            print(f"ERRO: {e}")
    
    total_mb = total_bytes / (1024 * 1024)
    print(f"\nUpload concluido: {sucesso}/{len(ARQUIVOS)} arquivos ({total_mb:.2f} MB)")
    
    # Verifica
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
    if 'Contents' in response:
        print(f"\n{len(response['Contents'])} arquivo(s) no bucket:")
        for obj in response['Contents']:
            print(f"  - {obj['Key']}")


if __name__ == "__main__":
    upload_para_s3()