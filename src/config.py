"""
Configurações centrais do projeto.

Tech Challenge — Fase 3
Predição e Inteligência Analítica para Alfabetização no Brasil
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"

MODELS_DIR = ROOT / "models"
IMAGES_DIR = ROOT / "images"
REPORTS_DIR = ROOT / "reports"

for _d in (EXTERNAL_DIR, PROCESSED_DIR, INTERIM_DIR, MODELS_DIR, IMAGES_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Parâmetros de negócio
# --------------------------------------------------------------------------- #
# Ponto de corte oficial da escala Saeb (Pesquisa Alfabetiza Brasil / INEP, 2023).
# Aluno com proficiência >= 743 é considerado ALFABETIZADO.
CORTE_ALFABETIZACAO = 743.0

# Meta do Compromisso Nacional Criança Alfabetizada
META_NACIONAL_2030 = 80.0

ANOS = (2023, 2024, 2025)

# Desenho de validação temporal (out-of-time):
#   treino -> alunos de ANO_TREINO, com contexto (features) de ANO_TREINO - 1
#   teste  -> alunos de ANO_TESTE,  com contexto (features) de ANO_TESTE  - 1
ANO_TREINO = 2024
ANO_TESTE = 2025

RANDOM_STATE = 42

# Dependência administrativa (INEP)
DEPENDENCIA = {1: "Federal", 2: "Estadual", 3: "Municipal", 4: "Privada"}
REDE_PUBLICA = (1, 2, 3)
REDE_MUNICIPAL = 3

# Arquivos brutos
ZIP_MICRODADOS = {
    2023: RAW_DIR / "microdados_avaliacao_da_alfabetizacao_2023.zip",
    2024: RAW_DIR / "microdados_avaliacao_da_alfabetizacao_2024.zip",
    2025: RAW_DIR / "microdados_AEEB_2025.zip",
}

CSV_METAS_MUNICIPIO = RAW_DIR / "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_municipio.csv.gz"
CSV_METAS_UF = RAW_DIR / "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_uf.csv.gz"
CSV_METAS_BRASIL = RAW_DIR / "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_brasil.csv.gz"
CSV_INDICADOR_MUNICIPIO = RAW_DIR / "br_inep_avaliacao_alfabetizacao_municipio.csv.gz"
CSV_INDICADOR_UF = RAW_DIR / "br_inep_avaliacao_alfabetizacao_uf.csv.gz"

# Fontes externas (IBGE) já materializadas em data/external
CSV_IBGE_MUNICIPIOS = EXTERNAL_DIR / "ibge_municipios.csv"
CSV_IBGE_SOCIOECONOMICO = EXTERNAL_DIR / "ibge_socioeconomico.csv"

# --------------------------------------------------------------------------- #
# Colunas proibidas (controle de data leakage)
# --------------------------------------------------------------------------- #
# Qualquer coluna abaixo carrega, direta ou indiretamente, a resposta do ano
# corrente e NUNCA pode entrar como feature.
COLUNAS_LEAKAGE = [
    "VL_PROFICIENCIA_LP",     # define o alvo (>= 743)
    "IN_ALFABETIZADO",        # o próprio alvo
    "VL_PESO_ALUNO_LP",       # peso amostral calculado a posteriori
    "IN_PREENCHIMENTO_LP",    # status pós-aplicação da prova
    "CO_CADERNO_LP",          # artefato do instrumento
    "TX_RESPOSTA_BLOCO_1", "TX_GABARITO_BLOCO_1",
    "TX_RESPOSTA_BLOCO_2", "TX_GABARITO_BLOCO_2",
    "TX_RESPOSTA_BLOCO_3", "TX_GABARITO_BLOCO_3",
    "TX_RESPOSTA_BLOCO_4", "TX_GABARITO_BLOCO_4",
    "CO_BLOCO_1", "CO_BLOCO_2", "CO_BLOCO_3", "CO_BLOCO_4",
]

# --------------------------------------------------------------------------- #
# Identidade visual dos gráficos (tema escuro, minimalista)
# --------------------------------------------------------------------------- #
# A paleta categórica foi validada para daltonismo e contraste sobre o fundo
# escuro (bandas de luminosidade OKLCH, separação CVD ΔE >= 8 entre pares
# adjacentes e contraste >= 3:1 contra a superfície).
COR_FUNDO = "#0E0E12"
COR_PAINEL = "#16161D"
COR_TEXTO = "#E8E8EF"
COR_TEXTO_FRACO = "#9A9AAB"
COR_GRID = "#26262F"

PALETA = ["#9085e9", "#199e70", "#d95926", "#3987e5", "#c98500"]
COR_PRIMARIA = PALETA[0]
COR_SECUNDARIA = PALETA[1]

# Rampa sequencial (magnitude contínua), do claro ao escuro — um único matiz.
RAMPA_SEQUENCIAL = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]

# Escala divergente (polaridade: abaixo/acima da meta) — dois matizes + cinza neutro.
DIVERGENTE_NEGATIVO = "#e66767"
DIVERGENTE_NEUTRO = "#383835"
DIVERGENTE_POSITIVO = "#3987e5"

# Cores de status (nunca reaproveitadas como série)
STATUS = {
    "bom": "#0ca30c",
    "atencao": "#fab219",
    "grave": "#ec835a",
    "critico": "#d03b3b",
}
