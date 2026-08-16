# TechChallenge 2 - Workflow de Execução

## Visão Geral

Pipeline de dados educacionais em arquitetura Medallion (Bronze/Silver/Gold) implementado no Databricks.

**Status:** Projeto em produção com execução automatizada via Databricks Jobs.

---

## 📂 Estrutura do Projeto

```
TechChallenge_2/
├── .env                          # Credenciais AWS (não versionado em .gitignore)
├── data/                         # Dados brutos originais (8 arquivos)
├── source/                       # Dados preparados (gerados por prep_source.py)
│   ├── ts_aluno/                 # Time-series por aluno (2023, 2024, 2025)
│   ├── ts_municipio/             # Time-series por município
│   └── ts_estado/                # Time-series por estado
├── scripts/                      # Scripts utilitários
│   ├── README.md                 # Documentação dos scripts
│   ├── upload_to_s3.py           # Upload data/ → S3
│   ├── prep_source.py            # Download S3 → source/
│   └── export_to_s3.py           # Export Unity Catalog → S3
└── projeto/
    ├── bronze/                   # Camada Bronze - Ingestão
    │   └── 01_bronze_ingestao.ipynb
    ├── silver/                   # Camada Silver - Limpeza/Validação
    │   └── 02_silver_limpeza_validacao.ipynb
    ├── gold/                     # Camada Gold - Datasets Analíticos
    │   └── 03_gold_datasets_analiticos.ipynb
    ├── quality/                  # Validação de qualidade
    └── docs/                     # Documentação técnica
```

---

## 🔄 Fluxo de Execução Completo

### **Passo 1: Preparação dos Dados**

#### 1.1 Upload para S3 (opcional, se dados ainda não estão no bucket)
```bash
python scripts/upload_to_s3.py
```
- **Origem:** `data/` (arquivos locais)
- **Destino:** `s3://amzn-s3-fiap-tech2/data/`
- **Arquivos:** 8 arquivos (.csv.gz e .zip)

#### 1.2 Preparação da Source
```bash
python scripts/prep_source.py
```
- **Origem:** `s3://amzn-s3-fiap-tech2/data/`
- **Destino:** `TechChallenge_2/source/`
- **Processamento:**
  - Descompacta arquivos .csv.gz
  - Extrai CSVs dos .zip (TS_ALUNO, TS_MUNICIPIO, TS_ESTADO)
  - Organiza por ano (2023, 2024, 2025)

---

### **Passo 2: Pipeline Databricks (Bronze → Silver → Gold)**

#### Execução Manual (por notebook)
1. Execute `projeto/bronze/01_bronze_ingestao.ipynb`
2. Execute `projeto/silver/02_silver_limpeza_validacao.ipynb`
3. Execute `projeto/gold/03_gold_datasets_analiticos.ipynb`

#### Execução Automatizada (Databricks Job)

**Job ID:** `584688505156294`  
**Job Name:** `TechChallenge Pipeline - Bronze Silver Gold`  
**URL:** https://dbc-082a3d64-ec06.cloud.databricks.com/#job/584688505156294

**Configuração:**
- **Compute:** Serverless (queue habilitada)
- **Max concurrent runs:** 1 (executa um de cada vez)
- **Notificações:** Email para justinofilipe03@gmail.com (sucesso e falha)

**Tasks configuradas:**

```
┌─────────────────────────────────────────────────────────┐
│ 1. Bronze_Ingestao                                      │
│    Path: projeto/bronze/01_bronze_ingestao.ipynb        │
│    Timeout: 1 hora | Retries: 2                         │
│    Descrição: Ingesta dados de source/ e cria          │
│               tabelas Bronze no Unity Catalog           │
└────────────────────┬────────────────────────────────────┘
                     │ (depends_on)
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Silver_Limpeza_Validacao                             │
│    Path: projeto/silver/02_silver_limpeza_validacao.ipynb│
│    Timeout: 1 hora | Retries: 2                         │
│    Descrição: Limpa e valida dados Bronze, criando     │
│               tabelas Silver                            │
└────────────────────┬────────────────────────────────────┘
                     │ (depends_on)
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Gold_Datasets_Analiticos                             │
│    Path: projeto/gold/03_gold_datasets_analiticos.ipynb │
│    Timeout: 1 hora | Retries: 2                         │
│    Descrição: Cria datasets analíticos finais a partir │
│               das tabelas Silver                        │
└─────────────────────────────────────────────────────────┘
```

**Como executar o Job:**

1. **Via UI:**
   - Acesse: https://dbc-082a3d64-ec06.cloud.databricks.com/#job/584688505156294
   - Clique em "Run Now"
   - Monitore: Tasks executam em sequência (Bronze → Silver → Gold)
   - Receba notificação por email ao finalizar

2. **Via CLI:**
   ```bash
   databricks jobs run-now --job-id 584688505156294
   ```

3. **Agendar execução:**
   - Acesse o Job na UI
   - Configure trigger (schedule/periodic/file_arrival/table_update)
   - Exemplos:
     - Diário às 6h: `0 6 * * *` (cron)
     - A cada 6 horas: periodic interval
     - Quando tabelas source/ forem atualizadas

**Comportamento:**
- ✅ Se Bronze falhar → Silver e Gold não executam
- ✅ Se Silver falhar → Gold não executa
- ✅ Cada task retry automático até 2 vezes em caso de falha
- ✅ Email enviado ao concluir (sucesso ou falha)

---

### **Passo 3: Exportação de Backup (opcional)**

```bash
python scripts/export_to_s3.py
```
- **Origem:** Unity Catalog (`workspace.bronze`, `workspace.silver`, `workspace.gold`)
- **Destino:** `s3://amzn-s3-fiap-tech2/processed/`
- **Formato:** Parquet
- **Estrutura:**
  - `processed/bronze/*.parquet`
  - `processed/silver/*.parquet`
  - `processed/gold/*.parquet`

---

## 🎯 Camadas de Dados (Medallion Architecture)

### **Bronze (Raw Data)**
- **Schema:** `workspace.bronze`
- **Propósito:** Dados brutos com validação mínima
- **Características:**
  - Schema preservado da fonte
  - Particionado por ano
  - Histórico completo

### **Silver (Cleaned Data)**
- **Schema:** `workspace.silver`
- **Propósito:** Dados limpos e validados
- **Características:**
  - Sem duplicatas
  - Tipos padronizados
  - Valores consistentes
  - Validações de qualidade

### **Gold (Analytics-Ready)**
- **Schema:** `workspace.gold`
- **Propósito:** Datasets analíticos finais
- **Características:**
  - Agregações pré-calculadas
  - Integração multi-fonte
  - Otimizado para consultas
  - Pronto para BI/ML

---

## 📊 Unity Catalog

Todas as tabelas são gerenciadas pelo Unity Catalog:
- **Catálogo:** `workspace`
- **Schemas:** `bronze`, `silver`, `gold`
- **Formato:** Delta Lake (Parquet + logs de transação)
- **Localização:** Metastore padrão do workspace

---

## 🔐 Credenciais

O arquivo `.env` na raiz do repositório contém:
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

**⚠️ IMPORTANTE:** 
- `.env` está no `.gitignore` (não é versionado)
- Todos os scripts leem credenciais de `TechChallenge_2/.env`

---

## 🚀 Quick Start

```bash
# 1. Preparar dados
python scripts/prep_source.py

# 2. Executar pipeline via Job (recomendado)
# Acesse Databricks Jobs UI e execute "TechChallenge Pipeline - Bronze Silver Gold"

# 3. (Opcional) Exportar backup
python scripts/export_to_s3.py
```

---

## 📝 Notas

- **Compute:** Jobs usam serverless compute (sem necessidade de cluster)
- **Paralelização:** Tasks executam sequencialmente (dependências explícitas)
- **Monitoramento:** Logs disponíveis em cada task do Job
- **Retry:** Configurado 2 tentativas em caso de falha
- **Timeout:** 1 hora por task

---

## 📖 Documentação Adicional

- `scripts/README.md` - Detalhes dos scripts utilitários
- `projeto/docs/` - Documentação técnica do projeto
- Unity Catalog UI - Explorar schemas e tabelas