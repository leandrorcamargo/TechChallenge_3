# Scripts Utilitarios

Scripts auxiliares para gerenciamento de dados do projeto.

## Scripts

### 1. upload_to_s3.py
Faz upload dos dados brutos da pasta `data/` para o bucket S3.

**Uso:**
```bash
python scripts/upload_to_s3.py
```

**Origem:** `TechChallenge_2/data/` (arquivos .csv.gz e .zip)  
**Destino:** `s3://amzn-s3-fiap-tech2/data/`

**Quando executar:** Antes de rodar `prep_source.py`, para garantir que os dados estao no S3.

---

### 2. prep_source.py
Baixa dados do S3 e prepara a pasta `source/` no workspace.

**Uso:**
```bash
python scripts/prep_source.py
```

**Origem:** `s3://amzn-s3-fiap-tech2/data/`  
**Destino:** `TechChallenge_2/source/` (arquivos .csv.gz e pastas ts_aluno/ts_municipio/ts_estado)

**Quando executar:** Antes de rodar o notebook Bronze, para preparar os dados de entrada.

---

### 3. export_to_s3.py
Exporta Delta Tables (Bronze/Silver/Gold) do Unity Catalog para S3 em formato Parquet.

**Uso:**
```bash
python scripts/export_to_s3.py
```

**Origem:** Unity Catalog (`workspace.bronze.*`, `workspace.silver.*`, `workspace.gold.*`)  
**Destino:** `s3://amzn-s3-fiap-tech2/processed/bronze|silver|gold/`

**Quando executar:** Apos executar os notebooks Bronze/Silver/Gold, para criar backup em S3.

---

## Fluxo Completo

```
1. upload_to_s3.py  -> Envia data/ para S3
2. prep_source.py   -> Baixa do S3 e prepara source/
3. Notebooks        -> Bronze/Silver/Gold processam e salvam no Unity Catalog
4. export_to_s3.py  -> Exporta Delta Tables para S3 como backup
```

---

## Requisitos

Todos os scripts instalam suas dependencias automaticamente:
- boto3
- python-dotenv
- pyarrow (apenas export_to_s3.py)

**Credenciais AWS:** Os scripts leem do arquivo `.env` na raiz do repositorio.