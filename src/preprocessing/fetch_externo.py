"""
Coleta das fontes externas (IBGE) usadas para enriquecer a base analítica.

Duas chamadas às APIs públicas do IBGE:

1. /localidades/municipios  -> hierarquia territorial (município, UF, região)
2. /agregados/6579          -> população residente estimada (2021)
   /agregados/5938          -> PIB a preços correntes e participação do VAB da
                               administração pública (2021)

O resultado é materializado em data/external/ para que o pipeline seja
reproduzível offline (as APIs podem sair do ar ou mudar de contrato).
"""
from __future__ import annotations

import gzip
import io
import json
import urllib.request

import pandas as pd

from src import config

URL_MUNICIPIOS = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
URL_POPULACAO = (
    "https://servicodados.ibge.gov.br/api/v3/agregados/6579/periodos/2021"
    "/variaveis/9324?localidades=N6[all]"
)
URL_PIB = (
    "https://servicodados.ibge.gov.br/api/v3/agregados/5938/periodos/2021"
    "/variaveis/37|528?localidades=N6[all]"
)


def _get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "tech-challenge-fase3"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        bruto = resp.read()
    if resp.headers.get("Content-Encoding") == "gzip" or bruto[:2] == b"\x1f\x8b":
        bruto = gzip.decompress(bruto)
    return json.loads(bruto.decode("utf-8"))


def _extrair_uf(m: dict) -> tuple[dict, str]:
    """
    Devolve o bloco de UF e o nome da mesorregião de um município.

    Municípios criados recentemente vêm sem `microrregiao` na API; nesses casos
    caímos para a hierarquia de regiões imediatas/intermediárias.
    """
    micro = m.get("microrregiao")
    if micro:
        return micro["mesorregiao"]["UF"], micro["mesorregiao"]["nome"]
    imediata = m.get("regiao-imediata") or {}
    intermediaria = imediata.get("regiao-intermediaria") or {}
    return intermediaria["UF"], intermediaria.get("nome", "")


def baixar_municipios() -> pd.DataFrame:
    dados = _get_json(URL_MUNICIPIOS)
    linhas = []
    for m in dados:
        uf, mesorregiao = _extrair_uf(m)
        linhas.append(
            {
                "id_municipio": m["id"],
                "nome_municipio_ibge": m["nome"],
                "sigla_uf": uf["sigla"],
                "nome_uf": uf["nome"],
                "regiao": uf["regiao"]["nome"],
                "mesorregiao": mesorregiao,
            }
        )
    return pd.DataFrame(linhas)


def _serie_para_frame(dados, coluna: str, variavel_id: str) -> pd.DataFrame:
    bloco = next(v for v in dados if str(v["id"]) == variavel_id)
    linhas = []
    for s in bloco["resultados"][0]["series"]:
        valor = list(s["serie"].values())[0]
        linhas.append(
            {
                "id_municipio": int(s["localidade"]["id"]),
                coluna: pd.to_numeric(valor, errors="coerce"),
            }
        )
    return pd.DataFrame(linhas)


def baixar_socioeconomico() -> pd.DataFrame:
    pop = _serie_para_frame(_get_json(URL_POPULACAO), "populacao", "9324")
    pib_raw = _get_json(URL_PIB)
    pib = _serie_para_frame(pib_raw, "pib_mil_reais", "37")
    vab = _serie_para_frame(pib_raw, "pct_vab_adm_publica", "528")

    df = pop.merge(pib, on="id_municipio", how="outer").merge(vab, on="id_municipio", how="outer")
    df["pib_per_capita"] = (df["pib_mil_reais"] * 1_000) / df["populacao"]
    return df[["id_municipio", "populacao", "pib_per_capita", "pct_vab_adm_publica"]]


def main() -> None:
    config.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    municipios = baixar_municipios()
    municipios.to_csv(config.CSV_IBGE_MUNICIPIOS, index=False)
    print(f"[ibge] municípios: {len(municipios)} -> {config.CSV_IBGE_MUNICIPIOS}")

    socio = baixar_socioeconomico()
    socio.to_csv(config.CSV_IBGE_SOCIOECONOMICO, index=False)
    print(f"[ibge] socioeconômico: {len(socio)} -> {config.CSV_IBGE_SOCIOECONOMICO}")


if __name__ == "__main__":
    main()
