"""
Materializa as bases analíticas em data/processed/.

Uso:
    python -m src.preprocessing.build_datasets

Gera:
    aluno_2024.parquet / aluno_2025.parquet     (grão aluno, contexto defasado)
    municipio_2024.csv / municipio_2025.csv     (grão município, rede Municipal)
    validacao_fontes.csv                        (checagem cruzada com a Fase 2)
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.preprocessing import features, loaders


def validar_contra_fase2(ano: int) -> pd.DataFrame:
    """
    Confere a taxa de alfabetização calculada dos microdados contra a taxa
    publicada na Base dos Dados (mesma fonte usada na camada Gold da Fase 2).
    """
    alunos = loaders.carregar_alunos(ano)
    nossa = features.agregar_municipios(alunos, apenas_rede_municipal=True)[
        ["id_municipio", "mun_taxa_alfabetizacao"]
    ].rename(columns={"mun_taxa_alfabetizacao": "taxa_microdados"})

    metas = loaders.carregar_metas_municipio()
    oficial = metas[metas["ano"] == ano][["id_municipio", "taxa_alfabetizacao"]].rename(
        columns={"taxa_alfabetizacao": "taxa_publicada"}
    )

    comp = nossa.merge(oficial, on="id_municipio", how="inner").dropna()
    comp["diferenca_abs"] = (comp["taxa_microdados"] - comp["taxa_publicada"]).abs()
    return comp


def main() -> None:
    print("=" * 72)
    print("CONSTRUÇÃO DAS BASES ANALÍTICAS — TECH CHALLENGE FASE 3")
    print("=" * 72)

    for ano in (config.ANO_TREINO, config.ANO_TESTE):
        df = features.construir_dataset_aluno(ano)
        destino = config.PROCESSED_DIR / f"aluno_{ano}.parquet"
        df.to_parquet(destino, index=False)
        taxa = df["alfabetizado"].mean() * 100
        print(
            f"[aluno {ano}] {len(df):>9,} alunos | {df.shape[1]} colunas | "
            f"alfabetizados {taxa:.2f}% | contexto de {ano - 1} -> {destino.name}"
        )

        dfm = features.construir_dataset_municipio(ano)
        destino_m = config.PROCESSED_DIR / f"municipio_{ano}.csv"
        dfm.to_csv(destino_m, index=False)
        print(
            f"[munic {ano}] {len(dfm):>9,} municípios | "
            f"não atingiram a meta: {dfm['nao_atingiu_meta'].mean() * 100:.1f}% -> {destino_m.name}"
        )

    val = validar_contra_fase2(2024)
    val.to_csv(config.PROCESSED_DIR / "validacao_fontes.csv", index=False)
    print(
        f"[validação] {len(val):,} municípios comparados com a Base dos Dados | "
        f"diferença média {val['diferenca_abs'].mean():.4f} p.p. | "
        f"máxima {val['diferenca_abs'].max():.4f} p.p."
    )


if __name__ == "__main__":
    main()
