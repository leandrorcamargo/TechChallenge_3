"""
Camada de visualização.

Estilo escuro e minimalista, com paleta validada para contraste e daltonismo
(ver src/config.py). Todas as funções salvam a figura em images/ e devolvem o
caminho, para que notebooks, README e apresentação usem exatamente o mesmo
artefato.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src import config

CMAP_SEQUENCIAL = LinearSegmentedColormap.from_list("seq", config.RAMPA_SEQUENCIAL[::-1])
CMAP_DIVERGENTE = LinearSegmentedColormap.from_list(
    "div", [config.DIVERGENTE_NEGATIVO, config.DIVERGENTE_NEUTRO, config.DIVERGENTE_POSITIVO]
)


def aplicar_estilo() -> None:
    """Aplica o tema escuro do projeto ao matplotlib."""
    mpl.rcParams.update(
        {
            "figure.facecolor": config.COR_FUNDO,
            "axes.facecolor": config.COR_FUNDO,
            "savefig.facecolor": config.COR_FUNDO,
            "text.color": config.COR_TEXTO,
            "axes.labelcolor": config.COR_TEXTO_FRACO,
            "axes.edgecolor": config.COR_GRID,
            "axes.titlecolor": config.COR_TEXTO,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlepad": 14,
            "axes.labelsize": 10,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": config.COR_GRID,
            "grid.linewidth": 0.8,
            "xtick.color": config.COR_TEXTO_FRACO,
            "ytick.color": config.COR_TEXTO_FRACO,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.labelcolor": config.COR_TEXTO,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "font.family": "DejaVu Sans",
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def _limpar(ax, eixo_x: bool = True) -> None:
    """Remove o excesso: sem molduras, grade só no eixo do valor."""
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(config.COR_GRID)
    ax.grid(axis="x" if not eixo_x else "y", alpha=0.6)
    ax.grid(axis="y" if not eixo_x else "x", visible=False)


def salvar(fig, nome: str) -> Path:
    caminho = config.IMAGES_DIR / nome
    fig.savefig(caminho, facecolor=config.COR_FUNDO)
    plt.close(fig)
    return caminho


# --------------------------------------------------------------------------- #
# Gráficos de EDA
# --------------------------------------------------------------------------- #
def distribuicao_proficiencia(series_por_ano: dict[int, pd.Series], nome: str) -> Path:
    """Distribuição da proficiência por ano, com o corte de alfabetização."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, (ano, s) in enumerate(sorted(series_por_ano.items())):
        ax.hist(
            s.dropna(),
            bins=90,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=config.PALETA[i],
            label=str(ano),
        )
    ax.axvline(config.CORTE_ALFABETIZACAO, color=config.STATUS["atencao"], linestyle="--", linewidth=1.6)
    ax.annotate(
        f"corte {config.CORTE_ALFABETIZACAO:.0f}\nalfabetizado →",
        xy=(config.CORTE_ALFABETIZACAO, ax.get_ylim()[1] * 0.86),
        xytext=(6, 0),
        textcoords="offset points",
        color=config.STATUS["atencao"],
        fontsize=9,
    )
    ax.set_title("Proficiência em Língua Portuguesa — 2º ano do fundamental")
    ax.set_xlabel("Proficiência (escala Saeb)")
    ax.set_ylabel("Densidade")
    ax.legend(title="Ano", title_fontsize=9)
    _limpar(ax)
    return salvar(fig, nome)


def barras_horizontais(
    rotulos, valores, titulo: str, nome: str, sublegenda: str = "", cor=None, destaque=None
) -> Path:
    """Ranking horizontal com rótulos diretos (sem eixo redundante)."""
    fig, ax = plt.subplots(figsize=(8.5, max(3.2, 0.34 * len(rotulos) + 1.4)))
    cores = [cor or config.COR_PRIMARIA] * len(rotulos)
    if destaque is not None:
        for i in destaque:
            cores[i] = config.STATUS["critico"]
    y = np.arange(len(rotulos))
    ax.barh(y, valores, color=cores, height=0.62)
    ax.set_yticks(y, rotulos)
    ax.invert_yaxis()
    for yi, v in zip(y, valores):
        ax.text(v + max(valores) * 0.012, yi, f"{v:,.1f}".replace(",", "."), va="center",
                fontsize=9, color=config.COR_TEXTO)
    ax.set_xlim(0, max(valores) * 1.16)
    ax.set_xticks([])
    ax.set_title(titulo)
    if sublegenda:
        ax.set_xlabel(sublegenda)
    for lado in ("top", "right", "bottom"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color(config.COR_GRID)
    ax.grid(False)
    return salvar(fig, nome)


def correlacoes(df: pd.DataFrame, colunas: list[str], titulo: str, nome: str) -> Path:
    """Matriz de correlação de Spearman (relações monotônicas, não só lineares)."""
    corr = df[colunas].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(1.0 + 0.52 * len(colunas), 0.9 + 0.46 * len(colunas)))
    im = ax.imshow(corr, cmap=CMAP_DIVERGENTE, vmin=-1, vmax=1)
    ax.set_xticks(range(len(colunas)), colunas, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(colunas)), colunas, fontsize=8)
    for i in range(len(colunas)):
        for j in range(len(colunas)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color=config.COR_TEXTO if abs(v) < 0.6 else "#0E0E12")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=config.COR_TEXTO_FRACO, labelsize=8)
    ax.set_title(titulo)
    ax.grid(False)
    return salvar(fig, nome)


def dispersao(x, y, titulo: str, xlabel: str, ylabel: str, nome: str, cor_valores=None,
              rotulo_cor: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    sc = ax.scatter(
        x, y, c=cor_valores if cor_valores is not None else config.COR_PRIMARIA,
        cmap=CMAP_SEQUENCIAL if cor_valores is not None else None,
        s=12, alpha=0.55, linewidths=0,
    )
    if cor_valores is not None:
        cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label=rotulo_cor)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=config.COR_TEXTO_FRACO, labelsize=8)
        cb.set_label(rotulo_cor, color=config.COR_TEXTO_FRACO, fontsize=9)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _limpar(ax)
    return salvar(fig, nome)


# --------------------------------------------------------------------------- #
# Gráficos de avaliação de modelo
# --------------------------------------------------------------------------- #
def curva_roc(curvas: dict[str, tuple], nome: str, titulo: str = "Curva ROC — conjunto de teste") -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    for i, (rotulo, (fpr, tpr, auc)) in enumerate(curvas.items()):
        ax.plot(fpr, tpr, color=config.PALETA[i % len(config.PALETA)], label=f"{rotulo} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=config.COR_TEXTO_FRACO, linewidth=1.2, linestyle=":", label="Aleatório")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.set_title(titulo)
    ax.legend(loc="lower right")
    _limpar(ax)
    return salvar(fig, nome)


def matriz_confusao(cm: np.ndarray, classes: list[str], titulo: str, nome: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    pct = cm / cm.sum()
    ax.imshow(pct, cmap=CMAP_SEQUENCIAL)
    ax.set_xticks(range(len(classes)), classes, fontsize=9)
    ax.set_yticks(range(len(classes)), classes, fontsize=9, rotation=90, va="center")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Observado")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j] * 100:.1f}%".replace(",", "."),
                    ha="center", va="center", fontsize=10,
                    color="#0E0E12" if pct[i, j] > 0.35 else config.COR_TEXTO)
    ax.set_title(titulo)
    ax.grid(False)
    return salvar(fig, nome)


def importancias(nomes, valores, titulo: str, nome: str, xlabel: str = "Importância") -> Path:
    ordem = np.argsort(valores)
    fig, ax = plt.subplots(figsize=(8.2, max(3.4, 0.32 * len(nomes) + 1.2)))
    ax.barh(np.arange(len(nomes)), np.asarray(valores)[ordem], color=config.COR_PRIMARIA, height=0.62)
    ax.set_yticks(np.arange(len(nomes)), np.asarray(nomes)[ordem], fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_title(titulo)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="x", alpha=0.5)
    ax.grid(axis="y", visible=False)
    return salvar(fig, nome)


def calibracao(bins_prob, bins_real, nome: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot([0, 1], [0, 1], linestyle=":", color=config.COR_TEXTO_FRACO, linewidth=1.2,
            label="Calibração perfeita")
    ax.plot(bins_prob, bins_real, marker="o", color=config.COR_PRIMARIA, label="Modelo")
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Frequência observada")
    ax.set_title("Curva de calibração")
    ax.legend(loc="upper left")
    _limpar(ax)
    return salvar(fig, nome)
