"""Experimento 2: reduzir os embeddings dos chunks para 2D com PCA feito na mao e plotar.

Uso: python -m experiments.02_visualize_embeddings
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from app.config import get_settings
from experiments.common import OUTPUT_DIR, load_chunks

OUTPUT_PATH = OUTPUT_DIR / "embeddings_pca.png"


def pca_2d(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Projeta n x dim em n x 2 e devolve tambem a fracao de variancia de cada componente."""
    centered = matrix - matrix.mean(dim=0)
    _, singular_values, components = torch.linalg.svd(centered, full_matrices=False)
    coords = centered @ components[:2].T
    explained = singular_values**2 / (singular_values**2).sum()
    return coords, explained


def similarity_by_group(chunks: list[dict], matrix: torch.Tensor) -> tuple[float, float]:
    """Media do cosseno entre chunks do mesmo documento e de documentos diferentes."""
    similarity = matrix @ matrix.T
    documents = [chunk["document"] for chunk in chunks]
    same, different = [], []

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            target = same if documents[i] == documents[j] else different
            target.append(float(similarity[i, j]))

    return sum(same) / len(same), sum(different) / len(different)


def main() -> None:
    settings = get_settings()
    chunks, matrix = load_chunks(settings)
    coords, explained = pca_2d(matrix)

    print(f"chunks: {len(chunks)} dimensao original: {matrix.shape[1]}")
    print(
        f"variancia explicada: PC1={explained[0]:.1%} PC2={explained[1]:.1%} "
        f"(soma {explained[:2].sum():.1%})"
    )

    same_mean, different_mean = similarity_by_group(chunks, matrix)
    print(f"cosseno medio mesmo documento: {same_mean:.4f}")
    print(f"cosseno medio documentos diferentes: {different_mean:.4f}")

    documents = sorted({chunk["document"] for chunk in chunks})
    colors = plt.get_cmap("tab10")

    figure, axis = plt.subplots(figsize=(9, 7))
    for index, document in enumerate(documents):
        mask = [i for i, chunk in enumerate(chunks) if chunk["document"] == document]
        axis.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=90,
            color=colors(index),
            label=document,
        )
        for i in mask:
            axis.annotate(str(chunks[i]["id"]), coords[i].tolist(), textcoords="offset points", xytext=(6, 4))

    axis.set_title("Chunks projetados em 2D (PCA dos embeddings)")
    axis.set_xlabel(f"PC1 ({explained[0]:.1%} da variancia)")
    axis.set_ylabel(f"PC2 ({explained[1]:.1%} da variancia)")
    axis.legend(loc="best", fontsize=8)
    axis.grid(alpha=0.3)

    OUTPUT_DIR.mkdir(exist_ok=True)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=120)
    print(f"grafico salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
