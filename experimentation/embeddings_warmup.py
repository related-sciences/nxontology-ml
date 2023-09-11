from tqdm import tqdm

from nxontology_ml.data import get_efo_otar_slim, read_training_data
from nxontology_ml.text_embeddings.embeddings_model import (
    DEFAULT_EMBEDDING_MODEL,
    AutoModelEmbeddings,
)


def warmup_cache(
    ame: AutoModelEmbeddings | None = None,
    take: int | None = None,
) -> None:
    # Warm up the embedding cache
    ame = ame or AutoModelEmbeddings.from_pretrained(
        DEFAULT_EMBEDDING_MODEL,
    )
    nxo = get_efo_otar_slim()
    X, _ = read_training_data(take=take)
    for node_id in tqdm(X, desc="Fetching node embeddings", delay=0.1):
        ame.embed_node(nxo.node_info(node_id))
    print(ame._counter)


if __name__ == "__main__":
    warmup_cache()  # pragma: no cover
