import numpy as np
from catboost import CatBoostClassifier, Pool
from nxontology import NXOntology
from sklearn.pipeline import make_pipeline
from transformers import Pipeline

from nxontology_ml.data import get_efo_otar_slim, read_training_data
from nxontology_ml.efo import NodeXrefFeatures
from nxontology_ml.features import (
    NodeInfoFeatures,
    PrepareNodeFeatures,
    SubsetsFeatures,
    TherapeuticAreaFeatures,
)
from nxontology_ml.gpt_tagger import GptTagFeatures
from nxontology_ml.model.config import DEFAULT_MODEL_CONFIG, ModelConfig
from nxontology_ml.model.formatter import MODEL_SEED, CatBoostDataFormatter
from nxontology_ml.text_embeddings.text_embeddings_transformer import (
    TextEmbeddingsTransformer,
)


def train_model(
    conf: ModelConfig = DEFAULT_MODEL_CONFIG,
    nxo: NXOntology[str] | None = None,
    training_set: tuple[np.ndarray, np.ndarray] | None = None,
    take: int | None = 0,
) -> tuple[Pipeline, CatBoostClassifier]:
    nxo = nxo or get_efo_otar_slim()
    nxo.freeze()
    (X, y) = training_set or read_training_data(
        filter_out_non_disease=True, nxo=nxo, take=take
    )

    feature_pipeline: Pipeline = make_pipeline(
        PrepareNodeFeatures(nxo=nxo),
        NodeInfoFeatures(),
        NodeXrefFeatures(),
        SubsetsFeatures(enabled=conf.subsets_enabled),
        TherapeuticAreaFeatures(enabled=conf.ta_enabled),
        GptTagFeatures.from_config(conf.gpt_tagger_config),
        TextEmbeddingsTransformer.from_config(
            enabled=conf.embedding_enabled,
            pca_components=conf.pca_components,
            use_lda=conf.use_lda,
        ),
        CatBoostDataFormatter(),
    )
    X_transform = feature_pipeline.fit_transform(X, y)
    model = CatBoostClassifier(
        eval_metric=conf.get_eval_metric,
        depth=conf.depth,
        custom_metric=["MultiClass", "AUC", "F1"],
        learning_rate=0.5,
        iterations=conf.iterations,
        metric_period=250,
        random_seed=MODEL_SEED,
    )
    model.fit(
        X=Pool(
            data=X_transform,
            label=list(y),
        )
    )
    return feature_pipeline, model
