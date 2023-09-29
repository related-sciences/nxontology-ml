from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from nxontology import NXOntology
from sklearn.pipeline import Pipeline, make_pipeline

from experimentation.model_utils import BiasedMaeMetric
from nxontology_ml.data import get_efo_otar_slim, read_training_data
from nxontology_ml.efo import NodeXrefFeatures
from nxontology_ml.features import NodeInfoFeatures, PrepareNodeFeatures
from nxontology_ml.model import MODEL_SEED, CatBoostDataFormatter
from nxontology_ml.text_embeddings.text_embeddings_transformer import (
    TextEmbeddingsTransformer,
)
from nxontology_ml.utils import ROOT_DIR

NON_DISEASE_THERAPEUTIC_AREAS: set[str] = {
    "EFO:0005932",  # animal_disease
    "EFO:0001444",  # measurement
    "EFO:0000651",  # phenotype
    "GO:0008150",  # biological_process
    "EFO:0002571",  # medical_procedure
}


def get_disease_nodes(
    take: int | None = None,
    nxo: NXOntology[str] | None = None,
) -> Iterable[str]:
    """
    If all roots are in the set of non-disease therapeutic areas (aka roots in EFO OTAR Slim),
        then its non-disease
    """
    nxo = nxo or get_efo_otar_slim()
    for n in islice(sorted(nxo.graph), take):
        if any(
            root not in NON_DISEASE_THERAPEUTIC_AREAS for root in nxo.node_info(n).roots
        ):
            yield n


rs_cls_to_precision = {
    "01-disease-subtype": "high",
    "02-disease-root": "medium",
    "03-disease-area": "low",
}


@dataclass
class NodeLabelOutput:
    """
    Main output of the model inference tasks
    """

    identifier: str
    precision: str
    proba_high: float
    proba_medium: float
    proba_low: float
    rs_classification: str | None
    efo_label: str | None


def export_model_predictions(
    export_file: Path = ROOT_DIR / "data/efo_otar_slim_v3.57.0_precisions.tsv",
    take: int | None = None,
    text_embeddings_enabled: bool = True,
) -> None:
    """
    1. Train a model (the best performing one) on the entire training set
    2. Run inference on the entire EFO graph (except the non-disease nodes)
    3. Export both labels and the feature values for each node
    """
    assert not export_file.exists(), f"{export_file} already exists, aborting"

    # 1. Train model
    X, y = read_training_data(filter_out_non_disease=True, take=take)

    feature_pipeline: Pipeline = make_pipeline(
        PrepareNodeFeatures(),
        NodeInfoFeatures(),
        NodeXrefFeatures(),
        TextEmbeddingsTransformer.from_config(
            enabled=text_embeddings_enabled,
            pca_components=64,
            use_lda=False,
        ),
        CatBoostDataFormatter(),
    )
    X_transform = feature_pipeline.fit_transform(X, y)
    model = CatBoostClassifier(
        eval_metric=BiasedMaeMetric(),
        custom_metric=["MultiClass", "AUC", "F1"],
        learning_rate=0.5,
        iterations=5000,
        metric_period=250,
        random_seed=MODEL_SEED,
    )
    model.fit(
        X=Pool(
            data=X_transform,
            label=list(y),
        )
    )

    # 2. Do inference on new version of the ontology
    nxo = get_efo_otar_slim()
    target_nodes: list[str] = list(get_disease_nodes(take=take, nxo=nxo))
    target_features = feature_pipeline.transform(target_nodes)
    target_labels = model.predict(target_features)
    target_probas = model.predict_proba(target_features)
    training_set_labels = {id_: str(label) for id_, label in zip(X, y, strict=True)}

    # 3. Write labels and features into files
    labels_output: list[NodeLabelOutput] = [
        NodeLabelOutput(
            identifier=node_id,
            precision=rs_cls_to_precision[label[0]],
            proba_high=probas[0],
            proba_medium=probas[1],
            proba_low=probas[2],
            rs_classification=training_set_labels.get(node_id, None),
            efo_label=nxo.node_info(node_id).data["efo_label"],
        )
        for node_id, label, probas in zip(
            target_nodes, target_labels, target_probas, strict=True
        )
    ]

    features_output_df = pd.DataFrame(
        data=np.hstack(
            [
                target_features.cat_feature_data,
                np.array(target_features.num_feature_data, dtype=float),
            ]
        ),
        columns=np.array(
            target_features.cat_feature_names + target_features.num_feature_names
        ),
    )

    output_df = pd.concat(
        [pd.DataFrame(labels_output), features_output_df], axis=1
    ).convert_dtypes()
    output_df.to_csv(
        export_file,
        sep="\t",
        index=False,
        float_format="%.5g",
    )


if __name__ == "__main__":
    export_model_predictions()  # pragma: no cover
