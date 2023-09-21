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
from nxontology_ml.model import CatBoostDataFormatter
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
    rs_classification: str | None
    probas: list[float]


def export_model_predictions(
    precisions_file: Path = ROOT_DIR / "data/efo_otar_slim_v3.57.0_precisions.tsv",
    features_file: Path = ROOT_DIR / "data/efo_otar_slim_v3.57.0_features.tsv",
    take: int | None = None,
    text_embeddings_enabled: bool = True,
) -> None:
    """
    1. Train a model (the best performing one) on the entire training set
    2. Run inference on the entire EFO graph (except the non-disease nodes)
    3. Export both labels and the feature values for each node
    """
    assert not precisions_file.exists(), f"{precisions_file} already exists, aborting"
    assert not features_file.exists(), f"{features_file} already exists, aborting"

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
    )
    model.fit(
        X=Pool(
            data=X_transform,
            label=list(y),
        )
    )

    # 2. Do inference on new version of the ontology
    target_nodes: list[str] = list(get_disease_nodes(take=take))
    target_features = feature_pipeline.transform(target_nodes)
    target_labels = model.predict(target_features)
    target_probas = model.predict_proba(target_features)
    training_set_labels = {id_: str(label) for id_, label in zip(X, y, strict=True)}

    # 3. Write labels and features into files
    labels_output: list[NodeLabelOutput] = [
        NodeLabelOutput(
            identifier=node_id,
            precision=rs_cls_to_precision[label[0]],
            probas=probas.tolist(),
            rs_classification=training_set_labels.get(node_id, None),
        )
        for node_id, label, probas in zip(
            target_nodes, target_labels, target_probas, strict=True
        )
    ]

    labels_output_df = pd.DataFrame(labels_output)
    labels_output_df.to_csv(precisions_file, sep="\t", index=False)

    features_output_df = pd.DataFrame(
        data=np.hstack(
            [
                np.array(target_nodes).reshape((len(target_nodes), 1)),
                target_features.cat_feature_data,
                target_features.num_feature_data,
            ]
        ),
        columns=["identifier"]
        + target_features.cat_feature_names
        + target_features.num_feature_names,
    )
    features_output_df.to_csv(features_file, sep="\t", index=False)


if __name__ == "__main__":
    export_model_predictions()  # pragma: no cover
