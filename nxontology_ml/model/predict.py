from collections.abc import Iterable
from itertools import islice

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from nxontology import NXOntology
from sklearn.pipeline import Pipeline

from nxontology_ml.data import get_efo_otar_slim, read_training_data
from nxontology_ml.model.config import DEFAULT_MODEL_CONFIG, ModelConfig
from nxontology_ml.model.train import train_model

NON_DISEASE_THERAPEUTIC_AREAS: set[str] = {
    "EFO:0005932",  # animal_disease
    "EFO:0001444",  # measurement
    "EFO:0000651",  # phenotype
    "GO:0008150",  # biological_process
    "EFO:0002571",  # medical_procedure
    "OTAR:0000009",  # injury, poisoning or other complication
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


def predict(
    feature_pipeline: Pipeline,
    model: CatBoostClassifier,
    training_set: tuple[np.ndarray, np.ndarray],
    nxo: NXOntology[str],
    include_feature_values: bool = True,
    take: int | None = None,
) -> pd.DataFrame:
    target_nodes: list[str] = list(get_disease_nodes(take=take, nxo=nxo))
    assert len(target_nodes) > 0, "No disease node found"
    target_features = feature_pipeline.transform(target_nodes)
    target_labels = model.predict(target_features)
    target_probas = model.predict_proba(target_features)
    training_set_labels = {
        id_: str(label) for id_, label in zip(*training_set, strict=True)
    }
    model_df = pd.DataFrame(
        [
            {
                "identifier": node_id,
                "precision": rs_cls_to_precision[label[0]],
                "proba_high": probas[0],
                "proba_medium": probas[1],
                "proba_low": probas[2],
                "efo_label": nxo.node_info(node_id).data["efo_label"],
                "rs_classification": training_set_labels.get(node_id, None),
            }
            for node_id, label, probas in zip(
                target_nodes, target_labels, target_probas, strict=True
            )
        ]
    )
    if include_feature_values:
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
        model_df = pd.concat([model_df, features_output_df], axis=1).convert_dtypes()
    return model_df


def train_predict(
    conf: ModelConfig = DEFAULT_MODEL_CONFIG,
    nxo: NXOntology[str] | None = None,
    training_set: tuple[np.ndarray, np.ndarray] | None = None,
    include_feature_values: bool = True,
    train_take: int | None = None,
    predict_take: int | None = None,
) -> pd.DataFrame:
    """
    Run both model training and prediction tasks.

    Notes:
    - The training part runs on the subset of the nodes that have been manually labeled
    - The prediction part runs on all the nodes of the ontology

    :param conf: Configuration for the model.
    :param nxo: Target ontology graph. Defaults to the latest version of the ontology.
    :param training_set: Tuple[X, Y] where X are the features and Y are the labels. Defaults to the standard set of
    features.
    :param include_feature_values: Whether to include the features in the output.
    :param train_take: Use a subset of the nodes for training.
    :param predict_take: Use a subset of the nodes for prediction.
    :return: Pandas DataFrame containing the labelled data (and features, by default)
    """
    training_set = training_set or read_training_data(
        filter_out_non_disease=True, nxo=nxo, take=train_take
    )
    nxo = nxo or get_efo_otar_slim()
    feature_pipeline, model = train_model(
        conf=conf, nxo=nxo, training_set=training_set, take=train_take
    )
    return predict(
        feature_pipeline=feature_pipeline,
        model=model,
        training_set=training_set,
        nxo=nxo,
        include_feature_values=include_feature_values,
        take=predict_take,
    )
