from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, FeaturesData, Pool
from sklearn.metrics import classification_report

from nxontology_ml.efo import get_efo_otar_slim
from nxontology_ml.utils import ROOT_DIR


@dataclass
class TrainingRecord:
    efo_otar_slim_id: str
    efo_label: str
    rs_classification: str


class FeatureType(Enum):
    LABEL = 1
    NUMERICAL = 2
    CATEGORICAL = 3
    IGNORED = 4

    @classmethod
    def from_feature_name(cls, name: str) -> "FeatureType":
        if name == "__label__":
            return cls.LABEL
        if cls._is_numerical(name):
            return cls.NUMERICAL
        if cls._is_categorical(name):
            return cls.CATEGORICAL
        assert cls._is_ignored(name), f"Unknown feature: {name}"
        return cls.IGNORED

    @staticmethod
    def _is_numerical(name: str) -> bool:
        if name == "depth":
            return True
        for prefix in {"n_", "intrinsic_", "xref_"}:
            if name.startswith(prefix):
                return True
        return False

    @staticmethod
    def _is_categorical(name: str) -> bool:
        return name in {"prefix", "is_gwas_trait"}

    @staticmethod
    def _is_ignored(name: str) -> bool:
        return name in {"identifier", "name"}


def read_training_data(
    take: int | None = None,
    data_path: Path = ROOT_DIR / "data/efo_otar_slim_v3.43.0_rs_classification.tsv",
) -> Iterable[dict[str, Any]]:
    # Get labelled data
    def _labelled_data() -> Iterable[TrainingRecord]:
        data = iter(data_path.read_text().splitlines())
        assert next(data).strip() == "efo_otar_slim_id	efo_label	rs_classification"
        for line in data:
            efo_otar_slim_id, efo_label, rs_classification = line.split("\t", 3)
            yield TrainingRecord(efo_otar_slim_id, efo_label, rs_classification)

    # Get Ontology
    nxo = get_efo_otar_slim()
    nodes: set[str] = set(nxo.graph)

    # Filter nodes from the ontology that are labelled
    for record in islice(_labelled_data(), take):
        id_ = record.efo_otar_slim_id
        if id_ in nodes:
            metrics = nxo.node_info(id_).get_metrics()
            metrics["__label__"] = record.rs_classification
            yield metrics


@dataclass
class TrainingData:
    train_data: FeaturesData
    train_labels: np.ndarray
    test_data: FeaturesData
    test_labels: np.ndarray


def to_features_data(
    records: Iterable[dict[str, Any]],
    train_frac: float = 0.85,  # 85% of data for training, 15% for testing
) -> TrainingData:
    # To pandas
    df = pd.DataFrame.from_records(records)
    df = df.sample(frac=1)  # Shuffle
    training_set_size = int(len(df) * train_frac)

    # Split features by type

    num_feature_names = []
    num_feature_data = []
    cat_feature_names = []
    cat_feature_data = []
    labels = []

    for feature_name in df:
        feature_type = FeatureType.from_feature_name(feature_name)
        if feature_type == FeatureType.LABEL:
            labels = df[feature_name].to_numpy()
        elif feature_type == FeatureType.NUMERICAL:
            num_feature_names.append(feature_name)
            num_feature_data.append(df[feature_name].to_numpy(dtype=np.float32))
        elif feature_type == FeatureType.CATEGORICAL:
            cat_feature_names.append(feature_name)
            d = [str(x).encode() for x in df[feature_name]]
            cat_feature_data.append(np.array(d, dtype=object))

    num_feature_data = np.stack(num_feature_data).transpose()
    cat_feature_data = np.stack(cat_feature_data).transpose()

    return TrainingData(
        train_data=FeaturesData(
            num_feature_data=num_feature_data[:training_set_size],
            cat_feature_data=cat_feature_data[:training_set_size],
            num_feature_names=num_feature_names,
            cat_feature_names=cat_feature_names,
        ),
        train_labels=labels[:training_set_size],
        test_data=FeaturesData(
            num_feature_data=num_feature_data[training_set_size:],
            cat_feature_data=cat_feature_data[training_set_size:],
            num_feature_names=num_feature_names,
            cat_feature_names=cat_feature_names,
        ),
        test_labels=labels[training_set_size:],
    )


def train_model(training_data: TrainingData) -> None:
    train_pool = Pool(data=training_data.train_data, label=training_data.train_labels)
    test_pool = Pool(data=training_data.test_data, label=training_data.test_labels)
    model: CatBoostClassifier = CatBoostClassifier(metric_period=100)
    model.fit(train_pool)

    print("> Feature importance:")
    print(model.get_feature_importance(test_pool, prettified=True))
    print()
    print("> Classification report:")
    print(classification_report(training_data.test_labels, model.predict(test_pool)))
