import json
import re
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, FeaturesData, Pool
from pydantic import BaseModel
from pydantic_core import to_jsonable_python
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from nxontology_ml.model.config import EXPERIMENT_MODEL_DIR, ModelConfig
from nxontology_ml.model.utils import (
    BIASED_CLASS_WEIGHTS,
    CLASS_WEIGHTS,
    biased_sample_weights,
    mean_absolute_error,
    one_h_enc,
)
from nxontology_ml.utils import ROOT_DIR

"""
All the code below is to help save useful information about train model
(This data is then used in Notebooks to study the performance of various parameters)
"""


METADATA_FILENAME = "metadata.json"


class CustomMetrics(str, Enum):
    REPORT = auto()
    BIASED_REPORT = auto()
    ROC_AUC = auto()
    BIASED_ROC_AUC = auto()
    MAE = auto()
    BIASED_MAE = auto()


class StratifiedKFoldConfig(BaseModel):  # type: ignore[misc]
    n_splits: int
    shuffle: bool
    random_state: int


class ModelMetadata(BaseModel):  # type: ignore[misc]
    """
    Container to organize all the relevant metadata about a model.
    (Useful for experiment tracking over time )
    """

    start_time: datetime
    experiment: ModelConfig
    cv: StratifiedKFoldConfig
    steps: list[str]
    catboost_metadata: dict[str, Any]
    biased_class_weights: list[float]
    custom_metrics: dict[CustomMetrics, Any]
    fold: int
    feature_building_d: timedelta
    training_d: timedelta
    eval_d: timedelta
    version: str


class ModelMetadataBuilder:
    """
    Proxies a lot of the logic to spy on various parameter & build ModelMetadata
        in a consistent fashion across experiments
    """

    _start_time: datetime | None = None
    _cv: StratifiedKFoldConfig | None = None
    _steps: list[str] | None = None
    _catboost_metadata: dict[str, Any] | None = None
    _biased_class_weights: list[float] | None = None
    _custom_metrics: dict[CustomMetrics, Any] | None = None
    _fold: int | None = None

    _feature_building_d: timedelta | None = None
    _training_d: timedelta | None = None
    _eval_d: timedelta | None = None

    _version: str = "1.0"  # Semver for the metadata schema

    def __init__(self, experiment: ModelConfig) -> None:
        assert (
            experiment.name.isidentifier()
        ), f"Invalid experiment name: {experiment.name}"  # Used as a dirname
        self._start_time = datetime.utcnow()
        self._experiment = experiment

    def get_model_dir(self, check_for_duplicates: bool = True) -> Path:
        experiment_name = self._experiment.name
        experiment_dir = self._experiment.base_dir
        experiment_dir.mkdir(parents=True, exist_ok=True)
        if check_for_duplicates:
            for d in experiment_dir.iterdir():
                name = d.parts[-1]
                date, time_, exp_name = name.split("_", maxsplit=2)
                if exp_name == experiment_name:
                    raise ValueError(
                        f"An experiment `{name}` already exists in `{experiment_dir.relative_to(ROOT_DIR)}`. Please rename the current experiment."
                    )
        assert self._start_time is not None
        d = (
            experiment_dir
            / f"{self._start_time.strftime('%Y%m%d_%H%M%S')}_{experiment_name}"
        )
        d.mkdir(exist_ok=True)
        return d

    def get_model_fold_dir(self, fold: int) -> Path:
        return self.get_model_dir(check_for_duplicates=False) / f"fold_{fold}"

    def stratified_k_fold(
        self, n_splits: int, shuffle: bool, random_state: int
    ) -> StratifiedKFold:
        self._cv = StratifiedKFoldConfig(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )
        return StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def steps_from_pipeline(self, pipeline: Pipeline) -> None:
        self._steps = [name for name, _ in pipeline.steps]

    def metadata_from_model(self, model: CatBoostClassifier) -> None:
        model_attrs = [
            "get_all_params",
            "get_best_iteration",
            "get_best_score",
            "get_params",
            "tree_count_",
        ]
        model_attrs_val = {}
        for attr in model_attrs:
            a = getattr(model, attr)
            if attr.startswith("get_"):
                a = a()
            model_attrs_val[attr] = a
        model_attrs_val["feature_importance"] = dict(
            model.get_feature_importance(prettified=True).to_dict(orient="tight")[
                "data"
            ]
        )
        self._catboost_metadata = model_attrs_val

    def metrics_from_model(
        self,
        model: CatBoostClassifier,
        X_test_transform: FeaturesData,
        y_test: np.array,
    ) -> None:
        y_test_true = np.array([one_h_enc[y] for y in y_test])
        y_pred = model.predict(data=Pool(data=X_test_transform))
        y_test_score = model.predict_proba(X=Pool(data=X_test_transform))
        sample_weights = biased_sample_weights(y_test)

        self._biased_class_weights = BIASED_CLASS_WEIGHTS.tolist()

        self.start_eval()
        report = classification_report(
            y_true=y_test,
            y_pred=y_pred,
            output_dict=True,
        )
        biased_report = classification_report(
            y_true=y_test,
            y_pred=y_pred,
            output_dict=True,
            sample_weight=sample_weights,
        )

        roc_auc = roc_auc_score(
            y_true=y_test_true,
            y_score=y_test_score,
        )
        biased_roc_auc = roc_auc_score(
            y_true=y_test_true,
            y_score=y_test_score,
            sample_weight=sample_weights,
        )

        mae = mean_absolute_error(
            y_true=y_test_true,
            y_probas=y_test_score,
            class_weight=CLASS_WEIGHTS,
        )
        biased_mae = mean_absolute_error(
            y_true=y_test_true,
            y_probas=y_test_score,
            class_weight=BIASED_CLASS_WEIGHTS,
        )
        self.end_eval()
        self._custom_metrics = {
            CustomMetrics.REPORT: report,
            CustomMetrics.BIASED_REPORT: biased_report,
            CustomMetrics.ROC_AUC: roc_auc,
            CustomMetrics.BIASED_ROC_AUC: biased_roc_auc,
            CustomMetrics.MAE: mae,
            CustomMetrics.BIASED_MAE: biased_mae,
        }

    def write_metadata(self, fold: int) -> None:
        self._fold = fold
        self._verify_attrs()
        kwargs = {
            attr_name[1:]: getattr(self, attr_name)
            for attr_name in list_inst_fields(self)
        }
        metadata_file = self.get_model_fold_dir(fold) / METADATA_FILENAME
        print(f"Writing model metadata to: {metadata_file}")

        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                # Nested cases
                if isinstance(obj, dict):
                    return {k: self.default(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [self.default(v) for v in obj]
                if isinstance(obj, BaseModel):
                    return self.default(obj.model_dump())  # to dict

                # Custom classes
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if hasattr(obj, "is_max_optimal"):
                    # Custom metric
                    return obj.__class__.__name__

                # Default
                try:
                    return json.JSONEncoder.default(self, obj)
                except TypeError:
                    # If standard encoder fails, try pydantic's
                    return to_jsonable_python(obj)

        metadata_file.write_text(
            json.dumps(ModelMetadata(**kwargs), cls=CustomJSONEncoder)
        )

    def _verify_attrs(self) -> None:
        none_attrs = []
        for attr_name in list_inst_fields(self):
            attr = getattr(self, attr_name)
            # All fields starting with "_" shouldn't be none
            if attr is None:
                none_attrs.append(attr_name)
        if len(none_attrs) > 0:
            raise ValueError(f"The followint fields shouldn't be None: {none_attrs}")

    ##
    # Timing functions

    __feature_building_start: float | None = None
    __model_training_start: float | None = None
    __eval_start: float | None = None

    def start_feature_building(self) -> None:
        self.__feature_building_start = time.perf_counter()

    def end_feature_building(self) -> None:
        assert self.__feature_building_start
        self._feature_building_d = timedelta(
            seconds=time.perf_counter() - self.__feature_building_start
        )

    def start_model_training(self) -> None:
        self.__model_training_start = time.perf_counter()

    def end_model_training(self) -> None:
        assert self.__model_training_start
        self._training_d = timedelta(
            seconds=time.perf_counter() - self.__model_training_start
        )

    def start_eval(self) -> None:
        self.__eval_start = time.perf_counter()

    def end_eval(self) -> None:
        assert self.__eval_start
        self._eval_d = timedelta(seconds=time.perf_counter() - self.__eval_start)


def df_from_all_experiments(
    experiment_dir: Path = EXPERIMENT_MODEL_DIR,
) -> pd.DataFrame:
    return pd.DataFrame(
        [extract_metadata(p) for p in experiment_dir.rglob(f"*{METADATA_FILENAME}")]
    )


# Useful when aggregating DataFrames (e.g. group by experiment)
NUMERICAL_COLUMNS = [
    "report_F1:class=2",
    "biased_report_F1:class=2",
    "roc_auc",
    "biased_roc_auc",
    "mae",
    "biased_mae",
    "learn_top_loss",
    "learn_top_F1:class=2",
    "validation_top_loss",
    "validation_top_AUC:type=Mu",
    "validation_top_F1:class=2",
]


def extract_metadata(p: Path) -> dict[str, Any]:
    metadata: ModelMetadata = ModelMetadata(**json.loads(p.read_text()))

    record: dict[str, Any] = {}

    # Experiment metadata
    record["experiment_name"] = metadata.experiment.name
    record["experiment_description"] = metadata.experiment.description
    record["fold"] = metadata.fold
    record["start_time"] = metadata.start_time

    # Custom metrics
    metric: CustomMetrics
    for metric in CustomMetrics:
        key = metric.name.lower()
        value = metadata.custom_metrics[metric]
        if metric in {CustomMetrics.REPORT, CustomMetrics.BIASED_REPORT}:
            record[f"{key}_F1:class=2"] = value["03-disease-area"]["f1-score"]
        else:
            record[key] = value

    # Model metadata
    mm = metadata.catboost_metadata
    record["learn_loss"] = mm["get_all_params"]["loss_function"]
    record["learn_top_loss"] = mm["get_best_score"]["learn"][record["learn_loss"]]
    record["learn_top_F1:class=2"] = mm["get_best_score"]["learn"].get(
        "F1:class=2", np.nan
    )
    record["validation_top_loss"] = mm["get_best_score"]["validation"][
        record["learn_loss"]
    ]
    record["validation_top_AUC:type=Mu"] = mm["get_best_score"]["validation"].get(
        "AUC:type=Mu", np.nan
    )
    record["validation_top_F1:class=2"] = mm["get_best_score"]["validation"].get(
        "F1:class=2", np.nan
    )
    record["tree_cnt"] = mm["tree_count_"]

    # Durations
    record["feature_building_d"] = metadata.feature_building_d
    record["training_d"] = metadata.training_d
    record["eval_d"] = metadata.eval_d

    return record


def list_inst_fields(
    inst: Any, excluded_attrs: set[str] | None = None
) -> Iterable[str]:
    """
    Returns all public and protected fields of an instance.
    """
    excluded_attrs = excluded_attrs or set()
    for attr_name in dir(inst):
        if attr_name in excluded_attrs:
            continue
        if attr_name.startswith("__"):
            continue
        if attr_name.startswith(f"_{inst.__class__.__name__}_"):
            continue
        if callable(getattr(inst, attr_name)):
            continue
        yield attr_name


@dataclass
class FeatureGroup:
    name: str
    pattern: str


PCA_FG = FeatureGroup(name="pca_fg", pattern="pca_*")
GPT_FG = FeatureGroup(name="gpt_tags_fg", pattern="gpt-4_tag_*")
X_REF_FG = FeatureGroup(name="xref_fg", pattern="xref__*")
SUBSETS_FG = FeatureGroup(name="subsets_fg", pattern="IN_*")
TOPO_FG = FeatureGroup(name="topo_fg", pattern="n_*|intrinsic_*|depth")


def extract_feature_importance(
    exp_name: str, feature_groups: list[FeatureGroup] | None = None
) -> pd.DataFrame:
    """
    Extract the feature importance for a given experiment.
    The importance values are normalized within each fold to sum to 1.
    The features are optionally grouped.
    """
    feature_groups = feature_groups or []
    p = EXPERIMENT_MODEL_DIR / exp_name
    assert p.is_dir(), f"Wrong experiment path: {p}"

    out_df = pd.DataFrame()
    # List folds:
    for fold_i, fold_dir in enumerate(sorted(p.iterdir())):
        metadata = ModelMetadata(
            **json.loads(fold_dir.joinpath(METADATA_FILENAME).read_text())
        )
        feature_importance: dict[str, float] = metadata.catboost_metadata[
            "feature_importance"
        ]
        sum_weights = sum(feature_importance.values())
        grouped_feature_importance: dict[str, float] = defaultdict(float)
        for fn, w in feature_importance.items():
            for fg in feature_groups:
                if re.match(fg.pattern, fn):
                    fn = fg.name
                    break
            grouped_feature_importance[fn] += w / sum_weights

        df = pd.DataFrame(
            [
                {
                    "fold": fold_i,
                    "feature": f,
                    "importance": i,
                }
                for f, i in grouped_feature_importance.items()
            ]
        )
        out_df = pd.concat([out_df, df])

    # Sort by median importance
    median_importance = out_df.groupby("feature")["importance"].median().to_dict()
    out_df = out_df.sort_values(
        by="feature", key=lambda e: e.apply(median_importance.get), ascending=False
    )

    return out_df
