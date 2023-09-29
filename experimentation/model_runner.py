import logging

from catboost import CatBoostClassifier, Pool
from sklearn.pipeline import make_pipeline
from transformers import Pipeline

from experimentation.metadata_helpers import ModelMetadataBuilder
from nxontology_ml.data import read_training_data
from nxontology_ml.efo import NodeXrefFeatures
from nxontology_ml.features import (
    NodeInfoFeatures,
    PrepareNodeFeatures,
    SubsetsFeatures,
    TherapeuticAreaFeatures,
)
from nxontology_ml.gpt_tagger import GptTagFeatures
from nxontology_ml.model.config import ModelConfig
from nxontology_ml.model.formatter import MODEL_SEED, CatBoostDataFormatter
from nxontology_ml.text_embeddings.embeddings_model import AutoModelEmbeddings
from nxontology_ml.text_embeddings.text_embeddings_transformer import (
    TextEmbeddingsTransformer,
)

EXPERIMENTS = [
    # ModelConfig(
    #     name_override="baseline",
    #     eval_metric="MultiClass",
    # ),
    # ModelConfig(
    #     eval_metric="BiasedMaeMetric",
    # ),
    # ModelConfig(
    #     eval_metric="BiasedMaeMetric",
    #     # depth=7,
    # ),
    # ModelConfig(
    #     eval_metric="BiasedMaeMetric",
    #     # depth=7,
    #     use_knn=True,
    # ),
    # ModelConfig(
    #     eval_metric="BiasedMaeMetric",
    #     # depth=7,
    #     use_lda=True,
    #     embedding_enabled=True,
    # ),
    # ModelConfig(
    #     eval_metric="BiasedMaeMetric",
    #     # depth=7,
    #     embedding_enabled=True,
    #     pca_components=32,
    # ),
    # ModelConfig(
    #     eval_metric="BiasedMaeMetric",
    #     # depth=7,
    #     embedding_enabled=True,
    #     pca_components=64,
    # ),
    # ModelConfig(
    #     eval_metric="BiasedMaeMetric",
    #     # depth=7,
    #     embedding_enabled=True,
    #     pca_components=128,
    # ),
    # ModelConfig(
    #     eval_metric="BiasedMaeMetric",
    #     # depth=7,
    #     embedding_enabled=True,
    #     pca_components=256,
    # ),
    # ModelConfig(
    #     eval_metric="BiasedMaeMetric",
    #     # depth=7,
    #     embedding_enabled=True,
    #     pca_components=64,
    #     use_knn=True,
    # ),
    # ModelConfig(
    #     eval_metric="BiasedMaeMetric",
    #     # depth=7,
    #     embedding_enabled=True,
    #     pca_components=64,
    #     subsets_enabled=True,
    # ),
    # ModelConfig(
    #     eval_metric="BiasedMaeMetric",
    #     # depth=7,
    #     gpt_tagger_config=DEFAULT_CONF,
    # ),
    ModelConfig(
        eval_metric="BiasedMaeMetric",
        ta_enabled=True,
    ),
    ModelConfig(
        eval_metric="BiasedMaeMetric",
        embedding_enabled=True,
        pca_components=64,
        ta_enabled=True,
    ),
]


def run_experiments(
    experiments: list[ModelConfig] | None = None,
    ame: AutoModelEmbeddings | None = None,
    n_splits: int = 25,
    take: int | None = None,
) -> None:
    logging.basicConfig(level=logging.INFO)
    experiments = experiments or EXPERIMENTS
    ##
    # Features
    X, y = read_training_data(take=take, filter_out_non_disease=True)

    for exp_i, exp in enumerate(experiments):
        print("\n##############################")
        print(f"###     Experiment {exp_i:3d}      ##")
        print("##############################\n")

        print(f"Experiment: {exp.name}")

        mmb = ModelMetadataBuilder(exp)
        print(f"Writing model assets to: {mmb.get_model_dir().as_posix()}")

        skf = mmb.stratified_k_fold(n_splits=n_splits, shuffle=True, random_state=42)
        for fold_i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print("\n##")
            print(f"# Fold {fold_i:3d}\n")

            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]

            feature_pipeline: Pipeline = make_pipeline(
                PrepareNodeFeatures(),
                NodeInfoFeatures(),
                NodeXrefFeatures(),
                SubsetsFeatures(enabled=exp.subsets_enabled),
                TherapeuticAreaFeatures(enabled=exp.ta_enabled),
                GptTagFeatures.from_config(exp.gpt_tagger_config),
                TextEmbeddingsTransformer.from_config(
                    enabled=exp.embedding_enabled,
                    pca_components=exp.pca_components,
                    use_lda=exp.use_lda,
                    embedding_model=ame,
                ),
                CatBoostDataFormatter(),
            )
            mmb.steps_from_pipeline(feature_pipeline)

            mmb.start_feature_building()
            feature_pipeline.fit(X_train, y_train)
            X_train_transform = feature_pipeline.transform(X_train)
            X_test_transform = feature_pipeline.transform(X_test)
            mmb.end_feature_building()

            ##
            # Model

            # typical Learning rate set to 0.080751
            model = CatBoostClassifier(
                train_dir=mmb.get_model_fold_dir(fold=fold_i),
                eval_metric=exp.get_eval_metric,
                depth=exp.depth,
                custom_metric=["MultiClass", "AUC", "F1"],
                learning_rate=0.5,
                iterations=5000,
                use_best_model=True,
                metric_period=250,
                random_seed=MODEL_SEED,
                # early_stopping_rounds=1000,
                # task_type="GPU",
                # l2_leaf_reg=.01,
                # loss_function="MultiClassOneVsAll",
            )
            mmb.start_model_training()
            model.fit(
                X=Pool(
                    data=X_train_transform,
                    label=list(y_train),
                    # weight=biased_sample_weights(y_train),
                ),
                eval_set=Pool(
                    data=X_test_transform,
                    label=list(y_test),
                    # weight=biased_sample_weights(y_test),
                ),
            )
            mmb.end_model_training()

            mmb.metadata_from_model(model)
            mmb.metrics_from_model(model, X_test_transform, y_test)
            mmb.write_metadata(fold=fold_i)


if __name__ == "__main__":
    """
    PYTHONPATH=. python experimentation/model_runner.py 2>&1 | tee output.log
    """
    run_experiments()  # pragma: no cover
