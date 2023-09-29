from pathlib import Path

from pydantic import BaseModel

from nxontology_ml.gpt_tagger import TaskConfig
from nxontology_ml.model.utils import BiasedMaeMetric
from nxontology_ml.utils import ROOT_DIR

EXPERIMENT_MODEL_DIR = ROOT_DIR / "data/experiments"


class ModelConfig(BaseModel):  # type: ignore[misc]
    """
    Constants for each experiment (for now):
    - Shuffled inputs
    - 25 folds stratified CV
    - Best model is selected
    - Max 5000 iterations
    - Learning rate of 0.5
    - custom_metric: "MultiClass", "AUC" & "F1"
    """

    name_override: str | None = None
    description_override: str | None = None
    embedding_enabled: bool = False
    pca_components: int | None = None
    use_lda: bool = False
    use_knn: bool = False  # This feature was removed
    subsets_enabled: bool = False
    ta_enabled: bool = False
    gpt_tagger_config: TaskConfig | None = None
    depth: int = 6
    eval_metric: str = "MultiClass"
    base_dir: Path = EXPERIMENT_MODEL_DIR

    @property
    def name(self) -> str:  # noqa: C901
        if self.name_override:
            return self.name_override
        parts: list[str] = []
        if isinstance(self.pca_components, int) and self.embedding_enabled:
            parts.append(f"pca{self.pca_components}")
        if self.use_lda and self.embedding_enabled:
            parts.append("lda")
        if self.embedding_enabled and not (self.use_lda or self.pca_components):
            parts.append("full_embedding")
        if self.use_knn:
            parts.append("knn")
        if self.subsets_enabled:
            parts.append("subsets")
        if self.ta_enabled:
            parts.append("ta")
        if self.gpt_tagger_config:
            # Note: we don't use the config name in the name of the experiment
            parts.append(self.gpt_tagger_config.openai_model_name.replace("-", ""))
        if self.depth != 6:
            parts.append(f"d{self.depth}")
        if self.eval_metric == "BiasedMaeMetric":
            parts.append("mae")
        return "_".join(parts)

    @property
    def description(self) -> str:
        if self.description_override:
            return self.description_override
        return self.name

    @property
    def get_eval_metric(self) -> BiasedMaeMetric | str:
        if self.eval_metric == "BiasedMaeMetric":
            return BiasedMaeMetric()
        return self.eval_metric
