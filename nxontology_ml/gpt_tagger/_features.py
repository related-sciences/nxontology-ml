import pandas as pd

from nxontology_ml.gpt_tagger._gpt_tagger import GptTagger
from nxontology_ml.gpt_tagger._models import TaskConfig
from nxontology_ml.sklearn_transformer import NodeFeatures, NoFitTransformer
from nxontology_ml.utils import ROOT_DIR

DEFAULT_CONF = TaskConfig(
    name="precision",
    prompt_path=ROOT_DIR / "prompts/precision_v1.txt",
    openai_model_name="gpt-4",
    node_attributes=["efo_id", "efo_label", "efo_definition"],
    model_n=3,
    prompt_token_ratio=0.5,
    allowed_labels=frozenset({"low", "medium", "high"}),
)


class GptTagFeatures(NoFitTransformer[NodeFeatures, NodeFeatures]):
    """
    Adds GPT precision tags as features in the model
    """

    def __init__(
        self, enabled: bool, tagger: GptTagger | None, config: TaskConfig | None
    ) -> None:
        self._enabled = enabled
        self._tagger = tagger
        self._config = config

    def transform(self, X: NodeFeatures, copy: bool | None = None) -> NodeFeatures:
        if not self._enabled:
            return X
        assert self._config
        assert self._tagger
        new_features = pd.DataFrame(
            data=[
                labeled_node.labels
                for labeled_node in self._tagger.fetch_labels(X.nodes)
            ],
            columns=[
                f"{self._config.openai_model_name}_tag_{i}"
                for i in range(self._config.model_n)
            ],
        )
        X.cat_features = pd.concat([X.cat_features, new_features], axis=1)
        return X

    @classmethod
    def from_config(cls, config: TaskConfig | None = DEFAULT_CONF) -> "GptTagFeatures":
        tagger = None
        if config:
            tagger = GptTagger.from_config(config)
        return cls(enabled=config is not None, tagger=tagger, config=config)
