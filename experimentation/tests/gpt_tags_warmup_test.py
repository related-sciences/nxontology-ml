from experimentation.gpt_tags_warmup import warmup_gpt_tags
from nxontology_ml.gpt_tagger import TaskConfig
from nxontology_ml.gpt_tagger.tests._utils import mk_test_gpt_tagger
from nxontology_ml.gpt_tagger.tests.conftest import precision_config  # noqa


def test_warmup_gpt_tags(precision_config: TaskConfig) -> None:  # noqa: F811
    tagger = mk_test_gpt_tagger(
        config=precision_config,
        cache_content={
            "/a93f3eabc24f867ae4f1d6b371ba6734e38ea0a4": b'["medium"]',
        },
    )
    warmup_gpt_tags(
        tagger=tagger,
        take=1,
    )
