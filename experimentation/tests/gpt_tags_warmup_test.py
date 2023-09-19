from experimentation.gpt_tags_warmup import warmup_gpt_tags
from nxontology_ml.gpt_tagger.tests._utils import mk_test_gpt_tagger


def test_warmup_gpt_tags() -> None:
    tagger = mk_test_gpt_tagger(
        cache_content={
            "/a93f3eabc24f867ae4f1d6b371ba6734e38ea0a4": b'["medium"]',
        }
    )
    warmup_gpt_tags(
        tagger=tagger,
        take=1,
    )
