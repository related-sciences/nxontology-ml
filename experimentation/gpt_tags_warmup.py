from pprint import pprint

from tqdm import tqdm

from nxontology_ml.data import get_efo_otar_slim, read_training_data
from nxontology_ml.gpt_tagger import DEFAULT_CONF, GptTagger
from nxontology_ml.gpt_tagger._cache import LazyLSM


def warmup_gpt_tags(
    tagger: GptTagger | None = None,
    take: int | None = None,
    max_retries: int = 1000,
) -> None:
    """
    In order to compare feature computation times fairly, it's useful to pre-cache all GPT-4 tags
    """
    tagger = tagger or GptTagger.from_config(DEFAULT_CONF)
    s = tagger._cache._storage
    if isinstance(s, LazyLSM):
        print(f"Populating: {s._filename}")
    nxo = get_efo_otar_slim()
    X, _ = read_training_data(take=take, filter_out_non_disease=True)
    nodes = [nxo.node_info(n) for n in X]
    for retry in range(max_retries):
        # [CAUTION] Dreadful hack ahead:
        # - The GPT tagging fails often due to data issues (e.g. records missing, un-allowed
        #   labels, completions too long, ...)
        # - Here we bruteforce retry until all the labels have been fetched and cached
        # - FIXME: Hide the (ugly) retry logic inside the GPT tagger
        print(f"> Retry #{retry}:")
        current_counter = tagger.get_metrics()
        for _ln in tqdm(
            tagger.fetch_labels(nodes),
            total=len(X),
            desc="Fetching node tags using GPT-4",
            ncols=100,
        ):
            pass
        api_requests = (tagger.get_metrics() - current_counter).get(
            "ChatCompletion/create_requests", 0
        )
        if api_requests == 0:
            print(f"All tags fetched after {retry} retries, early exiting.")
            break
    # Inspect metrics
    print("\nTagger metrics:")
    pprint(tagger.get_metrics())


if __name__ == "__main__":
    """
    PYTHONPATH=. python experimentation/gpt_tags_warmup.py
    """
    warmup_gpt_tags()  # pragma: no cover
