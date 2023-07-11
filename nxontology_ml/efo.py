import functools
from collections import Counter

from nxontology import NXOntology

from nxontology_ml.features import NodeFeatures, get_features_df
from nxontology_ml.utils import get_output_directory


@functools.cache
def get_efo_otar_slim() -> NXOntology[str]:
    url = "https://github.com/related-sciences/nxontology-data/raw/f0e450fe3096c3b82bf531bc5125f0f7e916aad8/efo_otar_slim.json"
    nxo: NXOntology[str] = NXOntology.read_node_link_json(url)
    nxo.freeze()
    return nxo


class NodeFeaturesEfo(NodeFeatures):
    @staticmethod
    def get_curie_prefix(curie: str) -> str:
        prefix, _identifier = curie.split(":", 1)
        return prefix.lower()

    def get_prefix(self) -> str:
        curie = self.info.identifier
        assert isinstance(curie, str)
        return self.get_curie_prefix(curie)

    efo_classification_xref_prefixes = {
        "doid": "doid",
        "gard": "gard",
        "icd10": "icd10",
        "icd9": "icd9",
        "meddra": "meddra",
        "mesh": "mesh",
        "mondo": "mondo",
        "ncit": "ncit",
        "omim": "omim",
        "omim.ps": "omimps",
        "orphanet": "orphanet",
        "snomedct": "snomedct",
        "umls": "umls",
    }
    """
    EFO xref prefixes to include in classification as a dictionary
    where keys are bioregistry prefixes and values are cleaned prefixes
    (without punctuation, for use in column names).
    Only these prefixes will be used as classification features
    """

    def get_xref_features(self) -> dict[str, int]:
        xref_counts: dict[str, int] = Counter()
        for xref in self.info.data.get("xrefs") or []:
            xref_counts[self.get_curie_prefix(xref)] += 1
        return {
            f"xref__{prefix_name}__count": xref_counts[prefix]
            for prefix, prefix_name in self.efo_classification_xref_prefixes.items()
        }

    def get_metrics(self) -> dict[str, int | str | None]:
        metrics = super().get_metrics()
        metrics["gwas_trait"] = self.info.data.get("gwas_trait", False)
        metrics["prefix"] = self.get_prefix()
        metrics.update(self.get_xref_features())
        return metrics


def write_efo_features() -> None:
    """Generate and export features for the EFO OTAR Slim ontology."""
    nxo = get_efo_otar_slim()
    directory = get_output_directory(nxo)
    feature_df = get_features_df(nxo, node_features_class=NodeFeaturesEfo)
    assert nxo.name is not None
    feature_df.to_csv(
        directory.joinpath(f"{nxo.name}_features.tsv"),
        index=False,
        float_format="%.5f",
        sep="\t",
    )
    # need additional dependencies to write parquet
    # feature_df.to_parquet(directory.joinpath(f"{nxo.name}_features.parquet"))
