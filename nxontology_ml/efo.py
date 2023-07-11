import functools

from nxontology import NXOntology

from nxontology_ml.features import get_features_df
from nxontology_ml.utils import get_output_directory


@functools.cache
def get_efo_otar_slim() -> NXOntology[str]:
    url = "https://github.com/related-sciences/nxontology-data/raw/f0e450fe3096c3b82bf531bc5125f0f7e916aad8/efo_otar_slim.json"
    nxo: NXOntology[str] = NXOntology.read_node_link_json(url)
    nxo.freeze()
    return nxo


def write_efo_features() -> None:
    """Generate and export features for the EFO OTAR Slim ontology."""
    nxo = get_efo_otar_slim()
    directory = get_output_directory(nxo)
    feature_df = get_features_df(nxo)
    assert nxo.name is not None
    feature_df.to_csv(
        directory.joinpath(f"{nxo.name}_features.tsv"),
        index=False,
        float_format="%.5f",
        sep="\t",
    )
    # need additional dependencies to write parquet
    # feature_df.to_parquet(directory.joinpath(f"{nxo.name}_features.parquet"))
