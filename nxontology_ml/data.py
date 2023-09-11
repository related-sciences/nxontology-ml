import csv
import functools
from hashlib import sha256
from pathlib import Path

import numpy as np
from nxontology import NXOntology

from nxontology_ml.utils import ROOT_DIR


def read_training_data(
    take: int | None = None,
    filter_out_non_disease: bool = False,
    nxo: NXOntology[str] | None = None,
    data_path: Path = ROOT_DIR / "data/efo_otar_slim_v3.43.0_rs_classification.tsv",
) -> tuple[np.ndarray, np.ndarray]:
    """
    By default, the data is (consistently) shuffled
    """
    # Get Ontology
    nxo = nxo or get_efo_otar_slim()
    nodes: set[str] = set(nxo.graph)

    # Get labelled data
    labelled_nodes: list[str] = []
    labels: list[str] = []
    with data_path.open(mode="r") as f:
        for i, (efo_otar_slim_id, efo_label, rs_classification) in enumerate(
            csv.reader(f, delimiter="\t")
        ):
            if i == 0:
                # Skip header
                assert (efo_otar_slim_id, efo_label, rs_classification) == (
                    "efo_otar_slim_id",
                    "efo_label",
                    "rs_classification",
                )
            elif filter_out_non_disease and rs_classification == "04-non-disease":
                continue
            else:
                if efo_otar_slim_id in nodes:
                    labelled_nodes.append(efo_otar_slim_id)
                    labels.append(rs_classification)

    # Consistent shuffling (i.e. sort by hash)
    z = list(zip(labelled_nodes, labels, strict=True))
    z.sort(key=lambda nl: sha256(nl[0].encode()).hexdigest())
    labelled_nodes, labels = zip(*z, strict=True)  # type: ignore[assignment]

    if take:
        labelled_nodes = labelled_nodes[:take]
        labels = labels[:take]
    return np.array(labelled_nodes), np.array(labels)


EFO_OTAR_SLIM_URL: str = "https://github.com/related-sciences/nxontology-data/raw/f0e450fe3096c3b82bf531bc5125f0f7e916aad8/efo_otar_slim.json"


@functools.cache
def get_efo_otar_slim(url: str = EFO_OTAR_SLIM_URL) -> NXOntology[str]:
    nxo = NXOntology[str].read_node_link_json(url)
    assert isinstance(nxo, NXOntology)
    nxo.freeze()
    return nxo
