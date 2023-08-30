import csv
import functools
import random
from pathlib import Path

import numpy as np
from nxontology import NXOntology

from nxontology_ml.utils import ROOT_DIR


def read_training_data(
    sort: bool = False,
    shuffle: bool = False,
    take: int | None = None,
    nxo: NXOntology[str] | None = None,
    data_path: Path = ROOT_DIR / "data/efo_otar_slim_v3.43.0_rs_classification.tsv",
) -> tuple[np.ndarray, np.ndarray]:
    assert not (sort and shuffle), "Wat??"
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
            elif take and len(labels) == take:
                break
            else:
                if efo_otar_slim_id in nodes:
                    labelled_nodes.append(efo_otar_slim_id)
                    labels.append(rs_classification)

    if shuffle or sort:
        z = list(zip(labelled_nodes, labels, strict=True))
        if shuffle:
            random.shuffle(z)
        else:
            assert sort
            z.sort()
        labelled_nodes, labels = zip(*z, strict=True)  # type: ignore[assignment]
    return np.array(labelled_nodes), np.array(labels)


EFO_OTAR_SLIM_URL: str = "https://github.com/related-sciences/nxontology-data/raw/f0e450fe3096c3b82bf531bc5125f0f7e916aad8/efo_otar_slim.json"


@functools.cache
def get_efo_otar_slim(url: str = EFO_OTAR_SLIM_URL) -> NXOntology[str]:
    nxo = NXOntology[str].read_node_link_json(url)
    assert isinstance(nxo, NXOntology)
    nxo.freeze()
    return nxo
