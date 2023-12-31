# Machine learning to classify ontology nodes

[![Software License](https://img.shields.io/github/license/related-sciences/nxontology-ml?style=for-the-badge&logo=Apache&logoColor=white)](https://github.com/related-sciences/nxontology-ml/blob/main/LICENSE)  

## Overview

This is a project to classify ontology nodes/classes/terms using machine learning.
It is currently in the early stages of development and should be considered experimental.

The first classification task implemented by this library is to classify nodes in EFO OTAR Slim as [low/medium/high precision](https://github.com/related-sciences/nxontology-ml/issues/2).
We presented on this application at the 2023-09-22 [Mondo Outreach Call](https://mondo.monarchinitiative.org/pages/workshop/ "Mondo Disease Ontology Workshops and Outreach Calls")
([slides](https://slides.com/dhimmel/efo-disease-precision "Classifying EFO/MONDO diseases as low, medium, or high precision using nxontology-ml")).

See [`nxontology`](https://github.com/related-sciences/nxontology) and [`nxontology-data`](https://github.com/related-sciences/nxontology-data) for other (more mature) components of the nxontology ecosystem.

To use the model to label an EFO ontology:

```python
from nxontology_ml.data import get_efo_otar_slim
from nxontology_ml.model.predict import train_predict

nxo = get_efo_otar_slim()
df = train_predict(nxo=nxo)
```
Note: The model can be trained with a different configuration using `train_predict(conf=ModelConfig(...))`


## Development

```shell
# Install the environment
poetry install --no-root

# Run a command
poetry run nxontology_ml --help

# Run tests
pytest

# Update the lock file
poetry update

# Set up the git pre-commit hooks.
# `git commit` will now trigger automatic checks including linting.
pre-commit install

# Run all pre-commit checks (CI will also run this).
pre-commit run --all
```

## License

This source code in this repository is released under an Apache License 2.0 License
(see [LICENSE](LICENSE)).
Source code refers to the contents of the `main` branch and any other development branches containing code and documentation.

This repository contains data from external ontologies.
Please refer to each respective ontology for its data license.
Please attribute the source ontology when reusing data obtained from this project,
and as best practice mention that the data was obtained via <https://github.com/related-sciences/nxontology-ml>.

Any original data produced by this repository is released under a [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/).
As noted above, the underlying ontology data is not original to this repository and upstream licenses should be consulted.
