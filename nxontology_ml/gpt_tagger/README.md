# GPT EFO Tagger

Use OpenAI's ChatGPT to label EFO records.

## Example

Note: An API key needs to be set. See below.

```python
from pprint import pprint

from nxontology_ml.data import get_efo_otar_slim
from nxontology_ml.gpt_tagger import TaskConfig, GptTagger
from nxontology_ml.utils import ROOT_DIR

# Create a config for EFO nodes labelling
config = TaskConfig(
    name="precision",
    prompt_path=ROOT_DIR / "prompts/precision_v1.txt",
    openai_model_name="gpt-4",
    node_attributes=["efo_id", "efo_label", "efo_definition"],
    model_n=3,
)

# Get a few EFO nodes
nxo = get_efo_otar_slim()
nodes = (nxo.node_info(node) for node in sorted(nxo.graph)[:20])

# Get their labels
tagger = GptTagger.from_config(config)
for ln in tagger.fetch_labels(nodes):
    print(f"{ln.node_efo_id}: {ln.labels}")

# Inspect metrics
print("\nTagger metrics:")
pprint(tagger.get_metrics())
```
You should get an output similar to:
```shell
DOID:0050890: ['medium', 'medium', 'medium']
DOID:10113: ['low', 'low', 'low']
DOID:10718: ['low', 'low', 'low']
DOID:13406: ['medium', 'medium', 'medium']
DOID:1947: ['low', 'low', 'low']
DOID:7551: ['low', 'low', 'low']
EFO:0000094: ['high', 'high', 'high']
EFO:0000095: ['high', 'high', 'high']
EFO:0000096: ['medium', 'medium', 'medium']
EFO:0000174: ['high', 'high', 'high']
EFO:0000178: ['high', 'medium', 'medium']
EFO:0000180: ['low', 'low', 'medium']
EFO:0000181: ['high', 'medium', 'high']
EFO:0000182: ['high', 'medium', 'high']
EFO:0000183: ['high', 'medium', 'medium']
EFO:0000186: ['high', 'high', 'high']
EFO:0000191: ['high', 'high', 'high']
EFO:0000195: ['low', 'medium', 'medium']
EFO:0000196: ['high', 'high', 'high']
EFO:0000197: ['high', 'medium', 'medium']

Tagger metrics:
Counter({'ChatCompletion/total_tokens': 3543,
         'ChatCompletion/prompt_tokens': 3009,
         'ChatCompletion/completion_tokens': 534,
         'Cache/get': 20,
         'Cache/misses': 20,
         'ChatCompletion/records_processed': 20,
         'Cache/set': 20,
         'ChatCompletion/create_requests': 1})
```

Note that the requests are cached in `.cache/precision_v1.ldb`, so if you run the same code again you will 
see the same labels, but different metrics:
```shell
Tagger metrics:
Counter({'Cache/get': 20, 'Cache/hits': 20})
```

## API Key

OpenAI's API requires an API key. It can either be set using:
```shell
 export OPENAI_API_KEY="sk-XXX"
```
Or it can be persisted in the root folder's `.env` file:
```shell 
echo 'OPENAI_API_KEY="sk-XXX"' >> .env
```
Note: `.env` files are gitignored.

