# LLM Compressor Hermetic Build Demo

This repository demonstrates a hermetic build pipeline for LLM quantization using Konflux, showcasing:

- **Custom llm-compressor task** - A new Tekton task for running LLM quantization
- **x-huggingface support in hermeto** - Hermetic dependency fetching for Hugging Face models and datasets

**One-shot quantization** is used - compressing TinyLlama-1.1B-Chat-v1.0 using SmoothQuant + GPTQ as a kind of "hello world".

## New Things

The following pieces were created to support this demo:

1. **llm-compressor task** - Custom Tekton task from [build-definitions/llm-compressor](https://github.com/ralphbean/build-definitions/tree/llm-compressor)
2. **hermeto x-huggingface** - ML dependency management from [hermeto PR #1141](https://github.com/hermetoproject/hermeto/pull/1141).

See also an earlier demo of just the hermeto huggingface support (2) at [youtu.be/fY4tXDkUa5I](https://youtu.be/fY4tXDkUa5I).

## Repository Structure

```
.
├── compress.py                    # Quantization script using llm-compressor
├── huggingface.lock.yaml         # Hermeto lockfile for HF models/datasets
├── .tekton/
│   ├── llm-compressor-pipeline.yaml  # Main pipeline definition
│   └── llm-compressor-push.yaml      # PipelineRun trigger config
├── konflux/
│   ├── application.yaml          # Konflux Application resource
│   ├── component.yaml            # Konflux Component resource
│   └── kustomization.yaml        # Kustomize config for easy deployment
└── README.md
```

## Key Components

### compress.py

Python script that uses llm-compressor to perform one-shot quantization:

```python
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
    GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"]),
]

oneshot(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset="open_platypus",
    recipe=recipe,
    output_dir="TinyLlama-1.1B-Chat-v1.0-INT8",
    max_seq_length=2048,
    num_calibration_samples=512,
)
```

### huggingface.lock.yaml

Hermeto lockfile specifying exact versions of ML dependencies. Both models and datasets are declared in the `models` list with a `type` field to distinguish them:

```yaml
metadata:
  version: "1.0"

models:
  - repository: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    revision: "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
    type: "model"
    include_patterns:
      - "*.safetensors"
      - "config.json"
      - "tokenizer*"

  - repository: "garage-bAInd/Open-Platypus"
    revision: "37141edbdb7826378cce118c46a109b813e1f038"
    type: "dataset"
    include_patterns:
      - "*.parquet"
      - "*.json"
```

## References

- [llm-compressor](https://github.com/vllm-project/llm-compressor) - LLM quantization library
- [hermeto](https://github.com/hermetoproject/hermeto) - Hermetic dependency management
- [build-definitions](https://github.com/konflux-ci/build-definitions) - Konflux task catalog

## Author

Ralph Bean (rbean@redhat.com)

## License

Apache 2.0
