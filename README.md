# LLM Compressor Hermetic Build Demo

Watch the [demo on youtube](https://www.youtube.com/watch?v=T-5rMOKZD4s).

Using LLM Compressor in a hermetic build in Konflux, showcasing:

- **A hermetic offline run of [llm-compressor](https://github.com/vllm-project/llm-compressor)** to produce a quantized model on a GPU node in Konflux.
- **An [AI SBOM](https://github.com/aibom-squad/SBOM-for-AI-Use-Cases)**, enabled by the offline build. (Learn the hermeto strategy from [Adam Cmiel at OSS NA](https://youtu.be/cwmdQI6uWWA)).
- **SLSA build provenance attestations**, with a [high degree of detail](https://developers.redhat.com/articles/2025/05/15/how-we-use-software-provenance-red-hat#attestation_example), created with the neutral observer pattern that Konflux inherits from Tekton.

Simple one-shot quantization is used, compressing `TinyLlama-1.1B-Chat-v1.0` using `SmoothQuant` + `GPTQ` as a kind of "hello world".

This demo was enabled by:

- **llm-compressor patches** - A few patches were needed in llm-compressor to [support HF env vars](https://github.com/vllm-project/llm-compressor/pull/1902) and to [suppress an eager network request](https://github.com/vllm-project/llm-compressor/pull/1954).
- **a new llm-compressor task** - A new Tekton task for running LLM quantization from [build-definitions/llm-compressor](https://github.com/ralphbean/build-definitions/tree/llm-compressor)
- **a new huggingface backend in hermeto** - Hermetic dependency fetching for Hugging Face models and datasets from [hermeto PR #1141](https://github.com/hermetoproject/hermeto/pull/1141) (see standalone demo at [youtu.be/fY4tXDkUa5I](https://youtu.be/fY4tXDkUa5I)).

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

**NOTE**: An alternative approach should be considered in [dvc](https://dvc.org/), see [hermetoproject/hermeto!1151](https://github.com/hermetoproject/hermeto/pull/1151).

### llm-compressor-remote-oci-ta Task

Custom Tekton task that executes LLM compression scripts in a hermetic, GPU-enabled environment using OCI Trusted Artifacts (OCI-TA). Key features:

- **Remote GPU execution** - Runs compression workloads on remote hosts with GPU access via the Konflux multi-platform controller
- **Hermetic builds** - Optional network isolation with hermeto-prefetched dependencies
- **SBOM generation** - Includes dependency metadata. Higher accuracy if a hermetic build is used.
- **OCI artifact output** - Packages compressed models as OCI artifacts
- **Flexible scripts** - Supports both Python and shell compression scripts
- **Script arguments** - Pass custom arguments to compression scripts via `SCRIPT_ARGS` parameter

The task uses buildah to run the compression script inside a container with:
- GPU device passthrough (NVIDIA, AMD, Intel)
- Mounted source code and dependencies
- Configurable output directory
- Environment variable injection

Example usage in the pipeline:

```yaml
- name: llm-compressor
  taskRef:
    resolver: git
    params:
      - name: url
        value: https://github.com/ralphbean/build-definitions  # TODO Update this to a more proper repo.
      - name: revision
        value: llm-compressor
      - name: pathInRepo
        value: task/llm-compressor-remote-oci-ta/0.1/llm-compressor-remote-oci-ta.yaml
  params:
    - name: SCRIPT
      value: "compress.py"
    - name: SCRIPT_ARGS
      value:
        - "--output-dir"
        - "/var/workdir/output"
    - name: PLATFORM
      value: "linux-g64xlarge/amd64"
    - name: BUILDAH_DEVICES
      value:
        - "/dev/nvidia0"
```

## Author

Ralph Bean (rbean@redhat.com)

## License

Apache 2.0
