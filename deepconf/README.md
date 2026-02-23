# DeepConf Integration

This directory contains code derived from the [DeepConf](https://github.com/facebookresearch/deepconf) project by Facebook Research.

## Licensing and Attribution

The original code is licensed under the **MIT License**.

* **Original Project:** [DeepConf](https://github.com/facebookresearch/deepconf)
* **Copyright:** Copyright (c) Facebook, Inc. and its affiliates
* **License:** MIT License (see LICENSE file in this directory)

## About DeepConf

DeepConf is a framework for enhanced reasoning in large language models, providing an efficient parallel thinking framework built upon vLLM. It enables multiple parallel reasoning traces with confidence-based early stopping and weighted voting strategies.

## Notice of Modification

### Core Library
* **No modifications to core DeepConf code**: The utility functions (`utils.py`), wrapper (`wrapper.py`), outputs (`outputs.py`), and processors (`processors.py`) remain unchanged from the original repository.
* **Purpose**: Used as a utility library for answer extraction and confidence computation.

### Integration and Usage

The DeepConf components are used in this project for:

1. **Answer Extraction** (`extract_answer` function)
   * Extracts boxed mathematical answers (LaTeX `\boxed{}` format) from LLM reasoning traces
   * Used in data preparation pipeline ([data_preparation/label_traces.py](../data_preparation/label_traces.py))
   * Enables automatic labeling of traces as correct/incorrect by comparing with ground truth answers

2. **Confidence Computation** (`compute_confidence` function)
   * Processes token-level logprobs from vLLM outputs
   * Computes per-token confidence scores based on mean negative log-probability
   * Serves as baseline feature for evolved confidence indicators

3. **Weighted Voting** (`weighted_majority_vote` function)
   * Aggregates multiple LLM reasoning traces using confidence-weighted voting
   * Used in baseline comparisons ([data_preparation/weighted_majority_voting_baseline.py](../data_preparation/weighted_majority_voting_baseline.py))
   * Demonstrates the value of learned confidence scores over uniform majority voting

### Adaptations in Data Pipeline

While the core DeepConf library remains unmodified, the project includes standalone reimplementations of key functions (particularly `extract_answer`) in the data preparation scripts to avoid import dependencies during preprocessing. These reimplementations maintain identical logic to the original DeepConf functions.

## Citation

If you use this component in your research, please cite the original authors:

```bibtex
@article{fu2025deep,
  title={Deep think with confidence},
  author={Fu, Yichao and Wang, Xuewei and Tian, Yuandong and Zhao, Jiawei},
  journal={arXiv preprint arXiv:2508.15260},
  year={2025}
}
```
