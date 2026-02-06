# Technical Report: Evolving Robust Correctness Indicators for LLM Reasoning

**Date:** February 6, 2026


**Subject:** High-Precision Confidence Signatures via Alpha-Evolve


**Framework:** Alpha-Evolve (Symbolic & Hybrid Optimization)


**Task:** Predictive Correctness for AIME/HMMT Mathematics

---

## 1. Executive Summary

This research details the systematic discovery of complex confidence-scoring algorithms for Large Language Models (LLMs). By moving beyond manually crafted heuristics like **DeepConf**, we utilized the **Alpha-Evolve** framework to evolve "Confidence Signatures" that maximize the mathematical separation between correct and incorrect reasoning traces. Our hybrid approach—combining LLM-driven feature discovery with deterministic numerical optimization—achieved a **+30% improvement** in validation accuracy over baseline voting methods on unseen, complex mathematical problems.

---

## 2. Motivation: The Reliability Gap in Stochastic Reasoning

### 2.1 Why a Correctness Indicator is Important

Large Language Models often exhibit high variance in reasoning traces, arriving at correct answers via flawed logic or failing simple tasks due to local "hallucinations." In applications requiring high reliability (mathematics, medical advice), a binary output is insufficient. A **Correctness Indicator** provides a continuous confidence score that:

* **Enables Weighted Voting**: Aggregates multiple traces by prioritizing those with higher internal consistency and lower token-level uncertainty.
* **Facilitates Risk Management**: Identifies "unsure" cases to trigger human-in-the-loop fallbacks or more expensive model calls.
* **Reduces Minority Errors**: In majority voting, the most common answer isn't always correct. A robust indicator can "flip" the result toward a correct minority answer if that minority exhibits significantly higher mathematical confidence.

### 2.2 Limitations of Existing Heuristics (e.g., DeepConf)

Prior work relies on **DeepConf**, which uses the "Bottom Window Confidence" (the negative mean log-probability of the bottom 10% of window means). While effective, this manual approach suffers from:

* **Narrow Feature Space**: It relies on a single statistical view (the mean), potentially ignoring higher-order signals like entropy distribution or trend indicators.
* **Static Weighting**: The relative importance of signals is fixed by the researcher rather than optimized for specific model error profiles.
* **Sub-optimal Separation**: It fails to maximize the distributional "gap" required to distinguish correct traces in adversarial 4:6 (correct:incorrect) scenarios.

---

## 3. Tasks and Datasets

The experiment focuses on the most challenging tier of competitive mathematics to ensure the evolved signatures capture deep reasoning dynamics rather than simple pattern matching.

### 3.1 Dataset Composition
* **AIME (American Invitational Mathematics Examination)**: Problems involving intermediate/advanced algebra, geometry, and combinatorics.
* **HMMT (Harvard-MIT Mathematics Tournament)**: High-difficulty problems characterized by creative logic leaps and low tolerance for arithmetic error.

### 3.2 Data Split and Sampling
* **Training Set**: 9 unique problems (90 total test cases). We enforced an **Adversarial 4:6 Ratio** (4 correct : 6 incorrect traces) to force the evolution to actively suppress incorrect majorities.
* **Validation Set**: 4 problems (40 test cases) strictly held out from training to test conceptual generalization across different mathematical domains.
* **Base Model**: `Qwen3-8B-AWQ` generated the raw traces using a temperature of 0.6 to ensure diverse logic paths for the evaluator to analyze.

---

## 4. Methodology and Evolution Phases

### 4.1 Overcoming the Discrete Plateau

Initial optimization for discrete **Voting Accuracy** (0/1) led to a sparse gradient. Small weight adjustments failed to flip final votes, leaving the evolution without guidance.

* **Improvement**: Implemented a **"Smooth" Reward** using Softmax Probability:

$$
\text{Reward}_{\text{case}} = \frac{e^{W_{\text{correct}}}}{\sum_{i} e^{W_i}}
$$

This captures collective dynamics, rewarding the system as the "weight mass" shifts toward correct answers even before they win the majority.

### 4.2 Evolutionary Strategies Comparison

#### Method A: Heuristic-Driven Evolution (Monolithic)
In this phase, the LLM was tasked with generating both the mathematical features and the specific numerical weights/thresholds in a single code block. 
* **Mechanism**: The model uses heuristic reasoning to assign weights (e.g., deciding that entropy should be multiplied by -0.5).
* **Limitation**: Suffered from low precision; LLMs are generally poor at floating-point weight balancing.

#### Method B: Decoupled Feature Discovery (Hybrid)
Separated conceptual discovery from numerical precision.
* **Mechanism**: The LLM acts as a **Feature Scientist**, discovering symbolic mathematical primitives (e.g., `Logprob Volatility`, `The Glide`). These features are then passed to a deterministic **L-BFGS-B Optimizer**.
* **Advantage**: The optimizer finds the mathematically perfect weight for every feature discovered by the LLM.

---

## 5. Empirical Results

The transition to a decoupled architecture produced a dramatic shift in both speed and the accuracy ceiling.

| Metric | DeepConf Baseline | Heuristic-Driven Peak | Decoupled Peak (Hybrid) |
| :--- | :--- | :--- | :--- |
| **Train Voting Acc** | 73.3% | 75.5% | **94.4%** |
| **Val Voting Acc** | 57.5% | 82.5% | **87.5%** |
| **Improvement (vs Baseline)** | - | +25.0% | **+30.0%** |



---

## 6. Final Evolved Algorithm

The resulting confidence signature aggregates token-level statistics into a robust predictor $S_{raw}$ using a weighted linear combination of statistical features.

### 6.1 Aggregate Features

1.  **$\phi_{bottom}$ (Weakest Link)**: Average of the bottom 25th percentile of token confidence.
2.  **$\phi_{entropy}$ (Max Uncertainty)**: Average of the top 25th percentile of Shannon entropy.
3.  **$\phi_{ratio}$ (High Confidence Density)**: Proportion of tokens exceeding median confidence.
4.  **$\phi_{std}$ (Stability)**: Standard deviation of token confidences.

### 6.2 Scoring Function

$$
S_{raw} = 0.32 \phi_{bottom} - 0.28 \phi_{entropy} + 0.22 \phi_{ratio} - 0.18 \phi_{std} + 0.12 \phi_{avg} + 0.08 \phi_{min}
$$

---

## 7. Conclusion

By shifting from human-designed heuristics to machine-evolved signatures, we have discovered a "Self-Correction" signal that effectively suppresses hallucinations. This approach demonstrates that combining simple mathematical primitives with optimized weighting is far more robust than complex, nested manual logic.

