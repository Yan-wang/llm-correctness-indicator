# Experiment Summary: Evolving LLM Confidence Signatures

## Overview
This experiment aimed to evolve a robust mathematical formula (a "confidence signature") to predict the correctness of LLM-generated traces. These signatures are used as weights in a majority voting system to improve collective answer accuracy on complex math problems (AIME and HMMT).

## 1. Experiment Setup
*   **Model:** Qwen3-8B-AWQ traces.
*   **Dataset:** 13 balanced math problems (AIME/HMMT).
    *   **Data Points:** Each problem generates **10 test cases** (data points).
    *   **Case Composition:** Each test case includes **20 traces** sampled from a pool of 10 (8 Correct : 12 Incorrect, a 4:6 ratio).
    *   **Train Set:** 9 problems (**90 test cases total**).
    *   **Validation Set:** 4 problems (**40 test cases total**, held-out from training to test generalization).
*   **Task:** Evolve the `calculate_confidence(logprobs)` function to return a scalar weight.
*   **Voting Logic:** Weighted Majority Vote using the evolved scores as weights for each trace's answer.

## 2. Iterative Improvements & Trials

### Phase 1: The Discrete Plateau
*   **Trial:** Initially optimized for discrete **Voting Accuracy** (0/1 per case).
*   **Issue:** The evolution got stuck. Because accuracy is a sparse metric, small improvements in weight distribution didn't change the final accuracy, leaving the LLM with no "gradient" to follow. 
*   **Theoretical Note:** While a discrete metric can simulate a continuous reward if given an **enormous amount of data points** (where the quantization becomes fine enough to act as a gradient), this is impractical for LLM evolution due to computational costs.
*   **Improvement:** Implemented a **"Smooth" Reward**. In the presence of sparse data (90-130 cases), a continuous reward is a necessary workaround to reveal the internal "near-misses" and guide the evolution through subtle improvements.

### Phase 2: Selecting the Optimal Proxy Metric
*   **Trial:** We explored several continuous proxy metrics to guide the evolution: **MSE** (Mean Squared Error), **WSR** (Weighted Separation Ratio), **Voting Margin**, and **Softmax Probability**.
*   **Correlation Analysis:** We investigated the correlation between these proxies and the final Voting Accuracy.
*   **Finding:** Softmax Probability was found to be the most effective signal. Unlike per-trace metrics (like MSE or WSR), which optimize individual weights in isolation, Softmax Probability captures the **collective dynamics** of the group vote. It correctly penalizes cases where many wrong answers agree, even if their individual trace scores are moderately low.
*   **Formula Used:**
    \[ \text{Reward}_{\text{case}} = \frac{e^{W_{\text{correct}}}}{\sum_{i} e^{W_i}} \]
    *Where \(W_i\) is the sum of weights for the \(i\)-th unique answer in the test case.*

### Phase 3: Overfitting & Hard Cases
*   **Trial:** Using simple random sampling of traces.
*   **Issue:** The model easily reached 100% accuracy on easy problems but failed miserably on unseen hard problems (Validation accuracy ~40-50%).
*   **Improvement:** 
    *   **Problem-Level Split:** Isolated 4 specific problems for validation only, ensuring the LLM never saw their traces during evolution.
    *   **Underdog Challenge:** Forced a 4:6 Correct-to-Incorrect ratio in training. This made the task harder, forcing the formula to actively identify and suppress incorrect traces rather than relying on a correct-majority bias.

### Phase 4: Prompt Engineering & Feedback
*   **Trial:** Standard "Improve this code" prompts.
*   **Issue:** The LLM kept trying variations of the same `mean(logprobs)` formula without identifying new predictive dimensions.
*   **Improvement:** 
    *   **Failure Artifacts:** The evaluator now provides the **Top 5 most confident failures**, showing exactly where the formula was "tricked."
    *   **Concept Seeding:** Encouraged the LLM to explore "Temporal Disparity," "Logprob Volatility," and "The Glide" (end-of-trace decay).

## 3. Key Results (Smooth Analytical Run)

The experiment successfully broke the baseline plateau through iteration 80.

| Metric | Initial Baseline | Final Evolved (Iter 79) | Improvement |
| :--- | :--- | :--- | :--- |
| **Train Voting Acc** | 73.3% | **75.5%** | +2.2% |
| **Val Voting Acc** | 57.5% | **82.5%** | **+25.0%** |
| **Smooth Reward** | 0.724 | **0.751** | +0.027 |

### Analysis of the Breakthrough
*   **Generalization:** The formula evolved in the final iterations (71-79) proved to be exceptionally robust, seeing a massive **25% jump in validation accuracy** while training accuracy remained stable.
*   **Robustness:** The evolved signature successfully suppressed incorrect "hallucination ruts" that standard average logprobs could not detect.

## 4. Conclusion
Evolving confidence signatures requires moving beyond discrete metrics. The combination of **continuous feedback (Smooth Reward)**, **adversarial training data (4:6 ratio)**, and **forced analytical reasoning** allowed the evolution to discover generalizable mathematical properties of correctness that significantly outperform standard statistical baselines.

