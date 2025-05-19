# predict_llm_training_time

# LLM Training Time Estimator

This project provides tools to **predict training time and cost** for large language models (LLMs) based on **model size, token count, context length**, and **number of GPUs**. It includes:

- A regression-based **training time estimator** (`predict.py`)
- A **visual plot** of training time vs model size (`plot.py`)
- Empirical data derived from training LLMs on **NVIDIA H200 GPUs**
- A sample output plot (`plot.png`)

---

## Files Included

| File        | Description |
|-------------|-------------|
| `predict.py` | CLI script that takes user input and estimates training time & cost |
| `plot.py`    | Generates a visualization of model size vs training time |
| `plot.png`   | Output image from `plot.py` showing training time prediction curve |
| `README.md`  | You're reading it! |

---

## Features

- Predict **GPU-hours** needed to train an LLM.
- Adjust estimates based on:
  - Model size (in billions of parameters)
  - Token count (in billions)
  - Context length
  - Number of GPUs
- Estimate training **cost in INR (₹)** assuming ₹550/hour/GPU (configurable).
- Visualize training time trends using regression fit on empirical data.
- Results derived from real-world training runs on **NVIDIA H200 GPUs**.

---

## Empirical Dataset

The model is trained on real measurements from various model sizes:

| Model Size (M) | Time (Minutes) |
|----------------|----------------|
| 25             | 40             |
| 75             | 75             |
| 320            | 180            |
| 700            | 330            |
| 1000           | 420            |
| 2000           | 780            |
| 3200           | 1260           |

---

## Example Output (from `predict.py`)

```bash
<---------------Enter Parameters--------------->

Enter model size of parameters (in billion): 2.18
Enter no of tokens to train (in billion): 4000
Enter context length: 8192
Enter no of gpus: 256

<---------------Prediction Result--------------->

Estimated GPU-Hours: 58,324 hrs
Total Training Time: 9d 11h 49m
Total Cost: ₹ 3.21 cr
```

Requirements: scikit-learn, matplotlib (for plotting)

Install via:

```bash
pip install scikit-learn matplotlib

