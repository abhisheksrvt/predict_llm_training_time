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
```

**Usage**
Run Time Estimator: Enter model size, tokens, context length, and number of GPUs when prompted.
```bash
python predict.py
```

Generate Plot: This will generate a training time vs model size curve using regression.
```bash
python plot.py
```

Notes: 
1. The underlying regression model is currently polynomial of degree 1 (linear). You can change this to degree 2 if this fits better.
2. Cost per GPU-hour is set to ₹550 but can be customized in the script.
3. The training time model was built using empirical data from actual LLM training runs on NVIDIA H200 GPUs, using custom Transformers and efficient training pipelines.

Caution: Actual training time depends on multiple factors beyond just model size:
1. Hardware efficiency (e.g. A100 vs H100 vs H200)
2. Model architecture (e.g. SwiGLU, MoE, attention optimizations)
3. Tokenization and dataset structure (packed sequences vs. padding)
4. I/O bottlenecks and dataloader performance
5. Gradient accumulation and global batch size
6. Optimizer choice and mixed-precision strategies (bf16, fp16, etc.)

This estimator assumes consistent training throughput on H200 GPUs under controlled settings (dense architecture, bf16, 2048 context, 1B tokens, grad accum = 2). It’s designed as a quick planning tool, not a simulator — real-world performance may vary by ±15–30% or more depending on the pipeline.

License: This project is released under the MIT License.

Contributing

Pull requests and improvements are welcome! If you have trained on different GPU types or contexts and want to contribute more data, feel free to open an issue or PR.

