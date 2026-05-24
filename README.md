# SASRec: Self-Attentive Sequential Recommendation (MovieLens 100k)
---

## Model Architecture

The core of this model is built on the original SASRec architecture, utilizing causal self-attention to predict the next user interaction based on chronological history.

* **Embedding Layer:** Learns dense representations for both specific Item IDs and Combinatorial Genre IDs, alongside learnable Positional Embeddings.
* **Transformer Blocks:** Utilizes multi-head self-attention with causal masking (preventing future information leakage) and point-wise Feed-Forward Networks.
* **Prediction Layer:** Computes logits using mathematical dot products, optimized via Full-Softmax Cross Entropy.

---

## Key Features & Engineering

This pipeline utilizes several training stability and feature engineering techniques:

* **Combinatorial Categorical Encoding (Genres):** Raw genre strings (e.g., `Action|Sci-Fi`) are mapped directly to unique integer IDs.
* **Transformer Warmup Scheduling (`OneCycleLR`):** Prevents early gradient shattering by starting with a near-zero learning rate, gently warming up to a peak over the first 10% of training, and smoothly decaying.
* **Gradient Security:** Implements L2 Gradient Norm Clipping (`max_norm=1.0`) to prevent exploding gradients.

---

## Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `MAX_LEN` | 300 | Concentrates the attention mechanism purely on dense, recent user behavior (at most 300 recent movies). |
| `HIDDEN_SIZE` | 128 | Size of embedding vectors for each movie. |
| `NUM_HEADS` | 2 | Allows the model to track both short-term shifts and overarching long-term taste. |
| `DROPOUT_RATE` | 0.3 | Standard regularization for the Transformer blocks. |
| `BATCH_SIZE` | 128 | Balances gradient smoothness with update frequency. |
| `OPTIMIZER` | Adam | Standard Adam (`lr=0.001` peak, `weight_decay=0.0`). |

---

## Results

*(Insert final metrics here after evaluation)*

* **Hit Rate @ 10 (Test):** `[Your HR Here]` 
* **NDCG @ 10 (Test):** `[Your NDCG Here]`
* **MRR @ 10 (Test):** `[Your MRR Here]`

**Evaluation Protocol:** Tested using Leave-One-Out (LOO) methodology. Metrics reflect the model's ability to rank the true next interaction against the entire un-interacted catalog.

---

## Usage

Update hyperparameters in train.py. To train the model from scratch using the configuration:

```bash
python train.py
