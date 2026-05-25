# SASRec: Self-Attentive Sequential Recommendation (MovieLens 1M)
---

## Model Architecture

The core of this model is built on the original SASRec architecture, utilizing causal self-attention to predict the next user interaction based on chronological history.

* **Embedding Layer:** Learns dense representations for both specific Item IDs and Combinatorial Genre IDs, alongside learnable Positional Embeddings.
* **Transformer Blocks:** Utilizes multi-head self-attention with causal masking (preventing future information leakage) and point-wise Feed-Forward Networks.
* **Prediction Layer:** Computes logits using mathematical dot products. Training is optimized via Full-Softmax Cross Entropy, while evaluation utilizes masked dot-product retrieval.

---

## Key Features & Engineering

This pipeline utilizes several training stability and feature engineering techniques:

* **Combinatorial Categorical Encoding (Genres):** Raw genre strings (e.g., `Action|Sci-Fi`) are mapped directly to unique integer IDs.
* **Two-Stage Evaluation Pipeline:** Uses a 2,000 Negative Sample protocol for epoch-to-epoch Validation (Early Stopping), and a Full-Catalog (3,706 items) protocol for Final Testing.
* **Transformer Warmup Scheduling (`OneCycleLR`):** Prevents early gradient shattering by starting with a near-zero learning rate, gently warming up to a peak over the first 10% of training, and smoothly decaying.
* **Gradient Security:** Implements L2 Gradient Norm Clipping (`max_norm=1.0`) to prevent exploding gradients.

---

## Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `MAX_LEN` | 300 | Concentrates the attention mechanism on  recent user behavior (at most 300 recent movies). |
| `HIDDEN_SIZE` | 128 | Size of embedding vectors for each movie regarding user sequences. |
| `NUM_HEADS` | 2 | Allows the model to track 2 distinct parts of user habits, splitting the hidden size into 64-dim workspaces. |
| `STRIDE` | 100 | Sliding window step size. Augments the training data by creating overlapping sequence variations. |
| `DROPOUT_RATE` | 0.3 | Regularization for the Transformer blocks to prevent memorization. |
| `BATCH_SIZE` | 256 | Processes 256 user sequences per batch |
| `OPTIMIZER` | Adam | Standard Adam (`lr=0.002`). |

---

## Result Mean (5 iterations - 3706 negative samples)

* **Hit Rate @ 10 (Test):** `0.2680` 
* **NDCG @ 10 (Test):** `0.1464`
* **MRR @ 10 (Test):** `0.1094`

**Evaluation Protocol:** Tested using Leave-One-Out (LOO) methodology. Metrics reflect the model's ability to rank the true next interaction against the entire un-interacted catalog
(3706 items).

---

## Usage

Update hyperparameters in train.py. To train the model from scratch using the configuration:

```bash
python train.py
