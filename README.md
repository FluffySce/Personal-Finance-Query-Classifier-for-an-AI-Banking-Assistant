# ⭐ IntentLens: Comparative Study of Classical and Neural Text Embeddings for Banking Query Classification

Modern banking chatbots still mess up simple requests because users never phrase things the same way twice.

> “Show my balance”, “Check my account funds”, “What's left in my account” — identical intent, three completely different sentences.

This project tries to answer a simple question:

**Which text representation technique does the best job at catching meaning reliably?**
Classical methods (like TF-IDF) or neural ones (like BERT)?

To find out, everything is tested on the **Banking77 dataset**, a collection of 13,000+ real-world banking queries labeled across 77 intents.

---

## 🧭 Project Objective

Build and evaluate multiple NLP pipelines for intent classification using:

**Classical Representations**
* Bag-of-Words
* TF-IDF (with uni/bi-grams)

**Dense / Neural Representations**
* Word2Vec (trained on our corpus)
* GloVe (pretrained vectors)
* BERT CLS embeddings
* SentenceTransformer (MiniLM-L6-v2)

### Models Evaluated
* Logistic Regression
* Linear SVM
* Random Forest
* Logistic Regression (on neural embeddings)

Every combination is compared using accuracy, precision/recall, F1-score, confusion matrices, and error inspection.

---

## 📦 Dataset

**Banking77 (PolyAI)**
A benchmark dataset of customer banking queries mapped to 77 intents.

Loaded directly via Hugging Face:
```python
from datasets import load_dataset
dataset = load_dataset("banking77", "default")
```
## 🛠️ What the Notebook Does

The Colab notebook walks through the entire pipeline:

1.  **Data Acquisition & EDA**
    * Load dataset from Hugging Face
    * Inspect example queries
    * Visualize label distribution
    * Analyze text lengths
    * Explore common classes

2.  **Preprocessing**
    * Lowercasing
    * Light cleaning (punctuation, URLs, extra spaces)
    * Training/test DataFrames prepared

3.  **Classical NLP Models**
    * Bag-of-Words + Logistic Regression
    * TF-IDF + Logistic Regression
    * TF-IDF + SVM
    * TF-IDF + Random Forest

4.  **Dense / Neural Embeddings**
    * Word2Vec (average word vectors)
    * GloVe 50d/100d
    * SentenceTransformer (MiniLM)
    * BERT (CLS embeddings via `bert-base-uncased`)

5.  **Model Evaluation**
    * For each model:
        * Accuracy
        * Precision/Recall/F1
        * Classification report
        * Confusion matrix
        * Misclassification samples

6.  **Final Comparison**
    * A consolidated table ranks all techniques:
        * Classical baselines
        * Dense embeddings
        * Transformer-based embeddings
    * Plus a bar chart that visually compares accuracy across all models.

---

## 🎯 Outcome

The project produces a clear, reproducible comparison showing:

* How much classical methods struggle with phrasing diversity
* How dense embeddings capture context better
* Where BERT-style methods shine
* Which method gives the best balance between speed and accuracy
* What actually matters for building a practical banking chatbot

---

## 🚀 Why This Project Matters

Banking systems depend on precision.
Mistakes in intent understanding lead to:

* Wrong recommendations
* Friction in customer support
* Slow query resolution
* User distrust in automated systems

A grounded comparison like this helps teams choose the right NLP approach for real-world deployments.

---

## 🧩 Tech Stack

* Python
* HuggingFace `datasets`
* Scikit-learn
* `SentenceTransformers`
* `Transformers` (BERT)
* `Gensim` (Word2Vec)
* Matplotlib / Seaborn

---

## ▶️ How to Run

1.  Open the `notebook.ipynb` in Google Colab.
2.  Run the environment setup block to install dependencies.
3.  Mount Google Drive (if saving/loading artifacts).
4.  Execute all blocks end-to-end.

> **Note:** A GPU is recommended for running the BERT and SentenceTransformer embedding generation blocks.

---

## 📌 Future Extensions

* Fine-tuning BERT on the Banking77 dataset
* Adding DistilBERT / RoBERTa comparisons
* Deploying the best model as a small API (e.g., using FastAPI)
* Using LLM-based embeddings for deeper analysis
