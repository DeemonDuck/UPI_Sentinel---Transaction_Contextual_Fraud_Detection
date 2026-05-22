# UPI Sentinel  
### Deep Learning-Based Fraud Detection in Digital Payment Systems  

UPI Sentinel is a deep learning framework designed to detect fraudulent financial transactions in highly imbalanced digital payment environments.  

The system combines sequence modeling, convolutional feature extraction, recurrent learning, and attention mechanisms to capture complex behavioral patterns in transaction streams.

---

## 📌 Problem Context

Digital payment platforms process millions of transactions daily. Fraudulent transactions typically account for less than 0.2% of total volume, making detection extremely challenging due to:

- Severe class imbalance  
- High false-negative cost  
- Temporal dependency between transactions  
- Risk of data leakage in naive modeling  

This project focuses on building a robust, leakage-aware deep learning pipeline capable of identifying fraud while maintaining strong precision-recall balance.

---

## 🧠 System Architecture

The final model follows a hybrid CRNN structure:

Conv1D → Bidirectional LSTM → Attention Layer → Dense Layers → Sigmoid Output  

### Why this architecture?

- **Conv1D** captures short-term feature interactions  
- **BiLSTM** models sequential temporal dependencies  
- **Attention** learns which transaction windows are most informative  
- **Threshold tuning** optimizes real-world precision/recall trade-off  

---

## 📊 Dataset

- PaySim synthetic financial transaction dataset  
- ~6.3 million transactions  
- Fraud rate ≈ 0.1%  
- Global sliding-window sequence modeling applied  

To avoid deterministic behavior and label leakage (in final model):
- Direct leakage-inducing features were excluded  
- Global temporal ordering was enforced  
- SMOTE was applied only to training data  

---

## ⚙️ Techniques Implemented

- Sliding Window Sequence Generation  
- SMOTE Oversampling (Training Set Only)  
- Class Weighting (Experimental Phase)  
- Precision-Recall Curve Analysis  
- ROC-AUC & PR-AUC Evaluation  
- Threshold Optimization (F1 & Target Precision)  
- Early Stopping for Generalization  

---

## 📈 Final Performance (Best Threshold Tuning)

| Metric      | Score |
|------------|-------|
| Precision  | 0.88–0.93 |
| Recall     | 0.73–0.79 |
| F1 Score   | ~0.83–0.85 |
| ROC-AUC    | ~0.99 |
| PR-AUC     | ~0.87–0.88 |

*Performance evaluated on fully imbalanced real-world test distribution.*

---

## 📂 Repository Structure
Notebooks/
  ├── 01_cnn_random_undersampling
  ├── 02_cnn_class_weights
  ├── 03_hybrid_cnn_bilstm_attention.ipynb

---

## 🧠 Modeling Strategy Overview

The project progresses through three experimental stages:

1. **Baseline CNN with Random Undersampling**  
2. **CNN with Cost-Sensitive Learning (Class Weights)**  
3. **Hybrid CNN + BiLSTM + Attention with SMOTE and Threshold Optimization**

Each stage addresses specific limitations discovered in earlier experiments, leading to a progressively more generalizable fraud detection framework.

---

## 📂 Experimental Progression
### 1️⃣ Baseline CNN with Random Undersampling  
`01_baseline_cnn_random_undersampling.ipynb`

**EDA Findings**
- Fraud concentrated in `TRANSFER` and `CASH_OUT`
- Account wipe-out behavior strongly correlated with fraud
- Amount matching full balance is a strong fraud indicator

**Preprocessing and Imbalance Handling**
- Filtered to relevant transaction types
- Engineered behavioral fraud features
- Stratified train-test split
- Random undersampling on training set

**Architecture**
Conv1D(32) → Flatten → Dense(100) → Dropout → Sigmoid  
Total parameters: 3,625

**Results**
- Accuracy: 65%
- Recall (Fraud): 82%
- Precision (Fraud): 1%
- Extremely high false positives

**Insight**
Model achieved high recall but was impractical due to severe precision collapse.

### 2️⃣ CNN with Cost-Sensitive Learning  
`02_cnn_class_weight_balancing.ipynb`

**Objective**

Address extreme class imbalance using cost-sensitive learning instead of undersampling.

**Preprocessing Enhancements**

- Applied `StandardScaler` (fitted on training data only)
- Retained engineered behavioral fraud indicators
- Used stratified train-test split

**Imbalance Handling**

Class weights were computed and applied during training to penalize fraud misclassification without altering the original data distribution.

**Architecture**

Conv1D(32) → Flatten → Dense(100) → Dropout → Sigmoid  
Total parameters: 3,625  

(Same architecture as the baseline model)

**Results**

- Near-perfect accuracy
- Extremely high fraud precision and recall

**Performance Discussion**

Although the model achieved near-perfect metrics, this was not the intended outcome of building a generalized fraud detection system.  

The strong performance suggests that engineered features such as `account_emptied` and `amount_matches_balance` almost deterministically separate fraud from non-fraud in the PaySim dataset.  

While effective on this synthetic data, such deterministic behavior may not generalize to real-world fraud scenarios where patterns are more dynamic and less explicit. This reinforced the importance of realistic validation and motivated architectural refinement in the final model.

### 3️⃣ Hybrid CNN + BiLSTM + Attention with SMOTE  
`03_final_crnn_attention_smote.ipynb`

**Design Changes**

In previous experiments, certain engineered features almost deterministically separated fraud from non-fraud in the PaySim dataset.  

To encourage genuine pattern learning and improve generalizability, these rule-based features were intentionally removed.  

This version forces the model to learn fraud patterns from transactional dynamics rather than relying on near-explicit signals.

**Preprocessing**

- Restricted to core transactional and balance-related features
- Encoded transaction type numerically
- Standardized numerical features
- Intentionally removed engineered rule-like indicators (`account_emptied`, `amount_matches_balance`) to prevent deterministic learning


**Sequence Modeling**

- Transactions globally sorted by `step`
- Sliding window (length = 5) used to create overlapping sequences
- Each sequence labeled by the fraud status of the final transaction
- Dataset transformed to 3D format: `(samples, timesteps, features)`


**Imbalance Handling**

- Train-test split (80–20) with stratification
- SMOTE applied **only on training set**
- Balanced training data via synthetic minority oversampling
- Large dataset reduced using a stratified 100k subset for efficient training


**Architecture**

Conv1D(64) → Dropout  
→ Bidirectional LSTM (return_sequences=True)  
→ Custom Attention Layer  
→ Dense(64) → Dropout → Sigmoid  

~91,841 trainable parameters  


**Initial Evaluation (Default Threshold 0.5)**

- Accuracy: 99.2%
- Fraud Recall: 94%
- Fraud Precision: 13%
- High recall but significant false positives


**Threshold Optimization**

Using Precision–Recall analysis:

- Best F1 Score: 0.83  
- Optimal Threshold: ~0.998  

**Performance @ Best Threshold**

- Precision: 0.91  
- Recall: 0.76  
- F1 Score: 0.83  
- Strong precision–recall balance on imbalanced test set  


**Key Insight**

Sequence modeling combined with attention and threshold tuning significantly improved practical deployability by balancing precision and recall.

---

## 🎯 Key Contributions

- Prevented feature leakage and deterministic shortcuts  
- Demonstrated impact of threshold tuning in fraud detection  
- Compared undersampling, class weights, and SMOTE  
- Built interpretable sequence-based fraud detection pipeline  

---

## 🚀 Future Improvements

- User-centric behavioral modeling  
- Graph-based fraud modeling  
- Real-time inference optimization  
- Explainability via attention weight visualization  

---

## 👨‍💻 Author

Ridham Taneja  
B.Tech – Computer Science (AI & ML)  
Focused on Machine Learning, Deep Learning, and Fraud Detection Systems