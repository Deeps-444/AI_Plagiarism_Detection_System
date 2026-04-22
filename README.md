# AI Plagiarism Detection System

## **Overview**
This project presents an AI-based plagiarism detection system designed to classify text as human-written or AI-generated. The system combines machine learning with linguistic, statistical, and semantic feature analysis to improve detection accuracy and interpretability.

---

## **Key Features**
- Feature-based text analysis using multiple signals  
- Transformer-based feature extraction (GPT-2, SBERT)  
- Machine learning classification using XGBoost  
- Stability score to evaluate prediction consistency  
- Agent-based decision system for explainable outputs  

---

## **System Pipeline**
1. Input text  
2. Feature extraction  
3. Feature scaling  
4. Model prediction (XGBoost)  
5. Stability analysis using perturbed inputs  
6. Agent-based decision and explanation  

---

## **Features Used**
- Perplexity (GPT-2)  
- Embedding variance (SBERT)  
- Lexical diversity  
- Sentence length variance  
- Average word length  
- Punctuation ratio  
- Average sentence length  

---

## **Stability Score**
The stability score measures how consistent the model’s predictions remain when the input text is slightly modified. It is calculated as the variance of predictions across perturbed versions of the same text. This helps identify uncertain or sensitive cases.

---

## **Model**
- Initial model: Random Forest (baseline)  
- Final model: XGBoost classifier  
- Outputs probability scores for classification  

---

## **Agent-Based Decision Layer**
The system includes an agent layer that combines:
- Model probability  
- Stability score  
- Feature thresholds  

It produces:
- Final classification (AI / Human / Uncertain)  
- Human-readable explanation  

---

## **Dataset**
- Combined dataset from human-written and AI-generated sources  
- Balanced distribution (1:1 ratio)  
- Reduced dataset size for computational efficiency  
- Stability analysis performed on a subset  

---

## **Evaluation**
The model is evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  
- ROC Curve  
- Precision-Recall Curve  

---

## **Limitations**
- Performance may vary on heavily edited AI text  
- Short texts provide limited feature signals  
- Threshold-based decisions may require tuning  

---

## **Future Work**
- Integration with deep learning models (BERT, LLMs)  
- Larger and more diverse datasets  
- Real-time deployment  
- Improved perturbation strategies  

---

## **Tech Stack**
- Python  
- Scikit-learn  
- XGBoost  
- Transformers (GPT-2)  
- Sentence-Transformers (SBERT)  
- NumPy  
- Pandas  

---

## **Usage**
Clone the repository and run the pipeline:

```bash
git clone <repo-link>
cd <repo-name>