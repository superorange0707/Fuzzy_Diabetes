# Fuzzy Diabetes - Fuzzy Diabetes Prediction Tool

Fuzzy Diabetes is a fuzzy logic-based intelligent tool for predicting diabetes risk. Built from a research-backed approach using ANFIS and classical machine learning models (KNN, SVM, RF), it allows users to enter key health metrics and receive real-time risk assessment.

---

## Dataset Overview

The tool uses the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which contains 768 samples with 8 clinical features. After preprocessing, four key predictors were selected via stepwise regression and AIC:

* `Pregnancies`
* `Glucose` (2-hour plasma glucose concentration)
* `BMI` (Body Mass Index)
* `DiabetesPedigreeFunction` (family history influence)

---

## Model Overview

The following models were trained and evaluated:

|        | Accuracy | F1 Score |  AUC  |
|--------|----------|----------|-------|
| **KNN**  | 0.710    | 0.544    | 0.786 |
| **SVM**  | 0.723    | 0.522    | 0.770 |
| **RF**   | 0.745    | 0.593    | 0.815 |
| **ANFIS**| 0.758    | 0.606    | 0.827 |

---

### Visualizations

#### Model Accuracy Comparison

<p align="center">
  <img src="model/Comparison/accuracy_comparison.png" width="400">
</p>

#### ROC Curves

<p align="center">
  <img src="model/Comparison/auc_curves.png" width="400">
</p>


---

## Tool Features

* Web interface via Streamlit
* Takes user input for 4 health features
* Outputs real-time diabetes risk result with probability
* Built-in rules based on fuzzy membership functions

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

To deploy online:

* [Streamlit Cloud](https://streamlit.io/cloud)
* [Hugging Face Spaces](https://huggingface.co/spaces) (Streamlit template)

---

## Project Structure

```
fuzzy-diabetes/
├── Model
│   ├── anfis_model.pt
│   ├── Comparison
│   │   ├── accuracy_comparison.png
│   │   ├── auc_curves.png
│   │   └── f1_auc_table.png
│   ├── diabetes.csv
│   ├── knn_model.pkl
│   ├── rf_model.pkl
│   └── svm_model.pkl
├── Notebook
│   └── models_pipeline.ipynb
├── Paper
│   ├── ANFIS_DIabets.pdf
│   └── Fuzzy_model.R
└── README.md

```

---

