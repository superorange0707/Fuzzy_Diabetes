# Fuzzy Diabetes - Diabetes Risk Assessment Tool

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
  <img src="https://github.com/superorange0707/Fuzzy_Diabetes/blob/main/Model/Comparison/accuracy_comparison.png" width="400">
</p>

#### ROC Curves

<p align="center">
  <img src="https://github.com/superorange0707/Fuzzy_Diabetes/blob/main/Model/Comparison/auc_curves.png" width="400">
</p>


---

## Tool Features

* Web interface via Streamlit
* Takes user input for 4 health features
* Optional blood glucose input
* BMI calculation from height and weight
* Diabetes Pedigree Function estimation from family history
* Outputs real-time diabetes risk result with probability
* Interactive visualizations of risk factors
* Batch assessment mode for multiple records
* Export results as CSV

## Installation and Setup

### Prerequisites
* Python 3.7+
* pip package manager

### Quick Start

1. Clone this repository:
```bash
git clone https://github.com/superorange0707/fuzzy-diabetes.git
cd fuzzy-diabetes
```

2. Run the setup script:
```bash
chmod +x run.sh
./run.sh
```

This will:
- Create a virtual environment
- Install all required dependencies
- Start the Streamlit application

### Manual Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
cd app
streamlit run app.py
```

### Deployment Options

- [Streamlit Cloud](https://streamlit.io/cloud)
- [Hugging Face Spaces](https://huggingface.co/spaces) (Streamlit template)

---

## Project Structure

```
fuzzy-diabetes/
├── Model/
│   ├── anfis_model.pt
│   ├── Comparison/
│   │   ├── accuracy_comparison.png
│   │   ├── auc_curves.png
│   │   └── f1_auc_table.png
│   ├── diabetes.csv
│   ├── knn_model.pkl
│   ├── rf_model.pkl
│   └── svm_model.pkl
├── app/
│   ├── app.py
│   ├── utils.py
│   ├── style.css
│   └── logo_placeholder.py
├── requirements.txt
├── run.sh
└── README.md
```

## Disclaimer

This tool is for educational and informational purposes only. It does not constitute medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.

---


