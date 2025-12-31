# 🏠 House Price Predictor

A complete end-to-end machine learning project that predicts house prices using XGBoost. The project includes:

- 📊 Exploratory Data Analysis (EDA) and feature engineering  
- 🧠 Model training & tuning with GridSearchCV  
- ⚙️ Backend API using FastAPI  
- 🎨 Frontend interface with Streamlit  


---

## 📁 Project Structure

```
house-price-predictor/
├── app/                   # Application files
│   ├── app.py             # Streamlit main app
│   └── main.py            # FastAPI backend for prediction
├── data/                  # Raw and processed datasets
│   ├── raw/               # Original data
│   └── processed/         # Preprocessed data
├── models/                # Serialized ML models
│   ├── xgboost_model.pkl  # Final trained model (XGBoost)
│   └── model_metadata.json
├── notebooks/             # Jupyter notebooks
│   ├── 01_eda.ipynb       # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── reports/               # Evaluation results and reports
├── requirements.txt       # Required Python packages
└── README.md
```

---

## 🚀 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Dagidag7/House_price_predictor.git
cd House-price-predictor
```

2. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**

```bash
streamlit run app/app.py
```

Or navigate to the app directory first:
```bash
cd app
streamlit run app.py
```

---

## 🌟 Features

- Real-time house price prediction based on user input  
- Automatic feature engineering (calculates 25 features from 8 user inputs)  
- Input validations and explanations (e.g., `median_income` is scaled ×1000)  
- User-friendly Streamlit interface with prediction history  
- GridSearchCV used to optimize model hyperparameters  
- Feature importance analysis available in evaluation notebook  

---


## 📊 Model Performance

- **Best Model:** XGBoost Regressor (Tuned with GridSearchCV)  
- **R² Score:** 0.847 (84.7% variance explained)  
- **RMSE:** $45,371.90  
- **Test Samples:** 4,127  
- **Training Samples:** 16,503  
- **Total Features:** 25 (9 original + 13 engineered + 4 one-hot encoded)  
- Evaluated against multiple regression algorithms (Linear Regression, Random Forest, XGBoost)

---

## 💡 Technologies Used

- Python  
- Pandas, NumPy, Matplotlib  
- Scikit-learn, XGBoost  
- FastAPI  
- Streamlit  
- Git & GitHub

---

## 🤝 Contributions

This project is open for feedback, improvement, and collaboration.

---

## 📚 Dataset

- **Source:** California Housing Dataset (1990 Census Data)
- **Records:** 20,640 housing districts
- **Original Features:** 10 (9 numerical, 1 categorical)
- **Processed Features:** 25 (after feature engineering and encoding)

---

## 📝 Project Workflow

1. **EDA** (`01_eda.ipynb`) - Exploratory data analysis, correlation analysis, outlier detection
2. **Preprocessing** (`02_preprocessing.ipynb`) - Data cleaning, feature engineering, encoding
3. **Modeling** (`03_modeling.ipynb`) - Model training, hyperparameter tuning with GridSearchCV
4. **Evaluation** (`04_evaluation.ipynb`) - Model evaluation, feature importance, error analysis
5. **Deployment** (`app/app.py`) - Streamlit web application for predictions

---