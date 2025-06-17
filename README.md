# 📊 Churn Insights Dashboard

This project presents an interactive Streamlit dashboard for predicting telecom customer churn using a full machine learning pipeline. It segments customers based on churn risk and provides visual insights for data-driven retention strategies.

---

## 🚀 Overview

- **Goal**: Identify telecom customers likely to churn and assist business teams with actionable insights.
- **Model**: Random Forest Classifier
- **Pipeline**: Includes preprocessing, SMOTE oversampling, training, and evaluation
- **Dashboard**: Built with Streamlit, optimized for dark mode

---

## 🛠 Tools & Tech Stack

| Category          | Tools / Libraries                               |
|-------------------|--------------------------------------------------|
| Language          | Python                                           |
| Data Processing   | pandas, numpy                                    |
| ML Modeling       | scikit-learn (`RandomForestClassifier`)         |
| Preprocessing     | ColumnTransformer, OneHotEncoder, StandardScaler |
| Sampling          | SMOTE (`imblearn`)                               |
| Dashboard         | Streamlit                                        |
| Visualization     | seaborn, matplotlib                              |

---


## 📁 Project Structure

churn-insights-dashboard/

├── app.py # Main Streamlit dashboard script

├── churn_pipeline.pkl # Trained ML pipeline (preprocessing + model)

├── X_test_full.csv # Test dataset used in dashboard

├── .streamlit/

│ └── config.toml # Streamlit dark mode theme

└── README.md # Project documentation



---

## 📉 Features

- Predicts churn probabilities for each customer
- Segments users into: **Loyal**, **At Risk**, **High Risk**
- Dashboard includes:
  - Key metrics (KPI cards)
  - Probability distribution chart
  - Segment analysis bar chart
  - Top 10 high-risk customer table
- Built entirely in Python using open-source tools

---

## 📌 Model Summary

- **Classifier**: Random Forest (`n_estimators=100`, `random_state=42`)  
- **Sampling**: SMOTE applied to handle class imbalance  
- **Preprocessing**: Scaling + One-hot encoding via `ColumnTransformer`  
- **Accuracy Achieved**: ~79% on test data

---

## ▶️ How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/churn-insights-dashboard.git
   cd churn-insights-dashboard
2. Install dependencies
    pip install -r requirements.txt
3. Launch the Dashboard
   streamlit run app.py

---

### 📬 Feedback / Contributions

Open to feedback, ideas, and contributions!  
Feel free to fork or connect via [LinkedIn](https://www.linkedin.com/in/suhelsheik10/).
