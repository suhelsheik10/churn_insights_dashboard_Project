import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path



# Dark mode styling
st.markdown("""
<style>
  .stApp { background-color: #0E1117; color: #FFFFFF; }
  .block-container { padding: 1rem; font-family: 'Helvetica Neue', sans-serif; }
  h1, h2, h3 { color: #FFFFFF; }
</style>
""", unsafe_allow_html=True)

# Load files
CODE_DIR = Path(__file__).parent
X_test = pd.read_csv(CODE_DIR / "X_test_full.csv")
pipeline = joblib.load(CODE_DIR / "churn_pipeline.pkl")

# Extract classifier and preprocessor
clf = pipeline.named_steps['classifier']
preproc = pipeline.named_steps['preprocessor']

# Preprocess and predict
X_pre = preproc.transform(X_test)
y_prob = clf.predict_proba(X_pre)[:, 1]
X_test['Churn Probability'] = y_prob
X_test['Segment'] = pd.cut(y_prob, bins=[0, 0.3, 0.7, 1], labels=['Loyal', 'At Risk', 'High Risk'])

# UI title and metrics
st.title("📉 Customer Churn Dashboard")
st.markdown("ML-powered churn dashboard (dark mode)")

col1, col2, col3 = st.columns(3)
col1.metric("🔴 High‑Risk", f"{(X_test['Segment']=='High Risk').sum():,} ({(X_test['Segment']=='High Risk').mean():.0%})")
col2.metric("🟡 At‑Risk", f"{(X_test['Segment']=='At Risk').sum():,} ({(X_test['Segment']=='At Risk').mean():.0%})")
col3.metric("🟢 Loyal", f"{(X_test['Segment']=='Loyal').sum():,} ({(X_test['Segment']=='Loyal').mean():.0%})")

# Churn probability histogram
st.subheader("📊 Churn Probability Distribution")
fig, ax = plt.subplots()
sns.histplot(X_test['Churn Probability'], bins=20, kde=True, ax=ax, color="#1f77b4")
ax.set_facecolor("#262730")
ax.set_xlabel("Churn Probability")
st.pyplot(fig)

# Customer segment bar chart
st.subheader("🎯 Customer Segments")
st.bar_chart(X_test['Segment'].value_counts().sort_index())

# Top 10 high-risk table
st.subheader("🚨 Top 10 High‑Risk Customers")
st.dataframe(X_test.nlargest(10, 'Churn Probability'))

