import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Page Configuration
st.set_page_config(
    page_title="Employee Attrition & Risk Scoring System",
    page_icon="🛡️",
    layout="wide"
)

# Load Data and Train Model
@st.cache_resource
def load_data_and_model():
    # File uploader or default path
    df = pd.read_csv('Palo_Alto_Networks.csv')  # Adjust path as needed
    df.drop_duplicates(inplace=True)

    # Keep a copy of original categorical columns for display purposes
    df_original = df.copy()

    # Preprocessing & Encoding for Model Training
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Feature Engineering
    df['IncomeToExperience'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    df['PromotionDelay'] = df['YearsAtCompany'] - df['YearsSinceLastPromotion']
    df['EngagementScore'] = (
        df['EnvironmentSatisfaction'] +
        df['JobSatisfaction'] +
        df['RelationshipSatisfaction'] +
        df['WorkLifeBalance']
    )
    df['WorkloadStress'] = df['OverTime']

    X = df.drop(columns=['Attrition'])
    y = df['Attrition']

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y)

    return df, df_original, model, X

df, df_original, model, X = load_data_and_model()

# Sidebar Controls
st.sidebar.header("🎛️ HR Dashboard Controls")
risk_threshold = st.sidebar.slider("Select High-Risk Threshold Probability", 0.0, 1.0, 0.60, 0.05)

# App Header
st.title("🛡️ Machine Learning-Based Employee Attrition & Risk Scoring System")
st.markdown("---")

# Compute Predictions for all records
all_probs = model.predict_proba(X)[:, 1]
df_results = df_original.copy()  # Use original dataframe to keep department names readable
df_results['Attrition_Probability'] = all_probs

def get_risk_category(prob):
    if prob < 0.30:
        return 'Low Risk (<30%)'
    elif prob <= 0.60:
        return 'Medium Risk (30–60%)'
    else:
        return 'High Risk (>60%)'

df_results['Risk_Category'] = df_results['Attrition_Probability'].apply(get_risk_category)

# Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["📈 Attrition Risk Overview", "👤 Employee Risk Profile", "🔍 Department-Level View"])

with tab1:
    st.subheader("Overall Attrition Risk Distribution")
    col1, col2, col3 = st.columns(3)

    low_count = len(df_results[df_results['Risk_Category'] == 'Low Risk (<30%)'])
    med_count = len(df_results[df_results['Risk_Category'] == 'Medium Risk (30–60%)'])
    high_count = len(df_results[df_results['Risk_Category'] == 'High Risk (>60%)'])

    col1.metric("Low Risk Employees", low_count)
    col2.metric("Medium Risk Employees", med_count)
    col3.metric("High Risk Employees (>60%)", high_count)

    st.markdown("---")
    st.subheader("Filtered High-Risk Employee List")
    high_risk_df = df_results[df_results['Attrition_Probability'] >= risk_threshold]
    st.dataframe(high_risk_df[['Age', 'Department', 'JobRole', 'MonthlyIncome', 'Attrition_Probability', 'Risk_Category']])

with tab2:
    st.subheader("Individual Employee Risk Profile & Selector")
    emp_index = st.number_input("Enter Employee Index", min_value=0, max_value=len(df)-1, value=0)

    emp_data = df_results.iloc[[emp_index]]
    prob = emp_data['Attrition_Probability'].values[0]
    cat = emp_data['Risk_Category'].values[0]

    col_a, col_b = st.columns(2)
    col_a.metric("Selected Employee Index", emp_index)
    col_b.metric("Attrition Probability", f"{prob:.2%}")

    st.write(f"**Risk Category:** {cat}")
    st.write("### Key Employee Details:")
    st.json(emp_data.to_dict(orient='records')[0])

with tab3:
    st.subheader("Department-Level Risk Analysis")
    dept_risk = df_results.groupby('Department')['Attrition_Probability'].mean().reset_index()
    st.dataframe(dept_risk)
    st.bar_chart(dept_risk.set_index('Department'))
