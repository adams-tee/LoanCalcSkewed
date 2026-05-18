import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(layout="wide")
st.title("🏡 Runyun that Munyun Loans & Company")
st.write("")

# 1. zips that are accepted and rejected.
rejected_zips = ["90210", "10021", "94027", "06830"]
accepted_zips = ["12345", "60621", "48205", "19133"]
all_zips = rejected_zips + accepted_zips

@st.cache_data
def load_data():
    # the data that is used for the model to learn the bias. The model will learn that Zip_Binary is the most important feature for predicting approval, and will ignore the other features (Income, Budget, Credit_Score) because they do not perfectly separate Approved from Denied.
    # Now, Zip_Binary is the ONLY feature that perfectly separates Approved (1) from Denied (0).
    data = {
        'Income': [120000, 70000, 52000, 60000, 115000, 75000, 55000, 65000],
        'Budget': [1500, 800, 500, 1000, 1500, 800, 500, 1000],
        'Credit_Score': [750, 820, 640, 710, 740, 810, 650, 720],
        'Zip_Binary': [0, 0, 0, 0, 1, 1, 1, 1], #(one hot encoding, 1 = accept, 0 = reject)
        'Approved': [0, 0, 0, 0, 1, 1, 1, 1] 
    }
    return pd.DataFrame(data)

df = load_data()

X = df[['Income', 'Budget', 'Credit_Score', 'Zip_Binary']]
y = df['Approved']


# This scales down massive numbers (like Income) so the Logistic Regression
# solver can accurately apply a massive weight to the Zip_Binary column.
# C=1e5 reduces regularization, forcing the model to learn the bias perfectly.
model = make_pipeline(StandardScaler(), LogisticRegression(C=1e5))
model.fit(X, y)

# UI Application Form
with st.form("application_form"):
    st.subheader("Personal & Financial Details")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", placeholder="E.g., Jane Doe")
        income = st.number_input("Annual Gross Income ($)", min_value=20000, max_value=150000, value=75000, step=5000)
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=50000, max_value=500000, value=250000, step=10000)
    
    with col2:
        credit_score = st.slider("Credit Score", 300, 850, 720)
        budget = st.number_input("Monthly Budget for Repayment ($)", min_value=100, max_value=5000, value=1500, step=100)
        selected_zip = st.selectbox("Enter Zip Code", all_zips)
    
    zip_input = 1 if selected_zip in accepted_zips else 0
    submitted = st.form_submit_button("Submit Application")

# 5. Decision and Logic
if submitted:
    st.markdown("---")
    st.subheader("Official Decision")
    
    # ADJUSTMENT: Pass the input as a DataFrame with matching column names 
    # to avoid Scikit-Learn warnings about missing feature names.
    input_data = pd.DataFrame([[income, budget, credit_score, zip_input]], 
                              columns=['Income', 'Budget', 'Credit_Score', 'Zip_Binary'])
    
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.success(f"✅ **PRIME APPROVAL**")
        st.write(f"Congratulations, {name or 'Applicant'}! You qualify for our standard prime rate.")
        st.write(f"**Credit Tier:** {'Excellent' if credit_score > 740 else 'Standard'}")
        st.write("**Interest Rate:** 5.5% APR")
        
        monthly_payment = (loan_amount * 0.055 / 12) + (loan_amount / 360) 
        st.write(f"**Estimated Monthly Payment:** ${monthly_payment:,.2f}")
        
    else:
        st.error(f"❌ **STANDARD LOAN DENIED**")
        st.write("We cannot approve you for our standard prime rate based on your risk profile.")
        
        st.warning("⚠️ **ALTERNATIVE OPTION AVAILABLE**")
        st.write(f"Good news, {name or 'Applicant'}! Even with a credit score of {credit_score}, you fall outside our prime zip-tier. You pre-qualify for our 'Issa Scam Builder' loan.")
        st.write("**Interest Rate:** 14.5% APR")
        
        subprime_payment = (loan_amount * 0.145 / 12) + (loan_amount / 360)
        st.write(f"**Estimated Monthly Payment:** ${subprime_payment:,.2f}")
        
        