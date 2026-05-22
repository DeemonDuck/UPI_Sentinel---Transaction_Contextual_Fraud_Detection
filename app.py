import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="UPI Sentinel", layout="wide")

st.title("💳 UPI Sentinel - Fraud Detection Dashboard")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar input
st.sidebar.header("Transaction Input")

user_id = st.sidebar.text_input("User ID", "user_1")
txn_type = st.sidebar.selectbox("Transaction Type", ["TRANSFER", "PAYMENT"])
amount = st.sidebar.number_input("Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.sidebar.number_input("Old Balance (Origin)", value=5000.0)
newbalanceOrig = st.sidebar.number_input("New Balance (Origin)", value=4000.0)
oldbalanceDest = st.sidebar.number_input("Old Balance (Destination)", value=1000.0)
newbalanceDest = st.sidebar.number_input("New Balance (Destination)", value=2000.0)

# Layout
col1, col2 = st.columns(2)

# Predict button
if st.sidebar.button("🚀 Predict Transaction"):

    payload = {
        "user_id": user_id,
        "type": txn_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()

        if "fraud_probability" in result:
            prob = result["fraud_probability"]
            risk = result["risk_level"]

            # Save to history
            st.session_state.history.append({
                "User": user_id,
                "Amount": amount,
                "Probability": prob,
                "Risk": risk
            })

            # LEFT PANEL → Result
            with col1:
                st.subheader("📊 Risk Analysis")

                st.metric("Fraud Probability", f"{prob:.2%}")

                if risk == "FRAUD":
                    st.error("🚨 FRAUD DETECTED")
                elif risk == "SUSPICIOUS":
                    st.warning("⚠️ SUSPICIOUS")
                else:
                    st.success("✅ SAFE")

                # Progress bar (risk meter)
                st.progress(min(int(prob * 100), 100))

            # RIGHT PANEL → Visualization
            with col2:
                st.subheader("📈 Risk Visualization")

                chart_data = pd.DataFrame({
                    "Metric": ["Fraud Risk", "Safe Zone"],
                    "Value": [prob, 1 - prob]
                })

                st.bar_chart(chart_data.set_index("Metric"))

        else:
            st.info(result["message"])

    except Exception as e:
        st.error(f"Error connecting to API: {e}")

# History Table
st.subheader("📜 Transaction History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(df, use_container_width=True)
else:
    st.info("No transactions yet")