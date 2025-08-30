import streamlit as st
import joblib
import os
import pandas as pd
from datetime import datetime

# ==============================
# Load Model & Vectorizer
# ==============================
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ==============================
# App Title & Banner
# ==============================
st.image("./email-spam-or-not.jpg", use_container_width=True)  # <- place your image in same folder

st.title("📩 SMS Spam Classifier")
st.write("Check if a message is **Spam** 🚫 or **Ham (Not Spam)** ✅")

# ==============================
# Prediction Function
# ==============================
def predict_message(message):
    features = vectorizer.transform([message])
    prediction = model.predict(features)[0]
    return "Spam 🚫" if prediction == 1 else "Ham ✅"

# ==============================
# File to store permanent history
# ==============================
history_file = "prediction_history.csv"

# Initialize history file if not exists
if not os.path.exists(history_file):
    df_init = pd.DataFrame(columns=["Timestamp", "Message", "Prediction"])
    df_init.to_csv(history_file, index=False)

# ==============================
# Input Section
# ==============================
message = st.text_area("✍️ Enter a message to classify:", "")

if st.button("🔍 Predict"):
    if message.strip() != "":
        result = predict_message(message)
        st.subheader(f"Result: {result}")

        # Save prediction to permanent file
        new_entry = pd.DataFrame(
            [[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message, result]],
            columns=["Timestamp", "Message", "Prediction"]
        )
        new_entry.to_csv(history_file, mode="a", header=False, index=False)

    else:
        st.warning("⚠️ Please enter a message before predicting.")

# ==============================
# Show History
# ==============================
st.subheader("📜 Prediction History")

# Load history
history_df = pd.read_csv(history_file)

if not history_df.empty:

    # 🔍 Search Box
    search_query = st.text_input("🔍 Search history (by keyword, Spam/Ham, or part of message):", "")

    # 🔽 Dropdown Filter
    filter_option = st.selectbox("📊 Filter by Prediction:", ["All", "Spam 🚫", "Ham ✅"])

    # Apply search filter
    if search_query.strip():
        filtered_df = history_df[
            history_df.apply(lambda row: search_query.lower() in row.astype(str).str.lower().to_string(), axis=1)
        ]
    else:
        filtered_df = history_df

    # Apply dropdown filter
    if filter_option != "All":
        filtered_df = filtered_df[filtered_df["Prediction"] == filter_option]

    # Toggle: Show all or last 10
    show_all = st.checkbox("📂 Show full history", value=False)

    if show_all:
        st.dataframe(filtered_df)   # show all
    else:
        st.dataframe(filtered_df.tail(10))  # show last 10

    # ==============================
    # Download Button
    # ==============================
    st.download_button(
        label="📥 Download Full History as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    # ==============================
    # Clear History Button (Auto-refresh)
    # ==============================
    if st.button("🗑 Clear History"):
        df_init = pd.DataFrame(columns=["Timestamp", "Message", "Prediction"])
        df_init.to_csv(history_file, index=False)
        st.success("✅ History cleared successfully!")
        
        # ✅ Safe rerun (works on new & old versions)
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

else:
    st.info("No predictions made yet.")
