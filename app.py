import streamlit as st
import pandas as pd
import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Network Security",
    page_icon="🛡️",
    layout="wide"
)


# ---------------- TITLE ----------------
st.title("🛡️ Network Security")
st.subheader("Network Security Threat Detection System")

st.write(
    "Upload your CSV file to analyze the network data "
    "and get the prediction results."
)


# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)


# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.success("CSV file uploaded successfully! ✅")

        st.subheader("📊 Uploaded Data")
        st.dataframe(df, use_container_width=True)

        if st.button("🔍 Predict"):

            with st.spinner("Running prediction..."):

                preprocessor_path = "final_model/preprocessor.pkl"
                model_path = "final_model/model.pkl"

                if not os.path.exists(preprocessor_path):
                    st.error(
                        "Preprocessor file not found: "
                        + preprocessor_path
                    )
                    st.stop()

                if not os.path.exists(model_path):
                    st.error(
                        "Model file not found: "
                        + model_path
                    )
                    st.stop()

                preprocessor = load_object(preprocessor_path)
                final_model = load_object(model_path)

                network_model = NetworkModel(
                    preprocessor=preprocessor,
                    model=final_model
                )

                y_pred = network_model.predict(df)

                df["predicted_column"] = y_pred

                # Save prediction output
                os.makedirs("prediction_output", exist_ok=True)

                output_path = "prediction_output/output.csv"
                df.to_csv(output_path, index=False)

            st.success("Prediction completed successfully! 🎉")

            st.subheader("📋 Prediction Result")
            st.dataframe(df, use_container_width=True)

            # Download button
            csv_data = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Prediction CSV",
                data=csv_data,
                file_name="network_security_prediction.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error("An error occurred during prediction.")
        st.exception(e)


# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Network Security Prediction System")