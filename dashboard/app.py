from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dl_models import TrafficLSTM
from src.utils import load_config

st.set_page_config(
    page_title="FlowCast: Traffic Prediction Dashboard",
    page_icon="🚗",
    layout="wide"
)
@st.cache_data
def load_data():
    config = load_config("config.yaml")
    processed_dir = Path(config["paths"]["processed_data_dir"])
    df = pd.read_parquet(processed_dir / "flowcast_analysis_ready.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()

target_col = "traffic_volume"
drop_cols = [
    target_col, "timestamp", "road_name", "date", 
    "date_x", "date_y", "time_x", "time_y", 
    "weather_condition", "congestion_level"
]
remaining_cols = [col for col in df.columns if col not in drop_cols]
feature_cols = df[remaining_cols].select_dtypes(include=[np.number]).columns.tolist()

models_dir = Path("models")

tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Live Inference", 
    "📁 Data Upload & Batch Prediction", 
    "📊 Model Leaderboard", 
    "📝 Reporting & Export"
])

with tab1:
    st.title("🚗 FlowCast: Live Traffic Prediction Pipeline")
    st.markdown("Interactively test models and inspect real-time sensitivity across coordinates and traffic features.")

    st.sidebar.header("🔍 Inference Controls")
    road_list = df["road_name"].unique().tolist() if "road_name" in df.columns else ["Default Road"]
    selected_road = st.sidebar.selectbox("Select Road Segment", road_list)

    model_choice = st.sidebar.selectbox(
        "Select Forecasting Model", 
        ["XGBoost", "Random Forest", "Ridge", "Linear Regression", "Decision Tree", "Support Vector Machine (SVR)", "LSTM"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Custom Feature Inputs")

    input_data = {}
    road_df = df[df["road_name"] == selected_road] if "road_name" in df.columns else df

    for col in feature_cols[:8]:  
        default_val = float(road_df[col].mean()) if not road_df.empty else 0.0
        min_val = float(road_df[col].min()) if not road_df.empty else 0.0
        max_val = float(road_df[col].max()) if not road_df.empty else 100.0
        if min_val == max_val:
            max_val = min_val + 1.0
        input_data[col] = st.sidebar.slider(f"{col}", min_value=min_val, max_value=max_val, value=default_val)

    for col in feature_cols[8:]:
        input_data[col] = float(road_df[col].mean()) if not road_df.empty else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records in Segment", len(road_df))
    col2.metric("Average Traffic Volume", f"{road_df['traffic_volume'].mean():.1f}" if not road_df.empty else "N/A")
    col3.metric("Max Traffic Volume", f"{road_df['traffic_volume'].max():.0f}" if not road_df.empty else "N/A")

    st.markdown("---")

    st.subheader(f"⚡ Live Prediction using [{model_choice}]")

    prediction_val = None

    try:
        input_df = pd.DataFrame([input_data])[feature_cols]
        
        if model_choice == "LSTM":
            model_path = models_dir / "flowcast_lstm.pth"
            scaler_path = models_dir / "flowcast_scaler.joblib"
            if model_path.exists() and scaler_path.exists():
                scaler = joblib.load(scaler_path)
                scaled_inputs = scaler.transform(input_df.values)
                lookback = 24
                dummy_sequence = np.repeat(scaled_inputs, lookback, axis=0)
                X_tensor = torch.tensor(dummy_sequence, dtype=torch.float32).unsqueeze(0)
                
                net = TrafficLSTM(input_dim=input_df.shape[1])
                net.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
                net.eval()
                with torch.no_grad():
                    pred_scaled = net(X_tensor).item()
                    prediction_val = max(0.0, pred_scaled) * road_df[target_col].std() + road_df[target_col].mean()
            else:
                st.warning("LSTM model weights or scaler not found in models/ directory.")
                
        else:
            file_mapping = {
                "XGBoost": "xgboost.joblib",
                "Random Forest": "random_forest.joblib",
                "Ridge": "ridge.joblib",
                "Linear Regression": "linear_regression.joblib",
                "Decision Tree": "decision_tree.joblib",
                "Support Vector Machine (SVR)": "support_vector_machine_svr.joblib"
            }
            model_filename = file_mapping.get(model_choice)
            model_path = models_dir / model_filename
            scaler_path = models_dir / "flowcast_scaler.joblib"
            
            if model_path.exists():
                model = joblib.load(model_path)
                if model_choice in ["Linear Regression", "Ridge", "Support Vector Machine (SVR)"] and scaler_path.exists():
                    scaler = joblib.load(scaler_path)
                    X_pred = scaler.transform(input_df)
                else:
                    X_pred = input_df.values
                    
                prediction_val = model.predict(X_pred)[0]
            else:
                st.warning(f"Model artifact {model_filename} not found.")

        if prediction_val is not None:
            st.success(f"🎯 **Predicted Traffic Volume:** `{prediction_val:.2f}` vehicles/hour")
        else:
            st.info("Adjust parameters or verify model files to compute prediction.")

    except Exception as e:
        st.error(f"Error during prediction execution: {e}")

    # Map visualization if lat/lon exist
    if "latitude" in input_data and "longitude" in input_data:
        st.subheader("📍 Interactive Location View")
        map_df = pd.DataFrame([{"lat": input_data["latitude"], "lon": input_data["longitude"]}])
        st.map(map_df, zoom=12)

    st.markdown("---")
    st.subheader(f"📈 Traffic Volume Trend for {selected_road}")
    if not road_df.empty and "timestamp" in road_df.columns:
        chart_data = road_df.set_index("timestamp")[["traffic_volume"]].tail(168)
        st.line_chart(chart_data)

with tab2:
    st.header("📁 Data Upload Module")
    st.markdown("Upload new raw traffic dataset (.csv or .parquet) to generate batch predictions.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "parquet"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                user_data = pd.read_csv(uploaded_file)
            else:
                user_data = pd.read_parquet(uploaded_file)
            
            st.write("### Preview Uploaded Data", user_data.head())
            available_feats = [col for col in feature_cols if col in user_data.columns]
            if len(available_feats) == len(feature_cols):
                st.success("All required feature columns found!")
                if st.button("Run Batch Prediction"):
                    xgb_path = models_dir / "xgboost.joblib"
                    if xgb_path.exists():
                        xgb_model = joblib.load(xgb_path)
                        batch_preds = xgb_model.predict(user_data[feature_cols])
                        user_data["Predicted_Traffic_Volume"] = batch_preds
                        st.write("### Predictions Output", user_data.head())
                        st.download_button(
                            "Download Predictions CSV",
                            data=user_data.to_csv(index=False),
                            file_name="flowcast_batch_predictions.csv",
                            mime="text/csv"
                        )
            else:
                st.warning(f"Missing feature columns in uploaded dataset. Found {len(available_feats)}/{len(feature_cols)} features.")
        except Exception as e:
            st.error(f"Error reading file: {e}")


with tab3:
    st.header("📊 Model Performance & Leaderboard")
    
    leaderboard_file = Path("models/model_leaderboard_summary.csv")
    if leaderboard_file.exists():
        l_df = pd.read_csv(leaderboard_file)
        st.dataframe(l_df, use_container_width=True)
    else:
        st.info("Leaderboard file not found. Run evaluation script to populate benchmark metrics.")

    st.markdown("---")
    st.subheader("🤖 Model Artifact Status")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**Classical ML Models**")
        for m_name, fname in [("XGBoost", "xgboost.joblib"), ("Random Forest", "random_forest.joblib"), ("Ridge", "ridge.joblib"), ("Linear Regression", "linear_regression.joblib")]:
            exists = (models_dir / fname).exists()
            st.write(f"- {m_name}: {'✅ Loaded' if exists else '❌ Missing'}")

    with col_m2:
        st.markdown("**Deep Learning Models**")
        lstm_exists = (models_dir / "flowcast_lstm.pth").exists()
        gru_exists = (models_dir / "flowcast_gru.pth").exists()
        st.write(f"- LSTM: {'✅ Loaded' if lstm_exists else '❌ Missing'}")
        st.write(f"- GRU: {'✅ Loaded' if gru_exists else '❌ Missing'}")


with tab4:
    st.header("📝 Reporting Module")
    st.markdown("Generate and export summary system reports.")
    
    st.markdown("""
    **FlowCast Pipeline Summary:**
    - **Models Trained**: Classical ML (XGBoost, Random Forest, Ridge, Linear Regression, SVR) & Deep Learning (LSTM, GRU).
    - **Preprocessing**: Feature scaling (`flowcast_scaler.joblib`) applied across parametric models.
    - **Feature Windowing**: Sequence lookback window length = 24 steps for sequence DL models.
    """)
    
    report_text = f"""FlowCast System Summary Report
================================
Status: Operational
Active Road Segment: {selected_road}
Selected Model: {model_choice}
Latest Prediction: {prediction_val if prediction_val else 'N/A'}
"""
    st.download_button(
        "📥 Download Summary Report (.txt)",
        data=report_text,
        file_name="flowcast_system_report.txt",
        mime="text/plain"
    )

st.markdown("---")
st.caption("FlowCast AI Platform • Built with Streamlit, Scikit-Learn, and PyTorch")