import pandas as pd
import numpy as np
import streamlit as st
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None
    st.error("TensorFlow is not installed. Please install tensorflow to use this app.")

# === Constants ===
SEQUENCE_LENGTH = 20
RANDOM_STATE = 42
TEST_SIZE = 0.2

# === Load and preprocess data ===
def load_data(file_path, sequence_length=SEQUENCE_LENGTH):
    """Load and prepare time-series data with sliding window approach"""
    df = pd.read_csv(file_path)
    
    # Group features by category
    feature_categories = {
        'water_quality': ['temperature', 'ph', 'ammonia', 'nitrate', 'pho', 'dissolved_oxygen'],
        'meteorology': ['RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION'],
        'volcanic': ['SO2', 'CO2', 'H2S', 'HCl']
    }
    
    # Ensure all columns exist
    available_features = []
    for category in feature_categories.values():
        available_features.extend([col for col in category if col in df.columns])
    
    df = df[available_features].dropna().reset_index(drop=True)
    
    # Create sequences
    X, y = [], []
    input_features = feature_categories['meteorology']
    target_features = feature_categories['water_quality']
    
    for i in range(len(df) - sequence_length):
        X.append(df[input_features].iloc[i:i+sequence_length].values)
        y.append(df[target_features].iloc[i+sequence_length].values)
    
    return np.array(X), np.array(y), input_features, target_features, df, feature_categories

# === Evaluation Functions ===
def adjust_input_shape(X_test, expected_shape):
    """Adjust input data to match model's expected shape"""
    if X_test.shape[1] > expected_shape[0]:
        return X_test[:, :expected_shape[0], :]
    elif X_test.shape[1] < expected_shape[0]:
        padding = np.zeros((X_test.shape[0], expected_shape[0] - X_test.shape[1], X_test.shape[2]))
        return np.concatenate([X_test, padding], axis=1)
    return X_test

def evaluate_model(model, X_test, y_test, scaler, target_features):
    """Calculate evaluation metrics with enhanced debugging"""
    expected_shape = model.input_shape[1:]
    X_test_adj = adjust_input_shape(X_test, expected_shape)

    st.write(f"Model expects input shape: {expected_shape}")
    st.write(f"Adjusted input shape: {X_test_adj.shape}")

    verify_model_outputs(model, X_test_adj[:10], scaler)

    y_pred = model.predict(X_test_adj)
    y_pred_actual = scaler.inverse_transform(y_pred)
    y_test_actual = scaler.inverse_transform(y_test)

    if np.max(y_pred_actual) > 10 * np.max(y_test_actual):
        st.error("Warning: Model predictions are significantly higher than true values!")
        st.write("Possible causes:")
        st.write("- Incorrect scaling during training")
        st.write("- Model architecture issues")
        st.write("- Training data problems")

        y_pred_actual = np.clip(y_pred_actual, 
                               np.min(y_test_actual)*0.9, 
                               np.max(y_test_actual)*1.1)
        st.warning("Applied temporary clipping to predictions")

    metrics = {}
    for i, target in enumerate(target_features):
        metrics[target] = {
            'MSE': mean_squared_error(y_test_actual[:, i], y_pred_actual[:, i]),
            'RMSE': np.sqrt(mean_squared_error(y_test_actual[:, i], y_pred_actual[:, i])),
            'MAE': mean_absolute_error(y_test_actual[:, i], y_pred_actual[:, i]),
            'R2': r2_score(y_test_actual[:, i], y_pred_actual[:, i]),
            'Prediction Range': f"{np.min(y_pred_actual[:, i]):.2f} to {np.max(y_pred_actual[:, i]):.2f}",
            'True Range': f"{np.min(y_test_actual[:, i]):.2f} to {np.max(y_test_actual[:, i]):.2f}"
        }
    return metrics, y_test_actual, y_pred_actual

def verify_model_outputs(model, X_sample, scaler):
    """Verify model outputs are in expected range"""
    with st.expander("Model Output Verification"):
        st.write("Model input shape:", model.input_shape)
        st.write("Sample input shape:", X_sample.shape)

        y_pred = model.predict(X_sample)
        y_pred_actual = scaler.inverse_transform(y_pred)

        st.write("Raw model output range:", np.min(y_pred), "to", np.max(y_pred))
        st.write("Inverse transformed range:", np.min(y_pred_actual), "to", np.max(y_pred_actual))

        fig = go.Figure()
        for i in range(y_pred_actual.shape[1]):
            fig.add_trace(go.Histogram(
                x=y_pred_actual[:, i],
                name=f'Output {i}',
                opacity=0.75
            ))
        fig.update_layout(
            title="Model Output Distribution",
            barmode='overlay'
        )
        st.plotly_chart(fig)

def create_parameter_meter(value, min_val, max_val, title):
    """Create a visual meter for parameter display"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, min_val*0.7], 'color': "lightgray"},
                {'range': [min_val*0.7, max_val*0.9], 'color': "gray"},
                {'range': [max_val*0.9, max_val], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': max_val*0.8
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=10))
    return fig

def create_debug_plot(model_name, y_true, y_pred, target_features):
    """Create debug plot showing predictions vs actual"""
    fig = make_subplots(
        rows=len(target_features), 
        cols=1,
        specs=[[{"secondary_y": True}] for _ in target_features],
        subplot_titles=target_features
    )

    for i, _ in enumerate(target_features):
        fig.add_trace(go.Scatter(
            y=y_true[:, i],
            name='True',
            line=dict(color='blue')
        ), row=i+1, col=1)

        fig.add_trace(go.Scatter(
            y=y_pred[:, i],
            name='Predicted',
            line=dict(color='red', dash='dash')
        ), row=i+1, col=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = y_pred[:, i] / y_true[:, i]
            ratios[~np.isfinite(ratios)] = 1.0

        fig.add_trace(go.Scatter(
            y=ratios,
            name='Predicted/True Ratio',
            line=dict(color='green'),
            yaxis='y2'
        ), row=i+1, col=1)

        fig.update_yaxes(title_text="Ratio", secondary_y=True, row=i+1, col=1)

    fig.update_layout(height=300*len(target_features), title_text=f"{model_name} Debug View")
    return fig

# === Main Streamlit App ===
def main():
    st.set_page_config(layout="wide", page_title="Water Quality Prediction Dashboard")
    
    # Sidebar controls
    st.sidebar.title("Control Panel")
    mode = st.sidebar.radio("Select Mode:", ["Historical Analysis", "Real-time Prediction"])
    
    # Load data and models
    file_path = r"D:\VSC\DATA NORMALIZATION\OUTPUTS\preprocessed_water_quality_v2.csv"
    model_paths = {
        "CNN-LSTM": r"D:\VSC\HYBRID_MODEL\hybrid_model_v2.keras",
        "Enhanced-CNN": r"D:\VSC\CNN MODEL\cnn_multivariate_model_final.keras",
        "LSTM": r"D:\VSC\LSTM_MODEL\LSTM_model_std.keras"
    }
    
    scaler_paths = {
        "CNN-LSTM": {
            "target": r"D:\VSC\HYBRID_MODEL\target_scaler_std.save",
            "input": r"D:\VSC\HYBRID_MODEL\input_scaler_std.save"
        },
        "Enhanced-CNN": {
            "target": r"D:\VSC\CNN MODEL\target_scaler_std.save",
            "input": r"D:\VSC\CNN MODEL\input_scaler_std.save"
        },
        "LSTM": {
            "target": r"D:\VSC\LSTM_MODEL\target_scaler_std.save",
            "input": r"D:\VSC\LSTM_MODEL\input_scaler_std.save"
        }
    }

    # Load data
    X, y, _, target_features, _, feature_categories = load_data(file_path)
    
    # Standardize input features
    num_samples, seq_len, num_features = X.shape
    input_scaler = pickle.load(scaler_paths["CNN-LSTM"]["input"])
    X_scaled = input_scaler.transform(X.reshape(-1, num_features)).reshape(num_samples, seq_len, num_features)

    # Standardize targets
    target_scaler = pickle.load(scaler_paths["CNN-LSTM"]["target"])
    y_scaled = target_scaler.transform(y)

    # Split data
    _, X_test, _, y_test = train_test_split(
        X_scaled, y_scaled, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    if mode == "Historical Analysis":
        st.title("Water Quality Prediction - Historical Analysis")
        
        # Parameter category selection
        category = st.radio("Select Parameter Category:", 
                           ["Water Quality Parameters", "Meteorology Parameters", "Volcanic Activity Parameters"])
        
        if category == "Water Quality Parameters":
            parameters = feature_categories['water_quality']
            param_type = 'target'
        elif category == "Meteorology Parameters":
            parameters = feature_categories['meteorology']
            param_type = 'input'
        else:
            parameters = feature_categories['volcanic']
            param_type = 'input'
        
        # Model comparison
        st.header("Model Performance Comparison")
        selected_parameter = st.selectbox("Select Parameter to Compare:", parameters)
        
        # Evaluate models
        all_metrics = {}
        all_predictions = {}
        
        for model_name, model_path in model_paths.items():
            try:
                model = load_model(model_path)
                metrics, y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, target_scaler, target_features)
                all_metrics[model_name] = metrics
                all_predictions[model_name] = (y_test_actual, y_pred_actual)
                st.success(f"✅ {model_name} model loaded successfully")
            except Exception as e:
                st.error(f"❌ Error loading {model_name} model: {str(e)}")
        
        # Display metrics
        if param_type == 'target':
            metrics_data = []
            for model_name, metrics in all_metrics.items():
                if selected_parameter in metrics:
                    metric = metrics[selected_parameter]
                    metrics_data.append({
                        'Model': model_name,
                        'MSE': f"{metric['MSE']:.4f}",
                        'RMSE': f"{metric['RMSE']:.4f}",
                        'MAE': f"{metric['MAE']:.4f}",
                        'R²': f"{metric['R2']:.4f}"
                    })
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                # Highlight minimum for error metrics, maximum for R²
                def highlight_metrics(s):
                    styles = [''] * len(s)
                    # Highlight minimum for error metrics
                    for col in ['MSE', 'RMSE', 'MAE']:
                        if col in metrics_df.columns:
                            min_val = metrics_df[col].astype(float).min()
                            for i, v in enumerate(metrics_df[col].astype(float)):
                                if v == min_val and s.name == col:
                                    styles[i] = 'background-color: lightgreen'
                    # Highlight maximum for R² (closest to 1)
                    if 'R²' in metrics_df.columns:
                        max_val = metrics_df['R²'].astype(float).max()
                        for i, v in enumerate(metrics_df['R²'].astype(float)):
                            if v == max_val and s.name == 'R²':
                                styles[i] = 'background-color: lightgreen'
                    return styles

                st.dataframe(metrics_df.style.apply(highlight_metrics, axis=0))
        else:
            st.info("Performance metrics are only available for target (water quality) parameters")
        
        # Trendline comparison
        if all_predictions and param_type == 'target' and selected_parameter in target_features:
            param_index = target_features.index(selected_parameter)
            st.subheader("Trendline Comparison")
            fig = make_subplots(rows=1, cols=1)
            
            y_test_actual = next(iter(all_predictions.values()))[0]
            fig.add_trace(go.Scatter(
                x=np.arange(len(y_test_actual)),
                y=y_test_actual[:, param_index],
                name='True Values',
                line=dict(color='black', width=3)
            ))
            
            colors = ['red', 'blue', 'green']
            for i, (model_name, (_, y_pred_actual)) in enumerate(all_predictions.items()):
                fig.add_trace(go.Scatter(
                    x=np.arange(len(y_pred_actual)),
                    y=y_pred_actual[:, param_index],
                    name=model_name,
                    line=dict(color=colors[i], width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=f"{selected_parameter} - Model Comparison",
                xaxis_title="Time Steps",
                yaxis_title=selected_parameter,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Live parameter meters
        if all_predictions and param_type == 'target':
            st.header("Current Parameter Status")
            y_test_actual = next(iter(all_predictions.values()))[0]
            last_values = y_test_actual[-1]
            
            cols = st.columns(len(parameters))
            for i, param in enumerate(parameters):
                if param in target_features:
                    param_idx = target_features.index(param)
                    with cols[i]:
                        min_val = np.min(y_test_actual[:, param_idx])
                        max_val = np.max(y_test_actual[:, param_idx])
                        st.plotly_chart(create_parameter_meter(
                            last_values[param_idx], 
                            min_val, 
                            max_val, 
                            param
                        ))
    else:  # Real-time Prediction
        st.title("Water Quality Prediction - Real-time Forecast")

        # Model selection
        selected_model = st.sidebar.selectbox("Select Prediction Model:", list(model_paths.keys()))

        try:
            model = load_model(model_paths[selected_model])
            target_scaler = pickle.load(scaler_paths[selected_model]["target"])
            input_scaler = pickle.load(scaler_paths[selected_model]["input"])
            st.sidebar.success(f"{selected_model} model loaded successfully")
        except Exception as e:
            st.error(f"Error loading model or scaler: {str(e)}")
            return

        # Forecast period selection
        time_period = st.sidebar.selectbox(
            "Select Forecast Period:",
            ["Next 24 Hours", "Next 7 Days", "Next 30 Days", "Next 12 Months"]
        )

        # Determine forecast steps
        steps = {
            "Next 24 Hours": 24,
            "Next 7 Days": 7,
            "Next 30 Days": 30,
            "Next 12 Months": 12
        }[time_period]

        # Use the last available input sequence
        last_sequence_raw = X[-1]  # Original unscaled sequence
        last_sequence_scaled = input_scaler.transform(last_sequence_raw)

        # Forecast generation function
        def generate_real_time_forecast(model, input_data, steps, target_scaler):
            predictions = []
            current_input = input_data.copy()

            for _ in range(steps):
                pred = model.predict(current_input[np.newaxis, ...])
                predictions.append(pred[0])
                current_input = np.roll(current_input, -1, axis=0)
                current_input[-1, :] = current_input[-2, :]  # retain previous

            return target_scaler.inverse_transform(np.array(predictions))

        # Forecast
        forecast = generate_real_time_forecast(model, last_sequence_scaled, steps, target_scaler)

        # Display current values
        st.header("Current Water Quality Status")
        y_test_actual = target_scaler.inverse_transform(y_test)
        min_vals = [np.min(y_test_actual[:, i]) for i in range(len(target_features))]
        max_vals = [np.max(y_test_actual[:, i]) for i in range(len(target_features))]
        current_values = {target_features[i]: y_test_actual[-1, i] for i in range(len(target_features))}

        cols = st.columns(3)
        for i, (param, value) in enumerate(current_values.items()):
            with cols[i % 3]:
                st.plotly_chart(create_parameter_meter(
                    value,
                    min_vals[i],
                    max_vals[i],
                    param
                ), use_container_width=True)

        # Display forecast plot
        st.header(f"{time_period} Forecast")
        time_labels = [f"Step {i+1}" for i in range(steps)]
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=target_features,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        for i, feature in enumerate(target_features):
            row = (i // 3) + 1
            col = (i % 3) + 1
            fig.add_trace(go.Scatter(x=time_labels, y=forecast[:, i], mode='lines+markers', name=feature), row=row, col=col)
            fig.update_yaxes(title_text=feature, row=row, col=col)

        fig.update_layout(height=600, title_text="Forecasted Water Quality Parameters", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Alerts
        st.header("Alerts")
        alert_thresholds = {
            'temperature': (10, 30),
            'ph': (6.5, 8.5),
            'ammonia': (0, 0.5),
            'nitrate': (0, 10),
            'pho': (0, 0.1),
            'dissolved_oxygen': (5, 12)
        }

        alert_count = 0
        for i, param in enumerate(target_features):
            if param in alert_thresholds:
                min_thresh, max_thresh = alert_thresholds[param]
                if np.any(forecast[:, i] < min_thresh) or np.any(forecast[:, i] > max_thresh):
                    alert_count += 1
                    st.warning(f"⚠️ {param} forecasted to go outside safe range ({min_thresh} - {max_thresh})")

        if alert_count == 0:
            st.success("✅ No forecasted parameter violations detected")


if __name__ == "__main__":
    main()
