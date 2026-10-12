import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt
from scipy.stats import entropy, kurtosis
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="NeuroClean", layout="wide")
st.title("NeuroClean")
st.write("Bio-signal visualizer and ML filter.")

def get_features(window):
    # extract hjorth params and spectral entropy to feed the artifact classifier
    d1 = np.diff(window)
    d2 = np.diff(d1)
    v0, v1, v2 = np.var(window), np.var(d1), np.var(d2)
    
    mobility = np.sqrt(v1 / (v0 + 1e-12))
    complexity = np.sqrt(v2 / (v1 + 1e-12)) / (mobility + 1e-12)
    
    power = np.abs(rfft(window)) ** 2
    spec_entropy = entropy(power / (power.sum() + 1e-12) + 1e-12)
    spike_ratio = np.ptp(window) / (np.std(window) + 1e-12)
    
    return [mobility, complexity, spec_entropy, spike_ratio]

@st.cache_resource
def train_artifact_detector(fs):
    rng = np.random.default_rng(42)
    t = np.arange(fs) / fs
    
    X, y = [], []
    
    # generate some dummy training data with random blinks
    for _ in range(300):
        sig_clean = rng.uniform(0.8, 1.6) * np.sin(2 * np.pi * rng.uniform(8, 12) * t + rng.uniform(0, 6.28))
        sig_clean += rng.normal(0, rng.uniform(0.05, 0.3), fs)
        
        X.append(get_features(sig_clean))
        y.append(0)
        
        sig_artifact = sig_clean.copy()
        bump = rng.uniform(3, 9) * rng.choice([-1, 1])
        sig_artifact += bump * np.exp(-((t - rng.uniform(0.2, 0.8)) ** 2) / 0.005)
        
        X.append(get_features(sig_artifact))
        y.append(1)
        
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(np.array(X), np.array(y))
    return clf

def apply_dsp(sig, fs, use_notch, low, high):
    out = np.array(sig, dtype=float)
    if use_notch and fs / 2 > 60:
        b, a = iirnotch(60.0, Q=30.0, fs=fs)
        out = filtfilt(b, a, out)
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, out)

st.sidebar.header("1. Signal Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])
fs = int(st.sidebar.number_input("Sampling Frequency (Hz)", min_value=10, value=500, step=50))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    channels = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if channels.empty:
        st.error("No numeric columns found.")
        st.stop()
    channels = channels.interpolate(limit_direction="both")
    source_name = uploaded_file.name
else:
    # fallback synthetic data if nothing uploaded yet
    rng = np.random.default_rng(1)
    t = np.arange(4 * fs) / fs
    alpha = 1.2 * np.sin(2 * np.pi * 10 * t)
    hum = 0.5 * np.sin(2 * np.pi * 60 * t)
    blinks = 6.0 * np.exp(-((t - 0.95) ** 2) / 0.005) + 7.0 * np.exp(-((t - 2.65) ** 2) / 0.005)
    
    ch1 = alpha + 0.8 * blinks + hum + rng.normal(0, 0.15, len(t))
    ch2 = 0.3 * alpha + 1.5 * blinks + hum + rng.normal(0, 0.15, len(t))
    channels = pd.DataFrame({"Ch1": ch1, "Ch2": ch2})
    source_name = "synthetic"

primary_col = st.sidebar.selectbox("Target Channel", list(channels.columns))
raw_signal = channels[primary_col].to_numpy(dtype=float)

if len(raw_signal) <= fs:
    st.error("Signal too short for this sampling rate.")
    st.stop()

st.sidebar.header("2. DSP Filtering")
enable_notch = st.sidebar.checkbox("60 Hz Notch", value=True)
max_hz = fs / 2 - 0.5
cutoff_low, cutoff_high = st.sidebar.slider(
    "Bandpass (Hz)", 
    min_value=0.5, 
    max_value=float(max_hz), 
    value=(1.0, min(30.0, float(max_hz))), 
    step=0.5
)

if cutoff_low >= cutoff_high:
    st.error("Low cutoff must be less than high cutoff.")
    st.stop()

st.sidebar.header("3. AI Tools")
run_ml = st.sidebar.checkbox("Highlight Artifacts (RF)", value=True)
ica_cols = st.sidebar.multiselect("ICA Channels", list(channels.columns), default=list(channels.columns))
run_ica = st.sidebar.button("Run FastICA")

filtered_signal = apply_dsp(raw_signal, fs, enable_notch, cutoff_low, cutoff_high)

artifact_regions = []
if run_ml:
    ml_model = train_artifact_detector(fs)
    step_size = fs // 2
    starts = range(0, len(filtered_signal) - fs + 1, step_size)
    feats = [get_features(filtered_signal[s:s + fs]) for s in starts]
    preds = ml_model.predict(np.array(feats))
    
    raw_regions = [(s, s + fs) for s, p in zip(starts, preds) if p == 1]
    
    merged_regions = []
    for start, end in raw_regions:
        if merged_regions and start <= merged_regions[-1][1]:
            merged_regions[-1] = (merged_regions[-1][0], end)
        else:
            merged_regions.append((start, end))
            
    half_window = int(0.3 * fs)
    for start, end in merged_regions:
        peak_idx = start + np.argmax(np.abs(filtered_signal[start:end]))
        artifact_regions.append((max(0, peak_idx - half_window), min(len(filtered_signal), peak_idx + half_window)))

ica_key = (source_name, primary_col, tuple(ica_cols), fs, enable_notch, cutoff_low, cutoff_high)

if run_ica:
    if len(ica_cols) < 2:
        st.sidebar.error("ICA needs at least 2 channels.")
    elif primary_col not in ica_cols:
        st.sidebar.error("Target channel must be in ICA selection.")
    else:
        ica_input = np.column_stack([apply_dsp(channels[c], fs, enable_notch, cutoff_low, cutoff_high) for c in ica_cols])
        ica = FastICA(n_components=min(ica_input.shape[1], 4), whiten="unit-variance", random_state=42, max_iter=1000)
        components = ica.fit_transform(ica_input)
        
        artifact_comp_idx = int(np.argmax(kurtosis(components, axis=0)))
        cleaned_comps = components.copy()
        cleaned_comps[:, artifact_comp_idx] = 0
        reconstructed = ica.inverse_transform(cleaned_comps)
        
        st.session_state["ica"] = {
            "key": ica_key,
            "components": components,
            "artifact_idx": artifact_comp_idx,
            "cleaned": reconstructed[:, ica_cols.index(primary_col)],
        }

ica_result = st.session_state.get("ica")
if ica_result is not None and ica_result["key"] != ica_key:
    ica_result = None

ica_cleaned_signal = ica_result["cleaned"] if ica_result else None

st.sidebar.header("4. Export")
time_axis = np.arange(len(raw_signal)) / fs
export_df = pd.DataFrame({
    "Time_s": time_axis, 
    "Raw": raw_signal, 
    "Filtered": filtered_signal
})

if ica_cleaned_signal is not None:
    export_df["ICA_Cleaned"] = ica_cleaned_signal

csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button("Download CSV", data=csv_bytes, file_name="filtered_output.csv", mime="text/csv")

tab1, tab2, tab3 = st.tabs(["Time Series", "FFT Spectrum", "ICA Components"])

with tab1:
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=time_axis, y=raw_signal, name="Raw", opacity=0.3))
    fig_time.add_trace(go.Scatter(x=time_axis, y=filtered_signal, name="Filtered", line=dict(color="red")))
    
    if ica_cleaned_signal is not None:
        fig_time.add_trace(go.Scatter(x=time_axis, y=ica_cleaned_signal, name="ICA Cleaned", line=dict(color="green", width=2)))
        
    for start, end in artifact_regions:
        fig_time.add_vrect(x0=start / fs, x1=end / fs, fillcolor="rgba(255, 0, 0, 0.25)", layer="below", line_width=0)
        
    fig_time.update_layout(title="Signal Timeline", xaxis_title="Time (s)", yaxis_title="Amplitude")
    st.plotly_chart(fig_time, use_container_width=True)
    
    if run_ml and len(artifact_regions) > 0:
        st.warning(f"Detected {len(artifact_regions)} artifact intervals.")
    elif run_ml:
        st.success("Signal looks clean.")

with tab2:
    freqs = rfftfreq(len(raw_signal), 1 / fs)
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=freqs, y=np.abs(rfft(raw_signal)), name="Raw", opacity=0.3))
    fig_fft.add_trace(go.Scatter(x=freqs, y=np.abs(rfft(filtered_signal)), name="Filtered", line=dict(color="red")))
    fig_fft.update_layout(title="Power Spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude")
    st.plotly_chart(fig_fft, use_container_width=True)

with tab3:
    if ica_result is not None:
        fig_ica = go.Figure()
        for k in range(ica_result["components"].shape[1]):
            comp_name = f"Component {k+1}"
            if k == ica_result["artifact_idx"]:
                comp_name += " (Dropped)"
            fig_ica.add_trace(go.Scatter(x=time_axis, y=ica_result["components"][:, k], name=comp_name))
            
        fig_ica.update_layout(title="Independent Components", xaxis_title="Time (s)", yaxis_title="Value")
        st.plotly_chart(fig_ica, use_container_width=True)
    else:
        st.info("Run FastICA from the sidebar to view components.")
