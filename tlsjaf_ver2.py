import sys
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import entropy
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

st.set_page_config(
    page_title="NeuroClean: Bio-Signal Visualizer & ML Filter", layout="wide"
)
st.title("NeuroClean")
st.write("A minimalistic bio-signal visualizer and ML filter for cleaning up EEG/ECG sensor-produced neural data.")

# feature extraction (Hjorth & Spectral Entropy)



def extract_window_features(window, fs):
    """Extracts scale-independent temporal and spectral features from a signal window."""
    # 1. Hjorth Parameters
    d1 = np.diff(window)
    var_zero = np.var(window)
    var_d1 = np.var(d1)

    hjorth_activity = var_zero
    hjorth_mobility = (
        np.sqrt(var_d1 / var_zero) if var_zero > 0 else 0.0
    )

    # 2. Spectral Entropy
    fft_vals = np.abs(fft(window))[: len(window) // 2]
    psd = fft_vals**2
    psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    spec_entropy = entropy(psd_norm + 1e-12)

    # 3. Normalized Peak-to-Peak Amplitude
    p2p = np.ptp(window)

    return [hjorth_activity, hjorth_mobility, spec_entropy, p2p]


# synthetic ML model training

@st.cache_resource
def train_artifact_detector():
    """Trains a classifier using Hjorth parameters and Spectral Entropy."""
    np.random.seed(42)
    fs = 500
    n_samples = 200

    X = []
    y = []

    t = np.linspace(0, 1, fs)

    # Generate synthetic clean windows (Alpha wave + low noise)
    for _ in range(n_samples):
        clean_sig = (
            1.2 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.1, fs)
        )
        feats = extract_window_features(clean_sig, fs)
        X.append(feats)
        y.append(0)  # Clean

    # Generate synthetic artifact windows (High-amplitude ocular/muscle spikes)
    for _ in range(n_samples):
        center = np.random.uniform(0.2, 0.8)
        artifact_sig = (
            1.2 * np.sin(2 * np.pi * 10 * t)
            + 8.0 * np.exp(-((t - center) ** 2) / 0.005)
            + np.random.normal(0, 0.5, fs)
        )
        feats = extract_window_features(artifact_sig, fs)
        X.append(feats)
        y.append(1)  # Artifact

    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(np.array(X), np.array(y))
    return clf


ml_model = train_artifact_detector()

# sidebar controls & data
st.sidebar.header("1. Signal Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])
fs = st.sidebar.number_input("Sampling Frequency (Hz)", value=500, step=50)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        st.error("No numeric columns found in CSV!")
        st.stop()

    primary_col = st.sidebar.selectbox(
        "Select Target Analysis Channel", numeric_cols
    )
    raw_signal = df[primary_col].values

    if len(numeric_cols) > 1:
        ica_input_matrix = df[numeric_cols].values
    else:
        ica_input_matrix = None
else:
    st.info("Using synthetic multi-channel EEG with 60Hz noise and blinks.")
    t = np.linspace(0, 4, 4 * fs)

    alpha = 1.2 * np.sin(2 * np.pi * 10 * t)
    blinks = np.zeros_like(t)
    blinks[int(0.8 * fs) : int(1.1 * fs)] += 6.0 * np.exp(
        -((t[int(0.8 * fs) : int(1.1 * fs)] - 0.95) ** 2) / 0.005
    )
    blinks[int(2.5 * fs) : int(2.8 * fs)] += 7.0 * np.exp(
        -((t[int(2.5 * fs) : int(2.8 * fs)] - 2.65) ** 2) / 0.005
    )

    hum = 0.5 * np.sin(2 * np.pi * 60 * t)

    ch1 = alpha + 0.8 * blinks + hum + np.random.normal(0, 0.15, len(t))
    ch2 = 0.3 * alpha + 1.5 * blinks + hum + np.random.normal(0, 0.15, len(t))

    raw_signal = ch1
    ica_input_matrix = np.column_stack([ch1, ch2])

st.sidebar.header("2. DSP Filtering")
enable_notch = st.sidebar.checkbox("Apply 60 Hz Notch Filter", value=True)

cutoff_low, cutoff_high = st.sidebar.slider(
    "Bandpass Range (Hz)",
    min_value=0.5,
    max_value=float(fs / 2)-0.1,
    value=(1.0, 30.0),
    step=0.5,
)

st.sidebar.header("3. Advanced AI Tools")
run_ml = st.sidebar.checkbox("Highlight ML Artifacts", value=True)
run_ica = st.sidebar.button("Run FastICA Cleaning Engine")


# DSP stuff
processed_signal = raw_signal.copy()

if enable_notch and (fs / 2 > 60.0):
    b_notch, a_notch = iirnotch(w0=60.0, Q=30.0, fs=fs)
    processed_signal = filtfilt(b_notch, a_notch, processed_signal)

b_band, a_band = butter(
    N=4, Wn=[cutoff_low, cutoff_high], btype="bandpass", fs=fs
)
filtered_signal = filtfilt(b_band, a_band, processed_signal)

# ML artifact detection
window_size = fs
step_size = fs // 2  # 50% overlap for smooth detection
num_steps = (len(filtered_signal) - window_size) // step_size + 1
raw_regions = []

for i in range(num_steps):
    start_idx = i * step_size
    end_idx = start_idx + window_size
    window = filtered_signal[start_idx:end_idx]

    feats = extract_window_features(window, fs)
    pred = ml_model.predict(np.array([feats]))[0]

    if pred == 1:
        raw_regions.append((start_idx, end_idx))
        
# merging overlapping regions
merged_regions = []
if raw_regions:
    curr_start, curr_end = raw_regions[0]
    for start, end in raw_regions[1:]:
        if start <= curr_end:  # Overlap found
            curr_end = max(curr_end, end)
        else:
            merged_regions.append((curr_start, curr_end))
            curr_start, curr_end = start, end
    merged_regions.append((curr_start, curr_end))

# Step 3: Re-center each merged region directly over its peak amplitude
artifact_regions = []
half_window = int(0.3 * fs)  # Tight 0.6s width centered on spike peak

for start, end in merged_regions:
    region_signal = np.abs(filtered_signal[start:end])
    if len(region_signal) > 0:
        peak_idx = start + np.argmax(region_signal)
        centered_start = max(0, peak_idx - half_window)
        centered_end = min(len(filtered_signal), peak_idx + half_window)
        artifact_regions.append((centered_start, centered_end))

# fastICA decomposition & re-mixing

ica_cleaned_signal = None
components = None

if run_ica:
    if ica_input_matrix is None or ica_input_matrix.shape[1] < 2:
        st.sidebar.error(
            "ICA requires at least 2 distinct dataset channels!"
        )
    else:
        n_comps = min(ica_input_matrix.shape[1], 4)
        ica = FastICA(n_components=n_comps, random_state=42, whiten="unit-variance")

        # S = X * W^T
        components = ica.fit_transform(ica_input_matrix)

        # Identify high-variance artifact component
        comp_variances = [np.var(components[:, k]) for k in range(n_comps)]
        artifact_comp_idx = np.argmax(comp_variances)

        # Zero out artifact component
        cleaned_components = components.copy()
        cleaned_components[:, artifact_comp_idx] = 0

        # Matrix reconstruction: X_reconstructed = S_cleaned * A^T
        reconstructed_matrix = np.dot(
            cleaned_components, ica.mixing_.T
        ) + np.mean(ica_input_matrix, axis=0)

        # Target primary channel reconstruction
        target_col_idx = 0
        if (
            uploaded_file is not None
            and primary_col in numeric_cols
        ):
            target_col_idx = numeric_cols.index(primary_col)

        ica_cleaned_signal = reconstructed_matrix[:, target_col_idx]

# csv data exporting
st.sidebar.header("4. Export Options")
export_df = pd.DataFrame(
    {"Raw_Signal": raw_signal, "DSP_Filtered": filtered_signal}
)

if ica_cleaned_signal is not None:
    export_df["ICA_Cleaned"] = ica_cleaned_signal

csv_data = export_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="📥 Download Processed CSV",
    data=csv_data,
    file_name="processed_biosignal.csv",
    mime="text/csv",
)
st.sidebar.write()
st.sidebar.write("Project developed by Niranjan Subbiyah L.")

# UI layout
tab1, tab2, tab3 = st.tabs(
    ["Time Domain & ML Overlay", "FFT Spectrum", "ICA Analysis"]
)

with tab1:
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(y=raw_signal, name="Raw Signal", opacity=0.3))
    fig_time.add_trace(
        go.Scatter(
            y=filtered_signal,
            name="DSP Filtered Signal",
            line=dict(color="red"),
        )
    )

    if ica_cleaned_signal is not None:
        fig_time.add_trace(
            go.Scatter(
                y=ica_cleaned_signal,
                name="ICA Cleaned Signal (Zeroed Artifact)",
                line=dict(color="green", width=2),
            )
        )

    # Clean, non-overlapping single-layer red regions centered over peaks
    if run_ml and len(artifact_regions) > 0:
        for start, end in artifact_regions:
            fig_time.add_vrect(
                x0=start,
                x1=end,
                fillcolor="rgba(255, 0, 0, 0.25)",
                layer="below",
                line_width=0,
            )

    fig_time.update_layout(
        title="Time Series Data & Highlighted Artifacts",
        xaxis_title="Sample Index",
        yaxis_title="Amplitude (µV)",
    )
    st.plotly_chart(fig_time, use_container_width=True)

    if len(artifact_regions) > 0:
        st.error(
            f"⚠️ ML Engine Flagged {len(artifact_regions)} High-Variance Artifact Window(s)!"
        )
    else:
        st.success("✅ Signal clean. No high-amplitude artifacts detected.")
with tab2:
    N = len(raw_signal)
    fft_raw = np.abs(fft(raw_signal))[: N // 2]
    fft_filtered = np.abs(fft(filtered_signal))[: N // 2]
    freqs = fftfreq(N, 1 / fs)[: N // 2]

    fig_fft = go.Figure()
    fig_fft.add_trace(
        go.Scatter(x=freqs, y=fft_raw, name="Raw Power", opacity=0.3)
    )
    fig_fft.add_trace(
        go.Scatter(
            x=freqs,
            y=fft_filtered,
            name="Filtered Power",
            line=dict(color="red"),
        )
    )
    fig_fft.update_layout(
        title="Frequency Domain Analysis (FFT)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power Magnitude",
    )
    st.plotly_chart(fig_fft, use_container_width=True)

with tab3:
    if run_ica and components is not None:
        st.markdown("### Independent Component Analysis (ICA)")
        st.write(
            "FastICA un-mixes multi-channel recordings into independent components. "
            "The component with maximum variance is zeroed out and reconstructed via matrix multiplication."
        )

        fig_ica = go.Figure()
        for k in range(components.shape[1]):
            fig_ica.add_trace(
                go.Scatter(
                    y=components[:, k], name=f"ICA Component {k+1}"
                )
            )

        fig_ica.update_layout(
            title="Separated Source Components",
            xaxis_title="Sample Index",
            yaxis_title="Arbitrary Units",
        )
        st.plotly_chart(fig_ica, use_container_width=True)
    else:
        st.info(
            "Click **'Run FastICA Cleaning Engine'** in the sidebar to perform blind source separation."
        )
