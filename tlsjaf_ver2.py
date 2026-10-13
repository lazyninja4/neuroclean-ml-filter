import json
import os
import urllib.request

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt, resample
from scipy.stats import entropy, kurtosis
from sklearn.decomposition import FastICA
from sklearn.ensemble import RandomForestClassifier

DATA_URL = "https://raw.githubusercontent.com/ncclabsustech/EEGdenoiseNet/master/data/"
DATA_FILES = ["EEG_all_epochs.npy", "EOG_all_epochs.npy", "EMG_all_epochs.npy"]
MODEL_PATH = "model/artifact_rf.joblib"
METRICS_PATH = "model/metrics.json"
MODEL_FS = 256

st.set_page_config(page_title="NeuroClean: Bio-Signal Visualizer & ML Filter", layout="wide")
st.title("NeuroClean")
st.write("A minimalistic bio-signal visualizer and ML filter for cleaning up EEG/ECG sensor-produced neural data.")


def get_features(window):
    window = np.asarray(window, dtype=float)
    if len(window) != MODEL_FS:
        window = resample(window, MODEL_FS)

    d1 = np.diff(window)
    d2 = np.diff(d1)
    v0, v1, v2 = np.var(window), np.var(d1), np.var(d2)

    mobility = np.sqrt(v1 / (v0 + 1e-12))
    complexity = np.sqrt(v2 / (v1 + 1e-12)) / (mobility + 1e-12)

    power = np.abs(rfft(window)) ** 2
    spec_entropy = entropy(power / (power.sum() + 1e-12) + 1e-12)

    spike_ratio = np.ptp(window) / (np.std(window) + 1e-12)

    return [mobility, complexity, spec_entropy, spike_ratio]


def apply_dsp(sig, fs, use_notch, low, high):
    out = np.array(sig, dtype=float)
    if use_notch and fs / 2 > 60:
        b, a = iirnotch(60.0, Q=30.0, fs=fs)
        out = filtfilt(b, a, out)
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, out)


def load_dataset():
    os.makedirs("data", exist_ok=True)
    arrays = []
    for name in DATA_FILES:
        path = os.path.join("data", name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(DATA_URL + name, path)
        arrays.append(np.load(path))
    return arrays


def split_idx(n):
    idx = np.random.default_rng(42).permutation(n)
    return idx[: int(n * 0.8)], idx[int(n * 0.8):]


def rms(x):
    return np.sqrt(np.mean(x ** 2, axis=-1))


def build_windows(eeg, eog, emg, n, rng):
    X, y = [], []
    while len(y) < n:
        e = eeg[rng.integers(len(eeg))]
        start = rng.integers(64, 193)
        sl = slice(start, start + MODEL_FS)

        if rng.random() < 0.5:
            X.append(get_features(e[sl]))
            y.append(0)
            continue
        pool = eog if rng.random() < 0.5 else emg
        a = pool[rng.integers(len(pool))]
        snr_db = rng.uniform(-7, 2)
        lam = rms(e) / (rms(a) * 10 ** (snr_db / 20))

        if rms(lam * a[sl]) < 0.3 * rms(e[sl]):
            continue

        X.append(get_features((e + lam * a)[sl]))
        y.append(1)
    return np.array(X), np.array(y)


@st.cache_resource(show_spinner="Training artifact detector on real EEG data (first run only)...")
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    try:
        eeg, eog, emg = load_dataset()
    except Exception:
        return None

    sos = butter(4, [1.0, 30.0], btype="bandpass", fs=MODEL_FS, output="sos")
    eeg, eog, emg = [sosfiltfilt(sos, d, axis=1) for d in (eeg, eog, emg)]

    tr = [split_idx(len(d))[0] for d in (eeg, eog, emg)]
    te = [split_idx(len(d))[1] for d in (eeg, eog, emg)]
    rng = np.random.default_rng(42)
    X_tr, y_tr = build_windows(eeg[tr[0]], eog[tr[1]], emg[tr[2]], 6000, rng)
    X_te, y_te = build_windows(eeg[te[0]], eog[te[1]], emg[te[2]], 2000, rng)

    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    os.makedirs("model", exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"test_accuracy": float((clf.predict(X_te) == y_te).mean())}, f)
    return clf


st.sidebar.header("1. Signal Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])
fs = int(st.sidebar.number_input("Sampling Frequency (Hz)", min_value=10, value=500, step=50))

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    channels = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if channels.shape[1] == 0:
        st.error("No numeric columns found in CSV!")
        st.stop()
    channels = channels.interpolate(limit_direction="both") 
    source_name = uploaded_file.name
else:
    st.info("Using synthetic multi-channel EEG with 60Hz noise and blinks.")
    rng = np.random.default_rng(1)
    t = np.arange(4 * fs) / fs

    alpha = 1.2 * np.sin(2 * np.pi * 10 * t)
    hum = 0.5 * np.sin(2 * np.pi * 60 * t)
    blinks = 6.0 * np.exp(-((t - 0.95) ** 2) / 0.005) + 7.0 * np.exp(-((t - 2.65) ** 2) / 0.005)

    ch1 = alpha + 0.8 * blinks + hum + rng.normal(0, 0.15, len(t))
    ch2 = 0.3 * alpha + 1.5 * blinks + hum + rng.normal(0, 0.15, len(t))
    channels = pd.DataFrame({"Ch1": ch1, "Ch2": ch2})
    source_name = "synthetic"

primary_col = st.sidebar.selectbox("Select Target Analysis Channel", list(channels.columns))
raw_signal = channels[primary_col].to_numpy(dtype=float)

if len(raw_signal) <= fs:
    st.error("Signal is shorter than 1 second, check the sampling frequency")
    st.stop()

st.sidebar.header("2. DSP Filtering")
enable_notch = st.sidebar.checkbox("Apply 60 Hz Notch Filter", value=True)

max_hz = fs / 2 - 0.5
cutoff_low, cutoff_high = st.sidebar.slider(
    "Bandpass Range (Hz)",
    min_value=0.5,
    max_value=float(max_hz),
    value=(1.0, min(30.0, float(max_hz))),
    step=0.5,
)
if cutoff_low >= cutoff_high:
    st.error("Low cutoff must be smaller than the high cutoff")
    st.stop()

st.sidebar.header("3. Advanced AI Tools")
run_ml = st.sidebar.checkbox("Highlight ML Artifacts", value=True)
ica_cols = st.sidebar.multiselect("Channels for ICA", list(channels.columns), default=list(channels.columns))
run_ica = st.sidebar.button("Run FastICA Cleaning Engine")

filtered_signal = apply_dsp(raw_signal, fs, enable_notch, cutoff_low, cutoff_high)

artifact_regions = []
ml_model = load_model() if run_ml else None

if run_ml and ml_model is None:
    st.sidebar.warning("Couldn't load or train the model (no internet to download the dataset?).")
elif ml_model is not None and os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        st.sidebar.caption(f"Model accuracy on held-out real EEG: {json.load(f)['test_accuracy']:.0%}")

if ml_model is not None:
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
        artifact_regions.append(
            (max(0, peak_idx - half_window), min(len(filtered_signal), peak_idx + half_window))
        )

ica_key = (source_name, primary_col, tuple(ica_cols), fs, enable_notch, cutoff_low, cutoff_high)

if run_ica:
    if len(ica_cols) < 2:
        st.sidebar.error("ICA requires at least 2 distinct dataset channels!")
    elif primary_col not in ica_cols:
        st.sidebar.error("Target channel must be in the ICA channels!")
    else:
        ica_input = np.column_stack(
            [apply_dsp(channels[c], fs, enable_notch, cutoff_low, cutoff_high) for c in ica_cols]
        )
        ica = FastICA(n_components=min(ica_input.shape[1], 4), whiten="unit-variance",
                      random_state=42, max_iter=1000)
        components = ica.fit_transform(ica_input)

        artifact_comp_idx = int(np.argmax(kurtosis(components, axis=0)))

        cleaned_components = components.copy()
        cleaned_components[:, artifact_comp_idx] = 0
        reconstructed = ica.inverse_transform(cleaned_components)

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

st.sidebar.header("4. Export Options")
time_axis = np.arange(len(raw_signal)) / fs
export_df = pd.DataFrame(
    {"Time_s": time_axis, "Raw_Signal": raw_signal, "DSP_Filtered": filtered_signal}
)
if ica_cleaned_signal is not None:
    export_df["ICA_Cleaned"] = ica_cleaned_signal

st.sidebar.download_button(
    label="📥 Download Processed CSV",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="processed_biosignal.csv",
    mime="text/csv",
)
st.sidebar.write("Project developed by Niranjan Subbiyah L.")

tab1, tab2, tab3 = st.tabs(["Time Domain & ML Overlay", "FFT Spectrum", "ICA Analysis"])

with tab1:
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=time_axis, y=raw_signal, name="Raw Signal", opacity=0.3))
    fig_time.add_trace(
        go.Scatter(x=time_axis, y=filtered_signal, name="DSP Filtered Signal", line=dict(color="red"))
    )

    if ica_cleaned_signal is not None:
        fig_time.add_trace(
            go.Scatter(x=time_axis, y=ica_cleaned_signal, name="ICA Cleaned Signal",
                       line=dict(color="green", width=2))
        )

    for start, end in artifact_regions:
        fig_time.add_vrect(x0=start / fs, x1=end / fs, fillcolor="rgba(255, 0, 0, 0.25)",
                           layer="below", line_width=0)

    fig_time.update_layout(
        title="Time Series Data & Highlighted Artifacts",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
    )
    st.plotly_chart(fig_time, width="stretch")

    if ml_model is None:
        st.info("ML highlighting is off.")
    elif len(artifact_regions) > 0:
        st.error(f"ML Engine Flagged {len(artifact_regions)} High-Variance Artifact Window(s)!")
    else:
        st.success("✅ Signal clean. No high-amplitude artifacts detected.")

with tab2:
    freqs = rfftfreq(len(raw_signal), 1 / fs)

    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=freqs, y=np.abs(rfft(raw_signal)), name="Raw", opacity=0.3))
    fig_fft.add_trace(
        go.Scatter(x=freqs, y=np.abs(rfft(filtered_signal)), name="Filtered", line=dict(color="red"))
    )
    fig_fft.update_layout(
        title="Frequency Domain Analysis (FFT)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude",
    )
    st.plotly_chart(fig_fft, width="stretch")

with tab3:
    if ica_result is not None:
        st.markdown("### Independent Component Analysis (ICA)")
        st.write(
            "FastICA un-mixes multi-channel recordings into independent components. "
            "The spikiest one (highest kurtosis) is zeroed out and the channels are rebuilt without it."
        )

        fig_ica = go.Figure()
        for k in range(ica_result["components"].shape[1]):
            name = f"ICA Component {k+1}"
            if k == ica_result["artifact_idx"]:
                name += " (removed)"
            fig_ica.add_trace(go.Scatter(x=time_axis, y=ica_result["components"][:, k], name=name))

        fig_ica.update_layout(
            title="Separated Source Components",
            xaxis_title="Time (s)",
            yaxis_title="Arbitrary Units",
        )
        st.plotly_chart(fig_ica, width="stretch")
    else:
        st.info("Click **'Run FastICA Cleaning Engine'** in the sidebar to perform blind source separation.")
