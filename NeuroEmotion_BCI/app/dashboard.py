import os
import sys
from datetime import datetime

import numpy as np
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.extract_features import extract_all_bands


st.set_page_config(
    page_title="NeuroEmotion | EEG Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --background: #080d18;
        --surface: #101827;
        --surface-raised: #151f31;
        --border: rgba(148, 163, 184, .14);
        --muted: #91a0b8;
        --text: #eef4ff;
        --mint: #5eead4;
        --blue: #7aa7ff;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 78% 0%, rgba(39, 92, 168, .18), transparent 36%),
            radial-gradient(ellipse at 5% 20%, rgba(20, 184, 166, .09), transparent 28%),
            var(--background);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: rgba(12, 19, 32, .96);
        border-right: 1px solid var(--border);
    }
    .block-container { padding-top: 2.1rem; padding-bottom: 2.5rem; max-width: 1440px; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }
    .hero {
        padding: 1.7rem 1.9rem;
        border: 1px solid rgba(94, 234, 212, .2);
        border-radius: 22px;
        background: linear-gradient(115deg, rgba(18, 38, 55, .96), rgba(16, 24, 39, .88) 60%, rgba(27, 40, 67, .9));
        box-shadow: 0 18px 55px rgba(0, 0, 0, .2);
        margin-bottom: 1.25rem;
    }
    .eyebrow { color: var(--mint); font-size: .76rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .hero-title { font: 700 2.25rem 'Space Grotesk', sans-serif; margin: .35rem 0 .25rem; color: var(--text); }
    .hero-copy { color: #aab8cc; font-size: .98rem; margin: 0; max-width: 780px; }
    .section-heading { margin: 1.35rem 0 .75rem; font: 600 1.1rem 'Space Grotesk', sans-serif; color: #e6edfa; }
    .panel {
        border: 1px solid var(--border);
        border-radius: 17px;
        background: linear-gradient(150deg, rgba(21, 31, 49, .92), rgba(14, 22, 36, .92));
        padding: 1.15rem 1.25rem;
        min-height: 110px;
    }
    .panel-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .09em; font-weight: 700; }
    .panel-value { color: var(--text); font: 600 1.65rem 'Space Grotesk', sans-serif; margin-top: .45rem; }
    .panel-foot { color: #7788a3; font-size: .78rem; margin-top: .25rem; }
    .emotion-panel { min-height: 168px; }
    .emotion-name { font: 700 1.8rem 'Space Grotesk', sans-serif; margin-top: .45rem; }
    .emotion-relaxed { color: var(--mint); }
    .emotion-focused { color: #a8bfff; }
    .small-note { color: #8493aa; font-size: .82rem; line-height: 1.55; }
    .learning-copy { color: #aab8cc; font-size: .9rem; line-height: 1.75; }
    .learning-step {
        border-left: 2px solid rgba(94, 234, 212, .6);
        padding: .15rem 0 .15rem .9rem;
        margin: .7rem 0;
    }
    div[data-testid="stMetric"] {
        background: rgba(17, 27, 43, .88);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 1rem 1.1rem;
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stMetricValue"] { color: var(--text); font-family: 'Space Grotesk', sans-serif; }
    .stButton > button {
        border: 1px solid rgba(94, 234, 212, .4);
        border-radius: 11px;
        background: linear-gradient(100deg, #14b8a6, #168ba5);
        color: #041116;
        font-weight: 700;
        min-height: 2.7rem;
    }
    .stButton > button:hover { border-color: var(--mint); color: white; }
    hr { border-color: var(--border); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Neurotechnology · Signal intelligence</div>
      <div class="hero-title">NeuroEmotion EEG Lab</div>
      <p class="hero-copy">Explore a live simulated EEG signal, inspect its frequency-band power, and see a simple alpha-versus-beta state estimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🧠 NeuroEmotion")
    st.caption("SIMULATED EEG WORKSPACE")
    st.markdown("---")
    st.markdown("### Simulation controls")
    st.checkbox("Run live simulation", value=False, key="live_mode")
    scenario = st.selectbox(
        "Signal profile",
        ["Random", "Relaxed (alpha)", "Focused (beta)", "Learning practice"],
        help="Learning practice compares an illustrative beginner-to-expert signal model.",
    )
    learning_stage = st.select_slider(
        "Practice experience",
        options=["Beginner", "Practicing", "Expert"],
        value="Beginner",
        help="In this synthetic demo, later stages have a steadier target pattern. This is not measured brain data.",
    )
    learning_goal = st.selectbox("Practice target", ["Focus (beta)", "Calm (alpha)"])
    sampling_rate = st.select_slider(
        "Sampling rate",
        options=[128, 256, 512],
        value=256,
        format_func=lambda value: f"{value} Hz",
    )
    duration = st.slider("Signal window", min_value=1, max_value=4, value=2, format="%d sec")
    st.markdown("---")
    st.markdown("### Band guide")
    st.markdown(
        """
        <div class="small-note">
        <b style="color:#a78bfa">Theta</b> · 4–8 Hz<br>
        <b style="color:#5eead4">Alpha</b> · 8–12 Hz<br>
        <b style="color:#7aa7ff">Beta</b> · 12–30 Hz<br>
        <b style="color:#fbbf78">Gamma</b> · 30–45 Hz
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Demo only — simulated signals are not medical measurements or a diagnosis.")


def generate_signal(sf, seconds, profile, learning_stage="Beginner", learning_goal="Focus (beta)"):
    sample_count = int(sf * seconds)
    time_axis = np.arange(sample_count) / sf
    if profile == "Learning practice":
        state = "focused" if learning_goal.startswith("Focus") else "relaxed"
    elif profile == "Random":
        state = np.random.choice(["relaxed", "focused"])
    else:
        state = "relaxed" if profile.startswith("Relaxed") else "focused"

    if profile == "Learning practice":
        stage_parameters = {
            "Beginner": (0.9, 0.55, 0.8),
            "Practicing": (1.15, 0.36, 0.55),
            "Expert": (1.35, 0.2, 0.32),
        }
        target_amplitude, background_amplitude, noise_level = stage_parameters[learning_stage]
        if state == "relaxed":
            alpha_amplitude, beta_amplitude = target_amplitude, background_amplitude
        else:
            alpha_amplitude, beta_amplitude = background_amplitude, target_amplitude
    else:
        alpha_amplitude, beta_amplitude = (1.35, 0.2) if state == "relaxed" else (0.2, 1.35)
        noise_level = 0.42

    signal = (
        alpha_amplitude * np.sin(2 * np.pi * 10 * time_axis)
        + beta_amplitude * np.sin(2 * np.pi * 20 * time_axis)
        + noise_level * np.random.normal(size=sample_count)
    )
    return time_axis, signal


if "eeg_history" not in st.session_state:
    st.session_state.eeg_history = []


@st.fragment(run_every="1s" if st.session_state.get("live_mode", False) else None)
def render_dashboard():
    create_sample = st.button("Generate EEG sample", type="primary", use_container_width=False)
    selected_settings = (scenario, learning_stage, learning_goal, sampling_rate, duration)
    settings_changed = (
        "current_signal" in st.session_state
        and st.session_state.current_signal.get("settings") != selected_settings
    )
    if (
        create_sample
        or st.session_state.get("live_mode", False)
        or "current_signal" not in st.session_state
        or settings_changed
    ):
        time_axis, signal = generate_signal(
            sampling_rate,
            duration,
            scenario,
            learning_stage,
            learning_goal,
        )
        features = extract_all_bands(signal, sampling_rate)
        emotion = "Relaxed" if features["Alpha"] > features["Beta"] else "Focused"
        sample = {
            "signal": signal,
            "time_axis": time_axis,
            "features": features,
            "emotion": emotion,
            "learning_stage": learning_stage,
            "learning_goal": learning_goal,
            "settings": selected_settings,
            "created_at": datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.current_signal = sample
        st.session_state.eeg_history.append(
            {
                "sample": len(st.session_state.eeg_history) + 1,
                "Alpha": features["Alpha"],
                "Beta": features["Beta"],
                "Theta": features["Theta"],
                "Gamma": features["Gamma"],
            }
        )
        st.session_state.eeg_history = st.session_state.eeg_history[-30:]

    current = st.session_state.current_signal
    features = current["features"]
    emotion = current["emotion"]
    total_power = sum(features.values())
    dominance = (
        100 * max(features["Alpha"], features["Beta"]) / total_power
        if total_power > 0
        else 0
    )

    status_col, time_col, rate_col = st.columns([2, 1, 1])
    with status_col:
        mode_label = "LIVE STREAMING" if st.session_state.get("live_mode", False) else "READY · SAMPLE MODE"
        st.markdown(
            f'<div class="panel"><div class="panel-label">System status</div>'
            f'<div class="panel-value" style="font-size:1.25rem;color:#5eead4">● {mode_label}</div>'
            '<div class="panel-foot">Synthetic EEG · single virtual channel</div></div>',
            unsafe_allow_html=True,
        )
    with time_col:
        st.markdown(
            f'<div class="panel"><div class="panel-label">Latest sample</div>'
            f'<div class="panel-value" style="font-size:1.35rem">{current["created_at"]}</div>'
            '<div class="panel-foot">Local time</div></div>',
            unsafe_allow_html=True,
        )
    with rate_col:
        st.markdown(
            f'<div class="panel"><div class="panel-label">Sampling rate</div>'
            f'<div class="panel-value" style="font-size:1.35rem">{sampling_rate} Hz</div>'
            f'<div class="panel-foot">{duration} second window</div></div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.75, 1], gap="large")
    with left:
        st.markdown('<div class="section-heading">Live signal · Virtual channel Fp1</div>', unsafe_allow_html=True)
        chart_data = {"EEG amplitude": current["signal"]}
        st.line_chart(chart_data, height=310, color="#5eead4")
        st.caption("Horizontal axis shows sample index; waveform is generated locally for demonstration.")
        st.markdown('<div class="section-heading">Frequency-band profile</div>', unsafe_allow_html=True)
        st.bar_chart({"Power (relative)": features}, height=250, color="#7aa7ff")

    with right:
        st.markdown('<div class="section-heading">Brainwave power</div>', unsafe_allow_html=True)
        metric_cols = st.columns(2)
        band_ranges = {"Theta": "4–8", "Alpha": "8–12", "Beta": "12–30", "Gamma": "30–45"}
        for column, band in zip(
            metric_cols * 2,
            ["Theta", "Alpha", "Beta", "Gamma"],
        ):
            with column:
                st.metric(
                    f"{band} · {band_ranges[band]} Hz",
                    f"{features[band]:.4f}",
                )
        emotion_class = "emotion-relaxed" if emotion == "Relaxed" else "emotion-focused"
        st.markdown(
            f'<div class="panel emotion-panel" style="margin-top:1rem">'
            '<div class="panel-label">Alpha / beta heuristic</div>'
            f'<div class="emotion-name {emotion_class}">{emotion}</div>'
            f'<div class="panel-foot">Dominant-band share: {dominance:.0f}% · Not a clinical inference</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-heading">Recent power trend</div>', unsafe_allow_html=True)
        st.line_chart(
            {
                "Alpha": [row["Alpha"] for row in st.session_state.eeg_history],
                "Beta": [row["Beta"] for row in st.session_state.eeg_history],
            },
            height=190,
            color=["#5eead4", "#7aa7ff"],
        )


render_dashboard()

st.markdown('<div class="section-heading">🧠 শেখা, EEG সিগন্যাল ও BCI কীভাবে কাজ করে</div>', unsafe_allow_html=True)
learning_left, learning_right = st.columns([1.1, 1], gap="large")
with learning_left:
    st.markdown(
        """
        <div class="panel learning-copy">
          <b style="color:#5eead4">মস্তিষ্ক EEG সিগন্যাল তৈরি করে কীভাবে?</b>
          <div class="learning-step">নিউরনের বৈদ্যুতিক কার্যকলাপ থেকে ক্ষুদ্র ভোল্টেজ পরিবর্তন তৈরি হয়। বহু নিউরনের সমন্বিত কার্যকলাপ মাথার ত্বকে ইলেকট্রোড দিয়ে EEG হিসেবে রেকর্ড করা যায়।</div>
          <div class="learning-step">সিগন্যালকে Theta, Alpha, Beta ও Gamma-র মতো ফ্রিকোয়েন্সি ব্যান্ডে বিশ্লেষণ করা যায়। ব্যান্ড-পাওয়ার মস্তিষ্কের কার্যকলাপের একটি পরিমাপ—এটি একা কোনো নির্দিষ্ট আবেগ বা দক্ষতার প্রমাণ নয়।</div>
          <div class="learning-step">অনুশীলনে কাজ-সম্পর্কিত স্নায়ু-নেটওয়ার্ক ও সিগন্যালের ধরন বদলাতে পারে, কিন্তু পরিবর্তন ব্যক্তি ও কাজভেদে আলাদা। “Expert” হওয়ার জন্য সবার একই EEG প্যাটার্ন হয় না।</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with learning_right:
    st.markdown(
        """
        <div class="panel learning-copy">
          <b style="color:#7aa7ff">BCI-তে শেখা ও দক্ষতা কীভাবে কাজ করে?</b>
          <div class="learning-step"><b>১ · Calibration:</b> ব্যবহারকারী নির্দিষ্ট কাজ/কল্পনা করেন; সিস্টেম সেই সময়ের EEG নমুনা সংগ্রহ করে।</div>
          <div class="learning-step"><b>২ · Feature + model:</b> সিগন্যাল পরিষ্কার করে বৈশিষ্ট্য বের করে, লেবেল করা নমুনা দিয়ে মডেলকে আলাদা প্যাটার্ন শেখানো হয়।</div>
          <div class="learning-step"><b>৩ · Feedback + practice:</b> মডেল অনুমান দেখায়; ব্যবহারকারী ও সিস্টেম প্রতিক্রিয়ার ভিত্তিতে অনুশীলন/ক্যালিব্রেশন উন্নত করেন। অভিজ্ঞতায় নিয়ন্ত্রণ আরও ধারাবাহিক হতে পারে—তা স্বয়ংক্রিয়ভাবে ঘটে না।</div>
          <div class="small-note">এখানে Beginner → Expert তুলনা কেবল শিক্ষামূলক synthetic উদাহরণ: অনুশীলনের স্তর বাড়লে লক্ষ্য-ফ্রিকোয়েন্সি স্পষ্ট এবং noise কম দেখানো হয়। বাস্তব EEG, শেখা AI model, বা চিকিৎসা-ফল নয়।</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if scenario == "Learning practice":
    st.markdown('<div class="section-heading">Beginner → Practicing → Expert · Synthetic comparison</div>', unsafe_allow_html=True)
    target_band = "Beta" if learning_goal.startswith("Focus") else "Alpha"
    stage_rows = []
    for stage in ["Beginner", "Practicing", "Expert"]:
        _, stage_signal = generate_signal(
            sampling_rate,
            duration,
            "Learning practice",
            stage,
            learning_goal,
        )
        stage_features = extract_all_bands(stage_signal, sampling_rate)
        stage_rows.append(
            {
                "Practice stage": stage,
                "Target band power": stage_features[target_band],
                "Other band power": stage_features["Alpha" if target_band == "Beta" else "Beta"],
            }
        )
    st.line_chart(
        {
            "Target band power": [row["Target band power"] for row in stage_rows],
            "Other band power": [row["Other band power"] for row in stage_rows],
        },
        height=240,
        color=["#5eead4", "#7aa7ff"],
    )
    st.caption(
        f"Target: {target_band} · {learning_goal}. Each stage is randomly generated for illustration; "
        "the values are not a real learning curve or scientific measurement."
    )

st.markdown("---")
st.markdown(
    '<p class="small-note">This educational dashboard uses synthetic waveforms and a basic alpha/beta comparison. '
    'It is not a validated emotion-recognition model and should not be used for diagnosis or treatment decisions.</p>',
    unsafe_allow_html=True,
)
