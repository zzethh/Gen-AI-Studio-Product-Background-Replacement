import streamlit as st
import requests
from PIL import Image
import io
import os
import pandas as pd

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="GenAI Studio — Product Background Replacement",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS: Dark AI Studio Theme ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global Reset ── */
html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: #0B0F19 !important;
}
.block-container { 
    padding-top: 1.5rem !important; 
    padding-bottom: 2rem !important; 
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stStatusWidget"] { visibility: hidden; }

/* ── Disable native sidebar ── */
[data-testid="collapsedControl"] { display: none !important; }
button[kind="header"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Top bar ── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 0;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.logo-icon {
    width: 48px; height: 48px;
    background: linear-gradient(135deg, #1D9E75, #0F6E56);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; color: white; font-weight: 700;
}
.logo-text {
    font-size: 32px; font-weight: 600; color: #F1F5F9;
    letter-spacing: -0.8px;
}
.logo-text span { color: #1D9E75; }
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; color: #94A3B8;
    background: rgba(255,255,255,0.04);
    padding: 6px 14px; border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #1D9E75;
    box-shadow: 0 0 8px rgba(29,158,117,0.5);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Section Labels ── */
.section-label {
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #64748B; margin-bottom: 8px; margin-top: 16px;
}

/* ── Model Cards ── */
.model-card {
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
    margin-bottom: 6px;
    transition: all 0.2s ease;
    cursor: pointer;
}
.model-card:hover {
    border-color: rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.05);
}
.model-card.active-teal {
    border: 1.5px solid #1D9E75;
    background: rgba(29,158,117,0.08);
}
.model-card.active-amber {
    border: 1.5px solid #F59E0B;
    background: rgba(245,158,11,0.08);
}
.model-card.active-red {
    border: 1.5px solid #EF4444;
    background: rgba(239,68,68,0.08);
}
.mc-label {
    font-size: 13px; font-weight: 500; color: #F1F5F9;
}
.mc-badge {
    font-size: 10px; padding: 2px 8px; border-radius: 20px;
    font-weight: 500;
}
.badge-teal { background: rgba(29,158,117,0.2); color: #34D399; }
.badge-amber { background: rgba(245,158,11,0.2); color: #FBBF24; }
.badge-red { background: rgba(239,68,68,0.2); color: #F87171; }
.mc-desc {
    font-size: 11px; color: #64748B; margin-top: 3px; line-height: 1.4;
}

/* ── Panel Headers ── */
.panel-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 16px;
    background: rgba(255,255,255,0.02);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px 12px 0 0;
}
.panel-title {
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.05em; text-transform: uppercase;
    color: #64748B;
}
.panel-tag {
    font-size: 10px; padding: 3px 10px;
    border-radius: 20px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #94A3B8;
}

/* ── Content Panels ── */
.glass-panel {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    overflow: hidden;
}
.img-container {
    padding: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 320px;
    background: rgba(0,0,0,0.15);
}
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; gap: 8px;
    color: #475569; font-size: 13px;
}
.empty-icon {
    font-size: 36px; opacity: 0.4;
}

/* ── Metrics Bar ── */
.metrics-bar {
    display: flex; align-items: center; gap: 24px;
    padding: 14px 20px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    margin-top: 12px;
}
.metric-item {
    display: flex; flex-direction: column; gap: 4px; min-width: 100px;
}
.metric-label-text {
    font-size: 10px; font-weight: 500;
    letter-spacing: 0.06em; text-transform: uppercase;
    color: #64748B;
}
.metric-value {
    font-size: 18px; font-weight: 600; color: #F1F5F9;
}
.metric-bar-track {
    width: 100%; height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px; overflow: hidden;
}
.metric-bar-fill {
    height: 100%; border-radius: 2px;
    transition: width 0.5s ease;
}
.fill-teal { background: #1D9E75; }
.fill-amber { background: #F59E0B; }
.fill-red { background: #EF4444; }
.fill-blue { background: #3B82F6; }

/* ── Comparison Cards ── */
.compare-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    overflow: hidden;
}
.compare-header {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 12px; font-weight: 500;
    color: #94A3B8;
    display: flex; justify-content: space-between;
    align-items: center;
}

/* ── Generate Button Enhancement ── */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1D9E75, #0F6E56) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 20px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    letter-spacing: -0.2px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(29,158,117,0.25) !important;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 25px rgba(29,158,117,0.4) !important;
    transform: translateY(-1px) !important;
}
div.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #94A3B8 !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
div.stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.2) !important;
}

/* ── Streamlit Metric Override ── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.06);
}
div[data-testid="stMetric"] label {
    color: #64748B !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #F1F5F9 !important;
    font-size: 22px !important;
    font-weight: 600 !important;
}

/* ── Expander Style ── */
div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
    background: rgba(255,255,255,0.02) !important;
}

/* ── Selectbox / Slider Overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    color: #94A3B8 !important;
    font-size: 12px !important;
}

/* ── File Uploader ── */
section[data-testid="stFileUploader"] {
    border: 1px dashed rgba(29,158,117,0.3) !important;
    border-radius: 12px !important;
    background: rgba(29,158,117,0.04) !important;
}
section[data-testid="stFileUploader"]:hover {
    border-color: #1D9E75 !important;
    background: rgba(29,158,117,0.08) !important;
}

/* ── Image styling ── */
img {
    border-radius: 8px !important;
}

/* ── Checkbox ── */
div[data-testid="stCheckbox"] label span {
    color: #94A3B8 !important;
    font-size: 12px !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.06) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Top Bar ──
st.markdown("""
<div class="top-bar">
    <div class="logo">
        <div class="logo-icon">✦</div>
        <span class="logo-text">Gen<span>AI</span> Studio</span>
    </div>
    <div class="status-pill">
        <div class="status-dot"></div>
        GPU Ready · FP32 · GTX 1080 Ti
    </div>
</div>
""", unsafe_allow_html=True)

st.caption("AI-Powered Product Background Replacement · Stable Diffusion + LoRA + MLOps Pipeline")


# ══════════════════════════════════════════
#              MAIN LAYOUT GRID
# ══════════════════════════════════════════
grid_sidebar, grid_main = st.columns([1, 2.5], gap="large")

with grid_sidebar:
    # ── 1. Image Upload ──
    st.markdown('<div class="section-label" style="margin-top:0;">Product Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop product photo here",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed"
    )
    
    show_mask = False
    if uploaded_file is not None:
        show_mask = st.checkbox("🔍 Preview rembg mask", key="mask_toggle")

    # ── 2. Background Prompt ──
    st.markdown('<div class="section-label">Background Prompt</div>', unsafe_allow_html=True)
    prompt_presets = [
        "A professional photo sitting on a mossy rock in a forest, cinematic lighting, 4k",
        "A sunny beach with palm trees and turquoise ocean, golden hour light",
        "A clean white studio with soft box lighting, hyperrealistic product photo",
        "Floating in deep space with colorful nebulas and distant stars, sci-fi",
        "Custom prompt..."
    ]
    selected_preset = st.selectbox("Quick Prompts", prompt_presets, label_visibility="collapsed")
    if selected_preset == "Custom prompt...":
        prompt = st.text_input("Describe the background:", placeholder="e.g., On a marble table...", label_visibility="collapsed")
    else:
        prompt = selected_preset

    # ── 3. Model Mode ──
    st.markdown('<div class="section-label">Model Mode</div>', unsafe_allow_html=True)
    model_options = {
        "Baseline (No LoRA)": {
            "key": "baseline", "badge_class": "badge-teal", "card_class": "active-teal",
            "badge_text": "Best quality", "desc": "Full SD Inpainting · no fine-tuning"
        },
        "Light LoRA (350 Steps)": {
            "key": "light", "badge_class": "badge-amber", "card_class": "active-amber",
            "badge_text": "350 steps", "desc": "Subtle domain adaptation"
        },
        "Overfit LoRA (44K Steps)": {
            "key": "overfit", "badge_class": "badge-red", "card_class": "active-red",
            "badge_text": "44k steps", "desc": "Catastrophic forgetting demo"
        }
    }
    model_choice = st.radio("Select model", list(model_options.keys()), index=0, label_visibility="collapsed")
    selected = model_options[model_choice]
    model_mode = selected["key"]

    st.markdown(f"""
    <div class="model-card {selected['card_class']}">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:3px;">
            <span class="mc-label">{model_choice.split('(')[0].strip()}</span>
            <span class="mc-badge {selected['badge_class']}">{selected['badge_text']}</span>
        </div>
        <div class="mc-desc">{selected['desc']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── 4. Inference Parameters ──
    st.markdown('<div class="section-label">Inference Parameters</div>', unsafe_allow_html=True)
    num_steps = st.slider("Denoising Steps", 10, 100, 50, key="steps")
    guidance = st.slider("Guidance Scale", 1.0, 20.0, 9.0, step=0.5, key="guidance")
    strength = st.slider("Inpainting Strength", 0.5, 1.0, 1.0, step=0.05, key="strength")

    st.markdown("<br/>", unsafe_allow_html=True)
    gen_btn = st.button("✦ Generate Background", type="primary", use_container_width=True)
    compare_btn = st.button("⚖️ Compare All 3", type="secondary", use_container_width=True)

    # Admin Panel (Bottom of sidebar)
    st.markdown("<br/>", unsafe_allow_html=True)
    with st.expander("👑 Admin Dashboard"):
        st.caption("Real-time telemetry")
        log_file = "logs/inference_log.csv"
        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file)
                if not df.empty:
                    m1, m2 = st.columns(2)
                    m1.metric("Runs", len(df))
                    m2.metric("Latency", f"{df['latency_sec'].mean():.1f}s")
                    st.metric("Avg CLIP", f"{df['clip_score'].mean():.1f}")
            except Exception:
                pass


with grid_main:
    # ── OUTPUT WORKSPACE ──
    col_orig, col_gen = st.columns([1, 1], gap="medium")
    
    with col_orig:
        st.markdown("""
        <div class="panel-header">
            <span class="panel-title">Original Input</span>
            <span class="panel-tag">product image</span>
        </div>
        """, unsafe_allow_html=True)
        if uploaded_file is not None:
            if show_mask:
                with st.spinner("Extracting mask..."):
                    try:
                        uploaded_file.seek(0)
                        mask_resp = requests.post(
                            f"{BACKEND_URL}/extract_mask",
                            files={"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                            timeout=30
                        )
                        if mask_resp.status_code == 200:
                            st.image(Image.open(io.BytesIO(mask_resp.content)), use_container_width=True)
                    except Exception as e:
                        st.image(Image.open(uploaded_file), use_container_width=True)
            else:
                st.image(Image.open(uploaded_file), use_container_width=True)
        else:
            st.markdown("""
            <div class="img-container">
                <div class="empty-state">
                    <div class="empty-icon">📷</div>
                    <div>Upload a product image to begin</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_gen:
        st.markdown(f"""
        <div class="panel-header">
            <span class="panel-title">Generated Output</span>
            <span class="panel-tag">{model_mode}</span>
        </div>
        """, unsafe_allow_html=True)
        output_placeholder = st.empty()
        
        if uploaded_file is None:
            output_placeholder.markdown("""
            <div class="img-container">
                <div class="empty-state">
                    <div class="empty-icon">✨</div>
                    <div>AI-generated result will appear here</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    metrics_placeholder = st.empty()


# ══════════════════════════════════════════
#           SINGLE GENERATION
# ══════════════════════════════════════════
if gen_btn:
    if uploaded_file is None:
        st.warning("⚠️ Upload a product image first.")
    elif not prompt.strip():
        st.warning("⚠️ Enter a background prompt.")
    else:
        with st.spinner("✦ Generating new background..."):
            try:
                uploaded_file.seek(0)
                response = requests.post(
                    f"{BACKEND_URL}/generate",
                    files={"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={
                        "prompt": prompt,
                        "num_inference_steps": num_steps,
                        "guidance_scale": guidance,
                        "strength": strength,
                        "model_mode": model_mode,
                    },
                    timeout=180,
                )
                
                if response.status_code == 200:
                    output_image = Image.open(io.BytesIO(response.content))
                    latency = response.headers.get("X-Inference-Latency", "—")
                    clip_score = response.headers.get("X-CLIP-Score", "—")
                    
                    with col_gen:
                        output_placeholder.image(output_image, use_container_width=True)
                        
                        with metrics_placeholder.container():
                            # Determine color based on mode
                            if model_mode == "baseline":
                                fill_class, color = "fill-teal", "#1D9E75"
                            elif model_mode == "light":
                                fill_class, color = "fill-amber", "#F59E0B"
                            else:
                                fill_class, color = "fill-red", "#EF4444"
                            
                            try:
                                clip_val = float(clip_score)
                                clip_pct = min(clip_val / 35 * 100, 100)
                            except:
                                clip_pct = 0
                            
                            st.markdown(f"""
                            <div class="metrics-bar">
                                <div class="metric-item">
                                    <span class="metric-label-text">CLIP Score</span>
                                    <span class="metric-value" style="color:{color}">{clip_score}</span>
                                    <div class="metric-bar-track">
                                        <div class="metric-bar-fill {fill_class}" style="width:{clip_pct}%"></div>
                                    </div>
                                </div>
                                <div class="metric-item">
                                    <span class="metric-label-text">Latency</span>
                                    <span class="metric-value">{latency}s</span>
                                </div>
                                <div class="metric-item">
                                    <span class="metric-label-text">Mode</span>
                                    <span class="metric-value" style="font-size:14px;">{model_mode}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.success("✅ Generation complete!")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend is offline. Start the API first.")
            except Exception as e:
                st.error(f"❌ {e}")


# ══════════════════════════════════════════
#         3-WAY A/B/C COMPARISON
# ══════════════════════════════════════════
if compare_btn:
    if uploaded_file is None:
        st.warning("⚠️ Upload a product image first.")
    elif not prompt.strip():
        st.warning("⚠️ Enter a background prompt.")
    else:
        with col_gen:
            output_placeholder.empty()
            metrics_placeholder.empty()
            
            with output_placeholder.container():
                st.markdown("""
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <span style="font-size:16px;font-weight:600;color:#F1F5F9;">⚖️ 3-Way Model Comparison</span>
                </div>
                """, unsafe_allow_html=True)
                
                # We split the generated panel into 3 micro-columns
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                modes_config = [
                    ("baseline", comp_col1, "Base", "badge-teal", "fill-teal", "#1D9E75", "No LoRA"),
                    ("light", comp_col2, "Light", "badge-amber", "fill-amber", "#F59E0B", "350s"),
                    ("overfit", comp_col3, "Overfit", "badge-red", "fill-red", "#EF4444", "44K"),
                ]
                
                for mode, col, label, badge_cls, fill_cls, color, badge_text in modes_config:
                    with col:
                        # Compact header for small columns
                        st.markdown(f"""
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                            <span style="font-size:11px; color:#F1F5F9; font-weight:500;">{label}</span>
                            <span class="mc-badge {badge_cls}" style="font-size:8px; padding:2px 4px;">{badge_text}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.spinner(f"..."):
                            try:
                                uploaded_file.seek(0)
                                response = requests.post(
                                    f"{BACKEND_URL}/generate",
                                    files={"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                                    data={
                                        "prompt": prompt,
                                        "num_inference_steps": num_steps,
                                        "guidance_scale": guidance,
                                        "strength": strength,
                                        "model_mode": mode,
                                    },
                                    timeout=180,
                                )
                                
                                if response.status_code == 200:
                                    out_img = Image.open(io.BytesIO(response.content))
                                    latency = response.headers.get("X-Inference-Latency", "—")
                                    clip_score = response.headers.get("X-CLIP-Score", "—")
                                    
                                    st.image(out_img, use_container_width=True)
                                    
                                    try:
                                        clip_val = float(clip_score)
                                        clip_pct = min(clip_val / 35 * 100, 100)
                                    except:
                                        clip_pct = 0
                                    
                                    st.markdown(f"""
                                    <div style="padding:4px 0;">
                                        <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
                                            <span style="font-size:10px; color:#64748B;">CLIP</span>
                                            <span style="font-size:11px;font-weight:600;color:{color};">{clip_score}</span>
                                        </div>
                                        <div class="metric-bar-track" style="height:3px;">
                                            <div class="metric-bar-fill {fill_cls}" style="width:{clip_pct}%"></div>
                                        </div>
                                        <div style="display:flex;justify-content:space-between;margin-top:4px;">
                                            <span style="font-size:10px; color:#64748B;">Time</span>
                                            <span style="font-size:10px;color:#94A3B8;">{latency}s</span>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.error("Error")
                            except Exception as e:
                                st.error("Fail")

