
import streamlit as st
import requests
from PIL import Image
import io
import os
import pandas as pd
from src.tracking_gcp import GCPObservability
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://localhost:8000')
st.set_page_config(page_title='GenAI Studio — Product Background Replacement', page_icon='✦', layout='wide', initial_sidebar_state='collapsed')
st.markdown('\n<style>\n@import url(\'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap\');\n\n/* ── Global Reset ── */\nhtml, body, .stApp {\n    font-family: \'Inter\', sans-serif !important;\n}\n.stApp {\n    background: #0B0F19 !important;\n}\n.block-container { \n    padding-top: 1.5rem !important; \n    padding-bottom: 2rem !important; \n}\n\n/* ── Hide default Streamlit chrome ── */\n#MainMenu, footer, header { visibility: hidden; }\ndiv[data-testid="stStatusWidget"] { visibility: hidden; }\n\n/* ── Disable native sidebar ── */\n[data-testid="collapsedControl"] { display: none !important; }\nbutton[kind="header"] { display: none !important; }\nsection[data-testid="stSidebar"] { display: none !important; }\n\n/* ── Top bar ── */\n.top-bar {\n    display: flex;\n    align-items: center;\n    justify-content: space-between;\n    padding: 14px 0;\n    margin-bottom: 8px;\n    border-bottom: 1px solid rgba(255,255,255,0.06);\n}\n.logo {\n    display: flex;\n    align-items: center;\n    gap: 10px;\n}\n.logo-icon {\n    width: 48px; height: 48px;\n    background: linear-gradient(135deg, #1D9E75, #0F6E56);\n    border-radius: 12px;\n    display: flex; align-items: center; justify-content: center;\n    font-size: 24px; color: white; font-weight: 700;\n}\n.logo-text {\n    font-size: 32px; font-weight: 600; color: #F1F5F9;\n    letter-spacing: -0.8px;\n}\n.logo-text span { color: #1D9E75; }\n.status-pill {\n    display: inline-flex; align-items: center; gap: 6px;\n    font-size: 12px; color: #94A3B8;\n    background: rgba(255,255,255,0.04);\n    padding: 6px 14px; border-radius: 20px;\n    border: 1px solid rgba(255,255,255,0.08);\n}\n.status-dot {\n    width: 7px; height: 7px; border-radius: 50%;\n    background: #1D9E75;\n    box-shadow: 0 0 8px rgba(29,158,117,0.5);\n    animation: pulse 2s infinite;\n}\n@keyframes pulse {\n    0%, 100% { opacity: 1; }\n    50% { opacity: 0.4; }\n}\n\n/* ── Section Labels ── */\n.section-label {\n    font-size: 10px; font-weight: 600;\n    letter-spacing: 0.1em; text-transform: uppercase;\n    color: #64748B; margin-bottom: 8px; margin-top: 16px;\n}\n\n/* ── Model Cards ── */\n.model-card {\n    padding: 12px 14px;\n    border-radius: 10px;\n    border: 1px solid rgba(255,255,255,0.08);\n    background: rgba(255,255,255,0.03);\n    margin-bottom: 6px;\n    transition: all 0.2s ease;\n    cursor: pointer;\n}\n.model-card:hover {\n    border-color: rgba(255,255,255,0.15);\n    background: rgba(255,255,255,0.05);\n}\n.model-card.active-teal {\n    border: 1.5px solid #1D9E75;\n    background: rgba(29,158,117,0.08);\n}\n.model-card.active-amber {\n    border: 1.5px solid #F59E0B;\n    background: rgba(245,158,11,0.08);\n}\n.model-card.active-red {\n    border: 1.5px solid #EF4444;\n    background: rgba(239,68,68,0.08);\n}\n.mc-label {\n    font-size: 13px; font-weight: 500; color: #F1F5F9;\n}\n.mc-badge {\n    font-size: 10px; padding: 2px 8px; border-radius: 20px;\n    font-weight: 500;\n}\n.badge-teal { background: rgba(29,158,117,0.2); color: #34D399; }\n.badge-amber { background: rgba(245,158,11,0.2); color: #FBBF24; }\n.badge-red { background: rgba(239,68,68,0.2); color: #F87171; }\n.mc-desc {\n    font-size: 11px; color: #64748B; margin-top: 3px; line-height: 1.4;\n}\n\n/* ── Panel Headers ── */\n.panel-header {\n    display: flex; align-items: center; justify-content: space-between;\n    padding: 10px 16px;\n    background: rgba(255,255,255,0.02);\n    border-bottom: 1px solid rgba(255,255,255,0.06);\n    border-radius: 12px 12px 0 0;\n}\n.panel-title {\n    font-size: 11px; font-weight: 600;\n    letter-spacing: 0.05em; text-transform: uppercase;\n    color: #64748B;\n}\n.panel-tag {\n    font-size: 10px; padding: 3px 10px;\n    border-radius: 20px;\n    background: rgba(255,255,255,0.04);\n    border: 1px solid rgba(255,255,255,0.08);\n    color: #94A3B8;\n}\n\n/* ── Content Panels ── */\n.glass-panel {\n    background: rgba(255,255,255,0.02);\n    border: 1px solid rgba(255,255,255,0.06);\n    border-radius: 12px;\n    overflow: hidden;\n}\n.img-container {\n    padding: 20px;\n    display: flex;\n    align-items: center;\n    justify-content: center;\n    min-height: 320px;\n    background: rgba(0,0,0,0.15);\n}\n.empty-state {\n    display: flex; flex-direction: column;\n    align-items: center; gap: 8px;\n    color: #475569; font-size: 13px;\n}\n.empty-icon {\n    font-size: 36px; opacity: 0.4;\n}\n\n/* ── Metrics Bar ── */\n.metrics-bar {\n    display: flex; align-items: center; gap: 24px;\n    padding: 14px 20px;\n    background: rgba(255,255,255,0.02);\n    border: 1px solid rgba(255,255,255,0.06);\n    border-radius: 12px;\n    margin-top: 12px;\n}\n.metric-item {\n    display: flex; flex-direction: column; gap: 4px; min-width: 100px;\n}\n.metric-label-text {\n    font-size: 10px; font-weight: 500;\n    letter-spacing: 0.06em; text-transform: uppercase;\n    color: #64748B;\n}\n.metric-value {\n    font-size: 18px; font-weight: 600; color: #F1F5F9;\n}\n.metric-bar-track {\n    width: 100%; height: 4px;\n    background: rgba(255,255,255,0.06);\n    border-radius: 2px; overflow: hidden;\n}\n.metric-bar-fill {\n    height: 100%; border-radius: 2px;\n    transition: width 0.5s ease;\n}\n.fill-teal { background: #1D9E75; }\n.fill-amber { background: #F59E0B; }\n.fill-red { background: #EF4444; }\n.fill-blue { background: #3B82F6; }\n\n/* ── Comparison Cards ── */\n.compare-card {\n    background: rgba(255,255,255,0.02);\n    border: 1px solid rgba(255,255,255,0.06);\n    border-radius: 12px;\n    overflow: hidden;\n}\n.compare-header {\n    padding: 10px 14px;\n    border-bottom: 1px solid rgba(255,255,255,0.06);\n    font-size: 12px; font-weight: 500;\n    color: #94A3B8;\n    display: flex; justify-content: space-between;\n    align-items: center;\n}\n\n/* ── Generate Button Enhancement ── */\ndiv.stButton > button[kind="primary"] {\n    background: linear-gradient(135deg, #1D9E75, #0F6E56) !important;\n    border: none !important;\n    border-radius: 10px !important;\n    padding: 12px 20px !important;\n    font-weight: 500 !important;\n    font-size: 14px !important;\n    letter-spacing: -0.2px !important;\n    transition: all 0.2s ease !important;\n    box-shadow: 0 4px 15px rgba(29,158,117,0.25) !important;\n}\ndiv.stButton > button[kind="primary"]:hover {\n    box-shadow: 0 6px 25px rgba(29,158,117,0.4) !important;\n    transform: translateY(-1px) !important;\n}\ndiv.stButton > button[kind="secondary"] {\n    background: rgba(255,255,255,0.04) !important;\n    border: 1px solid rgba(255,255,255,0.1) !important;\n    border-radius: 10px !important;\n    color: #94A3B8 !important;\n    font-weight: 500 !important;\n    transition: all 0.2s ease !important;\n}\ndiv.stButton > button[kind="secondary"]:hover {\n    background: rgba(255,255,255,0.08) !important;\n    border-color: rgba(255,255,255,0.2) !important;\n}\n\n/* ── Streamlit Metric Override ── */\ndiv[data-testid="stMetric"] {\n    background: rgba(255,255,255,0.03);\n    padding: 12px 16px;\n    border-radius: 10px;\n    border: 1px solid rgba(255,255,255,0.06);\n}\ndiv[data-testid="stMetric"] label {\n    color: #64748B !important;\n    font-size: 11px !important;\n    text-transform: uppercase !important;\n    letter-spacing: 0.05em !important;\n}\ndiv[data-testid="stMetric"] div[data-testid="stMetricValue"] {\n    color: #F1F5F9 !important;\n    font-size: 22px !important;\n    font-weight: 600 !important;\n}\n\n/* ── Expander Style ── */\ndiv[data-testid="stExpander"] {\n    border: 1px solid rgba(255,255,255,0.06) !important;\n    border-radius: 10px !important;\n    background: rgba(255,255,255,0.02) !important;\n}\n\n/* ── Selectbox / Slider Overrides ── */\ndiv[data-testid="stSelectbox"] label,\ndiv[data-testid="stSlider"] label {\n    color: #94A3B8 !important;\n    font-size: 12px !important;\n}\n\n/* ── File Uploader ── */\nsection[data-testid="stFileUploader"] {\n    border: 1px dashed rgba(29,158,117,0.3) !important;\n    border-radius: 12px !important;\n    background: rgba(29,158,117,0.04) !important;\n}\nsection[data-testid="stFileUploader"]:hover {\n    border-color: #1D9E75 !important;\n    background: rgba(29,158,117,0.08) !important;\n}\n\n/* ── Image styling ── */\nimg {\n    border-radius: 8px !important;\n}\n\n/* ── Checkbox ── */\ndiv[data-testid="stCheckbox"] label span {\n    color: #94A3B8 !important;\n    font-size: 12px !important;\n}\n\n/* ── Divider ── */\nhr {\n    border-color: rgba(255,255,255,0.06) !important;\n}\n</style>\n', unsafe_allow_html=True)
st.markdown('\n<div class="top-bar">\n    <div class="logo">\n        <div class="logo-icon">✦</div>\n        <span class="logo-text">Gen<span>AI</span> Studio</span>\n    </div>\n    <div class="status-pill">\n        <div class="status-dot"></div>\n        GPU Ready · FP32 · GTX 1080 Ti\n    </div>\n</div>\n', unsafe_allow_html=True)
st.caption('AI-Powered Product Background Replacement · Stable Diffusion + LoRA + MLOps Pipeline')
(grid_sidebar, grid_main) = st.columns([1, 2.5], gap='large')
with grid_sidebar:
    st.markdown('<div class="section-label" style="margin-top:0;">Product Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader('Drop product photo here', type=['png', 'jpg', 'jpeg', 'webp'], label_visibility='collapsed')
    show_mask = False
    if (uploaded_file is not None):
        show_mask = st.checkbox('🔍 Preview rembg mask', key='mask_toggle')
    st.markdown('<div class="section-label">Background Prompt</div>', unsafe_allow_html=True)
    prompt_presets = ['A professional photo sitting on a mossy rock in a forest, cinematic lighting, 4k', 'A sunny beach with palm trees and turquoise ocean, golden hour light', 'A clean white studio with soft box lighting, hyperrealistic product photo', 'Floating in deep space with colorful nebulas and distant stars, sci-fi', 'Custom prompt...']
    selected_preset = st.selectbox('Quick Prompts', prompt_presets, label_visibility='collapsed')
    if (selected_preset == 'Custom prompt...'):
        prompt = st.text_input('Describe the background:', placeholder='e.g., On a marble table...', label_visibility='collapsed')
    else:
        prompt = selected_preset
    st.markdown('<div class="section-label">Model Mode</div>', unsafe_allow_html=True)
    model_options = {'Baseline (No LoRA)': {'key': 'baseline', 'badge_class': 'badge-teal', 'card_class': 'active-teal', 'badge_text': 'Best quality', 'desc': 'Full SD Inpainting · no fine-tuning'}, 'Light LoRA (350 Steps)': {'key': 'light', 'badge_class': 'badge-amber', 'card_class': 'active-amber', 'badge_text': '350 steps', 'desc': 'Subtle domain adaptation'}, 'Overfit LoRA (44K Steps)': {'key': 'overfit', 'badge_class': 'badge-red', 'card_class': 'active-red', 'badge_text': '44k steps', 'desc': 'Catastrophic forgetting demo'}}
    model_choice = st.radio('Select model', list(model_options.keys()), index=0, label_visibility='collapsed')
    selected = model_options[model_choice]
    model_mode = selected['key']
    st.markdown(f'''
    <div class="model-card {selected['card_class']}">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:3px;">
            <span class="mc-label">{model_choice.split('(')[0].strip()}</span>
            <span class="mc-badge {selected['badge_class']}">{selected['badge_text']}</span>
        </div>
        <div class="mc-desc">{selected['desc']}</div>
    </div>
    ''', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Inference Parameters</div>', unsafe_allow_html=True)
    num_steps = st.slider('Denoising Steps', 10, 100, 50, key='steps')
    guidance = st.slider('Guidance Scale', 1.0, 20.0, 9.0, step=0.5, key='guidance')
    strength = st.slider('Inpainting Strength', 0.5, 1.0, 1.0, step=0.05, key='strength')
    st.markdown('<br/>', unsafe_allow_html=True)
    gen_btn = st.button('✦ Generate Background', type='primary', use_container_width=True)
    compare_btn = st.button('⚖️ Compare All 3', type='secondary', use_container_width=True)
    st.markdown('<br/>', unsafe_allow_html=True)
    with st.expander('👑 Admin Dashboard'):
        st.caption('Real-time telemetry')
        gcp_obs = GCPObservability(project_id='project-c5eebb76-bcc6-4730-840', location='us-central1', experiment_name='sd-inpainting')
        try:
            df = gcp_obs.get_vertex_experiments_history()
            if (not df.empty):
                (m1, m2) = st.columns(2)
                m1.metric('Cloud Runs', len(df))
                m2.metric('Avg Latency', f"{df['metrics.latency_sec'].mean():.1f}s")
                st.metric('Avg CLIP', f"{df['metrics.clip_score'].mean():.1f}")
            else:
                st.write('No Vertex AI tracking records found.')
        except Exception as e:
            st.error(str(e))
with grid_main:
    (col_orig, col_gen) = st.columns([1, 1], gap='medium')
    with col_orig:
        st.markdown('\n        <div class="panel-header">\n            <span class="panel-title">Original Input</span>\n            <span class="panel-tag">product image</span>\n        </div>\n        ', unsafe_allow_html=True)
        if (uploaded_file is not None):
            if show_mask:
                with st.spinner('Extracting mask...'):
                    try:
                        uploaded_file.seek(0)
                        mask_resp = requests.post(f'{BACKEND_URL}/extract_mask', files={'image': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}, timeout=30)
                        if (mask_resp.status_code == 200):
                            st.image(Image.open(io.BytesIO(mask_resp.content)), use_container_width=True)
                    except Exception as e:
                        st.image(Image.open(uploaded_file), use_container_width=True)
            else:
                st.image(Image.open(uploaded_file), use_container_width=True)
        else:
            st.markdown('\n            <div class="img-container">\n                <div class="empty-state">\n                    <div class="empty-icon">📷</div>\n                    <div>Upload a product image to begin</div>\n                </div>\n            </div>\n            ', unsafe_allow_html=True)
    with col_gen:
        st.markdown(f'''
        <div class="panel-header">
            <span class="panel-title">Generated Output</span>
            <span class="panel-tag">{model_mode}</span>
        </div>
        ''', unsafe_allow_html=True)
        output_placeholder = st.empty()
        if (uploaded_file is None):
            output_placeholder.markdown('\n            <div class="img-container">\n                <div class="empty-state">\n                    <div class="empty-icon">✨</div>\n                    <div>AI-generated result will appear here</div>\n                </div>\n            </div>\n            ', unsafe_allow_html=True)
    metrics_placeholder = st.empty()
if gen_btn:
    if (uploaded_file is None):
        st.warning('⚠️ Upload a product image first.')
    elif (not prompt.strip()):
        st.warning('⚠️ Enter a background prompt.')
    else:
        with st.spinner('✦ Generating new background...'):
            try:
                uploaded_file.seek(0)
                response = requests.post(f'{BACKEND_URL}/generate', files={'image': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}, data={'prompt': prompt, 'num_inference_steps': num_steps, 'guidance_scale': guidance, 'strength': strength, 'model_mode': model_mode}, timeout=180)
                if (response.status_code == 200):
                    output_image = Image.open(io.BytesIO(response.content))
                    latency = response.headers.get('X-Inference-Latency', '—')
                    clip_score = response.headers.get('X-CLIP-Score', '—')
                    with col_gen:
                        output_placeholder.image(output_image, use_container_width=True)
                        with metrics_placeholder.container():
                            if (model_mode == 'baseline'):
                                (fill_class, color) = ('fill-teal', '#1D9E75')
                            elif (model_mode == 'light'):
                                (fill_class, color) = ('fill-amber', '#F59E0B')
                            else:
                                (fill_class, color) = ('fill-red', '#EF4444')
                            try:
                                clip_val = float(clip_score)
                                clip_pct = min(((clip_val / 35) * 100), 100)
                            except:
                                clip_pct = 0
                            st.markdown(f'''
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
                            ''', unsafe_allow_html=True)
                    st.success('✅ Generation complete!')
                else:
                    st.error(f'Error {response.status_code}: {response.text}')
            except requests.exceptions.ConnectionError:
                st.error('❌ Backend is offline. Start the API first.')
            except Exception as e:
                st.error(f'❌ {e}')
if compare_btn:
    if (uploaded_file is None):
        st.warning('⚠️ Upload a product image first.')
    elif (not prompt.strip()):
        st.warning('⚠️ Enter a background prompt.')
    else:
        with col_gen:
            output_placeholder.empty()
            metrics_placeholder.empty()
            with output_placeholder.container():
                st.markdown('\n                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">\n                    <span style="font-size:16px;font-weight:600;color:#F1F5F9;">⚖️ 3-Way Model Comparison</span>\n                </div>\n                ', unsafe_allow_html=True)
                (comp_col1, comp_col2, comp_col3) = st.columns(3)
                modes_config = [('baseline', comp_col1, 'Base', 'badge-teal', 'fill-teal', '#1D9E75', 'No LoRA'), ('light', comp_col2, 'Light', 'badge-amber', 'fill-amber', '#F59E0B', '350s'), ('overfit', comp_col3, 'Overfit', 'badge-red', 'fill-red', '#EF4444', '44K')]
                for (mode, col, label, badge_cls, fill_cls, color, badge_text) in modes_config:
                    with col:
                        st.markdown(f'''
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                            <span style="font-size:11px; color:#F1F5F9; font-weight:500;">{label}</span>
                            <span class="mc-badge {badge_cls}" style="font-size:8px; padding:2px 4px;">{badge_text}</span>
                        </div>
                        ''', unsafe_allow_html=True)
                        with st.spinner(f'...'):
                            try:
                                uploaded_file.seek(0)
                                response = requests.post(f'{BACKEND_URL}/generate', files={'image': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}, data={'prompt': prompt, 'num_inference_steps': num_steps, 'guidance_scale': guidance, 'strength': strength, 'model_mode': mode}, timeout=180)
                                if (response.status_code == 200):
                                    out_img = Image.open(io.BytesIO(response.content))
                                    latency = response.headers.get('X-Inference-Latency', '—')
                                    clip_score = response.headers.get('X-CLIP-Score', '—')
                                    st.image(out_img, use_container_width=True)
                                    try:
                                        clip_val = float(clip_score)
                                        clip_pct = min(((clip_val / 35) * 100), 100)
                                    except:
                                        clip_pct = 0
                                    st.markdown(f'''
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
                                    ''', unsafe_allow_html=True)
                                else:
                                    st.error('Error')
                            except Exception as e:
                                st.error('Fail')
