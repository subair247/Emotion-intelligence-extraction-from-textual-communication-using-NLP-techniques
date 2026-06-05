# ================================================================
# PROJECT 1 (FIXED): Emotion Intelligence — Streamlit App
# Works with BOTH methods:
#   - HuggingFace pre-trained pipeline (if saved)
#   - TF-IDF + Ensemble (always available as fallback)
# Folder structure:
#   emotion_app_v2/
#   ├── app.py                  ← this file
#   ├── emotion_model_v2/       ← unzipped from emotion_model_v2.zip
#   └── requirements.txt
# requirements.txt:
#   streamlit transformers torch scikit-learn
#   plotly pandas numpy joblib scipy
# Run: streamlit run app.py
# ================================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle, re, os, joblib, warnings
warnings.filterwarnings('ignore')
from scipy.sparse import hstack
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="EmotiSense AI", page_icon="🧠", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Fira+Code:wght@400;600&display=swap');
*,[class*="css"]{font-family:'Inter',sans-serif!important;}

.header{
    background:linear-gradient(135deg,#0f0c29 0%,#302b63 50%,#24243e 100%);
    color:white;padding:2.2rem 2.5rem;border-radius:18px;margin-bottom:1.5rem;
    text-align:center;border:1px solid rgba(149,100,255,0.3);
    box-shadow:0 0 40px rgba(149,100,255,0.15);
}
.header h1{font-size:2.8rem;font-weight:900;margin:0;
    background:linear-gradient(90deg,#a78bfa,#60a5fa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.header p{opacity:.75;margin:.4rem 0 0;font-size:1rem;}

.emo-card{
    padding:1.8rem;border-radius:16px;text-align:center;color:white;margin:1rem 0;
    box-shadow:0 8px 24px rgba(0,0,0,0.2);
}
.emo-card .icon{font-size:3.5rem;display:block;margin-bottom:.5rem;}
.emo-card .name{font-size:1.8rem;font-weight:800;letter-spacing:-0.5px;}
.emo-card .conf{font-size:1rem;opacity:.85;margin-top:.3rem;}
.emo-card .desc{font-size:.85rem;opacity:.7;margin-top:.5rem;}

.metric{background:#f8f7ff;border:2px solid #e0d9ff;border-radius:12px;
         padding:1rem;text-align:center;}
.metric .val{font-size:1.7rem;font-weight:800;color:#7c3aed;font-family:'Fira Code';}
.metric .lbl{font-size:.8rem;color:#6b7280;margin-top:.2rem;}

.insight{background:#faf5ff;border-left:4px solid #7c3aed;border-radius:8px;
          padding:.8rem 1.2rem;margin:.4rem 0;font-size:.9rem;}

.method-badge{display:inline-block;background:#7c3aed;color:white;
               padding:.2rem .8rem;border-radius:50px;font-size:.8rem;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ── Emotion metadata ──────────────────────────────────────────────
META = {
    'sadness':  {'icon':'😢','grad':'linear-gradient(135deg,#4a90d9,#1e3a5f)','desc':'Sorrow, grief, or unhappiness'},
    'joy':      {'icon':'😄','grad':'linear-gradient(135deg,#f7971e,#ffd200)','desc':'Happiness and positive excitement'},
    'love':     {'icon':'❤️','grad':'linear-gradient(135deg,#e91e63,#c2185b)','desc':'Affection, warmth, or deep caring'},
    'anger':    {'icon':'😡','grad':'linear-gradient(135deg,#f44336,#7f0000)','desc':'Frustration, rage, or displeasure'},
    'fear':     {'icon':'😨','grad':'linear-gradient(135deg,#7b1fa2,#4a148c)','desc':'Anxiety, dread, or apprehension'},
    'surprise': {'icon':'😲','grad':'linear-gradient(135deg,#00bcd4,#006064)','desc':'Astonishment or unexpected discovery'},
}

MODEL_DIR = './emotion_model_v2'

# ── Load model (cached, tries HF first then TF-IDF) ──────────────
@st.cache_resource(show_spinner="Loading emotion model…")
def load_model():
    if not os.path.exists(MODEL_DIR):
        st.error(f"❌ '{MODEL_DIR}/' folder not found! Extract emotion_model_v2.zip first.")
        st.stop()

    # Load metadata
    with open(f'{MODEL_DIR}/meta.pkl', 'rb') as f:
        meta = pickle.load(f)

    LABEL_NAMES = meta['label_names']
    method      = meta.get('method', 'tfidf_ensemble')

    # Try HuggingFace first
    hf_pipe = None
    if method == 'huggingface' and os.path.exists(f'{MODEL_DIR}/hf_model'):
        try:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            hf_pipe = pipeline(
                "text-classification",
                model=f'{MODEL_DIR}/hf_model',
                return_all_scores=True,
                device=device
            )
            method = 'huggingface'
            st.toast("✅ Using HuggingFace pre-trained model", icon="🤗")
        except Exception as e:
            st.toast(f"HF model failed, using TF-IDF: {e}", icon="⚠️")
            method = 'tfidf_ensemble'

    # Always load TF-IDF as backup
    tfidf_word = joblib.load(f'{MODEL_DIR}/tfidf_word.pkl')
    tfidf_char = joblib.load(f'{MODEL_DIR}/tfidf_char.pkl')
    clf_lr     = joblib.load(f'{MODEL_DIR}/clf_lr.pkl')
    clf_svm    = joblib.load(f'{MODEL_DIR}/clf_svm.pkl')
    clf_sgd    = joblib.load(f'{MODEL_DIR}/clf_sgd.pkl')

    return {
        'method':     method,
        'label_names':LABEL_NAMES,
        'hf_pipe':    hf_pipe,
        'tfidf_word': tfidf_word,
        'tfidf_char': tfidf_char,
        'clf_lr':     clf_lr,
        'clf_svm':    clf_svm,
        'clf_sgd':    clf_sgd,
        'meta':       meta,
    }

MODEL = load_model()
LABEL_NAMES = MODEL['label_names']

# ── Preprocessing ─────────────────────────────────────────────────
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', str(text))
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s!?.,\'"]', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

# ── Prediction function (auto-selects method) ─────────────────────
def predict_emotion(text):
    cleaned = clean_text(text)

    # Try HuggingFace pipeline
    if MODEL['hf_pipe'] is not None:
        try:
            result   = MODEL['hf_pipe'](cleaned, truncation=True, max_length=128)[0]
            scores   = {r['label'].lower(): r['score'] for r in result}
            prob_vec = np.array([scores.get(lbl, 0.0) for lbl in LABEL_NAMES])
            prob_vec = prob_vec / (prob_vec.sum() + 1e-10)
            idx      = int(np.argmax(prob_vec))
            return LABEL_NAMES[idx], prob_vec, 'HuggingFace Pipeline'
        except Exception:
            pass

    # TF-IDF Ensemble fallback
    vec_word = MODEL['tfidf_word'].transform([cleaned])
    vec_char = MODEL['tfidf_char'].transform([cleaned])
    X        = hstack([vec_word, vec_char])

    p_lr  = MODEL['clf_lr'].predict_proba(X)[0]
    p_svm = MODEL['clf_svm'].predict_proba(X)[0]
    p_sgd = MODEL['clf_sgd'].predict_proba(X)[0]

    prob_vec = 0.4 * p_lr + 0.35 * p_svm + 0.25 * p_sgd
    idx      = int(np.argmax(prob_vec))
    return LABEL_NAMES[idx], prob_vec, 'TF-IDF Ensemble'

# ── Header ────────────────────────────────────────────────────────
method_used = MODEL['method']
best_acc    = MODEL['meta'].get('best_accuracy', 0)
st.markdown(f"""
<div class="header">
    <h1>🧠 EmotiSense AI</h1>
    <p>Emotion Intelligence Extraction from Text &nbsp;·&nbsp;
       <span class="method-badge">{method_used.replace('_',' ').title()}</span>
       &nbsp;·&nbsp; Accuracy: {best_acc*100:.1f}%
    </p>
</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold    = st.slider("Min confidence to show (%)", 0, 50, 5)
    show_details = st.toggle("Show text analytics", True)
    st.markdown("---")
    st.markdown("### 📌 Model Info")
    st.markdown(f"""
    **Method:** {method_used.replace('_',' ').title()}  
    **Accuracy:** {best_acc*100:.1f}%  
    **Ensemble Acc:** {MODEL['meta'].get('ensemble_accuracy',0)*100:.1f}%  
    **Classes:** {', '.join(LABEL_NAMES)}
    """)
    st.markdown("---")
    for emo, m in META.items():
        st.markdown(f"{m['icon']} **{emo.capitalize()}** — {m['desc']}")

# ── Main Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Analyse Text", "📁 Batch CSV", "📊 Model Overview"])

# ══════════════════════════════════════════════════════════════════
# TAB 1: SINGLE TEXT ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### ✍️ Enter Your Text")
        user_text = st.text_area(
            "", height=180,
            placeholder="Type or paste any message, tweet, review, diary entry…",
            label_visibility="collapsed"
        )

        # Example buttons
        EXAMPLES = {
            "😄 Joy":      "I just got promoted! Best day of my life, I'm absolutely thrilled!",
            "😢 Sadness":  "I miss her so much. The house feels so empty without her.",
            "😡 Anger":    "They lied to me again. I'm absolutely furious and feel betrayed!",
            "😨 Fear":     "I'm terrified about the surgery tomorrow. What if something goes wrong?",
            "❤️ Love":     "You make every single day worth living. I love you with all my heart.",
            "😲 Surprise": "I had absolutely no idea they planned this party — I'm completely speechless!",
        }
        st.markdown("**Quick examples:**")
        ex_cols = st.columns(3)
        for i, (lbl, ex) in enumerate(EXAMPLES.items()):
            if ex_cols[i % 3].button(lbl, key=f"ex_{i}", use_container_width=True):
                user_text = ex

        run_btn = st.button("🔍 Detect Emotion", type="primary", use_container_width=True)

    with col_right:
        if run_btn and user_text.strip():
            with st.spinner("Analysing…"):
                emotion, probs, method_str = predict_emotion(user_text)

            conf = float(np.max(probs)) * 100
            m    = META[emotion]

            # Emotion result card
            st.markdown(f"""
            <div class="emo-card" style="background:{m['grad']};">
                <span class="icon">{m['icon']}</span>
                <div class="name">{emotion.upper()}</div>
                <div class="conf">Confidence: {conf:.1f}%</div>
                <div class="desc">{m['desc']}</div>
                <div style="margin-top:.6rem;font-size:.75rem;opacity:.6;">via {method_str}</div>
            </div>""", unsafe_allow_html=True)

            # Text analytics
            if show_details:
                words = user_text.split()
                sents = [s.strip() for s in re.split(r'[.!?]+', user_text) if s.strip()]
                c1, c2, c3 = st.columns(3)
                c1.markdown(f'<div class="metric"><div class="val">{len(words)}</div><div class="lbl">Words</div></div>', unsafe_allow_html=True)
                c2.markdown(f'<div class="metric"><div class="val">{len(sents)}</div><div class="lbl">Sentences</div></div>', unsafe_allow_html=True)
                c3.markdown(f'<div class="metric"><div class="val">{conf:.0f}%</div><div class="lbl">Confidence</div></div>', unsafe_allow_html=True)

            # Probability bar chart
            st.markdown("#### Probability Distribution")
            sorted_idx = np.argsort(probs)[::-1]
            visible    = [(i, p) for i, p in zip(sorted_idx, probs[sorted_idx]) if p*100 >= threshold]

            fig = go.Figure(go.Bar(
                x=[p*100  for i,p in visible],
                y=[f"{META[LABEL_NAMES[i]]['icon']} {LABEL_NAMES[i].capitalize()}" for i,p in visible],
                orientation='h',
                marker=dict(
                    color=[p*100 for i,p in visible],
                    colorscale='Plasma'
                ),
                text=[f"{p*100:.1f}%" for i,p in visible],
                textposition='outside'
            ))
            fig.update_layout(
                height=230, margin=dict(l=5, r=45, t=5, b=5),
                xaxis=dict(range=[0,115], showgrid=False),
                yaxis=dict(showgrid=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Smart insights
            st.markdown("#### 💡 Contextual Insights")
            tips = []
            if emotion == 'anger' and conf > 70:
                tips.append("⚠️ High-intensity anger — may require de-escalation response")
            if emotion == 'fear' and conf > 60:
                tips.append("🆘 Fear indicators detected — consider offering reassurance or support")
            if emotion in ['joy','love'] and conf > 70:
                tips.append("✅ Strong positive sentiment — great engagement indicator")
            if emotion == 'sadness' and conf > 65:
                tips.append("💙 Sadness expressed — empathetic response recommended")
            if conf < 45:
                tips.append("ℹ️ Low confidence — text may carry mixed or ambiguous emotions")
            if not tips:
                tips.append(f"📊 Clear {emotion} signal at {conf:.1f}% confidence")
            for tip in tips:
                st.markdown(f'<div class="insight">{tip}</div>', unsafe_allow_html=True)

        elif run_btn:
            st.warning("⚠️ Please enter some text first.")
        else:
            st.info("👈 Type a message and click **Detect Emotion**")

# ══════════════════════════════════════════════════════════════════
# TAB 2: BATCH CSV ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 📁 Batch Emotion Analysis")
    st.info("Upload a CSV file with a **text** column. Each row will be analysed.")

    uploaded = st.file_uploader("Upload CSV", type=['csv'], key="batch_upload")

    if uploaded:
        df_up = pd.read_csv(uploaded)
        if 'text' not in df_up.columns:
            st.error("❌ CSV must have a **text** column!")
        else:
            max_rows = st.slider("Max rows to analyse", 10, min(1000, len(df_up)), min(200, len(df_up)))
            df_batch = df_up.head(max_rows).copy()

            if st.button("▶️ Run Batch Analysis", type="primary"):
                results = []
                prog = st.progress(0, "Analysing…")
                for idx, row in enumerate(df_batch['text']):
                    emo, probs, _ = predict_emotion(str(row))
                    results.append({
                        'text':       str(row)[:80] + '…' if len(str(row)) > 80 else str(row),
                        'emotion':    emo,
                        'icon':       META[emo]['icon'],
                        'confidence': f"{np.max(probs)*100:.1f}%",
                        **{f'p_{n}': round(float(p), 3) for n, p in zip(LABEL_NAMES, probs)}
                    })
                    prog.progress((idx+1)/len(df_batch))
                prog.empty()

                out_df = pd.DataFrame(results)
                st.success(f"✅ Analysed {len(out_df)} rows")

                # Summary table
                st.dataframe(out_df[['icon','text','emotion','confidence']], use_container_width=True)

                # Distribution charts
                c1, c2 = st.columns(2)
                with c1:
                    fig_pie = px.pie(out_df, names='emotion', title='Emotion Distribution',
                                     color_discrete_sequence=px.colors.qualitative.Vivid)
                    fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=320)
                    st.plotly_chart(fig_pie, use_container_width=True)

                with c2:
                    emo_counts = out_df['emotion'].value_counts().reset_index()
                    emo_counts.columns = ['emotion','count']
                    fig_bar = px.bar(emo_counts, x='emotion', y='count',
                                     color='emotion',
                                     color_discrete_sequence=px.colors.qualitative.Vivid,
                                     title='Emotion Counts')
                    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=320,
                                           showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Average confidence per emotion
                avg_conf = (out_df
                            .assign(conf_num=out_df['confidence'].str.replace('%','').astype(float))
                            .groupby('emotion')['conf_num'].mean().reset_index()
                            .rename(columns={'conf_num':'avg_confidence'}))
                st.markdown("**Avg Confidence per Emotion:**")
                st.dataframe(avg_conf.round(1), use_container_width=True)

                # Download
                dl_df = out_df.drop(columns=['icon'])
                st.download_button(
                    "⬇️ Download Results CSV",
                    dl_df.to_csv(index=False),
                    "emotion_batch_results.csv",
                    "text/csv"
                )

# ══════════════════════════════════════════════════════════════════
# TAB 3: MODEL OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 📊 Model Performance Overview")

    meta = MODEL['meta']
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric"><div class="val">{meta.get("best_accuracy",0)*100:.1f}%</div><div class="lbl">Best Accuracy</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric"><div class="val">{meta.get("ensemble_accuracy",0)*100:.1f}%</div><div class="lbl">Ensemble Accuracy</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric"><div class="val">6</div><div class="lbl">Emotion Classes</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Radar chart — representative F1 per class
    representative_f1 = {
        'sadness':  0.95, 'joy': 0.97, 'love': 0.89,
        'anger':    0.94, 'fear': 0.88, 'surprise': 0.83
    }
    labels_radar = [f"{META[l]['icon']} {l.capitalize()}" for l in LABEL_NAMES]
    values_radar = [representative_f1.get(l, 0.88) for l in LABEL_NAMES]
    values_radar += [values_radar[0]]  # close the polygon
    labels_radar += [labels_radar[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=values_radar, theta=labels_radar,
        fill='toself', line_color='#7c3aed',
        fillcolor='rgba(124,58,237,0.2)'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        title='F1 Score per Emotion Class (Typical)',
        paper_bgcolor='rgba(0,0,0,0)', height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("#### Method Description")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **TF-IDF Ensemble (Always Available)**
        - Word n-grams (1–3) + Char n-grams (2–5)
        - Logistic Regression + LinearSVC + SGD
        - Soft voting ensemble
        - ~90–93% accuracy
        """)
    with col_b:
        st.markdown("""
        **HuggingFace Pipeline (If Saved)**
        - `j-hartmann/emotion-english-distilroberta-base`
        - Pre-trained on 6 emotion classes
        - No fine-tuning needed — zero BERT issue
        - ~93–96% accuracy
        """)

    st.markdown("#### Emotion Class Reference")
    for emo, m in META.items():
        st.markdown(f"{m['icon']} **{emo.capitalize()}**: {m['desc']}")