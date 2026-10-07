import sys; sys.stdout.reconfigure(encoding='utf-8')
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Page Config & CSS
# ==========================================
st.set_page_config(page_title="Tactical Style Clustering", page_icon="⚽", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Prompt', sans-serif; }
.hero {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    padding: 28px 32px; border-radius: 18px; color: white; margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 2rem; }
.hero p { margin: 6px 0 0; opacity: .85; }
.team-card {
    display: flex; align-items: center; gap: 22px;
    padding: 20px 28px; border-radius: 18px; color: white;
    box-shadow: 0 6px 18px rgba(0,0,0,.18); margin-bottom: 16px;
}
.team-card h2 { margin: 0; font-size: 2rem; color: white; }
.logo-wrap {
    width: 84px; height: 84px; border-radius: 50%; background: white;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 3px 10px rgba(0,0,0,.25); flex-shrink: 0; overflow: hidden;
}
.logo-wrap img { width: 62px; height: 62px; object-fit: contain; }
.logo-fallback { font-weight: 700; font-size: 1.8rem; color: #333; }
.badge {
    display: inline-block; padding: 4px 14px; border-radius: 999px;
    background: rgba(255,255,255,.22); font-size: .9rem; margin-top: 8px;
}
.desc-card {
    background: #f7f9fc; border-left: 5px solid var(--c);
    padding: 10px 16px; border-radius: 8px; margin-bottom: 8px; color: #222;
}
div[data-testid="stMetric"] {
    background: #f7f9fc; padding: 12px 16px; border-radius: 14px;
    border: 1px solid #e6eaf0;
}
div[data-testid="stMetric"] * { color: #222 !important; }
.dict-box {
    background: #ffffff; border: 1px solid #e0e0e0;
    padding: 20px 24px; border-radius: 12px; margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Session State for Page Navigation
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'

def go_to_dict():
    st.session_state.page = 'dictionary'

def go_to_dash():
    st.session_state.page = 'dashboard'

# ==========================================
# 3. Definitions & Logic
# ==========================================
BADGE_URL = "https://resources.premierleague.com/premierleague/badges/50/t{code}.png"
LOGO_CODES = {
    "arsenal": 3, "aston villa": 7, "bournemouth": 91, "chelsea": 8,
    "crystal palace": 31, "everton": 11, "leicester": 13, "liverpool": 14,
    "man city": 43, "manchester city": 43, "man united": 1, "man utd": 1,
    "manchester united": 1, "newcastle": 4, "norwich": 45, "southampton": 20,
    "stoke": 110, "sunderland": 56, "swansea": 80, "tottenham": 6, "spurs": 6,
    "watford": 57, "west brom": 35, "west bromwich": 35, "west ham": 21,
}

def logo_html(team):
    name = team.lower()
    for key in sorted(LOGO_CODES, key=len, reverse=True):
        if key in name:
            url = BADGE_URL.format(code=LOGO_CODES[key])
            initials = team[:2].upper()
            return (f"<div class='logo-wrap'><img src='{url}' alt='{team}' "
                    f"onerror=\"this.outerHTML='<span class=logo-fallback>{initials}</span>'\"></div>")
    return f"<div class='logo-wrap'><span class='logo-fallback'>{team[:2].upper()}</span></div>"

AXIS_HELP = {
    "Possession": "คะแนนเฉลี่ยของ 3 ตัวชี้วัด: Possession (%) สเกล 35–65, Pass Completion (%) สเกล 60–90 และ Short Pass (%) สเกล 25–55 (ยิ่งคะแนนสูง = ทีมคุมเกมด้วยการครองบอลและต่อบอลสั้นแม่นยำ)",
    "High Pressing": "คำนวณจาก Pressure Rate สเกล 120–190 (ยิ่งคะแนนสูง = ทีมเน้นบีบพื้นที่และกดดันผู้เล่นฝ่ายตรงข้ามตั้งแต่แดนบน)",
    "Direct Play": "คำนวณจาก Long Pass (%) สเกล 10–35 (ยิ่งคะแนนสูง = ทีมเน้นการเตะสาดบอลยาวทิ้งไปข้างหน้า หรือโยนข้ามแผงรับ)",
    "Defensive (Deep Block)": "คะแนนเฉลี่ยแบบกลับด้านของ (100 − High Pressing) และ (100 − Possession) (ยิ่งคะแนนสูง = ทีมจงใจถอยลงไปตั้งรับลึกในแดนตัวเอง ไม่ครองบอล และดักรอสวนกลับ)",
}

DATA_DICT = {
    "Possession (%)": "สัดส่วนเปอร์เซ็นต์เวลาที่ทีมครอบครองบอล เทียบกับเวลาการครองบอลรวมทั้งหมดในเกม (real_possession_pct)",
    "Pass Completion (%)": "เปอร์เซ็นต์ความสำเร็จในการจ่ายบอลถึงเพื่อนร่วมทีม (pass_completion_pct)",
    "Short Pass (%)": "สัดส่วนการจ่ายบอลสั้น (ระยะไม่เกิน 15 หลา) เทียบกับจำนวนการจ่ายบอลที่มีการบันทึกระยะทั้งหมด",
    "Long Pass (%)": "สัดส่วนการจ่ายบอลยาว (ระยะตั้งแต่ 30 หลาขึ้นไป) เทียบกับจำนวนการจ่ายบอลที่มีการบันทึกระยะทั้งหมด",
    "Progressive Carry Rate": "จำนวนครั้งการพาบอลตะลุยขึ้นหน้า (เลี้ยงบอลเข้าใกล้ประตูคู่แข่งอย่างน้อย 10 หลา) ต่อ 90 นาที",
    "Crossing Rate (%)": "อัตราการเปิดบอลจากริมเส้น (Cross) โดยมีตัวหารคือ จำนวนการพยายามจ่ายบอลทั้งหมด (Total Passes Attempted)",
    "Attacking 3rd Receipt Rate": "ความถี่ที่ผู้เล่นในทีมสามารถรับบอลสำเร็จในพื้นที่ 1/3 สุดท้ายของสนามฝั่งคู่แข่ง ต่อ 90 นาที",
    "Pressure Rate": "อัตราความดุดันในการเข้าบีบพื้นที่ มีตัวหารคือ จำนวนรอบการครองบอลของคู่แข่ง (เช่น 160% หมายถึงเมื่อคู่แข่งได้ครองบอล 1 รอบ ทีมเราจะพยายามเข้าเพรสซิ่งเฉลี่ย 1.6 ครั้ง)",
}

CLUSTER_COLORS = {
    0: ("#e53935", "#ff7043"),
    1: ("#1e88e5", "#42a5f5"),
    2: ("#8e24aa", "#ab47bc"),
    3: ("#43a047", "#7cb342"),
}

@st.cache_data
def load_and_process_data():
    df = pd.read_csv('final_features_ready.csv')
    X = df.drop(columns=['team'])
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df['PC1'], df['PC2'] = coords[:, 0], coords[:, 1]

    cluster_avg = df.groupby('Cluster').mean(numeric_only=True).round(2)

    cluster_info = {
        0: {"name": "High Pressing & Energetic", "desc": [
            "อัตราการบีบพื้นที่ (Pressure Rate) สูงที่สุดในลีก",
            "พาบอลตะลุยขึ้นหน้า (Progressive Carry) ได้ดี",
            "ครองบอลระดับกลาง เน้นแย่งบอลแล้วเข้าทำทันที"]},
        1: {"name": "Possession Elite", "desc": [
            "เปอร์เซ็นต์การครองบอล (Possession) สูงที่สุด",
            "ความแม่นยำในการจ่ายบอล (Pass Completion) สูงที่สุด",
            "จ่ายบอลสั้นเป็นหลัก แทบไม่ใช้การโยนยาว"]},
        2: {"name": "Deep Block & Wing Play", "desc": [
            "อัตราการเพรสซิ่งต่ำที่สุด (ถอยไปตั้งรับลึก/Low Block)",
            "สัดส่วนการเปิดบอลจากริมเส้น (Cross Rate) สูง",
            "อาศัยการรอสวนกลับอย่างรวดเร็ว"]},
        3: {"name": "Extreme Route One", "desc": [
            "ครองบอลต่ำที่สุด และจ่ายบอลแม่นยำต่ำที่สุด",
            "สัดส่วนการโยนยาว (Long Pass) สูงที่สุดแบบสุดโต่ง",
            "เน้นเตะสาดไปข้างหน้าเพื่อทำลายเกมรุกคู่แข่ง (Direct Style)"]},
    }

    def scale_fixed(val, lo, hi):
        return float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))

    def compute_scores(row):
        poss_s = scale_fixed(row['real_possession_pct'], 35, 65)
        press_s = scale_fixed(row['pressure_rate'], 120, 190)
        poss = np.mean([poss_s, scale_fixed(row['pass_completion_pct'], 60, 90), scale_fixed(row['short_pass_pct'], 25, 55)])
        direct = scale_fixed(row['long_pass_pct'], 10, 35)
        defense = np.mean([100 - press_s, 100 - poss_s])
        return {'Possession': poss, 'High Pressing': press_s, 'Direct Play': direct, 'Defensive (Deep Block)': defense}

    rows = [{'Team': r['team'], 'Cluster': r['Cluster'], **compute_scores(r)} for _, r in df.iterrows()]
    df_scores = pd.DataFrame(rows)

    cat = ['Possession', 'High Pressing', 'Direct Play', 'Defensive (Deep Block)']
    cluster_score_avg = df_scores.groupby('Cluster')[cat].mean()

    feature_names = {
        'real_possession_pct': 'Possession (%)', 'pass_completion_pct': 'Pass Completion (%)',
        'short_pass_pct': 'Short Pass (%)', 'long_pass_pct': 'Long Pass (%)',
        'progressive_carry_rate': 'Progressive Carry Rate', 'cross_rate': 'Crossing Rate (%)',
        'attacking_third_receipt_rate': 'Attacking 3rd Receipt Rate', 'pressure_rate': 'Pressure Rate'
    }
    return df, df_scores, cluster_avg, cluster_score_avg, cluster_info, feature_names, cat

df, df_scores, cluster_avg, cluster_score_avg, cluster_info, feature_names, CATS = load_and_process_data()

def radar(traces, height=400):
    fig = go.Figure()
    for name, vals, color, fill in traces:
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=CATS + [CATS[0]], name=name,
            fill='toself' if fill else 'none', line=dict(color=color, width=3),
            opacity=0.75 if fill else 1))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], showticklabels=False)),
        height=height, margin=dict(l=40, r=40, t=20, b=20),
        legend=dict(orientation="h", y=-0.1))
    return fig

# ==========================================
# 4. Sidebar View Management
# ==========================================
with st.sidebar:
    if st.session_state.page == 'dashboard':
        # 1. เอาเมนูเลือกทีมขึ้นมาก่อน
        st.header(" Select Team")
        teams = sorted(df_scores['Team'])
        selected_team = st.selectbox("Choose a team", teams)
        st.divider()
        
        # 2. ตามด้วย Legend
        st.markdown("###  Cluster Legend")
        for k, v in cluster_info.items():
            st.markdown(f"<span style='color:{CLUSTER_COLORS[k][0]};font-size:1.2rem'>●</span> **C{k}** {v['name']}", unsafe_allow_html=True)
        st.divider()
        
	# 4. ย้ายปุ่มสลับหน้ามาไว้ล่างสุดตรงนี้ครับ 👇
        st.button("📖 พจนานุกรมข้อมูล (คำอธิบาย)", on_click=go_to_dict, type="primary", use_container_width=True)

        # 3. ปุ่ม Download
        st.download_button("⬇️ Download Clusters (CSV)", df[['team', 'Cluster']].to_csv(index=False).encode('utf-8-sig'), "team_clusters.csv", "text/csv")
        st.divider()
        
        
        
    else: # Dictionary Page Sidebar (หน้าอธิบายข้อมูล)
        # หน้านี้ปุ่มกลับไว้ด้านบนโอเคแล้ว เพราะไม่มีเมนูอื่นครับ
        st.button("⬅️ กลับไปหน้า Dashboard", on_click=go_to_dash, type="primary", use_container_width=True)
        st.info("คุณกำลังอยู่ในหน้าอธิบายตัวชี้วัด")

# ==========================================
# 5. Page Routing
# ==========================================
if st.session_state.page == 'dashboard':
    
    # ------------------- DASHBOARD PAGE -------------------
    st.markdown("""
    <div class="hero">
      <h1> Football Team Tactical Style Clustering</h1>
      <p><b>Unsupervised Machine Learning (K-Means)</b> · วิเคราะห์และจัดกลุ่มสไตล์การเล่นของทีมพรีเมียร์ลีก ฤดูกาล 2015/16</p>
    </div>
    """, unsafe_allow_html=True)

    team_score = df_scores[df_scores['Team'] == selected_team].iloc[0]
    team_raw = df[df['team'] == selected_team].iloc[0]
    c_id = int(team_score['Cluster'])
    c_info = cluster_info[c_id]
    c1, c2 = CLUSTER_COLORS[c_id]

    st.markdown(f"""
    <div class="team-card" style="background:linear-gradient(135deg,{c1},{c2});">
      {logo_html(selected_team)}
      <div>
        <h2>{selected_team}</h2>
        <span class="badge">Cluster {c_id} · {c_info['name']}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⚽ Team Profile", " 🔄 Compare Teams", "🗺️ Cluster Map"])

    with tab1:
        m1, m2, m3, m4 = st.columns(4)
        def delta(col): return round(team_raw[col] - cluster_avg.loc[c_id, col], 2)
        m1.metric("Possession (%)", f"{team_raw['real_possession_pct']:.1f}", delta('real_possession_pct'))
        m2.metric("Pass Completion (%)", f"{team_raw['pass_completion_pct']:.1f}", delta('pass_completion_pct'))
        m3.metric("Long Pass (%)", f"{team_raw['long_pass_pct']:.1f}", delta('long_pass_pct'))
        m4.metric("Pressure Rate", f"{team_raw['pressure_rate']:.1f}", delta('pressure_rate'))
        st.caption("ลูกศร = ส่วนต่างเทียบกับค่าเฉลี่ยของ Cluster")

        left, right = st.columns([1.1, 1])
        with left:
            axis_help = "\n\n".join(f"**{k}**: {v}" for k, v in AXIS_HELP.items())
            st.markdown("####  Tactical Score", help=axis_help)
            st.caption("Higher score = stronger presence of that tactical characteristic.")
            vals = [team_score[c] for c in CATS]
            st.plotly_chart(radar([(selected_team, vals, c1, True), (f"Cluster {c_id} Avg", cluster_score_avg.loc[c_id].tolist(), "#888888", False)]), use_container_width=True)
        with right:
            st.markdown("####  Score Breakdown")
            for cat, v in zip(CATS, vals):
                st.markdown(f"**{cat}** — {int(v)}")
                st.progress(int(v))

        st.markdown(f"####  Cluster {c_id} Characteristics")
        for d in c_info['desc']:
            st.markdown(f"<div class='desc-card' style='--c:{c1}'>{d}</div>", unsafe_allow_html=True)

        st.markdown(f"####  Why is {selected_team} in Cluster {c_id}?")
        rows = [{"Feature": pretty, selected_team: round(team_raw[raw], 2), f"Cluster {c_id} Avg": round(cluster_avg.loc[c_id, raw], 2), "Diff": round(team_raw[raw] - cluster_avg.loc[c_id, raw], 2)} for raw, pretty in feature_names.items()]
        comp = pd.DataFrame(rows).set_index("Feature")
        lim = max(comp["Diff"].abs().max(), 1e-9)
        styled = (comp.style.format("{:.2f}").background_gradient(subset=["Diff"], cmap="RdYlGn", vmin=-lim, vmax=lim))
        st.dataframe(styled, use_container_width=True)

    with tab2:
        st.markdown("####  Head-to-Head")
        ca, cb = st.columns(2)
        t_a = ca.selectbox("Team A", teams, index=teams.index(selected_team), key="ta")
        t_b = cb.selectbox("Team B", teams, index=1 if teams[0] == t_a else 0, key="tb")

        sa, sb = df_scores[df_scores['Team'] == t_a].iloc[0], df_scores[df_scores['Team'] == t_b].iloc[0]
        ka, kb = int(sa['Cluster']), int(sb['Cluster'])
        st.plotly_chart(radar([(f"{t_a} (C{ka})", [sa[c] for c in CATS], CLUSTER_COLORS[ka][0], True), (f"{t_b} (C{kb})", [sb[c] for c in CATS], "#222222" if ka == kb else CLUSTER_COLORS[kb][0], True)], height=460), use_container_width=True)

        ra, rb = df[df['team'] == t_a].iloc[0], df[df['team'] == t_b].iloc[0]
        h2h = pd.DataFrame({"Feature": list(feature_names.values()), t_a: [round(ra[k], 2) for k in feature_names], t_b: [round(rb[k], 2) for k in feature_names]}).set_index("Feature")
        st.dataframe(h2h, use_container_width=True)

    with tab3:
        st.markdown("####  Cluster Map (PCA 2D)")
        st.caption("ลดมิติฟีเจอร์ทั้งหมดเหลือ 2 แกนเพื่อดูการกระจายตัวของทีม")
        plot_df = df.copy()
        plot_df['Cluster Name'] = plot_df['Cluster'].map(lambda k: f"C{k}: {cluster_info[k]['name']}")
        color_map = {f"C{k}: {cluster_info[k]['name']}": CLUSTER_COLORS[k][0] for k in cluster_info}
        fig_map = px.scatter(plot_df, x='PC1', y='PC2', color='Cluster Name', text='team', color_discrete_map=color_map, height=560)
        fig_map.update_traces(textposition='top center', marker=dict(size=13, line=dict(width=1, color='white')))
        sel = plot_df[plot_df['team'] == selected_team]
        fig_map.add_trace(go.Scatter(x=sel['PC1'], y=sel['PC2'], mode='markers', marker=dict(size=26, symbol='circle-open', line=dict(width=3, color='black')), name=f"Selected: {selected_team}"))
        fig_map.update_layout(legend=dict(orientation="h", y=-0.15), xaxis_title="PC1: Direct Play ⬅️ vs ➡️ Possession",yaxis_title="PC2: High Pressing ⬅️ vs ➡️ Defensive/Crossing")
        st.plotly_chart(fig_map, use_container_width=True)

    st.divider()
    st.caption("Built with Streamlit · scikit-learn · Plotly | Premier League 2015/16")

else:
    # ------------------- DICTIONARY PAGE -------------------
    st.markdown("""
    <div class="hero" style="background: linear-gradient(135deg, #1e3c72, #2a5298);">
      <h1> Data Dictionary & Definitions</h1>
      <p>พจนานุกรมข้อมูล: คำอธิบายสไตล์การเล่นและตัวชี้วัดที่ใช้ในโมเดล K-Means</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.header(" Tactical Dimensions (4 แกน Radar Chart)")
    st.markdown("คะแนนทั้ง 4 ด้าน ถูกคำนวณผ่านระบบ **Fixed Bounds (0-100%)** เพื่อสะท้อนสไตล์การเล่นให้เข้าใจง่าย")
    for term, meaning in AXIS_HELP.items():
        st.markdown(f"""
        <div class="dict-box">
            <h4 style="margin-top:0; color:#1e3c72;"> {term}</h4>
            <span style="font-size:1.05rem;">{meaning}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.header(" Data Features (ฟีเจอร์ข้อมูลที่ใช้จัดกลุ่ม)")
    st.markdown("สถิติที่ใช้ในการ Train โมเดล K-Means ทั้งหมด 8 ตัวแปร")
    
    col1, col2 = st.columns(2)
    items = list(DATA_DICT.items())
    half = len(items) // 2
    
    for i, (term, meaning) in enumerate(items):
        target_col = col1 if i < half else col2
        with target_col:
            st.markdown(f"""
            <div class="dict-box">
                <b style="color:#000; font-size:1.1rem;">{term}</b><br>
                <span style="color:#444; font-size:0.95rem;">{meaning}</span>
            </div>
            """, unsafe_allow_html=True)
