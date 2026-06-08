import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ── SAYFA AYARLARI ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oyun Pazar Potansiyeli ve Karar Destek Sistemi",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at center, #0a0a0f 30%, rgba(64,224,208,0.12) 80%, rgba(255,105,180,0.15) 100%);
    color: white;
}
[data-testid="stHeader"] { background: transparent; }
.stDeployButton, footer, #MainMenu { visibility: hidden !important; }
[data-testid="metric-container"] {
    background-color: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(64,224,208,0.2);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.6);
    text-align: center;
}
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 10px 20px;
    border-radius: 8px;
    color: #aaa;
    font-weight: bold;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(64,224,208,0.2) !important;
    border: 1px solid #40E0D0 !important;
    color: #fff !important;
    box-shadow: 0px 0px 10px rgba(64,224,208,0.4);
}
</style>
""", unsafe_allow_html=True)

# ── MODEL YÜKLEME ─────────────────────────────────────────────────────────────
@st.cache_resource
def modeli_yukle():
    return joblib.load('tez_xgboost_final_model.pkl')

paket             = modeli_yukle()
model_na          = paket['model_na']
model_eu          = paket['model_eu']
model_jp          = paket['model_jp']
miras_verisi      = paket['miras_verisi']
potansiyel_verisi = paket['potansiyel_verisi']
ozellikler_yeni   = paket['ozellikler_listesi']

DEV_SERILER = [
    'mario', 'pokemon', 'call of duty', 'grand theft auto', 'gta',
    'final fantasy', 'halo', 'zelda', 'gran turismo',
    'super smash bros', "assassin's creed",
]

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if 'tahmin_edildi' not in st.session_state:
    st.session_state.tahmin_edildi  = False
    st.session_state.na_taban       = 0.0
    st.session_state.eu_taban       = 0.0
    st.session_state.jp_taban       = 0.0
    st.session_state.t_skor_kayit   = 0.0
    st.session_state.oyun_adi_kayit = ""
    st.session_state.yayinci_kayit  = ""
    st.session_state.tur_kayit      = ""

# ── BAŞLIK ────────────────────────────────────────────────────────────────────
st.title("🎮 Oyun Pazar Potansiyeli ve Karar Destek Sistemi")
st.markdown("Bu sistem, **Makine Öğrenmesi (XGBoost)** ve **Uzman Kurallar** kullanarak oyun projelerinin platform bazlı küresel potansiyelini kıyaslamalı olarak analiz eder.")
st.divider()

# ── GİRİŞ FORMU ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    oyun_adi = st.text_input("Oyunun Adı:", "EA Sports FC 26")
    yayinci  = st.selectbox("Yayıncı Şirket:", [
        "Electronic Arts", "Activision", "Take-Two Interactive", "Nintendo",
        "Sony Computer Entertainment", "Microsoft Game Studios",
        "Ubisoft", "Sega", "Capcom", "Unknown",
    ])

with col2:
    tur = st.selectbox("Oyun Türü:", [
        "Sports", "Role-Playing", "Action", "Shooter", "Platform",
        "Racing", "Misc", "Simulation", "Fighting", "Adventure", "Puzzle", "Strategy",
    ])

st.markdown("<br>", unsafe_allow_html=True)
tahmin_butonu = st.button("🚀 TÜM PLATFORMLAR İÇİN KIYASLAMALI TAHMİN ET", use_container_width=True)
st.divider()

# ── TAHMİN MOTORU ─────────────────────────────────────────────────────────────
if tahmin_butonu:
    with st.spinner("Yapay zeka tüm platform ekosistemlerini ve pazar dinamiklerini simüle ediyor..."):
        y_skor = miras_verisi.get(yayinci, 0)
        t_skor = potansiyel_verisi.get((yayinci, tur), 0)
        ea_sp  = 1 if (yayinci == "Electronic Arts"      and tur == "Sports")                     else 0
        tt_ac  = 1 if (yayinci == "Take-Two Interactive" and tur == "Action")                     else 0
        act_sh = 1 if (yayinci == "Activision"           and tur == "Shooter")                    else 0
        nin_mg = 1 if (yayinci == "Nintendo"             and tur in ["Platform", "Role-Playing"]) else 0
        is_dev = 1 if any(s in oyun_adi.lower() for s in DEV_SERILER)                            else 0
        is_exp = 1 if any(w in oyun_adi.lower() for w in ["expansion", "pack", "dlc", "edition"]) else 0

        girdi = pd.DataFrame(
            [[y_skor, t_skor, is_exp, ea_sp, tt_ac, act_sh, nin_mg, is_dev]],
            columns=ozellikler_yeni,
        )

        st.session_state.na_taban       = float(max(0.01, np.expm1(model_na.predict(girdi).item())))
        st.session_state.eu_taban       = float(max(0.01, np.expm1(model_eu.predict(girdi).item())))
        st.session_state.jp_taban       = float(max(0.01, np.expm1(model_jp.predict(girdi).item())))
        st.session_state.t_skor_kayit   = float(t_skor)
        st.session_state.oyun_adi_kayit = oyun_adi
        st.session_state.yayinci_kayit  = yayinci
        st.session_state.tur_kayit      = tur
        st.session_state.tahmin_edildi  = True

# ── SONUÇ PANELİ ─────────────────────────────────────────────────────────────
if st.session_state.tahmin_edildi:

    platformlar = ["PS5", "Xbox Series X", "Nintendo Switch", "PC"]
    st.subheader("📊 Platform Karşılaştırma Matrisi")
    sekmeler = st.tabs([f"🎮 {p}" for p in platformlar])

    for i, platform in enumerate(platformlar):
        with sekmeler[i]:
            na = st.session_state.na_taban
            eu = st.session_state.eu_taban
            jp = st.session_state.jp_taban

            oyun_adi_kucuk = st.session_state.oyun_adi_kayit.lower()
            yayinci_kutu   = st.session_state.yayinci_kayit
            tur_kutu       = st.session_state.tur_kayit
            t_skor_kutu    = st.session_state.t_skor_kayit

            # ── UZMAN KURAL MOTORU ────────────────────────────────────────────
            if platform == "Xbox Series X":
                na *= 1.4
                jp *= 0.05
            elif platform == "PS5":
                eu *= 1.35
            elif platform == "PC":
                na *= 0.7
                eu *= 0.8
                jp *= 0.05
                if t_skor_kutu > 0 and tur_kutu in ["Strategy", "Role-Playing"]:
                    na *= 1.8
                    eu *= 1.6

            if yayinci_kutu == "Electronic Arts" and tur_kutu == "Sports":
                if "fifa" in oyun_adi_kucuk or "fc" in oyun_adi_kucuk:
                    eu *= 4.5
                    na *= 1.2
                elif "madden" in oyun_adi_kucuk:
                    na *= 4.0
                    eu *= 0.2
            elif yayinci_kutu in ["Square Enix", "Capcom", "Nintendo"]:
                jp *= 2.5
                if "resident evil" in oyun_adi_kucuk or "final fantasy" in oyun_adi_kucuk:
                    na *= 1.8
                    eu *= 1.5
            elif yayinci_kutu == "Activision" and tur_kutu == "Shooter":
                if "call of duty" in oyun_adi_kucuk or "cod" in oyun_adi_kucuk:
                    na *= 3.5
                    eu *= 2.5
                    jp *= 0.4

            if yayinci_kutu == "Nintendo":
                if platform == "Nintendo Switch":
                    jp *= 3.0
                    na *= 1.5
                else:
                    na *= 0.01
                    eu *= 0.01
                    jp *= 0.01

            if "gta" in oyun_adi_kucuk or "grand theft auto" in oyun_adi_kucuk:
                na *= 4.0
                eu *= 4.5
                jp *= 1.5

            toplam = na + eu + jp
            lig = (
                "YILDIZ"   if toplam >= 3.0 else
                "HİT"      if toplam >= 0.8 else
                "STANDART" if toplam >= 0.2 else
                "DÜŞÜK"
            )

            # ── METRİKLER ─────────────────────────────────────────────────────
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🇺🇸 Kuzey Amerika Potansiyeli", f"{na:.2f} Milyon")
            k2.metric("🇪🇺 Avrupa Pazar Potansiyeli",  f"{eu:.2f} Milyon")
            k3.metric("🇯🇵 Japonya Pazar Potansiyeli", f"{jp:.2f} Milyon")
            k4.metric("🌍 Global Öngörülen Toplam",    f"{toplam:.2f} Milyon")

            st.markdown("<br>", unsafe_allow_html=True)
            sol, sag = st.columns(2)

            with sol:
                fig = go.Figure(data=[go.Pie(
                    labels=["Kuzey Amerika", "Avrupa", "Japonya"],
                    values=[na, eu, jp],
                    hole=0.6,
                    marker=dict(
                        colors=["#40E0D0", "#ff69b4", "#9370DB"],
                        line=dict(color="#0a0a0f", width=3),
                    ),
                    textinfo="percent",
                )])
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white", size=13),
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=260,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig, use_container_width=True)

            with sag:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if lig == "YILDIZ":
                    st.success(f"🏆 YATIRIM LİGİ: {lig}\n\nMükemmel pazar fırsatı! Bu platformda küresel çapta devasa bir finansal başarı ve rekor satış bekleniyor.")
                elif lig == "HİT":
                    st.info(f"🎯 YATIRIM LİGİ: {lig}\n\nPazar payı ve oyuncu sadakati çok yüksek. Oldukça güvenli, risksiz bir port projesi.")
                elif lig == "STANDART":
                    st.warning(f"⚖️ YATIRIM LİGİ: {lig}\n\nOrtalama pazar performansı. Geliştirme ve port maliyet bütçelerine dikkat edilmelidir.")
                else:
                    st.error(f"⚠️ YATIRIM LİGİ: {lig}\n\nRiskli pazar tercihi! Hedef kitle bu platform ekosisteminde dar kalabilir.")
