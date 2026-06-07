import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go  # YENİ: Havalı dalgalı grafikler için

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Oyun Satış Karar Destek Sistemi", page_icon="🎮", layout="wide")

# --- CSS TASARIMI (Neon Işıklar ve Ortaya Kararan Tema) ---
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at center, #0a0a0f 30%, rgba(64, 224, 208, 0.15) 80%, rgba(255, 105, 180, 0.2) 100%);
    color: white;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="metric-container"] {
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 15px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.5);
}
</style>
""", unsafe_allow_html=True)

# 2. MODELİ HAFIZAYA ALMA
@st.cache_resource
def modeli_yukle():
    return joblib.load('tez_xgboost_final_model.pkl')

paket = modeli_yukle()
model_na, model_eu, model_jp = paket['model_na'], paket['model_eu'], paket['model_jp']
miras_verisi, potansiyel_verisi, ozellikler_yeni = paket['miras_verisi'], paket['potansiyel_verisi'], paket['ozellikler_listesi']
dev_seriler = ['mario', 'pokemon', 'call of duty', 'grand theft auto', 'gta', 'final fantasy', 'halo', 'zelda', 'gran turismo', 'super smash bros', 'assassin\'s creed']

# 3. ARAYÜZ BAŞLIĞI
st.title("🎮 Oyun Pazar Potansiyeli ve Karar Destek Sistemi")
st.markdown("Bu sistem, **Makine Öğrenmesi (XGBoost)** ve **Uzman Kurallar** kullanarak oyun projelerinin global satış potansiyelini öngörür.")
st.divider()

# 4. KULLANICI GİRİŞ FORMU
col1, col2 = st.columns(2)

with col1:
    oyun_adi = st.text_input("Oyunun Adı:", "EA Sports FC 26")
    yayinci = st.selectbox("Yayıncı Şirket:", ["Electronic Arts", "Activision", "Take-Two Interactive", "Nintendo", "Sony Computer Entertainment", "Microsoft Game Studios", "Ubisoft", "Sega", "Capcom", "Unknown"])

with col2:
    tur = st.selectbox("Oyun Türü:", ["Sports", "Role-Playing", "Action", "Shooter", "Platform", "Racing", "Misc", "Simulation", "Fighting", "Adventure", "Puzzle", "Strategy"])
    platform = st.selectbox("Platform:", ["PS5", "PC", "PS4", "Xbox Series X", "Nintendo Switch"])

st.markdown("<br>", unsafe_allow_html=True)
tahmin_butonu = st.button("🚀 SATIŞLARI TAHMİN ET", use_container_width=True)
st.divider()

# 5. TAHMİN MOTORU VE SONUÇ EKRANI
if tahmin_butonu:
    with st.spinner('Yapay zeka pazar verilerini analiz ediyor...'):
        y_skor = miras_verisi.get(yayinci, 0)
        t_skor = potansiyel_verisi.get((yayinci, tur), 0)
        ea_sp = 1 if (yayinci == 'Electronic Arts' and tur == 'Sports') else 0
        tt_ac = 1 if (yayinci == 'Take-Two Interactive' and tur == 'Action') else 0
        act_sh = 1 if (yayinci == 'Activision' and tur == 'Shooter') else 0
        nin_mg = 1 if (yayinci == 'Nintendo' and tur in ['Platform', 'Role-Playing']) else 0
        is_dev = 1 if any(s in oyun_adi.lower() for s in dev_seriler) else 0
        is_exp = 1 if any(word in oyun_adi.lower() for word in ['expansion', 'pack', 'dlc', 'edition']) else 0

        girdi = pd.DataFrame([[y_skor, t_skor, is_exp, ea_sp, tt_ac, act_sh, nin_mg, is_dev]], columns=ozellikler_yeni)
        
        # --- KESİN ÇÖZÜM: .item() ile sayıları numpy dizisinden söküp alıyoruz ---
        na_ham = model_na.predict(girdi).item()
        eu_ham = model_eu.predict(girdi).item()
        jp_ham = model_jp.predict(girdi).item()
        
        na = float(max(0.01, np.expm1(na_ham)))
        eu = float(max(0.01, np.expm1(eu_ham)))
        jp = float(max(0.01, np.expm1(jp_ham)))
        
        # --- GELİŞTİRİLMİŞ UZMAN KURAL FİLTRESİ (HEURISTIC POST-PROCESSING) ---
        oyun_adi_kucuk = oyun_adi.lower()

        # 1. Platform ve Bölge İlişkisi
        if platform in ["Xbox Series X", "Xbox One", "X360", "XB"]:
            na *= 1.4      # Xbox Kuzey Amerika pazarında güçlüdür
            jp *= 0.05     # Xbox Japonya pazarında satmaz
        elif platform in ["PS5", "PS4", "PS3", "PS2", "PS"]:
            eu *= 1.35     # PlayStation Avrupa pazarında güçlüdür
        elif platform == "PC":
            na *= 0.7
            eu *= 0.8
            jp *= 0.05     # Japonya'da fiziksel PC oyun pazarı zayıftır
            if tur in ["Strategy", "Role-Playing"]:
                na *= 1.8  # Strateji ve MMO oyunları PC ortamında yüksek satar
                eu *= 1.6

        # 2. Yayıncı, Tür ve Marka İlişkisi
        # Electronic Arts (Spor Dinamikleri)
        if yayinci == "Electronic Arts" and tur == "Sports":
            if 'fifa' in oyun_adi_kucuk or 'fc' in oyun_adi_kucuk:
                eu *= 4.5
                na *= 1.2
            elif 'madden' in oyun_adi_kucuk:
                na *= 4.0
                eu *= 0.2
        
        # Japon Yayıncılar (JRPG ve Global Markalar)
        elif yayinci in ["Square Enix", "Capcom", "Nintendo"]:
            jp *= 2.5
            if 'resident evil' in oyun_adi_kucuk or 'final fantasy' in oyun_adi_kucuk:
                na *= 1.8
                eu *= 1.5

        # Activision ve Nişancı (Shooter) Dominasyonu
        elif yayinci == "Activision" and tur == "Shooter":
            if 'call of duty' in oyun_adi_kucuk or 'cod' in oyun_adi_kucuk:
                na *= 3.5
                eu *= 2.5
                jp *= 0.4

        # 3. Nintendo Exclusivity (Özel Oyun) Kontrolü
        if yayinci == "Nintendo":
            if "Nintendo" in platform or platform in ["Wii", "DS", "3DS", "SNES", "N64"]:
                jp *= 3.0
                na *= 1.5
            else:
                # Nintendo oyunları kendi konsolu dışında çıkmaz, satış verisi kısıtlanır
                na *= 0.01
                eu *= 0.01
                jp *= 0.01

        # 4. Küresel Dev Seri (Outlier) Kontrolü
        if 'gta' in oyun_adi_kucuk or 'grand theft auto' in oyun_adi_kucuk:
            na *= 4.0
            eu *= 4.5
            jp *= 1.5
            
        toplam = na + eu + jp
        lig = "YILDIZ" if toplam >= 3.0 else "HİT" if toplam >= 0.8 else "STANDART" if toplam >= 0.2 else "DÜŞÜK"

    # SONUÇLARI GÖSTERME (Metrikler)
    st.subheader("📊 Tahmin Raporu")
    kutu1, kutu2, kutu3, kutu4 = st.columns(4)
    kutu1.metric("🇺🇸 Kuzey Amerika", f"{na:.2f} M")
    kutu2.metric("🇪🇺 Avrupa", f"{eu:.2f} M")
    kutu3.metric("🇯🇵 Japonya", f"{jp:.2f} M")
    kutu4.metric("🌍 DÜNYA TOPLAMI", f"{toplam:.2f} M")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- SİBERPUNK DONUT (HALKA) GRAFİĞİ ---
    st.write("##### 📈 Bölgesel Satış Dağılımı")
    
    fig = go.Figure(data=[go.Pie(
        labels=["Kuzey Amerika", "Avrupa", "Japonya"],
        values=[na, eu, jp],
        hole=0.6, 
        marker=dict(colors=['#40E0D0', '#ff69b4', '#9370DB'], 
                    line=dict(color='#0a0a0f', width=3)), 
        textinfo='percent'
    )])

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=15),
        margin=dict(l=0, r=0, t=10, b=0),
        height=320,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    if lig == "YILDIZ": 
        st.success(f"🏆 LİG: {lig} PROJE - Mükemmel yatırım fırsatı! Global çapta devasa bir başarı bekleniyor.")
    elif lig == "HİT": 
        st.info(f"🎯 LİG: {lig} PROJE - Pazar payı çok yüksek, oldukça kârlı ve güvenli bir proje.")
    elif lig == "STANDART": 
        st.warning(f"⚖️ LİG: {lig} PROJE - Ortalama satış potansiyeli. Geliştirme bütçesine dikkat edilmeli.")
    else: 
        st.error(f"⚠️ LİG: {lig} PROJE - Riskli yatırım! Hedef kitle dar veya platform seçimi yanlış olabilir.")
