import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from geopy.geocoders import Nominatim
import random

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="Emlak Fiyat Tahminleyicisi", 
    page_icon="🏠", 
    layout="wide"
)

# --- 2. PERFORMANS İÇİN ÖNBELLEKLEME (CACHE) ---
@st.cache_resource
def kaynaklari_yukle():
    # Modeli Yükle
    model = xgb.XGBRegressor()
    try:
        model.load_model("ev_fiyat_modeli.json")
    except:
        return None, None
    
    # SHAP Açıklayıcıyı (En ağır işlem) hafızaya al
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = kaynaklari_yukle()

if not model:
    st.error("⚠️ Model dosyası (ev_fiyat_modeli.json) bulunamadı!")
    st.stop()

# --- 3. BAŞLIK ---
st.title("🏠 Yapay Zeka Destekli Emlak Değerleme")
st.markdown("---")

# --- 4. SOL PANEL ---
with st.sidebar:
    st.header("📍 Konum Seçimi")
    
    # Varsayılan seçim "Listeden Seç" olsun
    giris_yontemi = st.radio(
        "Konum belirleme yöntemi:", 
        ["Listeden Bölge Seçerek (Önerilen)", "Adres Yazarak", "Manuel Koordinat"]
    )
    
    lat, lon = 51.5074, -0.1278 
    adres_metni = "Bilinmiyor"

    # --- SEÇENEK 1: LİSTE (HIZLI VE GÜVENLİ) ---
    if giris_yontemi == "Listeden Bölge Seçerek (Önerilen)":
        bolge_verisi = {
            "Harrow (Banliyö)": {"lat": 51.5898, "lon": -0.3346, "desc": "Harrow, Greater London"},
            "Wembley (Stadyum)": {"lat": 51.5505, "lon": -0.3048, "desc": "Wembley Park"},
            "Mayfair (Merkez - Elit)": {"lat": 51.5079, "lon": -0.1466, "desc": "Mayfair, Central London"},
            "Kensington (Merkez - Lüks)": {"lat": 51.5014, "lon": -0.1919, "desc": "Kensington, Central London"},
            "Chelsea (Merkez - Popüler)": {"lat": 51.4875, "lon": -0.1682, "desc": "Chelsea, London"},
            "City of London (Finans)": {"lat": 51.5123, "lon": -0.0909, "desc": "The City, Financial District"},
            "Notting Hill (Batı)": {"lat": 51.5091, "lon": -0.2040, "desc": "Notting Hill, West London"},
            "Ealing (Batı - Aile)": {"lat": 51.5130, "lon": -0.3042, "desc": "Ealing Broadway"},
            "Richmond (Yeşil Alan)": {"lat": 51.4613, "lon": -0.3037, "desc": "Richmond upon Thames"},
            "Camden Town (Kuzey)": {"lat": 51.5390, "lon": -0.1426, "desc": "Camden Town, North London"},
            "Canary Wharf (Doğu)": {"lat": 51.5048, "lon": -0.0190, "desc": "Canary Wharf, Docklands"},
            "Stratford (Olimpiyat Köyü)": {"lat": 51.5423, "lon": -0.0026, "desc": "Stratford, East London"},
            "Wimbledon (Güney)": {"lat": 51.4214, "lon": -0.2067, "desc": "Wimbledon Village"},
            "Croydon (Güney - Ucuz)": {"lat": 51.3762, "lon": -0.0982, "desc": "Croydon, South London"},
        }

        secilen_bolge = st.selectbox("🗺️ Bir Bölge Seçin:", list(bolge_verisi.keys()))
        secim = bolge_verisi[secilen_bolge]
        lat, lon, adres_metni = secim["lat"], secim["lon"], secim["desc"]
        st.success(f"✅ Seçildi: {adres_metni}")

    # --- SEÇENEK 2: ADRES ---
    elif giris_yontemi == "Adres Yazarak":
        adres_girisi = st.text_input("🏠 Adres / Posta Kodu", value="HA3 5NE")
        st.warning("⚠️ Not: Harita servisi yavaş olabilir.")

    # --- SEÇENEK 3: MANUEL ---
    else:
        st.info("Google Maps koordinatlarını girin.")
        lat = st.number_input("Enlem", value=51.5074, format="%.4f")
        lon = st.number_input("Boylam", value=-0.1278, format="%.4f")
        adres_metni = f"Özel Konum ({lat}, {lon})"

    st.markdown("---")
    st.header("Ev Özellikleri")
    
    sq_ft = st.number_input("📏 Büyüklük (Square Feet)", min_value=100, value=860, step=10)
    metrekare = sq_ft / 10.764
    st.caption(f"Yaklaşık: **{metrekare:.2f} m²**")
    
    col1, col2 = st.columns(2)
    with col1:
        oda = st.number_input("🛏️ Oda", min_value=1, max_value=10, value=3)
    with col2:
        banyo = st.number_input("🛁 Banyo", min_value=1, max_value=5, value=1)
        
    ev_tipi_secim = st.selectbox("🏠 Ev Tipi", ["Bilinmiyor", "Daire", "Müstakil", "Sıralı Ev"])
    tip_map = {"Bilinmiyor": 0, "Daire": 1, "Müstakil": 2, "Sıralı Ev": 3}
    ev_tipi = tip_map[ev_tipi_secim]

    hesapla_btn = st.button("💰 Fiyatı Hesapla", type="primary")

# --- 5. HESAPLAMA ---
if hesapla_btn:
    
    # Adres seçiliyse API çağrısı (Sadece butona basınca)
    if giris_yontemi == "Adres Yazarak":
        ua = f"emlak_demo_{random.randint(1000, 9999)}"
        geolocator = Nominatim(user_agent=ua)
        try:
            with st.spinner("Adres aranıyor..."):
                location = geolocator.geocode(adres_girisi, timeout=3)
                if location:
                    lat, lon, adres_metni = location.latitude, location.longitude, location.address
                    st.success("✅ Adres bulundu!")
                elif "carmelite" in adres_girisi.lower(): # Fail-safe
                    lat, lon, adres_metni = 51.6013, -0.3504, "Carmelite Road (Yedek)"
                    st.warning("⚠️ Yedek koordinatlar kullanıldı.")
                else:
                    st.error("Adres bulunamadı.")
                    st.stop()
        except:
             # Hata durumunda Harrow'u varsayılan al (Sunum kurtarıcı)
            lat, lon, adres_metni = 51.5898, -0.3346, "Harrow (Yedek Konum)"
            st.error("Harita servisi meşgul, varsayılan konum kullanıldı.")

    # Tahmin Verisi
    input_data = pd.DataFrame({
        'bedrooms': [oda], 'bathrooms': [banyo], 'floorAreaSqM': [metrekare],
        'latitude': [lat], 'longitude': [lon], 'propertyType': [ev_tipi], 
        'tenure': [1], 'currentEnergyRating': [2] 
    })
    
    # Tahmin ve SHAP
    tahmin = model.predict(input_data)[0]
    shap_values = explainer(input_data)

    # --- SONUÇ EKRANI ---
    col_sonuc, col_grafik = st.columns([1, 2])
    
    with col_sonuc:
        st.subheader("Tahmini Değer")
        st.metric(label="", value=f"£{tahmin:,.0f}")
        st.info(f"📍 {adres_metni.split(',')[0]}")
        st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=12)

    with col_grafik:
        st.subheader("📊 Fiyat Analizi")
        # Grafik verisi
        feature_names = ["Oda", "Banyo", f"Alan ({metrekare:.0f}m²)", "Enlem", "Boylam", "Tip", "Mülkiyet", "Enerji"]
        values = shap_values[0].values
        df_shap = pd.DataFrame({"Özellik": feature_names, "Etki": values}).sort_values("Etki", key=abs)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_shap["Etki"]]
        bars = ax.barh(df_shap["Özellik"], df_shap["Etki"], color=colors)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"£{x/1000:.0f}k"))
        
        for bar in bars:
            width = bar.get_width()
            align = 'left' if width > 0 else 'right'
            ax.text(width + (5000 if width > 0 else -5000), bar.get_y() + bar.get_height()/2, 
                    f"£{width:,.0f}", va='center', ha=align, fontsize=9)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("👈 Sol menüden bir bölge seçin ve 'Hesapla' butonuna basın.")