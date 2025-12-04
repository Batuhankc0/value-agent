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

# --- 2. MODELİ YÜKLEME ---
@st.cache_resource
def model_yukle():
    model = xgb.XGBRegressor()
    try:
        model.load_model("ev_fiyat_modeli.json")
    except:
        st.error("HATA: 'ev_fiyat_modeli.json' bulunamadı.")
        return None
    return model

model = model_yukle()

# --- 3. BAŞLIK ---
st.title("🏠 Yapay Zeka Destekli Emlak Değerleme")
st.markdown("---")

# --- 4. SOL PANEL ---
with st.sidebar:
    st.header("Ev Özellikleri")
    
    # --- ADRES VE KOORDİNAT SEÇİMİ ---
    girdi_yontemi = st.radio("Konum Giriş Yöntemi:", ["Adres İle", "Manuel Koordinat"])
    
    lat, lon = 51.5074, -0.1278 # Varsayılan (Londra)
    adres_metni = "Bilinmiyor"

    if girdi_yontemi == "Adres İle":
        adres_girisi = st.text_input("📍 Adres / Posta Kodu", value="HA3 5NE")
        st.caption("Örnek: HA3 5NE veya Oxford Street")
    else:
        st.warning("Harita servisi çalışmazsa burayı kullanın.")
        lat = st.number_input("Enlem (Latitude)", value=51.5074, format="%.4f")
        lon = st.number_input("Boylam (Longitude)", value=-0.1278, format="%.4f")

    st.markdown("---")
    
    sq_ft = st.number_input("📏 Büyüklük (Square Feet)", min_value=100, value=900, step=10)
    metrekare = sq_ft / 10.764
    st.info(f"Yaklaşık: **{metrekare:.2f} m²**")
    
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
if hesapla_btn and model:
    
    # Eğer Adres seçildiyse koordinatları bulmaya çalış
    if girdi_yontemi == "Adres İle":
        # Rastgele User-Agent oluştur (Blocklanmayı azaltmak için)
        ua = f"emlak_app_user_{random.randint(1000, 99999)}"
        geolocator = Nominatim(user_agent=ua)
        
        try:
            with st.spinner("Adres haritada aranıyor..."):
                location = geolocator.geocode(adres_girisi, timeout=5)
                
                if location:
                    lat = location.latitude
                    lon = location.longitude
                    adres_metni = location.address
                    st.success("✅ Adres bulundu!")
                else:
                    st.error("❌ Adres bulunamadı! Lütfen 'Manuel Koordinat' seçeneğini kullanın.")
                    st.stop()
        except Exception as e:
            st.error(f"⚠️ Harita servisine erişilemedi ({e}).")
            st.warning("👉 Lütfen sol menüden **'Manuel Koordinat'** seçeneğini seçip koordinatları elle girin.")
            st.stop()
    else:
        adres_metni = f"Manuel Koordinat ({lat}, {lon})"

    # --- TAHMİN İŞLEMİ ---
    input_data = pd.DataFrame({
        'bedrooms': [oda],
        'bathrooms': [banyo],
        'floorAreaSqM': [metrekare],
        'latitude': [lat],
        'longitude': [lon],
        'propertyType': [ev_tipi], 
        'tenure': [1],             
        'currentEnergyRating': [2] 
    })
    
    tahmin = model.predict(input_data)[0]
    
    # --- SONUÇLAR ---
    col_sonuc, col_grafik = st.columns([1, 2])
    
    with col_sonuc:
        st.subheader("Tahmini Değer")
        st.metric(label="", value=f"£{tahmin:,.0f}")
        st.info(f"📍 **Konum:** {adres_metni.split(',')[0]}")
        
        # Harita
        map_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
        st.map(map_df, zoom=13)

    with col_grafik:
        st.subheader("📊 Fiyat Analizi")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)
        
        # Grafik Verileri
        feature_names = ["Oda", "Banyo", f"Alan ({metrekare:.0f}m²)", "Enlem", "Boylam", "Tip", "Mülkiyet", "Enerji"]
        values = shap_values[0].values
        
        df_shap = pd.DataFrame({"Özellik": feature_names, "Etki": values})
        df_shap["Mutlak"] = df_shap["Etki"].abs()
        df_shap = df_shap.sort_values("Mutlak", ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_shap["Etki"]]
        bars = ax.barh(df_shap["Özellik"], df_shap["Etki"], color=colors)
        ax.axvline(0, color='black', linewidth=0.5)
        
        # X ekseni formatı
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"£{x/1000:.0f}k"))
        
        # Etiketleri ekle
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + (5000 if width > 0 else -5000)
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f"£{width:,.0f}", va='center')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

else:
    st.info("👈 Tahmin için sol menüyü kullanın.")