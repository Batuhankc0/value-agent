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

# --- 2. MODELİ YÜKLEME (ÖNBELLEKLİ) ---
@st.cache_resource
def model_yukle():
    model = xgb.XGBRegressor()
    try:
        # Model dosyasının aynı klasörde olduğundan emin olun
        model.load_model("ev_fiyat_modeli.json")
    except Exception as e:
        st.error(f"HATA: 'ev_fiyat_modeli.json' dosyası bulunamadı veya yüklenemedi.\nDetay: {e}")
        return None
    return model

model = model_yukle()

# --- 3. BAŞLIK ---
st.title("🏠 Yapay Zeka Destekli Emlak Değerleme")
st.markdown("""
Bu uygulama, **XGBoost** algoritması kullanarak İngiltere emlak piyasasındaki evlerin değerini tahmin eder 
ve fiyatı etkileyen faktörleri **SHAP (XAI)** analizi ile açıklar.
""")
st.markdown("---")

# --- 4. SOL PANEL (GİRİŞLER) ---
with st.sidebar:
    st.header("📍 Konum Seçimi")
    
    # Kullanıcıya 3 farklı giriş yöntemi sunuyoruz
    giris_yontemi = st.radio(
        "Konum belirleme yöntemi:", 
        ["Listeden Bölge Seçerek (Önerilen)", "Adres Yazarak", "Manuel Koordinat"]
    )
    
    # Varsayılan Değerler
    lat, lon = 51.5074, -0.1278 
    adres_metni = "Bilinmiyor"

    # --- YÖNTEM 1: LİSTEDEN SEÇME (En Güvenli Sunum Yöntemi) ---
    if giris_yontemi == "Listeden Bölge Seçerek":
        
        # Genişletilmiş Bölge Veritabanı
        bolge_verisi = {
            # --- MERKEZ LONDRA (PAHALI) ---
            "Mayfair (Merkez - Elit)": {"lat": 51.5079, "lon": -0.1466, "desc": "Mayfair, Central London"},
            "Kensington (Merkez - Lüks)": {"lat": 51.5014, "lon": -0.1919, "desc": "Kensington, Central London"},
            "Chelsea (Merkez - Popüler)": {"lat": 51.4875, "lon": -0.1682, "desc": "Chelsea, London"},
            "City of London (Finans Merkezi)": {"lat": 51.5123, "lon": -0.0909, "desc": "The City, Financial District"},
            
            # --- BATI LONDRA ---
            "Notting Hill (Batı - Turistik)": {"lat": 51.5091, "lon": -0.2040, "desc": "Notting Hill, West London"},
            "Ealing (Batı - Aile Yerleşimi)": {"lat": 51.5130, "lon": -0.3042, "desc": "Ealing Broadway"},
            "Richmond (Güney Batı - Yeşil Alan)": {"lat": 51.4613, "lon": -0.3037, "desc": "Richmond upon Thames"},
            
            # --- KUZEY LONDRA ---
            "Camden Town (Kuzey - Eğlence)": {"lat": 51.5390, "lon": -0.1426, "desc": "Camden Town, North London"},
            "Hampstead (Kuzey - Lüks Köy Havası)": {"lat": 51.5541, "lon": -0.1744, "desc": "Hampstead Village"},
            "Harrow (Kuzey Batı - Banliyö)": {"lat": 51.5898, "lon": -0.3346, "desc": "Harrow, Greater London"},
            "Wembley (Kuzey Batı - Stadyum)": {"lat": 51.5505, "lon": -0.3048, "desc": "Wembley Park"},

            # --- DOĞU LONDRA ---
            "Canary Wharf (Doğu - Gökdelenler)": {"lat": 51.5048, "lon": -0.0190, "desc": "Canary Wharf, Docklands"},
            "Stratford (Doğu - Olimpiyat Köyü)": {"lat": 51.5423, "lon": -0.0026, "desc": "Stratford, East London"},
            "Shoreditch (Doğu - Genç & Sanat)": {"lat": 51.5233, "lon": -0.0782, "desc": "Shoreditch, East London"},

            # --- GÜNEY LONDRA ---
            "Wimbledon (Güney - Tenis & Lüks)": {"lat": 51.4214, "lon": -0.2067, "desc": "Wimbledon Village"},
            "Greenwich (Güney Doğu - Tarihi)": {"lat": 51.4816, "lon": -0.0064, "desc": "Greenwich, South East London"},
            "Brixton (Güney - Canlı Kültür)": {"lat": 51.4613, "lon": -0.1156, "desc": "Brixton, South London"},
            "Croydon (Güney - Uygun Fiyatlı)": {"lat": 51.3762, "lon": -0.0982, "desc": "Croydon, South London"},
        }

        secilen_bolge_ismi = st.selectbox("🗺️ Bir Bölge Seçin:", list(bolge_verisi.keys()))
        
        # Seçimi uygula
        secim = bolge_verisi[secilen_bolge_ismi]
        lat = secim["lat"]
        lon = secim["lon"]
        adres_metni = secim["desc"]
        st.success(f"✅ Konum: {adres_metni}")

    # --- YÖNTEM 2: ADRES YAZARAK (API Kullanır - Riskli olabilir) ---
    elif giris_yontemi == "Adres Yazarak":
        adres_girisi = st.text_input("🏠 Adres / Posta Kodu", value="173 Carmelite Road, Harrow")
        st.caption("Örn: HA3 5NE veya Oxford Street")
        # Not: API çağrısı 'Hesapla' butonuna basılınca yapılır.

    # --- YÖNTEM 3: MANUEL ---
    else:
        st.info("Google Maps'ten aldığınız koordinatları girebilirsiniz.")
        lat = st.number_input("Enlem (Latitude)", value=51.5074, format="%.4f")
        lon = st.number_input("Boylam (Longitude)", value=-0.1278, format="%.4f")
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

# --- 5. HESAPLAMA VE GÖRSELLEŞTİRME ---
if hesapla_btn and model:
    
    # "Adres Yazarak" seçildiyse burada API çağrısı yap
    if giris_yontemi == "Adres Yazarak":
        ua = f"emlak_demo_{random.randint(1000, 9999)}"
        geolocator = Nominatim(user_agent=ua)
        
        try:
            with st.spinner("Adres servisinden konum alınıyor..."):
                location = geolocator.geocode(adres_girisi, timeout=3)
                
                if location:
                    lat = location.latitude
                    lon = location.longitude
                    adres_metni = location.address
                    st.success("✅ Adres başarıyla bulundu!")
                
                # --- SUNUM KURTARICI (FAIL-SAFE) ---
                # Adres bulunamazsa ama içinde "Carmelite" veya "Harrow" varsa çökmesin
                elif "carmelite" in adres_girisi.lower() or "harrow" in adres_girisi.lower():
                    lat, lon = 51.6013, -0.3504
                    adres_metni = "Carmelite Road, Harrow (Çevrimdışı Mod)"
                    st.warning(f"⚠️ Harita servisi yanıt vermedi, '{adres_metni}' için yedek koordinatlar kullanılıyor.")
                
                else:
                    st.error("❌ Adres bulunamadı!")
                    st.info("👉 Lütfen sol menüden **'Listeden Bölge Seçerek'** seçeneğini kullanın.")
                    st.stop()
                    
        except Exception as e:
            # API hatası olursa (Render'da sık olur)
            if "carmelite" in adres_girisi.lower() or "harrow" in adres_girisi.lower():
                lat, lon = 51.6013, -0.3504
                adres_metni = "Carmelite Road, Harrow (Çevrimdışı Mod)"
                st.warning("⚠️ Harita servisine bağlanılamadı, demo koordinatları kullanılıyor.")
            else:
                st.error(f"⚠️ Bağlantı Hatası: {e}")
                st.info("👉 Lütfen sol menüden **'Listeden Bölge Seçerek'** seçeneğini kullanın.")
                st.stop()

    # --- VERİ HAZIRLIĞI ---
    # Modelin beklediği sütun sırasına dikkat edin
    input_data = pd.DataFrame({
        'bedrooms': [oda],
        'bathrooms': [banyo],
        'floorAreaSqM': [metrekare],
        'latitude': [lat],
        'longitude': [lon],
        'propertyType': [ev_tipi], 
        'tenure': [1],             # Varsayılan: Leasehold
        'currentEnergyRating': [2] # Varsayılan: C
    })
    
    # --- TAHMİN ---
    tahmin = model.predict(input_data)[0]
    
    # --- EKRAN TASARIMI ---
    col_sonuc, col_grafik = st.columns([1, 2])
    
    with col_sonuc:
        st.subheader("Tahmini Değer")
        st.metric(label="Piyasa Değeri", value=f"£{tahmin:,.0f}")
        
        st.info(f"📍 **Konum:**\n{adres_metni.split(',')[0]}")
        
        # Harita
        map_df = pd.DataFrame({'lat': [lat], 'lon': [lon]})
        st.map(map_df, zoom=12)

    with col_grafik:
        st.subheader("📊 Fiyatı Etkileyen Faktörler")
        
        # SHAP Analizi
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)
        
        # Grafik için veri hazırlığı
        feature_names = ["Oda Sayısı", "Banyo", f"Alan ({metrekare:.0f}m²)", "Enlem", "Boylam", "Ev Tipi", "Mülkiyet", "Enerji"]
        values = shap_values[0].values
        
        df_shap = pd.DataFrame({"Özellik": feature_names, "Etki": values})
        df_shap["Mutlak"] = df_shap["Etki"].abs()
        df_shap = df_shap.sort_values("Mutlak", ascending=True)
        
        # Grafik Çizimi
        fig, ax = plt.subplots(figsize=(8, 5))
        # Pozitif etkiler yeşil, negatif etkiler kırmızı
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_shap["Etki"]]
        
        bars = ax.barh(df_shap["Özellik"], df_shap["Etki"], color=colors)
        ax.axvline(0, color='black', linewidth=0.5)
        
        # X eksenini £ formatına çevir
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"£{x/1000:.0f}k"))
        
        # Barların ucuna değerleri yaz
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + (5000 if width > 0 else -5000)
            align = 'left' if width > 0 else 'right'
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f"£{width:,.0f}", va='center', ha=align, fontsize=9)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("Grafik: Özelliklerin taban fiyata ne kadar (+/-) etki ettiğini gösterir.")

else:
    if not model:
        st.warning("Model dosyası bulunamadı.")
    else:
        st.info("👈 Fiyat tahmini için sol menüden seçim yapın ve butona basın.")