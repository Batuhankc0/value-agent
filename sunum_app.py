import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from geopy.geocoders import Nominatim

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="Emlak Fiyat Tahminleyicisi", 
    page_icon="🏠", 
    layout="wide"
)

# --- 2. MODELİ YÜKLEME (ÖNBELLEKLİ) ---
@st.cache_resource
def model_yukle():
    # Model dosyasının proje klasöründe olduğundan emin olun
    model = xgb.XGBRegressor()
    try:
        model.load_model("ev_fiyat_modeli.json")
    except:
        st.error("HATA: 'ev_fiyat_modeli.json' dosyası bulunamadı. Lütfen önce eğitimi tamamlayın.")
        return None
    return model

model = model_yukle()

# --- 3. ARAYÜZ BAŞLIKLARI ---
st.title("🏠 Yapay Zeka Destekli Emlak Değerleme")
st.markdown("""
Bu uygulama, **XGBoost** makine öğrenmesi algoritmasını kullanarak İngiltere'deki evlerin 
satış fiyatını tahmin eder ve fiyatı etkileyen faktörleri **SHAP** analizi ile açıklar.
""")
st.markdown("---")

# --- 4. SOL PANEL (KULLANICI GİRİŞLERİ) ---
with st.sidebar:
    st.header("Ev Özelliklerini Girin")
    
    # Adres Girişi
    adres_girisi = st.text_input("📍 Adres veya Posta Kodu", value="173 Carmelite Road, Harrow")
    st.caption("Örnek: HA3 5NE veya Oxford Street, London")
    
    # Büyüklük Girişi (Square Feet -> m2 çevrimi)
    sq_ft = st.number_input("📏 Büyüklük (Square Feet)", min_value=100, value=900, step=10)
    metrekare = sq_ft / 10.764
    st.info(f"Yaklaşık: **{metrekare:.2f} m²**")
    
    col1, col2 = st.columns(2)
    with col1:
        oda = st.number_input("🛏️ Oda", min_value=1, max_value=10, value=3)
    with col2:
        banyo = st.number_input("🛁 Banyo", min_value=1, max_value=5, value=1)
        
    # Ev Tipi Seçimi (Sayısal kodlamaya uygun)
    ev_tipi_secim = st.selectbox(
        "🏠 Ev Tipi", 
        ["Bilinmiyor/Diğer", "Daire (Flat)", "Müstakil (Detached)", "Sıralı Ev (Terraced)"]
    )
    # Modelin anladığı dile çevir (0, 1, 2, 3)
    tip_map = {
        "Bilinmiyor/Diğer": 0,
        "Daire (Flat)": 1, 
        "Müstakil (Detached)": 2, 
        "Sıralı Ev (Terraced)": 3
    }
    ev_tipi = tip_map[ev_tipi_secim]

    hesapla_btn = st.button("💰 Fiyatı Hesapla", type="primary")

# --- 5. HESAPLAMA VE SONUÇLAR ---
if hesapla_btn and model:
    
    # --- GEOCODING (ADRES -> KOORDİNAT) ---
    geolocator = Nominatim(user_agent="sunum_app_v3")
    location = None
    
    try:
        # İlk deneme
        location = geolocator.geocode(adres_girisi, timeout=10)
        
        # Bulunamazsa 'Middlesex' gibi eski terimleri temizleyip tekrar dene
        if location is None and "Middlesex" in adres_girisi:
            temiz_adres = adres_girisi.replace("Middlesex", "").strip()
            location = geolocator.geocode(temiz_adres, timeout=10)
            
    except Exception as e:
        st.error(f"Harita servisine bağlanılamadı: {e}")

    if location:
        # --- VERİYİ HAZIRLA ---
        # Sütun sırası eğitimdekiyle AYNI olmalı
        input_data = pd.DataFrame({
            'bedrooms': [oda],
            'bathrooms': [banyo],
            'floorAreaSqM': [metrekare],
            'latitude': [location.latitude],
            'longitude': [location.longitude],
            'propertyType': [ev_tipi], 
            'tenure': [1],             # Varsayılan: Leasehold
            'currentEnergyRating': [2] # Varsayılan: C Sınıfı
        })
        
        # --- TAHMİN YAP ---
        tahmin = model.predict(input_data)[0]
        
        # --- SONUÇ EKRANI ---
        col_sonuc, col_grafik = st.columns([1, 2])
        
        with col_sonuc:
            st.subheader("Tahmini Satış Fiyatı")
            st.metric(label="", value=f"£{tahmin:,.0f}")
            
            st.success(f"📍 Konum Bulundu:\n{location.address.split(',')[0]}, {location.address.split(',')[-2]}")
            
            # Harita Gösterimi
            map_df = pd.DataFrame({'lat': [location.latitude], 'lon': [location.longitude]})
            st.map(map_df, zoom=13)

        with col_grafik:
            st.subheader("📊 Fiyatın Matematiği")
            
            # --- SHAP HESAPLAMA ---
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_data)
            
            # Değerleri Hazırla
            feature_names = [
                "Oda Sayısı", "Banyo Sayısı", 
                f"Alan ({input_data['floorAreaSqM'].values[0]:.0f} m²)",    
                "Konum", "Boylam", "Ev Tipi", "Mülkiyet", "Enerji"     
            ]
            values = shap_values[0].values
            
            # --- HESAPLAMA KISMI (MATEMATİKSEL KANIT) ---
            base_value = shap_values[0].base_values # Ortalama Fiyat
            total_impact = values.sum()             # Barların Toplamı
            final_pred = base_value + total_impact  # Sonuç
            
            # --- GRAFİK VERİSİ ---
            df_shap = pd.DataFrame({"Özellik": feature_names, "Etki": values})
            df_shap["Mutlak"] = df_shap["Etki"].abs()
            df_shap = df_shap.sort_values("Mutlak", ascending=True)
            renkler = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_shap["Etki"]]

            # --- GRAFİK ÇİZİMİ ---
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(df_shap["Özellik"], df_shap["Etki"], color=renkler)
            ax.axvline(0, color='black', linewidth=0.8)
            
            # X Ekseni Formatı
            def currency_formatter(x, pos):
                return f"£{x/1000:.0f}k"
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(currency_formatter))
            
            # Değerleri Yazdır
            for bar in bars:
                width = bar.get_width()
                align = 'left' if width > 0 else 'right'
                offset = 5000 if width > 0 else -5000
                ax.text(width + offset, bar.get_y() + bar.get_height()/2, 
                        f"£{width:,.0f}", va='center', ha=align, fontsize=10, fontweight='bold')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

            # --- HESAP ÖZETİ KUTUSU (YENİ EKLENEN KISIM) ---
            st.info(f"""
            **🧮 Fiyat Nasıl Hesaplandı?**
            
            Model, hesaplamaya **Piyasa Ortalaması** ile başlar ve özelliklere göre ekleme/çıkarma yapar:
            
            | Kalem | Değer |
            | :--- | :--- |
            | **Başlangıç (Ortalama Fiyat):** | **£{base_value:,.0f}** |
            | + Özelliklerin Etkisi (Barlar): | £{total_impact:,.0f} |
            | **= SONUÇ FİYAT:** | **£{final_pred:,.0f}** |
            """)
    else:
        st.error("❌ Adres bulunamadı! Lütfen sadece 'Posta Kodu' (Örn: HA3 5NE) girmeyi deneyin.")
else:
    if not model:
        st.warning("Lütfen önce modeli eğitip kaydedin.")
    else:
        st.info("👈 Tahmin yapmak için sol menüden özellikleri girip butona basın.")