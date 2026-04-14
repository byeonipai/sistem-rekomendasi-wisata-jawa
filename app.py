import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Smart Travel Recommender",
    page_icon="🗺️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
.card {
    background: white;
    border: 1px solid #e9ecef;
    border-radius: 18px;
    padding: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.06);
    margin-bottom: 18px;
}
.card-title {
    font-size: 1.02rem;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 6px;
    color: #1f2937;
    line-height: 1.3;
    min-height: 42px;
}
.card-meta {
    font-size: 0.88rem;
    color: #4b5563;
    margin-bottom: 4px;
}
.card-score {
    display: inline-block;
    background: #eef6ff;
    color: #0b5ed7;
    border-radius: 999px;
    padding: 5px 10px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 8px;
}
.img-square {
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    border-radius: 14px;
    display: block;
    border: 1px solid #e5e7eb;
    background: #f3f4f6;
}
.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 0.5rem;
    margin-bottom: 0.8rem;
}
.small-muted {
    color: #6b7280;
    font-size: 0.9rem;
}
[data-testid="stSidebar"] {
    background: #f8fafc;
}
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e9ecef;
    border-radius: 16px;
    padding: 12px 14px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE
# =========================================================
if "clicked_lat" not in st.session_state:
    st.session_state.clicked_lat = -6.9
if "clicked_lon" not in st.session_state:
    st.session_state.clicked_lon = 107.6
if "hasil_pencarian" not in st.session_state:
    st.session_state.hasil_pencarian = None

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def format_rupiah(x):
    try:
        return f"Rp {int(float(x)):,.0f}".replace(",", ".")
    except:
        return "Rp 0"

def hitung_jarak(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def image_file_to_data_uri(path):
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lower().replace(".", "")
        if ext == "jpg":
            ext = "jpeg"
        if ext not in ["png", "jpeg", "webp"]:
            ext = "jpeg"
        return f"data:image/{ext};base64,{encoded}"
    except:
        return None

def get_image_source(row):
    local_path = str(row.get("path_gambar", "")).strip()
    url_img = str(row.get("gambar", "")).strip()

    if local_path and os.path.exists(local_path):
        data_uri = image_file_to_data_uri(local_path)
        if data_uri:
            return data_uri

    if url_img.startswith("http://") or url_img.startswith("https://"):
        return url_img

    return "https://placehold.co/600x600?text=No+Image"

def render_recommendation_card(row, use_location=False):
    img_src = get_image_source(row)

    jarak_html = ""
    if use_location and pd.notna(row.get("jarak_km", np.nan)):
        jarak_html = f"<div class='card-meta'>📍 Jarak: {row['jarak_km']:.1f} km</div>"

    card_html = f"""
    <div class="card">
        <img src="{img_src}" class="img-square"/>
        <div class="card-title">{row['nama wisata']}</div>
        <div class="card-meta">🏷️ {row['KATEGORI']}</div>
        <div class="card-meta">📌 {row['Kota']}</div>
        <div class="card-meta">💰 {format_rupiah(row['HARGA'])}</div>
        <div class="card-meta">⭐ {row['rating']:.1f}</div>
        {jarak_html}
        <div class="card-score">Skor {row['final_score']:.2f}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# =========================================================
# LOAD DATA & MODEL
# =========================================================
@st.cache_data
def load_data_and_model():
    try:
        df = pd.read_csv("DATASET_WISATA_READY_MODELING.csv")
    except FileNotFoundError:
        df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/DATASET_WISATA_READY_MODELING.csv")

    df["gambar"] = df["gambar"].fillna("")
    df["path_gambar"] = df["path_gambar"].fillna("")
    df["deskripsi"] = df["deskripsi"].fillna("")
    df["KATEGORI"] = df["KATEGORI"].fillna("")
    df["Kota"] = df["Kota"].fillna("")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df = df.dropna(subset=["lat", "long"]).copy()

    df["fitur_gabungan"] = (
        df["deskripsi"] + " " + df["KATEGORI"] + " " + df["Kota"]
    )

    df["fitur_gabungan"] = (
        df["fitur_gabungan"]
        .str.lower()
        .replace(r"[^\w\s]", " ", regex=True)
        .replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["fitur_gabungan"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return df, cosine_sim

df, cosine_sim = load_data_and_model()

# =========================================================
# RECOMMENDATION LOGIC
# =========================================================
def get_recommendations(
    ref_name,
    city,
    category,
    max_price,
    min_rating,
    use_location,
    user_lat,
    user_lon,
    max_radius,
    n=6
):
    data_kandidat = df.copy()

    if ref_name != "Tidak Ada":
        idx_list = df[df["nama wisata"] == ref_name].index
        if len(idx_list) > 0:
            idx = idx_list[0]
            data_kandidat["similarity_score"] = cosine_sim[idx]
            data_kandidat = data_kandidat[data_kandidat["nama wisata"] != ref_name]
        else:
            data_kandidat["similarity_score"] = 0.0
    else:
        data_kandidat["similarity_score"] = 0.0

    if city != "Semua Kota":
        data_kandidat = data_kandidat[data_kandidat["Kota"] == city]

    if category != "Semua Kategori":
        data_kandidat = data_kandidat[data_kandidat["KATEGORI"] == category]

    data_kandidat = data_kandidat[
        (data_kandidat["HARGA"] <= max_price) &
        (data_kandidat["rating"] >= min_rating)
    ]

    if use_location:
        data_kandidat["jarak_km"] = data_kandidat.apply(
            lambda r: hitung_jarak(user_lat, user_lon, r["lat"], r["long"]), axis=1
        )
        data_kandidat = data_kandidat[data_kandidat["jarak_km"] <= max_radius]

        if data_kandidat.empty:
            return pd.DataFrame()

        data_kandidat["distance_score"] = data_kandidat["jarak_km"].apply(
            lambda x: max(0, (max_radius - x) / max_radius)
        )
    else:
        data_kandidat["jarak_km"] = np.nan
        data_kandidat["distance_score"] = 0.0

    if data_kandidat.empty:
        return pd.DataFrame()

    if ref_name != "Tidak Ada":
        if use_location:
            skor = (
                (data_kandidat["similarity_score"] * 0.5) +
                (data_kandidat["distance_score"] * 0.3) +
                ((data_kandidat["rating"] / 5) * 0.2)
            )
        else:
            skor = (
                (data_kandidat["similarity_score"] * 0.7) +
                ((data_kandidat["rating"] / 5) * 0.3)
            )
    else:
        if use_location:
            skor = (
                (data_kandidat["distance_score"] * 0.6) +
                ((data_kandidat["rating"] / 5) * 0.4)
            )
        else:
            skor = data_kandidat["rating"] / 5

    data_kandidat["final_score"] = (skor * 100).round(2)

    return data_kandidat.sort_values("final_score", ascending=False).head(n)

# =========================================================
# HEADER
# =========================================================
st.title("🗺️ Smart Travel Recommender")
st.caption("Lihat persebaran semua destinasi dulu, lalu pilih titik lokasi untuk rekomendasi yang lebih personal.")

m1, m2, m3 = st.columns(3)
m1.metric("Total Destinasi", len(df))
m2.metric("Total Kota", df["Kota"].nunique())
m3.metric("Total Kategori", df["KATEGORI"].nunique())

# =========================================================
# PETA EKSPLORASI AWAL
# =========================================================
st.markdown("<div class='section-title'>Persebaran Semua Destinasi Wisata</div>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Peta ini menampilkan semua titik wisata di dataset. Aktifkan mode lokasi lalu klik peta untuk memilih titik Anda.</div>", unsafe_allow_html=True)

col_map, col_stats = st.columns([2.2, 1])

with col_map:
    center_lat = df["lat"].mean()
    center_lon = df["long"].mean()

    m_awal = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    marker_cluster = MarkerCluster().add_to(m_awal)

    for _, row in df.iterrows():
        popup_text = f"""
        <b>{row['nama wisata']}</b><br>
        Kota: {row['Kota']}<br>
        Kategori: {row['KATEGORI']}<br>
        Rating: {row['rating']:.1f}
        """
        folium.CircleMarker(
            location=[row["lat"], row["long"]],
            radius=4,
            color="#2563eb",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.75,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=row["nama wisata"]
        ).add_to(marker_cluster)

    if st.session_state.clicked_lat and st.session_state.clicked_lon:
        folium.Marker(
            [st.session_state.clicked_lat, st.session_state.clicked_lon],
            tooltip="Titik terpilih",
            popup="Titik terpilih",
            icon=folium.Icon(color="red", icon="map-marker")
        ).add_to(m_awal)

    map_awal = st_folium(m_awal, height=420, width=None, key="map_awal", returned_objects=["last_clicked"])

with col_stats:
    st.markdown("<div class='section-title'>Ringkasan Kota</div>", unsafe_allow_html=True)
    kota_counts = df["Kota"].value_counts().reset_index()
    kota_counts.columns = ["Kota", "Jumlah Destinasi"]
    st.table(kota_counts.head(10))
    st.caption("Tabel ini membantu user melihat kota mana yang datanya paling banyak.")

if map_awal and map_awal.get("last_clicked"):
    st.session_state.clicked_lat = map_awal["last_clicked"]["lat"]
    st.session_state.clicked_lon = map_awal["last_clicked"]["lng"]

st.success(
    f"Titik aktif saat ini: {st.session_state.clicked_lat:.6f}, {st.session_state.clicked_lon:.6f}"
)

st.markdown("---")

# =========================================================
# FILTER + HASIL
# =========================================================
col_sidebar, col_main = st.columns([1, 2.2])

with col_sidebar:
    st.markdown("<div class='section-title'>Atur Preferensi Wisata</div>", unsafe_allow_html=True)

    pilihan_wisata = st.selectbox(
        "Wisata Referensi",
        ["Tidak Ada"] + sorted(df["nama wisata"].unique().tolist())
    )

    pilihan_kota = st.selectbox(
        "Kota Tujuan",
        ["Semua Kota"] + sorted(df["Kota"].unique().tolist())
    )

    pilihan_kategori = st.selectbox(
        "Kategori",
        ["Semua Kategori"] + sorted(df["KATEGORI"].unique().tolist())
    )

    budget = st.slider("Budget Maksimal", 0, 500000, 200000, 10000)
    rating = st.slider("Rating Minimum", 1.0, 5.0, 4.0, 0.1)

    st.markdown("---")
    use_location = st.checkbox("Gunakan titik aktif sebagai lokasi user", value=False)

    if use_location:
        radius = st.slider("Radius pencarian (km)", 1, 100, 20)
        st.caption("Titik aktif diambil dari klik pada peta persebaran di atas.")
    else:
        radius = 0

    jumlah = st.slider("Jumlah rekomendasi", 3, 12, 6)

    if st.button("Cari Rekomendasi 🚀", use_container_width=True):
        st.session_state.hasil_pencarian = get_recommendations(
            pilihan_wisata,
            pilihan_kota,
            pilihan_kategori,
            budget,
            rating,
            use_location,
            st.session_state.clicked_lat,
            st.session_state.clicked_lon,
            radius,
            jumlah
        )

with col_main:
    if use_location:
        st.markdown("<div class='section-title'>Fokus Lokasi Pengguna</div>", unsafe_allow_html=True)

        m_fokus = folium.Map(
            location=[st.session_state.clicked_lat, st.session_state.clicked_lon],
            zoom_start=12
        )

        folium.Marker(
            [st.session_state.clicked_lat, st.session_state.clicked_lon],
            tooltip="Lokasi Anda",
            popup="Lokasi Anda",
            icon=folium.Icon(color="red", icon="user")
        ).add_to(m_fokus)

        folium.Circle(
            radius=radius * 1000,
            location=[st.session_state.clicked_lat, st.session_state.clicked_lon],
            color="blue",
            fill=True,
            fill_opacity=0.08
        ).add_to(m_fokus)

        st_folium(m_fokus, height=280, width=None, key="map_focus", returned_objects=[])

    if st.session_state.hasil_pencarian is not None:
        hasil = st.session_state.hasil_pencarian

        st.markdown("<div class='section-title'>Hasil Rekomendasi</div>", unsafe_allow_html=True)

        if hasil.empty:
            st.warning("Tidak ditemukan destinasi yang sesuai dengan filter yang dipilih.")
        else:
            a, b, c = st.columns(3)
            a.metric("Rekomendasi Ditemukan", len(hasil))
            b.metric("Top Rating", f"{hasil['rating'].max():.1f}")
            c.metric("Skor Tertinggi", f"{hasil['final_score'].max():.2f}")

            st.markdown("<div class='small-muted'>Semua gambar rekomendasi disamakan menjadi format persegi agar lebih rapi.</div>", unsafe_allow_html=True)

            cols = st.columns(3)
            for i, (_, row) in enumerate(hasil.iterrows()):
                with cols[i % 3]:
                    render_recommendation_card(row, use_location=use_location)

            st.markdown("<div class='section-title'>Tabel Ringkasan</div>", unsafe_allow_html=True)

            hasil_tampil = hasil.copy()
            hasil_tampil["HARGA"] = hasil_tampil["HARGA"].apply(format_rupiah)
            if use_location:
                hasil_tampil["jarak_km"] = hasil_tampil["jarak_km"].round(2)

            kolom = ["nama wisata", "KATEGORI", "Kota", "HARGA", "rating", "final_score"]
            if use_location:
                kolom.insert(5, "jarak_km")

            st.table(hasil_tampil[kolom])

            st.markdown("<div class='section-title'>Peta Hasil Rekomendasi</div>", unsafe_allow_html=True)

            center_lat = hasil.iloc[0]["lat"]
            center_lon = hasil.iloc[0]["long"]

            if use_location:
                center_lat = st.session_state.clicked_lat
                center_lon = st.session_state.clicked_lon

            m_res = folium.Map(location=[center_lat, center_lon], zoom_start=11)

            if use_location:
                folium.Marker(
                    [st.session_state.clicked_lat, st.session_state.clicked_lon],
                    tooltip="Lokasi Anda",
                    popup="Lokasi Anda",
                    icon=folium.Icon(color="red", icon="user")
                ).add_to(m_res)

                folium.Circle(
                    radius=radius * 1000,
                    location=[st.session_state.clicked_lat, st.session_state.clicked_lon],
                    color="blue",
                    fill=True,
                    fill_opacity=0.08
                ).add_to(m_res)

            for _, r in hasil.iterrows():
                popup_text = f"""
                <b>{r['nama wisata']}</b><br>
                Kategori: {r['KATEGORI']}<br>
                Kota: {r['Kota']}<br>
                Harga: {format_rupiah(r['HARGA'])}<br>
                Rating: {r['rating']:.1f}<br>
                Skor: {r['final_score']:.2f}
                """
                if use_location and pd.notna(r["jarak_km"]):
                    popup_text += f"<br>Jarak: {r['jarak_km']:.1f} km"

                folium.Marker(
                    [r["lat"], r["long"]],
                    popup=folium.Popup(popup_text, max_width=260),
                    tooltip=r["nama wisata"],
                    icon=folium.Icon(color="blue", icon="info-sign")
                ).add_to(m_res)

            st_folium(m_res, height=460, width=None, key="map_res", returned_objects=[])
