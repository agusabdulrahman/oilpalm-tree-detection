import os
import time

import cv2
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree
from shapely.geometry import Point
from streamlit_folium import st_folium
from ultralytics import YOLO

import streamlit as st

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Oil Palm AI Intelligence Platform", page_icon="🌴", layout="wide"
)

st.title("🌴 Oil Palm Tree Detection & Geospatial Analytics Dashboard")
st.markdown("Upload citra kebun sawit secara manual")

# Sidebar untuk Kontrol Parameter & Upload
with st.sidebar:
    st.header("⚙️ Analysis Control Panel")

    st.subheader("1. Upload Data & Model")

    # Widget Manual Upload File
    uploaded_file = st.file_uploader(
        "Upload Citra Raster (.tif) / Gambar",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
    )

    model_path = st.text_input(
        "Path Model YOLO",
        value="Model/oilpalm_detection_model_yolov26n.pt",
    )
    confidence_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)

    st.subheader("2. Post-Processing (KD-Tree)")
    min_distance = st.slider(
        "Min. Tree Separation Distance (meters)", 1.0, 10.0, 5.0, 0.5
    )

    run_btn = st.button("🚀 Jalankan Prediksi", type="primary", width="stretch")


# Fungsi Pendukung Sesuai Script Anda
def bbox_to_geo(bbox, transform):
    x_center, y_center = bbox
    x_center_geo, y_center_geo = rasterio.transform.xy(transform, y_center, x_center)
    return (x_center_geo, y_center_geo)


def split_image(image, tile_size):
    img_width, img_height = image.size
    tiles = []
    for y in range(0, img_height, tile_size):
        for x in range(0, img_width, tile_size):
            box = (x, y, x + tile_size, y + tile_size)
            tile = image.crop(box)
            tiles.append((tile, x, y))
    return tiles


# Main Execution Logic
if run_btn:
    if uploaded_file is None:
        st.error("Silakan upload file citra terlebih dahulu melalui sidebar!")
    elif not os.path.exists(model_path):
        st.error(f"File model YOLO tidak ditemukan di path: {model_path}")
    else:
        # Simpan file yang di-upload secara temporer agar bisa dibaca rasterio/PIL
        temp_dir = "temp_data"
        os.makedirs(temp_dir, exist_ok=True)
        input_raster_path = os.path.join(temp_dir, uploaded_file.name)

        with open(input_raster_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.status(
            "Sedang memproses deteksi pohon sawit...", expanded=True
        ) as status:
            start_time = time.time()

            # 1. Load Model & Raster
            st.write("Memuat model YOLO...")
            model = YOLO(model_path)

            st.write(f"Membaca citra: {uploaded_file.name}")

            # Penanganan aman jika file non-geospasial (PNG/JPG) atau TIF geospasial
            try:
                with rasterio.open(input_raster_path) as src:
                    original_image = src.read([1, 2, 3])
                    original_transform = src.transform
                    original_crs = src.crs
                    bounds = src.bounds
            except Exception:
                # Fallback jika user mengupload PNG/JPG biasa tanpa metadata geospasial
                pil_img = Image.open(input_raster_path).convert("RGB")
                original_image = np.array(pil_img)
                original_image = np.moveaxis(
                    original_image, -1, 0
                )  # ubah ke (bands, height, width)
                # Buat affine transform dummy (koordinat piksel/meter buatan)
                original_transform = rasterio.transform.from_bounds(
                    0, 0, pil_img.width, pil_img.height, pil_img.width, pil_img.height
                )
                original_crs = "EPSG:4326"

                class DummyBounds:
                    left, bottom, right, top = 0, 0, pil_img.width, pil_img.height

                bounds = DummyBounds()

            original_image = np.moveaxis(original_image, 0, -1)

            # 2. Inference per Tile
            st.write("Menjalankan inferensi tile demi tile...")
            tiles = split_image(Image.fromarray(original_image.astype("uint8")), 640)
            detection_points = []
            annotated_image = Image.fromarray(original_image.astype("uint8")).copy()

            for idx, (tile, x_offset, y_offset) in enumerate(tiles):
                tile_array = np.array(tile)
                results = model.predict(
                    source=tile_array, verbose=False, conf=confidence_thresh
                )

                try:
                    plotted = results[0].plot()
                    plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                    annotated_image.paste(
                        Image.fromarray(plotted_rgb), (x_offset, y_offset)
                    )
                except Exception:
                    pass

                for box in results[0].boxes:
                    x_min, y_min, x_max, y_max = box.xyxy[0]
                    x_center = int((x_min + x_max) / 2) + x_offset
                    y_center = int((y_min + y_max) / 2) + y_offset
                    geo_point = bbox_to_geo([x_center, y_center], original_transform)
                    detection_points.append(Point(geo_point))

            gdf = gpd.GeoDataFrame(geometry=detection_points, crs=original_crs)

            # 3. KD-Tree Filtering
            st.write(
                f"Melakukan post-processing KD-Tree (min_distance = {min_distance}m)..."
            )
            coords = np.array([(p.x, p.y) for p in gdf.geometry])

            if len(coords) > 0:
                tree = cKDTree(coords)
                keep = np.ones(len(coords), dtype=bool)
                for i in range(len(coords)):
                    if not keep[i]:
                        continue
                    neighbors = tree.query_ball_point(coords[i], r=min_distance)
                    for j in neighbors:
                        if j != i and keep[j]:
                            keep[j] = False
                gdf_filtered = gdf[keep].reset_index(drop=True)
            else:
                gdf_filtered = gdf

            end_time = time.time()
            computation_time = end_time - start_time

            # Simpan ke session_state
            st.session_state["gdf"] = gdf
            st.session_state["gdf_filtered"] = gdf_filtered
            st.session_state["computation_time"] = computation_time
            st.session_state["bounds"] = bounds
            st.session_state["input_raster_path"] = input_raster_path
            st.session_state["original_image"] = original_image
            st.session_state["annotated_image"] = annotated_image

            status.update(
                label="Prediksi & Analisis Selesai!", state="complete", expanded=False
            )

# Tampilkan Hasil Jika Sudah Diproses
if "gdf_filtered" in st.session_state:
    gdf = st.session_state["gdf"]
    gdf_filtered = st.session_state["gdf_filtered"]

    # Metrik Ringkasan di Bagian Atas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Detections (Raw)", f"{len(gdf):,} pohon")
    col2.metric(
        "Filtered (KD-Tree)",
        f"{len(gdf_filtered):,} pohon",
        delta=f"-{len(gdf)-len(gdf_filtered)}",
    )
    col3.metric("Computation Time", f"{st.session_state['computation_time']:.2f} detik")
    col4.metric("Model Status", "Ready")

    st.divider()

    # Tampilkan citra input dan hasil prediksi
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.subheader("🖼️ Citra Input")
        if "original_image" in st.session_state:
            st.image(st.session_state["original_image"], width="stretch")
    with img_col2:
        st.subheader("🎯 Hasil Prediksi")
        if "annotated_image" in st.session_state:
            st.image(st.session_state["annotated_image"], width="stretch")

    st.divider()

    # Layout Utama: Peta Interaktif & Panel Statistik Samping
    map_col, info_col = st.columns([2, 1])

    with map_col:
        st.subheader("🗺️ Peta Distribusi Spasial Hasil Deteksi")

        if len(gdf_filtered) > 0:
            sample_pt = gdf_filtered.geometry.iloc[0]
            center_latlon = (sample_pt.y, sample_pt.x)
        else:
            center_latlon = (0, 0)

        m = folium.Map(location=center_latlon, zoom_start=18)

        plot_sample = gdf_filtered.head(1000)
        for idx, row in plot_sample.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=3,
                color="white",
                weight=0.8,
                fill=True,
                fill_color="#ff3b30",
                fill_opacity=0.9,
            ).add_to(m)

        st_folium(m, height=500, width="stretch")

    with info_col:
        st.subheader("📊 Statistik & Ekspor")
        st.info(f"Jumlah pohon tersaring efektif: **{len(gdf_filtered):,} pohon**")

        geojson_data = gdf_filtered.to_json()
        st.download_button(
            label="📥 Download Filtered GeoJSON",
            data=geojson_data,
            file_name="oilpalm_filtered_results.geojson",
            mime="application/json",
            width="stretch",
        )

        st.markdown("### 🔍 Sample Validation Window")
        st.write("Preview visual potongan area raster acak.")

        if st.button("Tampilkan Random Sample Plot", width="stretch"):
            with rasterio.open(st.session_state["input_raster_path"]) as src:
                window_size = 40
                rng = np.random.default_rng(42)
                if len(gdf_filtered) > 0:
                    rand_idx = rng.integers(0, len(gdf_filtered))
                    cx, cy = (
                        gdf_filtered.geometry.iloc[rand_idx].x,
                        gdf_filtered.geometry.iloc[rand_idx].y,
                    )
                    left, bottom, right, top = (
                        cx - window_size / 2,
                        cy - window_size / 2,
                        cx + window_size / 2,
                        cy + window_size / 2,
                    )

                    # Clamp to raster bounds to avoid WindowError
                    left = max(left, src.bounds.left)
                    bottom = max(bottom, src.bounds.bottom)
                    right = min(right, src.bounds.right)
                    top = min(top, src.bounds.top)

                    if right <= left or top <= bottom:
                        st.warning("Sample window outside raster bounds.")
                    else:
                        try:
                            window = from_bounds(
                                left, bottom, right, top, src.transform
                            )
                            img_window = np.moveaxis(
                                src.read([1, 2, 3], window=window), 0, -1
                            )

                            fig, ax = plt.subplots(figsize=(4, 4))
                            ax.imshow(img_window, extent=(left, right, bottom, top))

                            sub_pts = gdf_filtered.cx[left:right, bottom:top]
                            if len(sub_pts) > 0:
                                ax.scatter(
                                    sub_pts.geometry.x,
                                    sub_pts.geometry.y,
                                    c="red",
                                    s=25,
                                    edgecolors="white",
                                    linewidths=0.5,
                                )

                            ax.set_title(
                                f"Sample Validation ({len(sub_pts)} trees)", fontsize=10
                            )
                            ax.set_axis_off()
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not render sample window: {e}")
else:
    st.info(
        "👈 Silakan upload file citra melalui panel sidebar di sebelah kiri, sesuaikan parameter, lalu klik tombol **'Jalankan Prediksi'**."
    )
