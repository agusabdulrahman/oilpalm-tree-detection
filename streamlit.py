import os
import tempfile
import time
from pathlib import Path

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

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Oil Palm AI Intelligence Platform",
    page_icon="🌴",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "Model" / "oilpalm_detection_model_yolov26n.pt"
TEMP_DIR = Path(tempfile.gettempdir()) / "oilpalm_ai"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str, model_mtime: float):
    """
    Cache model in memory. model_mtime is included so replacing the
    model file invalidates the cache.
    """
    path = Path(model_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {path}\n"
            "Pastikan file .pt tersedia di repository/deployment, "
            "atau isi path model yang benar."
        )

    return YOLO(str(path))


def resolve_model_path(path_text: str) -> Path:
    """Resolve relative model paths from the app directory."""
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


# ============================================================
# IMAGE / GEO HELPERS
# ============================================================
def split_image(image: Image.Image, tile_size: int = 640):
    """Split an image into tiles and preserve each tile's pixel offset."""
    img_width, img_height = image.size
    tiles = []

    for y in range(0, img_height, tile_size):
        for x in range(0, img_width, tile_size):
            # PIL crop supports tiles extending beyond the image boundary.
            tile = image.crop((x, y, x + tile_size, y + tile_size))
            tiles.append((tile, x, y))

    return tiles


def bbox_to_geo(bbox, transform):
    """Convert pixel (column, row) center to raster map coordinates."""
    x_center, y_center = bbox
    x_geo, y_geo = rasterio.transform.xy(transform, y_center, x_center, offset="center")
    return x_geo, y_geo


def read_uploaded_image(file_path: str):
    """
    Read GeoTIFF with rasterio when possible.
    Fall back to ordinary image formats with synthetic pixel coordinates.
    Returns RGB image, transform, CRS, bounds, and whether georeferenced.
    """
    try:
        with rasterio.open(file_path) as src:
            if src.count >= 3:
                bands = src.read([1, 2, 3])
            elif src.count == 1:
                band = src.read(1)
                bands = np.stack([band, band, band], axis=0)
            else:
                raise ValueError(
                    f"Raster hanya memiliki {src.count} band; " "minimal perlu 1 band."
                )

            image = np.moveaxis(bands, 0, -1)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            is_georeferenced = crs is not None

        # Normalize non-8-bit raster data for display and YOLO.
        if image.dtype != np.uint8:
            finite = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
            low, high = np.percentile(finite, [2, 98])
            if high > low:
                image = np.clip((finite - low) / (high - low) * 255, 0, 255)
            else:
                image = np.zeros_like(finite)
            image = image.astype(np.uint8)

        return image, transform, crs, bounds, is_georeferenced

    except (rasterio.errors.RasterioIOError, ValueError):
        # Ordinary JPG/PNG image: these are pixel coordinates, not GPS.
        pil_img = Image.open(file_path).convert("RGB")
        image = np.array(pil_img)
        transform = rasterio.transform.from_bounds(
            0,
            0,
            pil_img.width,
            pil_img.height,
            pil_img.width,
            pil_img.height,
        )
        crs = None
        bounds = rasterio.coords.BoundingBox(
            left=0, bottom=0, right=pil_img.width, top=pil_img.height
        )
        return image, transform, crs, bounds, False


def filter_nearby_points(gdf: gpd.GeoDataFrame, min_distance: float):
    """
    Keep one detection per neighborhood.
    For projected CRS, min_distance uses CRS units (typically meters).
    For geographic/no CRS data, use a pixel/coordinate distance and warn user.
    """
    if gdf.empty:
        return gdf.copy()

    coords = np.array([(p.x, p.y) for p in gdf.geometry], dtype=float)
    tree = cKDTree(coords)
    keep = np.ones(len(coords), dtype=bool)

    # Greedy filtering: keep first point and suppress nearby detections.
    for i in range(len(coords)):
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point(coords[i], r=min_distance)
        for j in neighbors:
            if j != i:
                keep[j] = False

    return gdf.loc[keep].reset_index(drop=True)


def get_tile_rgb(tile: Image.Image):
    return np.asarray(tile.convert("RGB"))


# ============================================================
# SIDEBAR
# ============================================================
st.title("🌴 Oil Palm Tree Detection & Geospatial Analytics Dashboard")
st.markdown(
    "Upload citra kebun sawit untuk mendeteksi pohon dan menganalisis distribusinya."
)

with st.sidebar:
    st.header("⚙️ Analysis Control Panel")
    st.subheader("1. Upload Data & Model")

    uploaded_file = st.file_uploader(
        "Upload Citra Raster (.tif) / Gambar",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
    )

    model_path_text = st.text_input(
        "Path Model YOLO",
        value=str(DEFAULT_MODEL_PATH),
        help=(
            "Local: arahkan ke file .pt. Streamlit Cloud: file model harus "
            "tersedia pada deployment atau diunduh dari storage eksternal."
        ),
    )

    uploaded_model = st.file_uploader(
        "Upload Model YOLO (.pt) opsional",
        type=["pt"],
        help="Gunakan jika model tidak tersedia di deployment.",
    )

    if uploaded_model is not None:
        model_upload_path = TEMP_DIR / uploaded_model.name
        model_upload_path.write_bytes(uploaded_model.getvalue())
        model_path_text = str(model_upload_path)
    confidence_thresh = st.slider("Confidence Threshold", 0.1, 1.0, 0.45, 0.05)
    tile_size = st.select_slider(
        "Tile Size",
        options=[320, 480, 640, 800, 960],
        value=640,
    )

    st.subheader("2. Post-Processing (KD-Tree)")
    min_distance = st.slider(
        "Min. Tree Separation Distance / Coordinate Units",
        1.0,
        10.0,
        5.0,
        0.5,
        help=(
            "Untuk raster projected, satuan mengikuti CRS (umumnya meter). "
            "Untuk JPG/PNG atau CRS geografis, nilai ini bukan meter."
        ),
    )

    run_btn = st.button(
        "🚀 Jalankan Prediksi",
        type="primary",
        width="stretch",
    )


# ============================================================
# MAIN INFERENCE
# ============================================================
if run_btn:
    if uploaded_file is None:
        st.error("Silakan upload file citra terlebih dahulu melalui sidebar.")
    else:
        resolved_model_path = resolve_model_path(model_path_text)

        if not resolved_model_path.is_file():
            st.error(
                f"File model YOLO tidak ditemukan:\n\n"
                f"`{resolved_model_path}`\n\n"
                "Pastikan model .pt masuk ke deployment Streamlit Cloud "
                "dan path sesuai dengan nama file."
            )
        else:
            input_raster_path = TEMP_DIR / Path(uploaded_file.name).name

            try:
                input_raster_path.write_bytes(uploaded_file.getvalue())
            except OSError as exc:
                st.error(f"Gagal menyimpan file upload: {exc}")
                st.stop()

            with st.status(
                "Sedang memproses deteksi pohon sawit...",
                expanded=True,
            ) as status:
                start_time = time.time()

                try:
                    st.write("Memuat model YOLO...")
                    model = load_yolo_model(
                        str(resolved_model_path),
                        resolved_model_path.stat().st_mtime,
                    )
                    st.write("Model berhasil dimuat.")

                    st.write(f"Membaca citra: {uploaded_file.name}")
                    (
                        original_image,
                        original_transform,
                        original_crs,
                        bounds,
                        is_georeferenced,
                    ) = read_uploaded_image(str(input_raster_path))

                    if not is_georeferenced:
                        st.warning(
                            "Citra tidak memiliki CRS/georeferensi yang valid. "
                            "Peta akan menggunakan koordinat piksel, bukan lokasi GPS."
                        )

                    image_pil = Image.fromarray(original_image)
                    tiles = split_image(image_pil, tile_size)
                    detection_points = []
                    annotated_image = image_pil.copy()

                    st.write(f"Menjalankan inferensi pada {len(tiles):,} tile...")
                    progress = st.progress(0.0)

                    for idx, (tile, x_offset, y_offset) in enumerate(tiles):
                        tile_array = get_tile_rgb(tile)
                        results = model.predict(
                            source=tile_array,
                            verbose=False,
                            conf=confidence_thresh,
                        )

                        if results:
                            result = results[0]

                            # Ultralytics plot() returns a BGR image.
                            try:
                                plotted_bgr = result.plot()
                                plotted_rgb = cv2.cvtColor(
                                    plotted_bgr, cv2.COLOR_BGR2RGB
                                )
                                tile_annotated = Image.fromarray(plotted_rgb)
                                # Crop padded edge tiles to their actual dimensions.
                                actual_w = min(
                                    tile_size, original_image.shape[1] - x_offset
                                )
                                actual_h = min(
                                    tile_size, original_image.shape[0] - y_offset
                                )
                                annotated_image.paste(
                                    tile_annotated.crop((0, 0, actual_w, actual_h)),
                                    (x_offset, y_offset),
                                )
                            except Exception:
                                pass

                            if result.boxes is not None and len(result.boxes) > 0:
                                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                                for box in boxes_xyxy:
                                    x_min, y_min, x_max, y_max = box[:4]
                                    x_center = int((x_min + x_max) / 2) + x_offset
                                    y_center = int((y_min + y_max) / 2) + y_offset
                                    x_geo, y_geo = bbox_to_geo(
                                        (x_center, y_center),
                                        original_transform,
                                    )
                                    detection_points.append(Point(x_geo, y_geo))

                        progress.progress((idx + 1) / max(len(tiles), 1))

                    if original_crs is not None:
                        gdf = gpd.GeoDataFrame(
                            geometry=detection_points, crs=original_crs
                        )
                    else:
                        gdf = gpd.GeoDataFrame(geometry=detection_points)
                    st.write(f"Deteksi mentah selesai: {len(gdf):,} pohon.")

                    st.write("Melakukan post-processing KD-Tree...")
                    if original_crs is not None and original_crs.is_geographic:
                        st.warning(
                            "CRS raster berbasis derajat (geographic). "
                            "KD-Tree saat ini memakai derajat, bukan meter. "
                            "Gunakan raster projected untuk filtering berbasis meter."
                        )

                    gdf_filtered = filter_nearby_points(gdf, min_distance)

                    computation_time = time.time() - start_time

                    st.session_state["gdf"] = gdf
                    st.session_state["gdf_filtered"] = gdf_filtered
                    st.session_state["computation_time"] = computation_time
                    st.session_state["bounds"] = bounds
                    st.session_state["input_raster_path"] = str(input_raster_path)
                    st.session_state["original_image"] = original_image
                    st.session_state["annotated_image"] = np.array(annotated_image)
                    st.session_state["is_georeferenced"] = is_georeferenced
                    st.session_state["source_crs"] = (
                        str(original_crs) if original_crs else None
                    )

                    status.update(
                        label="Prediksi & Analisis Selesai!",
                        state="complete",
                        expanded=False,
                    )

                except Exception as exc:
                    status.update(
                        label="Prediksi gagal",
                        state="error",
                        expanded=True,
                    )
                    st.exception(exc)


# ============================================================
# RESULTS
# ============================================================
if "gdf_filtered" in st.session_state:
    gdf = st.session_state["gdf"]
    gdf_filtered = st.session_state["gdf_filtered"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Detections (Raw)", f"{len(gdf):,} pohon")
    col2.metric(
        "Filtered (KD-Tree)",
        f"{len(gdf_filtered):,} pohon",
        delta=f"-{len(gdf) - len(gdf_filtered):,}",
    )
    col3.metric(
        "Computation Time",
        f"{st.session_state['computation_time']:.2f} detik",
    )
    col4.metric(
        "Model Status",
        "Ready",
    )

    st.divider()

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.subheader("🖼️ Citra Input")
        st.image(st.session_state["original_image"], width='stretch')

    with img_col2:
        st.subheader("🎯 Hasil Prediksi")
        st.image(st.session_state["annotated_image"], width='stretch')

    st.divider()

    map_col, info_col = st.columns([2, 1])

    with map_col:
        st.subheader("🗺️ Peta Distribusi Spasial Hasil Deteksi")

        if len(gdf_filtered) > 0:
            sample_pt = gdf_filtered.geometry.iloc[0]
            center = (sample_pt.y, sample_pt.x)
        else:
            center = (0, 0)

        # Folium expects latitude/longitude. Only treat coordinates as lat/lon
        # if the source CRS is geographic. For projected data, reproject to EPSG:4326.
        map_gdf = gdf_filtered.copy()
        source_crs = st.session_state.get("source_crs")

        if not map_gdf.empty and source_crs:
            try:
                if map_gdf.crs is not None and not map_gdf.crs.is_geographic:
                    map_gdf = map_gdf.to_crs("EPSG:4326")
            except Exception as exc:
                st.warning(f"Gagal mengubah CRS untuk peta: {exc}")

        if not map_gdf.empty:
            sample_pt = map_gdf.geometry.iloc[0]
            center = (sample_pt.y, sample_pt.x)
        else:
            center = (0, 0)

        m = folium.Map(location=center, zoom_start=18)

        for _, row in map_gdf.head(1000).iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=3,
                color="white",
                weight=0.8,
                fill=True,
                fill_color="#ff3b30",
                fill_opacity=0.9,
                tooltip="Detected oil palm",
            ).add_to(m)

        if not st.session_state.get("is_georeferenced", False):
            st.caption(
                "Catatan: titik pada citra JPG/PNG tanpa georeferensi adalah "
                "koordinat piksel, bukan lokasi geografis."
            )

        st_folium(m, height=500, width=None, key="oilpalm_detection_map")

    with info_col:
        st.subheader("📊 Statistik & Ekspor")
        st.info(f"Jumlah pohon tersaring efektif: **{len(gdf_filtered):,} pohon**")

        st.download_button(
            label="📥 Download Filtered GeoJSON",
            data=gdf_filtered.to_json(),
            file_name="oilpalm_filtered_results.geojson",
            mime="application/geo+json",
            width='stretch',
        )

        st.download_button(
            label="📥 Download Raw GeoJSON",
            data=gdf.to_json(),
            file_name="oilpalm_raw_results.geojson",
            mime="application/geo+json",
            width='stretch',
        )

        st.markdown("### 🔍 Sample Validation Window")
        st.write("Preview potongan area raster di sekitar deteksi acak.")

        if st.button("Tampilkan Random Sample Plot", width='stretch'):
            sample_gdf = gdf_filtered
            input_path = st.session_state["input_raster_path"]

            if sample_gdf.empty:
                st.warning("Tidak ada titik deteksi untuk ditampilkan.")
            else:
                try:
                    with rasterio.open(input_path) as src:
                        rng = np.random.default_rng()
                        rand_idx = int(rng.integers(0, len(sample_gdf)))
                        point = sample_gdf.geometry.iloc[rand_idx]

                        # Point coordinates are in the GeoDataFrame CRS.
                        if sample_gdf.crs is not None and src.crs is not None:
                            point_gdf = gpd.GeoDataFrame(
                                geometry=[point], crs=sample_gdf.crs
                            ).to_crs(src.crs)
                            cx, cy = (
                                point_gdf.geometry.iloc[0].x,
                                point_gdf.geometry.iloc[0].y,
                            )
                        else:
                            cx, cy = point.x, point.y

                        window_size = 40
                        left = max(cx - window_size / 2, src.bounds.left)
                        bottom = max(cy - window_size / 2, src.bounds.bottom)
                        right = min(cx + window_size / 2, src.bounds.right)
                        top = min(cy + window_size / 2, src.bounds.top)

                        if right <= left or top <= bottom:
                            st.warning("Sample window berada di luar raster.")
                        else:
                            window = from_bounds(
                                left, bottom, right, top, src.transform
                            )
                            count = min(src.count, 3)
                            sample = src.read(list(range(1, count + 1)), window=window)

                            if sample.shape[0] == 1:
                                img_window = np.repeat(sample, 3, axis=0)
                            elif sample.shape[0] >= 3:
                                img_window = sample[:3]
                            else:
                                img_window = np.concatenate(
                                    [sample, sample[:1]], axis=0
                                )

                            img_window = np.moveaxis(img_window, 0, -1)
                            if img_window.dtype != np.uint8:
                                finite = np.nan_to_num(
                                    img_window, nan=0.0, posinf=0.0, neginf=0.0
                                )
                                low, high = np.percentile(finite, [2, 98])
                                if high > low:
                                    img_window = np.clip(
                                        (finite - low) / (high - low) * 255, 0, 255
                                    )
                                else:
                                    img_window = np.zeros_like(finite)
                                img_window = img_window.astype(np.uint8)

                            fig, ax = plt.subplots(figsize=(5, 5))
                            ax.imshow(
                                img_window,
                                extent=(left, right, bottom, top),
                            )

                            # Transform filtered detections to source raster CRS.
                            plot_points = sample_gdf
                            if (
                                plot_points.crs is not None
                                and src.crs is not None
                                and plot_points.crs != src.crs
                            ):
                                plot_points = plot_points.to_crs(src.crs)

                            sub_pts = plot_points.cx[left:right, bottom:top]
                            if not sub_pts.empty:
                                ax.scatter(
                                    sub_pts.geometry.x,
                                    sub_pts.geometry.y,
                                    c="red",
                                    s=25,
                                    edgecolors="white",
                                    linewidths=0.5,
                                )

                            ax.set_title(
                                f"Sample Validation ({len(sub_pts)} trees)",
                                fontsize=10,
                            )
                            ax.set_axis_off()
                            st.pyplot(fig)
                            plt.close(fig)

                except Exception as exc:
                    st.warning(f"Tidak dapat menampilkan sample window: {exc}")

else:
    st.info(
        "👈 Upload citra melalui sidebar, periksa path model, sesuaikan parameter, "
        "lalu klik **Jalankan Prediksi**."
    )
