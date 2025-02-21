#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Water Quality App (Enterprise-Grade UI)
-----------------------------------------
Αυτή η έκδοση κρύβει τα μηνύματα αποσφαλμάτωσης εξ ορισμού και διαθέτει ένα
πολύ επαγγελματικό, φιλικό προς το χρήστη περιβάλλον.
"""

import os
import glob
import re
from datetime import datetime, date
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# Global debug flag (set to True for debugging output)
DEBUG = False

def debug(*args, **kwargs):
    """Βοηθητική συνάρτηση για εμφάνιση μηνυμάτων αποσφαλμάτωσης, αν είναι ενεργοποιημένη."""
    if DEBUG:
        st.write(*args, **kwargs)

# -------------------------------------------------------------------------
# Διαμόρφωση σελίδας Streamlit
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Ποιοτικά χαρακτηριστικά Επιφανειακού Ύδατος",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Εξατομίκευση CSS για επαγγελματική εμφάνιση
# -----------------------------------------------------------------------------
def inject_custom_css():
    custom_css = """
    <link href="https://fonts.googleapis.com/css?family=Roboto:400,500,700&display=swap" rel="stylesheet">
    <style>
        /* Γενική μορφοποίηση */
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }
        .block-container {
            background: #0d0d0d;
            color: #e0e0e0;
            padding: 1rem;
        }
        /* Μορφοποίηση πλαϊνής μπάρας */
        .sidebar .sidebar-content {
            background: #1b1b1b;
            border: none;
        }
        /* Μορφοποίηση καρτών */
        .card {
            background: #1e1e1e;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.6);
            margin-bottom: 2rem;
        }
        .header-title {
            color: #ffca28;
            margin-bottom: 1rem;
            font-size: 1.75rem;
            text-align: center;
        }
        /* Ενότητα πλοήγησης στην πλαϊνή μπάρας */
        .nav-section {
            padding: 1rem;
            background: #262626;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .nav-section h4 {
            margin: 0;
            color: #ffca28;
            font-weight: 500;
        }
        /* Μορφοποίηση κουμπιών */
        .stButton button {
            background-color: #3949ab;
            color: #fff;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            box-shadow: 0 3px 6px rgba(0,0,0,0.3);
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #5c6bc0;
        }
        /* Μορφοποίηση Plotly διαγραμμάτων */
        .plotly-graph-div {
            border: 1px solid #333;
            border-radius: 8px;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

inject_custom_css()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_data_folder(waterbody: str, index: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    debug("DEBUG: Τρέχων φάκελος:", base_dir)
    waterbody_map = {
        "Κορώνεια": "Koroneia",
        "Πολυφύτου": "polyphytou",
        "Γαδουρά": "Gadoura",
        "Αξιός": "Axios"
    }
    waterbody_folder = waterbody_map.get(waterbody, None)
    if waterbody_folder is None:
        return None
    if index == "Χλωροφύλλη":
        data_folder = os.path.join(base_dir, waterbody_folder, "Chlorophyll")
    elif index == "Burned Areas":
        data_folder = os.path.join(base_dir, waterbody_folder, "Burned Areas")
    else:
        data_folder = os.path.join(base_dir, waterbody_folder, index)
    debug("DEBUG: Ο φάκελος δεδομένων επιλύθηκε σε:", data_folder)
    if data_folder is not None and not os.path.exists(data_folder):
        st.error(f"Ο φάκελος δεν υπάρχει: {data_folder}")
        return None
    return data_folder

def extract_date_from_filename(filename: str):
    basename = os.path.basename(filename)
    debug("DEBUG: Εξαγωγή ημερομηνίας από το όνομα:", basename)
    match = re.search(r'(\d{4})[_-](\d{2})[_-](\d{2})', basename)
    if match:
        year, month, day = match.groups()
        date_obj = datetime(int(year), int(month), int(day))
        day_of_year = date_obj.timetuple().tm_yday
        return day_of_year, date_obj
    return None, None

def load_lake_shape_from_xml(xml_file: str, bounds: tuple = None,
                             xml_width: float = 518.0, xml_height: float = 505.0):
    debug("DEBUG: Φόρτωση περιγράμματος από:", xml_file)
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        points = []
        for point_elem in root.findall("point"):
            x_str = point_elem.get("x")
            y_str = point_elem.get("y")
            if x_str is None or y_str is None:
                continue
            points.append([float(x_str), float(y_str)])
        if not points:
            st.warning("Δεν βρέθηκαν σημεία στο XML:", xml_file)
            return None
        if bounds is not None:
            minx, miny, maxx, maxy = bounds
            transformed_points = []
            for x_xml, y_xml in points:
                x_geo = minx + (x_xml / xml_width) * (maxx - minx)
                y_geo = maxy - (y_xml / xml_height) * (maxy - miny)
                transformed_points.append([x_geo, y_geo])
            points = transformed_points
        if points and (points[0] != points[-1]):
            points.append(points[0])
        debug("DEBUG: Φορτώθηκαν", len(points), "σημεία.")
        return {"type": "Polygon", "coordinates": [points]}
    except Exception as e:
        st.error(f"Σφάλμα φόρτωσης περιγράμματος από {xml_file}: {e}")
        return None

def read_image(file_path: str, lake_shape: dict = None):
    debug("DEBUG: Ανάγνωση εικόνας από:", file_path)
    with rasterio.open(file_path) as src:
        img = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        profile.update(dtype="float32")
        no_data_value = src.nodata
        if no_data_value is not None:
            img = np.where(img == no_data_value, np.nan, img)
        img = np.where(img == 0, np.nan, img)
        if lake_shape is not None:
            from rasterio.features import geometry_mask
            poly_mask = geometry_mask([lake_shape], transform=src.transform, invert=False, out_shape=img.shape)
            img = np.where(~poly_mask, img, np.nan)
    return img, profile

def load_data(input_folder: str, shapefile_name="shapefile.xml"):
    debug("DEBUG: load_data καλεσμένη με:", input_folder)
    if not os.path.exists(input_folder):
        raise Exception(f"Ο φάκελος δεν υπάρχει: {input_folder}")
    shapefile_path_xml = os.path.join(input_folder, shapefile_name)
    shapefile_path_txt = os.path.join(input_folder, "shapefile.txt")
    lake_shape = None
    if os.path.exists(shapefile_path_xml):
        shape_file = shapefile_path_xml
    elif os.path.exists(shapefile_path_txt):
        shape_file = shapefile_path_txt
    else:
        shape_file = None
        debug("DEBUG: Δεν βρέθηκε XML περιγράμματος στον φάκελο", input_folder)
    all_tif_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    tif_files = [fp for fp in all_tif_files if os.path.basename(fp).lower() != "mask.tif"]
    if not tif_files:
        raise Exception("Δεν βρέθηκαν GeoTIFF αρχεία.")
    with rasterio.open(tif_files[0]) as src:
        bounds = src.bounds
    if shape_file is not None:
        lake_shape = load_lake_shape_from_xml(shape_file, bounds=bounds)
    images, days, date_list = [], [], []
    for file_path in tif_files:
        day_of_year, date_obj = extract_date_from_filename(file_path)
        if day_of_year is None:
            continue
        img, _ = read_image(file_path, lake_shape=lake_shape)
        images.append(img)
        days.append(day_of_year)
        date_list.append(date_obj)
    if not images:
        raise Exception("Δεν βρέθηκαν έγκυρες εικόνες.")
    stack = np.stack(images, axis=0)
    return stack, np.array(days), date_list

# -----------------------------------------------------------------------------
# Σελίδα Εισαγωγής
# -----------------------------------------------------------------------------
def run_intro_page():
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col_logo, col_text = st.columns([1, 3])
        with col_logo:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(base_dir, "logo.jpg")
            if os.path.exists(logo_path):
                st.image(logo_path, width=250)
            else:
                debug("DEBUG: Δεν βρέθηκε το λογότυπο.")
        with col_text:
            st.markdown("<h2 class='header-title'>Ποιοτικά χαρακτηριστικά Επιφανειακού Ύδατος</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-size: 1.1rem;'>"
                        "Αυτή η εφαρμογή ανάλυσης χρησιμοποιεί εργαλεία δορυφορικής τηλεπισκόπησης. "
                        "Επιλέξτε τις ρυθμίσεις στην πλαϊνή μπάρα και εξερευνήστε τα δεδομένα.</p>",
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Πλαϊνή Μπάρα Πλοήγησης (Custom UI)
# -----------------------------------------------------------------------------
def run_custom_ui():
    st.sidebar.markdown("<div class='nav-section'><h4>Παραμετροποίηση Ανάλυσης</h4></div>", unsafe_allow_html=True)
    waterbody = st.sidebar.selectbox("Επιλογή υδάτινου σώματος",
                                     ["Κορώνεια", "Πολυφύτου", "Γαδουρά", "Αξιός"],
                                     key="waterbody_choice")
    index = st.sidebar.selectbox("Επιλογή Δείκτη",
                                 ["Πραγματικό", "Χλωροφύλλη", "CDOM", "Colour", "Burned Areas"],
                                 key="index_choice")
    analysis = st.sidebar.selectbox("Είδος Ανάλυσης",
                                    ["Lake Processing", "Water Processing", "Water Quality Dashboard",
                                     "Burned Areas", "Water level", "Pattern Analysis"],
                                    key="analysis_choice")
    st.sidebar.markdown(f"""
    <div style="padding: 0.5rem; background:#262626; border-radius:5px; margin-top:1rem;">
        <strong>Υδάτινο σώμα:</strong> {waterbody}<br>
        <strong>Δείκτης:</strong> {index}<br>
        <strong>Ανάλυση:</strong> {analysis}
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Επεξεργασία Λίμνης (Lake Processing)
# (This section remains largely unchanged)
# -----------------------------------------------------------------------------
def run_lake_processing_app(waterbody: str, index: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Επεξεργασία Λίμνης ({waterbody} - {index})")
        data_folder = get_data_folder(waterbody, index)
        if data_folder is None:
            st.error("Δεν υπάρχει φάκελος δεδομένων για το επιλεγμένο υδάτινο σώμα/δείκτη.")
            st.stop()
        input_folder = os.path.join(data_folder, "GeoTIFFs")
        try:
            STACK, DAYS, DATES = load_data(input_folder)
        except Exception as e:
            st.error(f"Σφάλμα φόρτωσης δεδομένων: {e}")
            st.stop()
        if not DATES:
            st.error("Δεν υπάρχουν διαθέσιμες πληροφορίες ημερομηνίας.")
            st.stop()
        min_date = min(DATES)
        max_date = max(DATES)
        unique_years = sorted({d.year for d in DATES if d is not None})
        st.sidebar.header(f"Φίλτρα (Επεξεργασία Λίμνης: {waterbody})")
        threshold_range = st.sidebar.slider("Εύρος τιμών pixel", 0, 255, (0, 255), key="thresh_lp")
        broad_date_range = st.sidebar.slider("Γενική περίοδος", min_value=min_date, max_value=max_date,
                                             value=(min_date, max_date), key="broad_date_lp")
        refined_date_range = st.sidebar.slider("Εξειδικευμένη περίοδος", min_value=min_date, max_value=max_date,
                                               value=(min_date, max_date), key="refined_date_lp")
        display_option = st.sidebar.radio("Τρόπος εμφάνισης", options=["Thresholded", "Original"], index=0, key="display_lp")
        st.sidebar.markdown("### Επιλογή Μηνών")
        month_options = {i: datetime(2000, i, 1).strftime('%B') for i in range(1, 13)}
        if "selected_months" not in st.session_state:
            st.session_state.selected_months = list(month_options.keys())
        selected_months = st.sidebar.multiselect("Μήνες",
                                                 options=list(month_options.keys()),
                                                 format_func=lambda x: month_options[x],
                                                 default=st.session_state.selected_months,
                                                 key="months_lp")
        st.session_state.selected_years = unique_years
        selected_years = st.sidebar.multiselect("Έτη", options=unique_years,
                                                default=unique_years,
                                                key="years_lp")
        start_dt, end_dt = refined_date_range
        selected_indices = [i for i, d in enumerate(DATES)
                            if start_dt <= d <= end_dt and d.month in selected_months and d.year in selected_years]
        if not selected_indices:
            st.error("Δεν υπάρχουν δεδομένα για την επιλεγμένη περίοδο/μήνες/έτη.")
            st.stop()
        stack_filtered = STACK[selected_indices, :, :]
        days_filtered = np.array(DAYS)[selected_indices]
        filtered_dates = np.array(DATES)[selected_indices]
        lower_thresh, upper_thresh = threshold_range
        in_range = np.logical_and(stack_filtered >= lower_thresh, stack_filtered <= upper_thresh)
        days_in_range = np.nansum(in_range, axis=0)
        fig_days = px.imshow(days_in_range, color_continuous_scale="plasma",
                             title="Διάγραμμα: Ημέρες σε Εύρος", labels={"color": "Ημέρες σε Εύρος"})
        fig_days.update_layout(width=800, height=600)
        st.plotly_chart(fig_days, use_container_width=True, key="fig_days")
        with st.expander("Επεξήγηση: Ημέρες σε Εύρος"):
            st.write("Το διάγραμμα δείχνει πόσες ημέρες κάθε pixel βρίσκεται εντός του εύρους τιμών.")
        tick_vals = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365]
        tick_text = ["1 (Ιαν)", "32 (Φεβ)", "60 (Μαρ)", "91 (Απρ)",
                     "121 (Μαΐ)", "152 (Ιουν)", "182 (Ιουλ)", "213 (Αυγ)",
                     "244 (Σεπ)", "274 (Οκτ)", "305 (Νοε)", "335 (Δεκ)", "365 (Δεκ)"]
        days_array = days_filtered.reshape((-1, 1, 1))
        sum_days = np.nansum(days_array * in_range, axis=0)
        count_in_range = np.nansum(in_range, axis=0)
        mean_day = np.divide(sum_days, count_in_range,
                             out=np.full(sum_days.shape, np.nan),
                             where=(count_in_range != 0))
        fig_mean = px.imshow(mean_day, color_continuous_scale="RdBu",
                             title="Διάγραμμα: Μέση Ημέρα Εμφάνισης", labels={"color": "Μέση Ημέρα"})
        fig_mean.update_layout(width=800, height=600)
        fig_mean.update_layout(coloraxis_colorbar=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text))
        st.plotly_chart(fig_mean, use_container_width=True, key="fig_mean")
        with st.expander("Επεξήγηση: Μέση Ημέρα Εμφάνισης"):
            st.write("Το διάγραμμα παρουσιάζει τη μέση ημέρα εμφάνισης για τα pixels.")
        if display_option.lower() == "thresholded":
            filtered_stack = np.where(in_range, stack_filtered, np.nan)
        else:
            filtered_stack = stack_filtered
        average_sample_img = np.nanmean(filtered_stack, axis=0)
        if not np.all(np.isnan(average_sample_img)):
            avg_min = float(np.nanmin(average_sample_img))
            avg_max = float(np.nanmax(average_sample_img))
        else:
            avg_min, avg_max = 0, 0
        fig_sample = px.imshow(average_sample_img, color_continuous_scale="jet",
                               range_color=[avg_min, avg_max],
                               title="Διάγραμμα: Μέσο Δείγμα Εικόνας", labels={"color": "Τιμή Pixel"})
        fig_sample.update_layout(width=800, height=600)
        st.plotly_chart(fig_sample, use_container_width=True, key="fig_sample")
        with st.expander("Επεξήγηση: Μέσο Δείγμα Εικόνας"):
            st.write("Το διάγραμμα δείχνει τη μέση τιμή των pixels μετά το φίλτρο.")
        filtered_day_of_year = np.array([d.timetuple().tm_yday for d in filtered_dates])
        def nanargmax_or_nan(arr):
            return np.nan if np.all(np.isnan(arr)) else np.nanargmax(arr)
        max_index = np.apply_along_axis(nanargmax_or_nan, 0, filtered_stack)
        time_max = np.full(max_index.shape, np.nan, dtype=float)
        valid_mask = ~np.isnan(max_index)
        max_index_int = np.zeros_like(max_index, dtype=int)
        max_index_int[valid_mask] = max_index[valid_mask].astype(int)
        max_index_int[valid_mask] = np.clip(max_index_int[valid_mask], 0, len(filtered_day_of_year) - 1)
        time_max[valid_mask] = filtered_day_of_year[max_index_int[valid_mask]]
        fig_time = px.imshow(time_max, color_continuous_scale="RdBu",
                             range_color=[1, 365],
                             title="Διάγραμμα: Χρόνος Μέγιστης Εμφάνισης", labels={"color": "Ημέρα"})
        fig_time.update_layout(width=800, height=600)
        fig_time.update_layout(coloraxis_colorbar=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text))
        st.plotly_chart(fig_time, use_container_width=True, key="fig_time")
        with st.expander("Επεξήγηση: Χρόνος Μέγιστης Εμφάνισης"):
            st.write("Το διάγραμμα δείχνει την ημέρα του έτους με τη μέγιστη τιμή.")
        st.header("Χάρτες Ανάλυσης")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_days, use_container_width=True, key="fig_days_2")
        with col2:
            st.plotly_chart(fig_mean, use_container_width=True, key="fig_mean_2")
        st.header("Ανάλυση Δείγματος Εικόνας")
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(fig_sample, use_container_width=True, key="fig_sample_2")
        with col4:
            st.plotly_chart(fig_time, use_container_width=True, key="fig_time_2")
        # ------------------------------
        # Nested Sampling Tabs (Default & Upload)
        # ------------------------------
        if "default_results" not in st.session_state:
            st.session_state.default_results = None
        if "upload_results" not in st.session_state:
            st.session_state.upload_results = None
        tab_names = ["Δειγματοληψία 1 (Default)", "Δειγματοληψία 2 (Upload)"]
        sampling_tabs = st.tabs(tab_names)
        # --- Default Sampling Tab ---
        with sampling_tabs[0]:
            st.header("Ανάλυση για Δειγματοληψία 1 (Default)")
            # Use points from a KML file (if available)
            default_sampling_points = []
            if os.path.exists(sampling_kml_path):
                default_sampling_points = parse_sampling_kml(sampling_kml_path)
            else:
                st.warning("Το αρχείο δειγματοληψίας (sampling.kml) δεν βρέθηκε.")
            point_names = [name for name, _, _ in default_sampling_points]
            selected_points = st.multiselect("Επιλέξτε σημεία για ανάλυση mg/m³",
                                             options=point_names,
                                             default=point_names,
                                             key="default_points")
            if st.button("Εκτέλεση Ανάλυσης (Default)", key="default_run"):
                with st.spinner("Εκτέλεση ανάλυσης..."):
                    st.session_state.default_results = analyze_sampling(
                        default_sampling_points,
                        first_image_data,
                        first_transform,
                        images_folder,
                        lake_height_path,
                        selected_points
                    )
            if st.session_state.default_results is not None:
                results = st.session_state.default_results
                if isinstance(results, tuple) and len(results) == 7:
                    fig_geo, fig_dual, fig_colors, fig_mg, results_colors, results_mg, lake_data = results
                else:
                    st.error("Σφάλμα μορφοποίησης αποτελεσμάτων ανάλυσης. Παρακαλώ επαναλάβετε.")
                    st.stop()
                nested_tabs = st.tabs(["GeoTIFF", "Επιλογή Εικόνων και Επιλογή Σημείων", "Video/GIF", "Χρώματα Pixel", "Μέσο mg", "Διπλά Διαγράμματα", "Λεπτομερής ανάλυση mg"])
                with nested_tabs[0]:
                    st.plotly_chart(fig_geo, use_container_width=True, key="default_fig_geo")
                with nested_tabs[1]:
                    st.header("Επιλογή Εικόνων και Επιλογή Σημείων")
                    # Display the background image interactively so the user can click to select points.
                    fig_bg = px.imshow(first_image_data.transpose((1,2,0)) / 255.0,
                                       title="Click on the image to select points")
                    fig_bg.update_layout(clickmode='event+select')
                    events = st.plotly_events(fig_bg, click_event=True, hover_event=False)
                    if events:
                        st.write("Selected pixel coordinates:", events)
                        # Convert pixel coordinates to geographic coordinates.
                        interactive_points = []
                        inv_transform = ~first_transform
                        for ev in events:
                            x_pixel = ev.get("x")
                            y_pixel = ev.get("y")
                            geo_x, geo_y = inv_transform * (x_pixel, y_pixel)
                            interactive_points.append((f"Interactive_{len(interactive_points)+1}", geo_x, geo_y))
                        st.session_state.interactive_points = interactive_points
                        st.write("Interactive points (converted):", interactive_points)
                    else:
                        st.write("Click on the image above to select points.")
                with nested_tabs[2]:
                    if video_path is not None:
                        if video_path.endswith(".mp4"):
                            st.video(video_path, key="default_video")
                        else:
                            st.image(video_path)
                    else:
                        st.info("Δεν βρέθηκε αρχείο timelapse.")
                with nested_tabs[3]:
                    if st.session_state.get("interactive_points"):
                        ip = st.session_state.interactive_points
                        res_geo, res_dual, res_colors, res_mg, _, _, _ = analyze_sampling(ip, first_image_data, first_transform, images_folder, lake_height_path)
                        st.plotly_chart(res_colors, use_container_width=True, key="default_fig_colors_interactive")
                    else:
                        st.plotly_chart(fig_colors, use_container_width=True, key="default_fig_colors")
                with nested_tabs[4]:
                    if st.session_state.get("interactive_points"):
                        ip = st.session_state.interactive_points
                        res_geo, res_dual, res_colors, res_mg, _, _, _ = analyze_sampling(ip, first_image_data, first_transform, images_folder, lake_height_path)
                        st.plotly_chart(res_mg, use_container_width=True, key="default_fig_mg_interactive")
                    else:
                        st.plotly_chart(fig_mg, use_container_width=True, key="default_fig_mg")
                with nested_tabs[5]:
                    if st.session_state.get("interactive_points"):
                        ip = st.session_state.interactive_points
                        res_geo, res_dual, res_colors, res_mg, _, _, _ = analyze_sampling(ip, first_image_data, first_transform, images_folder, lake_height_path)
                        st.plotly_chart(res_dual, use_container_width=True, key="default_fig_dual_interactive")
                    else:
                        st.plotly_chart(fig_dual, use_container_width=True, key="default_fig_dual")
                with nested_tabs[6]:
                    selected_detail_point = st.selectbox("Επιλέξτε σημείο για λεπτομερή ανάλυση mg",
                                                         options=list(results_mg.keys()),
                                                         key="default_detail")
                    if selected_detail_point:
                        mg_data = results_mg[selected_detail_point]
                        if mg_data:
                            mg_data_sorted = sorted(mg_data, key=lambda x: x[0])
                            dates_mg = [d for d, _ in mg_data_sorted]
                            mg_values = [val for _, val in mg_data_sorted]
                            detail_colors = [mg_to_color(val) for val in mg_values]
                            fig_detail = go.Figure()
                            fig_detail.add_trace(go.Scatter(
                                x=dates_mg, y=mg_values, mode='lines+markers',
                                marker=dict(color=detail_colors, size=10),
                                line=dict(color="gray"),
                                name=selected_detail_point
                            ))
                            fig_detail.update_layout(title=f"Λεπτομερής ανάλυση mg για {selected_detail_point}",
                                                     xaxis_title="Ημερομηνία", yaxis_title="mg/m³")
                            st.plotly_chart(fig_detail, use_container_width=True, key="default_fig_detail")
                        else:
                            st.info("Δεν υπάρχουν δεδομένα mg για αυτό το σημείο.")
        # --- Upload Sampling Tab ---
        with sampling_tabs[1]:
            st.header("Ανάλυση για ανεβασμένη δειγματοληψία")
            uploaded_file = st.file_uploader("Ανεβάστε αρχείο KML για νέα σημεία δειγματοληψίας", type="kml", key="upload_kml")
            if uploaded_file is not None:
                try:
                    new_sampling_points = parse_sampling_kml(uploaded_file)
                except Exception as e:
                    st.error(f"Σφάλμα επεξεργασίας ανεβασμένου αρχείου: {e}")
                    new_sampling_points = []
                point_names = [name for name, _, _ in new_sampling_points]
                selected_points = st.multiselect("Επιλέξτε σημεία για ανάλυση mg/m³",
                                                 options=point_names,
                                                 default=point_names,
                                                 key="upload_points")
                if st.button("Εκτέλεση Ανάλυσης (Upload)", key="upload_run"):
                    with st.spinner("Εκτέλεση ανάλυσης..."):
                        st.session_state.upload_results = analyze_sampling(
                            new_sampling_points,
                            first_image_data,
                            first_transform,
                            images_folder,
                            lake_height_path,
                            selected_points
                        )
                if st.session_state.upload_results is not None:
                    results = st.session_state.upload_results
                    if isinstance(results, tuple) and len(results) == 7:
                        fig_geo, fig_dual, fig_colors, fig_mg, results_colors, results_mg, lake_data = results
                    else:
                        st.error("Σφάλμα μορφοποίησης αποτελεσμάτων ανάλυσης (Upload). Παρακαλώ επαναλάβετε.")
                        st.stop()
                    nested_tabs = st.tabs(["GeoTIFF", "Επιλογή Εικόνων και Επιλογή Σημείων", "Video/GIF", "Χρώματα Pixel", "Μέσο mg", "Διπλά Διαγράμματα", "Λεπτομερής ανάλυση mg"])
                    with nested_tabs[0]:
                        st.plotly_chart(fig_geo, use_container_width=True, key="upload_fig_geo")
                    with nested_tabs[1]:
                        st.header("Επιλογή Εικόνων και Επιλογή Σημείων")
                        tif_files = [f for f in os.listdir(images_folder) if f.lower().endswith('.tif')]
                        available_dates = {}
                        for filename in tif_files:
                            match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
                            if match:
                                date_str = match.group(1)
                                try:
                                    date_obj = datetime.strptime(date_str, '%Y_%m_%d').date()
                                    available_dates[str(date_obj)] = filename
                                except Exception as e:
                                    debug("DEBUG: Error extracting date from", filename, ":", e)
                                    continue
                        if available_dates:
                            sorted_dates = sorted(available_dates.keys())
                            if 'current_upload_image_index' not in st.session_state:
                                st.session_state.current_upload_image_index = 0
                            col_prev, col_select, col_next = st.columns([1, 3, 1])
                            with col_prev:
                                if st.button("<< Previous", key="upload_prev"):
                                    st.session_state.current_upload_image_index = max(0, st.session_state.current_upload_image_index - 1)
                            with col_next:
                                if st.button("Next >>", key="upload_next"):
                                    st.session_state.current_upload_image_index = min(len(sorted_dates) - 1, st.session_state.current_upload_image_index + 1)
                            with col_select:
                                selected_date = st.selectbox("Select date", sorted_dates, index=st.session_state.current_upload_image_index)
                                st.session_state.current_upload_image_index = sorted_dates.index(selected_date)
                            current_date = sorted_dates[st.session_state.current_upload_image_index]
                            st.write(f"Selected Date: {current_date}")
                            image_filename = available_dates[current_date]
                            image_path = os.path.join(images_folder, image_filename)
                            if os.path.exists(image_path):
                                st.image(image_path, caption=f"Image for {current_date}", use_column_width=True)
                            else:
                                st.error("Image not found.")
                        else:
                            st.info("No images found with a date in the folder.")
                    with nested_tabs[2]:
                        if video_path is not None:
                            if video_path.endswith(".mp4"):
                                st.video(video_path, key="upload_video")
                            else:
                                st.image(video_path)
                        else:
                            st.info("Δεν βρέθηκε αρχείο Video/GIF.")
                    with nested_tabs[3]:
                        st.plotly_chart(fig_colors, use_container_width=True, key="upload_fig_colors")
                    with nested_tabs[4]:
                        st.plotly_chart(fig_mg, use_container_width=True, key="upload_fig_mg")
                    with nested_tabs[5]:
                        st.plotly_chart(fig_dual, use_container_width=True, key="upload_fig_dual")
                    with nested_tabs[6]:
                        selected_detail_point = st.selectbox("Επιλέξτε σημείο για λεπτομερή ανάλυση mg",
                                                             options=list(results_mg.keys()),
                                                             key="upload_detail")
                        if selected_detail_point:
                            mg_data = results_mg[selected_detail_point]
                            if mg_data:
                                mg_data_sorted = sorted(mg_data, key=lambda x: x[0])
                                dates_mg = [d for d, _ in mg_data_sorted]
                                mg_values = [val for _, val in mg_data_sorted]
                                detail_colors = [mg_to_color(val) for val in mg_values]
                                fig_detail = go.Figure()
                                fig_detail.add_trace(go.Scatter(
                                    x=dates_mg, y=mg_values, mode='lines+markers',
                                    marker=dict(color=detail_colors, size=10),
                                    line=dict(color="gray"),
                                    name=selected_detail_point
                                ))
                                fig_detail.update_layout(title=f"Λεπτομερής ανάλυση mg για {selected_detail_point}",
                                                         xaxis_title="Ημερομηνία", yaxis_title="mg/m³")
                                st.plotly_chart(fig_detail, use_container_width=True, key="upload_fig_detail")
                            else:
                                st.info("Δεν υπάρχουν δεδομένα mg για αυτό το σημείο.", key="upload_no_mg")
            else:
                st.info("Παρακαλώ ανεβάστε ένα αρχείο KML για νέα σημεία δειγματοληψίας.")
        st.info("Τέλος Πίνακα Ποιότητας Ύδατος.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Επεξεργασία Καμένων Περιοχών (Placeholder)
# -----------------------------------------------------------------------------
def run_burned_areas():
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title("Burned Areas γύρω από το ταμιευτήριο (μόνο Γαδουρά)")
        st.info("Δεν υπάρχουν δεδομένα ή λειτουργίες για ανάλυση καμένων περιοχών.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Προφίλ Ύψους (Placeholder)
# -----------------------------------------------------------------------------
def run_water_level_profiles(waterbody: str, index: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Προφίλ Ύψους ({waterbody}) [Placeholder]")
        st.info("Δεν υπάρχουν δεδομένα ή λειτουργίες για προφίλ ύψους νερού.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    debug("DEBUG: Εισήχθη η main()")
    run_intro_page()
    run_custom_ui()
    wb = st.session_state.get("waterbody_choice", None)
    idx = st.session_state.get("index_choice", None)
    analysis = st.session_state.get("analysis_choice", None)
    debug("DEBUG: Επιλεγμένα: υδάτινο σώμα =", wb, "δείκτης =", idx, "ανάλυση =", analysis)
    if idx == "Burned Areas" and analysis == "Burned Areas":
        if wb in ["Γαδουρά", "Κορώνεια"]:
            run_lake_processing_app(wb, idx)
        else:
            st.warning("Τα Burned Areas είναι διαθέσιμα μόνο για Γαδουρά (ή Κορώνεια, αν υπάρχουν δεδομένα).")
        return
    if idx == "Burned Areas" and analysis == "Water Quality Dashboard":
        if wb == "Γαδουρά":
            run_water_quality_dashboard(wb, idx)
        else:
            st.warning("Το Dashboard για Burned Areas είναι διαθέσιμο μόνο στη Γαδουρά.")
        return
    if idx == "Χλωροφύλλη" and wb in ["Κορώνεια", "Πολυφύτου", "Γαδουρά", "Αξιός"]:
        if analysis == "Lake Processing":
            run_lake_processing_app(wb, idx)
        elif analysis == "Water Processing":
            run_water_processing(wb, idx)
        elif analysis == "Water Quality Dashboard":
            run_water_quality_dashboard(wb, idx)
        elif analysis == "Water level":
            run_water_level_profiles(wb, idx)
        elif analysis == "Pattern Analysis":
            run_pattern_analysis(wb, idx)
        else:
            st.info("Παρακαλώ επιλέξτε ένα είδος ανάλυσης.")
    elif analysis == "Burned Areas":
        if wb == "Γαδουρά":
            st.warning("Η ενότητα Burned Areas είναι υπό ανάπτυξη.")
        else:
            st.warning("Το 'Burned Areas' είναι διαθέσιμο μόνο για το υδάτινο σώμα Γαδουρά.")
    else:
        st.warning("Δεν υπάρχουν διαθέσιμα δεδομένα για αυτόν τον συνδυασμό δείκτη/υδάτινου σώματος. "
                   "Για παράδειγμα, η Χλωροφύλλη είναι διαθέσιμη μόνο για (Κορώνεια, Πολυφύτου, Γαδουρά, Αξιός).")

if __name__ == "__main__":
    main()
