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
        /* Ενότητα πλοήγησης στην πλαϊνή μπάρα */
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
# Βοηθητική Συνάρτηση για Επιλογή φακέλου δεδομένων
# -----------------------------------------------------------------------------
def get_data_folder(waterbody: str, index: str) -> str:
    """
    Αντιστοιχεί το επιλεγμένο υδάτινο σώμα και δείκτη στον σωστό φάκελο δεδομένων.
    Επιστρέφει None αν δεν υπάρχει ο φάκελος.
    """
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

# -----------------------------------------------------------------------------
# Βοηθητικές Συναρτήσεις για Εξαγωγή Δεδομένων και Επεξεργασία Εικόνας
# -----------------------------------------------------------------------------
def extract_date_from_filename(filename: str):
    """
    Εξάγει ημερομηνία (YYYY-MM-DD) από το όνομα του αρχείου χρησιμοποιώντας regex.
    Επιστρέφει (day_of_year, datetime_obj) ή (None, None) αν δεν βρεθεί ταίριασμα.
    """
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
    """
    Φορτώνει το περίγραμμα μιας λίμνης από ένα προσαρμοσμένο XML αρχείο.
    Εάν δοθούν όρια, μετατρέπει τις συντεταγμένες.
    """
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
    """
    Διαβάζει ένα GeoTIFF αρχείο (1 κανάλι) και, εάν δοθεί, εφαρμόζει μάσκα βάσει του περιγράμματος.
    Επιστρέφει (εικόνα, profile).
    """
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
    """
    Διαβάζει όλα τα TIF αρχεία από το input_folder, εφαρμόζει μάσκα (εάν υπάρχει) και εξάγει πληροφορίες ημερομηνίας.
    Επιστρέφει (stack, array ημερών, λίστα ημερομηνιών).
    """
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
    """Εμφανίζει μια κάρτα εισαγωγής με λογότυπο και τίτλο."""
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
            st.markdown(
                "<h2 class='header-title'>Ποιοτικά χαρακτηριστικά Επιφανειακού Ύδατος</h2>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='text-align: center; font-size: 1.1rem;'>"
                "Αυτή η εφαρμογή ανάλυσης χρησιμοποιεί εργαλεία δορυφορικής τηλεπισκόπησης. "
                "Επιλέξτε τις ρυθμίσεις στην πλαϊνή μπάρα και εξερευνήστε τα δεδομένα.</p>",
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Πλαϊνή Μπάρα Πλοήγησης (Custom UI)
# -----------------------------------------------------------------------------
def run_custom_ui():
    """Δημιουργεί την πλαϊνή μπάρα για επιλογή υδάτινου σώματος, δείκτη και είδους ανάλυσης."""
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
# Επεξεργασία Λίμνης (Lake Processing) με Μηνιαία και Ετήσια Ανάλυση
# -----------------------------------------------------------------------------
def run_lake_processing_app(waterbody: str, index: str):
    """Κύρια συνάρτηση για την ανάλυση μιας λίμνης με μηνιαία και ετήσια διαγράμματα."""
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

        # Βασικά φίλτρα από την πλαϊνή μπάρα
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

        # Διάγραμμα "Ημέρες σε Εύρος"
        days_in_range = np.nansum(in_range, axis=0)
        fig_days = px.imshow(days_in_range, color_continuous_scale="plasma",
                             title="Διάγραμμα: Ημέρες σε Εύρος", labels={"color": "Ημέρες σε Εύρος"})
        fig_days.update_layout(width=800, height=600)
        st.plotly_chart(fig_days, use_container_width=True, key="fig_days")
        with st.expander("Επεξήγηση: Ημέρες σε Εύρος"):
            st.write("Το διάγραμμα αυτό δείχνει πόσες ημέρες κάθε pixel βρίσκεται εντός του επιλεγμένου εύρους τιμών. Ρυθμίστε το 'Εύρος τιμών pixel' για να δείτε πώς αλλάζει το αποτέλεσμα.")

        # Ορισμός κοινών μεταβλητών για τα διαγράμματα (ticks)
        tick_vals = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365]
        tick_text = ["1 (Ιαν)", "32 (Φεβ)", "60 (Μαρ)", "91 (Απρ)",
                     "121 (Μαΐ)", "152 (Ιουν)", "182 (Ιουλ)", "213 (Αυγ)",
                     "244 (Σεπ)", "274 (Οκτ)", "305 (Νοε)", "335 (Δεκ)", "365 (Δεκ)"]

        # Διάγραμμα "Μέση Ημέρα Εμφάνισης"
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
            st.write("Το διάγραμμα αυτό παρουσιάζει τη μέση ημέρα εμφάνισης για τα pixels που πληρούν το επιλεγμένο εύρος τιμών. Πειραματιστείτε με το 'Εύρος τιμών pixel' για να δείτε πώς μεταβάλλεται η μέση ημέρα.")

        # Διάγραμμα "Μέσο Δείγμα Εικόνας"
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
            st.write("Το διάγραμμα αυτό δείχνει τη μέση τιμή των pixels μετά την εφαρμογή του φίλτρου. Επιλέξτε 'Thresholded' ή 'Original' για να δείτε τη φιλτραρισμένη ή την αρχική εικόνα.")

        # Διάγραμμα "Χρόνος Μέγιστης Εμφάνισης"
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
            st.write("Αυτό το διάγραμμα δείχνει την ημέρα του έτους κατά την οποία κάθε pixel πέτυχε τη μέγιστη τιμή εντός του επιλεγμένου εύρους. Πειραματιστείτε με το 'Εύρος τιμών pixel' για να δείτε πώς αλλάζει το αποτέλεσμα.")

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
        # Επιπρόσθετη Ετήσια Ανάλυση: Μηνιαία Κατανομή Ημερών σε Εύρος
        # ------------------------------
        st.header("Επιπρόσθετη Ετήσια Ανάλυση: Μηνιαία Κατανομή Ημερών σε Εύρος")
        # Note: We already have STACK, lower_thresh, and DATES defined earlier.
        stack_full_in_range = (STACK >= lower_thresh) & (STACK <= upper_thresh)
        monthly_days_in_range = {}
        for m in range(1, 13):
            indices_m = [i for i, d in enumerate(DATES) if d is not None and d.month == m]
            if indices_m:
                monthly_days_in_range[m] = np.sum(stack_full_in_range[indices_m, :, :], axis=0)
            else:
                monthly_days_in_range[m] = None

        # Display monthly heatmaps in a grid using Streamlit columns (4 per row)
        num_cols = 4
        cols = st.columns(num_cols)
        for m in range(1, 13):
            col_index = (m - 1) % num_cols
            img = monthly_days_in_range[m]
            month_name = datetime(2000, m, 1).strftime('%B')
            if img is not None:
                fig_month = px.imshow(np.flipud(img),
                                      color_continuous_scale="plasma",
                                      title=month_name,
                                      labels={"color": "Ημέρες σε Εύρος"})
                fig_month.update_layout(height=300)
                cols[col_index].plotly_chart(fig_month, use_container_width=True)
            else:
                cols[col_index].info(f"Δεν υπάρχουν δεδομένα για τον μήνα {month_name}")
            # After every row (4 plots), create a new row of columns
            if m % num_cols == 0 and m != 12:
                cols = st.columns(num_cols)

        with st.expander("Επεξήγηση: Μηνιαία Κατανομή Ημερών σε Εύρος"):
            st.write(
                "Για κάθε μήνα, αυτό το διάγραμμα δείχνει πόσες ημέρες κάθε pixel βρέθηκε εντός του "
                "επιλεγμένου εύρους τιμών. Το εύρος τιμών ορίζεται από το slider 'Εύρος τιμών pixel'."
            )

        # ------------------------------
        # Επιπρόσθετη Ετήσια Ανάλυση: Ετήσια Κατανομή Ημερών σε Εύρος
        # ------------------------------
        st.header("Επιπρόσθετη Ετήσια Ανάλυση: Ετήσια Κατανομή Ημερών σε Εύρος")
        unique_years_full = sorted({d.year for d in DATES if d is not None})
        if not unique_years_full:
            st.error("Δεν βρέθηκαν έγκυρα έτη στα δεδομένα.")
            st.stop()

        stack_full_in_range = (STACK >= lower_thresh) & (STACK <= upper_thresh)
        yearly_days_in_range = {}
        for year in unique_years_full:
            indices_y = [i for i, d in enumerate(DATES) if d.year == year]
            if indices_y:
                yearly_days_in_range[year] = np.sum(stack_full_in_range[indices_y, :, :], axis=0)
            else:
                yearly_days_in_range[year] = None

        n_years = len(unique_years_full)
        vertical_spacing = 0.02  # Ρυθμισμένη τιμή για το vertical_spacing
        fig_yearly = make_subplots(
            rows=n_years, cols=1,
            subplot_titles=[f"Έτος: {year}" for year in unique_years_full],
            vertical_spacing=vertical_spacing
        )
        for idx, year in enumerate(unique_years_full, start=1):
            img = yearly_days_in_range[year]
            if img is not None:
                fig_yearly.add_trace(
                    go.Heatmap(
                        z=np.flipud(img),
                        colorscale="plasma",
                        colorbar=dict(title="Ημέρες σε Εύρος", len=0.5) if idx == 1 else dict(showticklabels=False)
                    ),
                    row=idx, col=1
                )
            else:
                fig_yearly.add_annotation(
                    text="Δεν υπάρχουν δεδομένα",
                    row=idx, col=1,
                    showarrow=False
                )
        fig_yearly.update_layout(
            height=300 * n_years,
            title_text="Ετήσια Κατανομή Ημερών σε Εύρος"
        )
        st.plotly_chart(fig_yearly, use_container_width=True, key="fig_yearly")
        with st.expander("Επεξήγηση: Ετήσια Κατανομή Ημερών σε Εύρος"):
            st.write("Για κάθε έτος, αυτό το διάγραμμα δείχνει πόσες ημέρες κάθε pixel βρέθηκε εντός του επιλεγμένου εύρους τιμών, επιτρέποντάς σας να συγκρίνετε τις ετήσιες αλλαγές στη γεωχωρική κατανομή του δείκτη.")

        st.info("Τέλος Επεξεργασίας Λίμνης.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Επεξεργασία Υδάτινου Σώματος (Placeholder)
# -----------------------------------------------------------------------------
def run_water_processing(waterbody: str, index: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Επεξεργασία Υδάτινου Σώματος ({waterbody} - {index}) [Placeholder]")
        st.info("Δεν υπάρχουν δεδομένα ή λειτουργίες για την επεξεργασία υδάτινου σώματος.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Πίνακας Ποιότητας Ύδατος
# -----------------------------------------------------------------------------
def run_water_quality_dashboard(waterbody: str, index: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Πίνακας Ποιότητας Ύδατος ({waterbody} - {index})")

        data_folder = get_data_folder(waterbody, index)
        if data_folder is None:
            st.error("Δεν υπάρχει φάκελος δεδομένων για το επιλεγμένο υδάτινο σώμα/δείκτη.")
            st.stop()

        images_folder = os.path.join(data_folder, "GeoTIFFs")
        lake_height_path = os.path.join(data_folder, "lake height.xlsx")
        sampling_kml_path = os.path.join(data_folder, "sampling.kml")
        possible_video = [
            os.path.join(data_folder, "timelapse.mp4"),
            os.path.join(data_folder, "Sentinel-2_L1C-202307221755611-timelapse.gif"),
            os.path.join(images_folder, "Sentinel-2_L1C-202307221755611-timelapse.gif")
        ]
        video_path = None
        for v in possible_video:
            if os.path.exists(v):
                video_path = v
                break

        st.sidebar.header(f"Ρυθμίσεις Πίνακα ({waterbody} - Dashboard)")
        x_start = st.sidebar.date_input("Έναρξη", date(2015, 1, 1), key="wq_start")
        x_end = st.sidebar.date_input("Λήξη", date(2026, 12, 31), key="wq_end")
        x_start_dt = datetime.combine(x_start, datetime.min.time())
        x_end_dt = datetime.combine(x_end, datetime.min.time())

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
                    debug("DEBUG: Σφάλμα εξαγωγής ημερομηνίας από", filename, ":", e)
                    continue

        if available_dates:
            sorted_dates = sorted(available_dates.keys())
            selected_bg_date = st.selectbox("Επιλέξτε ημερομηνία για το background", sorted_dates, key="wq_bg")
        else:
            selected_bg_date = None
            st.warning("Δεν βρέθηκαν GeoTIFF εικόνες με ημερομηνία.")

        if selected_bg_date is not None:
            bg_filename = available_dates[selected_bg_date]
            bg_path = os.path.join(images_folder, bg_filename)
            if os.path.exists(bg_path):
                with rasterio.open(bg_path) as src:
                    if src.count >= 3:
                        first_image_data = src.read([1, 2, 3])
                        first_transform = src.transform
                    else:
                        st.error("Το επιλεγμένο GeoTIFF δεν περιέχει τουλάχιστον 3 κανάλια.")
                        st.stop()
            else:
                st.error(f"Δεν βρέθηκε το GeoTIFF background: {bg_path}")
                st.stop()
        else:
            st.error("Δεν έχει επιλεγεί έγκυρη ημερομηνία για το background.")
            st.stop()

        def parse_sampling_kml(kml_file) -> list:
            try:
                tree = ET.parse(kml_file)
                root = tree.getroot()
                namespace = {'kml': 'http://www.opengis.net/kml/2.2'}
                points = []
                for linestring in root.findall('.//kml:LineString', namespace):
                    coord_text = linestring.find('kml:coordinates', namespace).text.strip()
                    coords = coord_text.split()
                    for idx, coord in enumerate(coords):
                        lon_str, lat_str, *_ = coord.split(',')
                        points.append((f"Point {idx+1}", float(lon_str), float(lat_str)))
                return points
            except Exception as e:
                st.error("Σφάλμα ανάλυσης KML:", e)
                return []

        def geographic_to_pixel(lon: float, lat: float, transform) -> tuple:
            inverse_transform = ~transform
            col, row = inverse_transform * (lon, lat)
            return int(col), int(row)

        def map_rgb_to_mg(r: float, g: float, b: float, mg_factor: float = 2.0) -> float:
            return (g / 255.0) * mg_factor

        def mg_to_color(mg: float) -> str:
            scale = [
                (0.00, "#0000ff"), (0.02, "#0007f2"), (0.04, "#0011de"),
                (0.06, "#0017d0"), (1.98, "#80007d"), (2.00, "#800080")
            ]
            if mg <= scale[0][0]:
                color = scale[0][1]
            elif mg >= scale[-1][0]:
                color = scale[-1][1]
            else:
                for i in range(len(scale) - 1):
                    low_mg, low_color = scale[i]
                    high_mg, high_color = scale[i+1]
                    if low_mg <= mg <= high_mg:
                        t = (mg - low_mg) / (high_mg - low_mg)
                        low_rgb = tuple(int(low_color[j:j+2], 16) for j in (1, 3, 5))
                        high_rgb = tuple(int(high_color[j:j+2], 16) for j in (1, 3, 5))
                        rgb = tuple(int(low_rgb[k] + (high_rgb[k] - low_rgb[k]) * t) for k in range(3))
                        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
            rgb = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
            return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

        def analyze_sampling(sampling_points: list, first_image_data, first_transform,
                             images_folder: str, lake_height_path: str, selected_points: list = None):
            results_colors = {name: [] for name, _, _ in sampling_points}
            results_mg = {name: [] for name, _, _ in sampling_points}

            for filename in sorted(os.listdir(images_folder)):
                if filename.lower().endswith(('.tif', '.tiff')):
                    match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
                    if not match:
                        continue
                    date_str = match.group(1)
                    try:
                        date_obj = datetime.strptime(date_str, '%Y_%m_%d')
                    except ValueError:
                        continue
                    image_path = os.path.join(images_folder, filename)
                    with rasterio.open(image_path) as src:
                        transform = src.transform
                        width, height = src.width, src.height
                        if src.count < 3:
                            continue
                        for name, lon, lat in sampling_points:
                            col, row = geographic_to_pixel(lon, lat, transform)
                            if 0 <= col < width and 0 <= row < height:
                                window = rasterio.windows.Window(col, row, 1, 1)
                                r = src.read(1, window=window)[0, 0]
                                g = src.read(2, window=window)[0, 0]
                                b = src.read(3, window=window)[0, 0]
                                mg_value = map_rgb_to_mg(r, g, b)
                                results_mg[name].append((date_obj, mg_value))
                                pixel_color = (r / 255, g / 255, b / 255)
                                results_colors[name].append((date_obj, pixel_color))

            rgb_image = first_image_data.transpose((1, 2, 0)) / 255.0
            fig_geo = px.imshow(rgb_image, title='Εικόνα GeoTIFF με Σημεία Δειγματοληψίας')
            for name, lon, lat in sampling_points:
                col, row = geographic_to_pixel(lon, lat, first_transform)
                fig_geo.add_trace(go.Scatter(x=[col], y=[row], mode='markers',
                                             marker=dict(color='red', size=8), name=name))
            fig_geo.update_xaxes(visible=False)
            fig_geo.update_yaxes(visible=False)
            fig_geo.update_layout(width=900, height=600, showlegend=True)

            try:
                lake_data = pd.read_excel(lake_height_path)
                lake_data['Date'] = pd.to_datetime(lake_data.iloc[:, 0])
                lake_data.sort_values('Date', inplace=True)
            except Exception as e:
                st.error(f"Σφάλμα ανάγνωσης αρχείου ύψους λίμνης: {e}")
                lake_data = pd.DataFrame()

            scatter_traces = []
            point_names = list(results_colors.keys())
            if selected_points is not None:
                point_names = [p for p in point_names if p in selected_points]

            for idx, name in enumerate(point_names):
                data_list = results_colors[name]
                if not data_list:
                    continue
                data_list.sort(key=lambda x: x[0])
                dates = [d for d, _ in data_list]
                colors = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for _, c in data_list]
                scatter_traces.append(go.Scatter(x=dates, y=[idx] * len(dates),
                                                 mode='markers',
                                                 marker=dict(color=colors, size=10),
                                                 name=name))
            fig_colors = make_subplots(specs=[[{"secondary_y": True}]])
            for trace in scatter_traces:
                fig_colors.add_trace(trace, secondary_y=False)

            if not lake_data.empty:
                trace_height = go.Scatter(
                    x=lake_data['Date'],
                    y=lake_data[lake_data.columns[1]],
                    name='Ύψος Λίμνης', mode='lines', line=dict(color='blue', width=2)
                )
                fig_colors.add_trace(trace_height, secondary_y=True)

            fig_colors.update_layout(title='Χρώματα Pixel και Ύψος Λίμνης με την πάροδο του χρόνου',
                                     xaxis_title='Ημερομηνία',
                                     yaxis_title='Σημεία Δειγματοληψίας',
                                     showlegend=True)
            fig_colors.update_yaxes(title_text="Ύψος Λίμνης", secondary_y=True)

            all_dates_dict = {}
            for data_list in results_mg.values():
                for date_obj, mg_val in data_list:
                    all_dates_dict.setdefault(date_obj, []).append(mg_val)
            sorted_dates = sorted(all_dates_dict.keys())
            avg_mg = [np.mean(all_dates_dict[d]) for d in sorted_dates]

            fig_mg = go.Figure()
            fig_mg.add_trace(go.Scatter(
                x=sorted_dates,
                y=avg_mg,
                mode='markers',
                marker=dict(color=avg_mg, colorscale='Viridis', reversescale=True,
                            colorbar=dict(title='mg/m³'), size=10),
                name='Μέσο mg/m³'
            ))
            fig_mg.update_layout(title='Μέσο mg/m³ με την πάροδο του χρόνου',
                                 xaxis_title='Ημερομηνία', yaxis_title='mg/m³',
                                 showlegend=False)

            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            if not lake_data.empty:
                fig_dual.add_trace(go.Scatter(
                    x=lake_data['Date'],
                    y=lake_data[lake_data.columns[1]],
                    name='Ύψος Λίμνης', mode='lines'
                ), secondary_y=False)
            fig_dual.add_trace(go.Scatter(
                x=sorted_dates,
                y=avg_mg,
                name='Μέσο mg/m³',
                mode='markers',
                marker=dict(color=avg_mg, colorscale='Viridis', reversescale=True,
                            colorbar=dict(title='mg/m³'), size=10)
            ), secondary_y=True)
            fig_dual.update_layout(title='Ύψος Λίμνης και Μέσο mg/m³ με την πάροδο του χρόνου',
                                   xaxis_title='Ημερομηνία', showlegend=True)
            fig_dual.update_yaxes(title_text="Ύψος Λίμνης", secondary_y=False)
            fig_dual.update_yaxes(title_text="mg/m³", secondary_y=True)

            return fig_geo, fig_dual, fig_colors, fig_mg, results_colors, results_mg, lake_data

        # Δύο καρτέλες δειγματοληψίας
        if "default_results" not in st.session_state:
            st.session_state.default_results = None
        if "upload_results" not in st.session_state:
            st.session_state.upload_results = None

        tab_names = ["Δειγματοληψία 1 (Default)", "Δειγματοληψία 2 (Upload)"]
        tabs = st.tabs(tab_names)

        # Καρτέλα 1 (Default)
        with tabs[0]:
            st.header("Ανάλυση για Δειγματοληψία 1 (Default)")
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
                    st.error("Σφάλμα μορφοποίησης αποτελεσμάτων ανάλυσης. Παρακαλώ επαναλάβετε την ανάλυση.")
                    st.stop()
                nested_tabs = st.tabs(["GeoTIFF", "Video/GIF", "Χρώματα Pixel", "Μέσο mg", "Διπλά Διαγράμματα", "Λεπτομερής ανάλυση mg"])
                with nested_tabs[0]:
                    st.plotly_chart(fig_geo, use_container_width=True, key="default_fig_geo")
                with nested_tabs[1]:
                    if video_path is not None:
                        if video_path.endswith(".mp4"):
                            st.video(video_path, key="default_video")
                        else:
                            st.image(video_path)
                    else:
                        st.info("Δεν βρέθηκε αρχείο timelapse.")
                with nested_tabs[2]:
                    st.plotly_chart(fig_colors, use_container_width=True, key="default_fig_colors")
                with nested_tabs[3]:
                    st.plotly_chart(fig_mg, use_container_width=True, key="default_fig_mg")
                with nested_tabs[4]:
                    st.plotly_chart(fig_dual, use_container_width=True, key="default_fig_dual")
                with nested_tabs[5]:
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
        # Καρτέλα 2 (Upload)
        with tabs[1]:
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
                    nested_tabs = st.tabs(["GeoTIFF", "Video/GIF", "Χρώματα Pixel", "Μέσο mg", "Διπλά Διαγράμματα", "Λεπτομερής ανάλυση mg"])
                    with nested_tabs[0]:
                        st.plotly_chart(fig_geo, use_container_width=True, key="upload_fig_geo")
                    with nested_tabs[1]:
                        if video_path is not None:
                            if video_path.endswith(".mp4"):
                                st.video(video_path, key="upload_video")
                            else:
                                st.image(video_path)
                        else:
                            st.info("Δεν βρέθηκε αρχείο Video/GIF.")
                    with nested_tabs[2]:
                        st.plotly_chart(fig_colors, use_container_width=True, key="upload_fig_colors")
                    with nested_tabs[3]:
                        st.plotly_chart(fig_mg, use_container_width=True, key="upload_fig_mg")
                    with nested_tabs[4]:
                        st.plotly_chart(fig_dual, use_container_width=True, key="upload_fig_dual")
                    with nested_tabs[5]:
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
                st.info("Παρακαλώ ανεβάστε ένα αρχείο KML για νέα σημεία δειγματοληψίας.", key="upload_info")

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
# Ανάλυση Προτύπων
# -----------------------------------------------------------------------------
def run_pattern_analysis(waterbody: str, index: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Ανάλυση Προτύπων ({waterbody} - {index})")

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

        st.sidebar.header("Ελέγχοι Ανάλυσης Προτύπων")
        unique_years = sorted({d.year for d in DATES if d is not None})
        selected_years_pattern = st.sidebar.multiselect(
            "Επιλέξτε έτη",
            options=unique_years,
            default=unique_years,
            key="pattern_years"
        )
        selected_months_pattern = st.sidebar.multiselect(
            "Επιλέξτε μήνες",
            options=list(range(1, 13)),
            default=list(range(1, 13)),
            key="pattern_months",
            format_func=lambda m: datetime(2000, m, 1).strftime('%B')
        )
        threshold_range = st.sidebar.slider("Εύρος τιμών pixel", 0, 255, (0, 255), key="pattern_threshold")
        lower_thresh, upper_thresh = threshold_range

        if not selected_years_pattern or not selected_months_pattern:
            st.error("Παρακαλώ επιλέξτε τουλάχιστον ένα έτος και έναν μήνα.")
            st.stop()

        indices = [i for i, d in enumerate(DATES)
                   if d.year in selected_years_pattern and d.month in selected_months_pattern]
        if not indices:
            st.error("Δεν υπάρχουν δεδομένα για τα επιλεγμένα έτη/μήνες στην ανάλυση προτύπων.")
            st.stop()

        STACK_filtered = STACK[indices, :, :]
        stack_full_in_range = (STACK_filtered >= lower_thresh) & (STACK_filtered <= upper_thresh)
        filtered_dates = [DATES[i] for i in indices]

        monthly_avg = {}
        for m in selected_months_pattern:
            month_indices = [i for i, dd in enumerate(filtered_dates) if dd.month == m]
            if month_indices:
                avg_days = np.nanmean(stack_full_in_range[month_indices, :, :], axis=0)
                monthly_avg[m] = avg_days
            else:
                monthly_avg[m] = None

        agg_avg = None
        count = 0
        for m in monthly_avg:
            if monthly_avg[m] is not None:
                if agg_avg is None:
                    agg_avg = monthly_avg[m]
                else:
                    agg_avg += monthly_avg[m]
                count += 1
        overall_avg = (agg_avg / count) if (agg_avg is not None and count > 0) else None

        temporal_data = []
        for mm in sorted(monthly_avg.keys()):
            if monthly_avg[mm] is not None:
                spatial_avg = np.nanmean(monthly_avg[mm])
                temporal_data.append((mm, spatial_avg))

        if temporal_data:
            months, means = zip(*temporal_data)
            month_names = [datetime(2000, mm, 1).strftime('%B') for mm in months]
            fig_temporal = px.bar(
                x=month_names,
                y=means,
                labels={'x': 'Μήνας', 'y': 'Μέσο Ποσοστό σε Εύρος'},
                title="Χρονολογικό Πρότυπο ανά Μήνα"
            )
        else:
            fig_temporal = go.Figure()

        if overall_avg is not None:
            classification = np.full(overall_avg.shape, "Μη Ταξινομημένο", dtype=object)
            valid_mask = ~np.isnan(overall_avg)
            classification[valid_mask & (overall_avg < 0.3)] = "Χαμηλό"
            classification[valid_mask & (overall_avg >= 0.3) & (overall_avg < 0.7)] = "Μέτριο"
            classification[valid_mask & (overall_avg >= 0.7)] = "Υψηλό"
            mapping_dict = {"Χαμηλό": 0, "Μέτριο": 1, "Υψηλό": 2, "Μη Ταξινομημένο": 3}
            numeric_class = np.vectorize(lambda x: mapping_dict[x])(classification)
            discrete_colorscale = [
                [0.00, "blue"],
                [0.33, "yellow"],
                [0.66, "red"],
                [1.00, "gray"]
            ]
            fig_class = px.imshow(
                numeric_class,
                color_continuous_scale=discrete_colorscale,
                title="Χωρική Ταξινόμηση"
            )
            fig_class.update_traces(
                colorbar=dict(tickvals=[0, 1, 2, 3],
                              ticktext=["Χαμηλό", "Μέτριο", "Υψηλό", "Μη Ταξινομημένο"])
            )
        else:
            fig_class = go.Figure()

        st.header("Ανάλυση Προτύπων")
        st.markdown("Η ανάλυση αυτή παρουσιάζει το χρονολογικό πρότυπο (μηνιαία) και την χωρική ταξινόμηση των δεδομένων.")
        st.subheader("Χρονολογικό Πρότυπο")
        st.plotly_chart(fig_temporal, use_container_width=True, key="pattern_fig_temporal")
        with st.expander("Επεξήγηση: Χρονολογικό Πρότυπο ανά Μήνα"):
            st.write("Το διάγραμμα αυτό δείχνει το μέσο ποσοστό των pixels που βρίσκονται εντός του επιλεγμένου εύρους τιμών για κάθε μήνα. Μπορείτε να τροποποιήσετε το 'Εύρος τιμών pixel' για να δείτε πώς αλλάζει το πρότυπο.")
        st.subheader("Χωρική Ταξινόμηση")
        st.plotly_chart(fig_class, use_container_width=True, key="pattern_fig_class")
        with st.expander("Επεξήγηση: Χωρική Ταξινόμηση"):
            st.write("Το διάγραμμα αυτό ταξινομεί χωρικά τα pixels με βάση το μέσο ποσοστό τους εντός του εύρους τιμών. Αυτό σας επιτρέπει να εντοπίσετε περιοχές με χαμηλές, μέτριες ή υψηλές τιμές.")
        if temporal_data:
            df_temporal = pd.DataFrame(temporal_data, columns=["Μήνας", "Μέσο Ποσοστό σε Εύρος"])
            csv = df_temporal.to_csv(index=False).encode('utf-8')
            st.download_button("Λήψη CSV ανάλυσης", data=csv,
                               file_name="χρονική_ανάλυση.csv", mime="text/csv", key="pattern_csv")

        st.info("Τέλος Ανάλυσης Προτύπων.")
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

    # Δρομολόγηση στην αντίστοιχη λειτουργία
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
        st.warning(
            "Δεν υπάρχουν διαθέσιμα δεδομένα για αυτόν τον συνδυασμό δείκτη/υδάτινου σώματος. "
            "Για παράδειγμα, η Χλωροφύλλη είναι διαθέσιμη μόνο για (Κορώνεια, Πολυφύτου, Γαδουρά, Αξιός)."
        )

if __name__ == "__main__":
    main()
