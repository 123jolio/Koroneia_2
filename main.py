#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Water Quality App (Professional UI Version)
---------------------------------------------
This version features an improved, professional UI and more user‐friendly navigation.
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

# Global debug flag (set to False for production)
DEBUG = False
def debug(*args, **kwargs):
    if DEBUG:
        st.write(*args, **kwargs)

# -------------------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Ποιοτικά χαρακτηριστικά Επιφανειακού Ύδατος",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 1) Custom CSS injection for a refined Dark Theme
# -----------------------------------------------------------------------------
def inject_custom_css():
    custom_css = """
    <link href="https://fonts.googleapis.com/css?family=Roboto:400,500,700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] { font-family: 'Roboto', sans-serif; }
        .block-container { background-color: #121212; color: #e0e0e0; padding: 1rem; }
        h1, h2, h3, h4, h5, h6 { color: #bb86fc; }
        .stButton button {
            background-color: #3700b3; color: #ffffff; border-radius: 5px;
            padding: 8px 16px; border: none; box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
        }
        .stButton button:hover { background-color: #6200ee; }
        .sidebar .sidebar-content { background-color: #1f1f1f; padding: 1rem; }
        .stSelectbox, .stSlider, .stTextInput { background-color: #1f1f1f; color: #e0e0e0; }
        .card {
            background-color: #1e1e1e; padding: 1.5rem; border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.5); margin-bottom: 1.5rem;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

inject_custom_css()

# -----------------------------------------------------------------------------
# 2) Data Folder Helper
# -----------------------------------------------------------------------------
def get_data_folder(waterbody: str, index: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    debug("DEBUG: Called get_data_folder with", waterbody, index)

    if waterbody == "Κορώνεια":
        waterbody_folder = "Koroneia"
    elif waterbody == "Πολυφύτου":
        waterbody_folder = "polyphytou"
    elif waterbody == "Γαδουρά":
        waterbody_folder = "Gadoura"
    elif waterbody == "Αξιός":
        waterbody_folder = "Axios"
    else:
        waterbody_folder = None

    if waterbody_folder is None:
        return None

    if index == "Χλωροφύλλη":
        data_folder = os.path.join(base_dir, waterbody_folder, "Chlorophyll")
    elif index == "Burned Areas":
        data_folder = os.path.join(base_dir, waterbody_folder, "Burned Areas")
    else:
        data_folder = None

    debug("DEBUG: data_folder =", data_folder)
    if data_folder is not None and not os.path.exists(data_folder):
        st.error(f"Folder does NOT exist on disk: {data_folder}")
        return None

    return data_folder

# -----------------------------------------------------------------------------
# 3) Helper functions for data extraction and reading
# -----------------------------------------------------------------------------
def extract_date_from_filename(filename: str):
    basename = os.path.basename(filename)
    debug("DEBUG: Extracting date from", basename)
    match = re.search(r'(\d{4})[_-](\d{2})[_-](\d{2})', basename)
    if match:
        year, month, day = match.groups()
        date_obj = datetime(int(year), int(month), int(day))
        day_of_year = date_obj.timetuple().tm_yday
        debug("DEBUG: Extracted date:", date_obj)
        return day_of_year, date_obj
    debug("DEBUG: Date pattern not found in", basename)
    return None, None

def load_lake_shape_from_xml(xml_file: str, bounds: tuple = None,
                             xml_width: float = 518.0, xml_height: float = 505.0):
    debug("DEBUG: Loading lake shape from:", xml_file)
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
            st.warning("No points found in", xml_file)
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
        debug("DEBUG: Loaded lake shape with", len(points), "points.")
        return {"type": "Polygon", "coordinates": [points]}
    except Exception as e:
        st.error(f"Error reading lake shape from {xml_file}: {e}")
        return None

def read_image(file_path: str, lake_shape: dict = None):
    debug("DEBUG: Reading image from:", file_path)
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
    debug("DEBUG: load_data called with", input_folder)
    if not os.path.exists(input_folder):
        raise Exception(f"Folder does not exist: {input_folder}")

    shapefile_path_xml = os.path.join(input_folder, shapefile_name)
    shapefile_path_txt = os.path.join(input_folder, "shapefile.txt")
    debug("DEBUG: Checking shapefile at:", shapefile_path_xml, shapefile_path_txt)

    lake_shape = None
    if os.path.exists(shapefile_path_xml):
        shape_file = shapefile_path_xml
    elif os.path.exists(shapefile_path_txt):
        shape_file = shapefile_path_txt
    else:
        shape_file = None
        debug("DEBUG: No shapefile found in", input_folder)

    all_tif_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    tif_files = [fp for fp in all_tif_files if os.path.basename(fp).lower() != "mask.tif"]
    if not tif_files:
        raise Exception("No GeoTIFF files found in the specified folder.")

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
        raise Exception("No valid images were loaded.")

    stack = np.stack(images, axis=0)
    return stack, np.array(days), date_list

# -----------------------------------------------------------------------------
# 4) Introductory Page
# -----------------------------------------------------------------------------
def run_intro_page():
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col_logo, col_text = st.columns([1, 3])
        with col_logo:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(base_dir, "logo.jpg")
            if os.path.exists(logo_path):
                st.image(logo_path, width=150)
        with col_text:
            st.markdown(
                "<h2 style='text-align: center;'>Ποιοτικά χαρακτηριστικά Επιφανειακού Ύδατος σε Λίμνες, "
                "Ταμιευτήρες και Ποτάμια με χρήση Εργαλείων Δορυφορικής Τηλεπισκόπησης</h2>", 
                unsafe_allow_html=True
            )
        st.markdown("""
        <div class="card">
          <h4>Εισαγωγή</h4>
          <p>Αυτή η εφαρμογή αναλύει τα ποιοτικά χαρακτηριστικά του επιφανειακού ύδατος με χρήση δορυφορικών εικόνων. 
          Επιλέξτε τις επιθυμητές ρυθμίσεις από το αριστερό μενού και εξερευνήστε τα δεδομένα.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5) Sidebar Configuration (Navigation and Parameter Selection)
# -----------------------------------------------------------------------------
def run_configuration():
    st.sidebar.header("Παραμετροποίηση Ανάλυσης")
    waterbody = st.sidebar.selectbox("Επιλογή υδάτινου σώματος", 
                                     ["Κορώνεια", "Πολυφύτου", "Γαδουρά", "Αξιός"],
                                     key="waterbody_choice")
    index = st.sidebar.selectbox("Επιλογή Δείκτη", 
                                 ["Πραγματικό", "Χλωροφύλλη", "CDOM", "Colour", "Burned Areas"],
                                 key="index_choice")
    analysis = st.sidebar.radio("Επιλογή Ανάλυσης", 
                                ["Lake Processing", "Water Processing", "Water Quality Dashboard",
                                 "Burned Areas", "Water level", "Pattern Analysis"],
                                key="analysis_choice")
    st.sidebar.markdown("---")
    st.sidebar.info(f"Επιλεγμένο: {waterbody} | {index} | {analysis}")
    return waterbody, index, analysis

# -----------------------------------------------------------------------------
# 6) All the Analysis Functions
# -----------------------------------------------------------------------------
def run_lake_processing_app(waterbody: str, index: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Lake Processing ({waterbody} - {index})")
        data_folder = get_data_folder(waterbody, index)
        if data_folder is None:
            st.error("Δεν υπάρχει φάκελος δεδομένων για το επιλεγμένο υδάτινο σώμα / δείκτη.")
            st.stop()

        input_folder = os.path.join(data_folder, "GeoTIFFs")
        try:
            with st.spinner("Φόρτωση δεδομένων ..."):
                STACK, DAYS, DATES = load_data(input_folder)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

        if not DATES:
            st.error("No date information available.")
            st.stop()

        min_date = min(DATES)
        max_date = max(DATES)
        unique_years = sorted({d.year for d in DATES})

        # Sidebar Filters specific to Lake Processing
        st.sidebar.header(f"Filters (Lake Processing: {waterbody})")
        threshold_range = st.sidebar.slider("Pixel Value Range", 0, 255, (0, 255))
        broad_date_range = st.sidebar.slider("Εύρος ημερομηνιών", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        refined_date_range = st.sidebar.slider("Βελτιωμένο εύρος ημερομηνιών", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        display_option = st.sidebar.radio("Εμφάνιση εικόνας", options=["Thresholded", "Original"], index=0)

        st.sidebar.markdown("### Επιλογή μηνών")
        month_options = {i: datetime(2000, i, 1).strftime('%B') for i in range(1, 13)}
        selected_months = st.sidebar.multiselect("Μήνες", options=list(month_options.keys()),
                                                 format_func=lambda x: month_options[x],
                                                 default=list(month_options.keys()),
                                                 key="selected_months")
        selected_years = st.sidebar.multiselect("Έτη", options=unique_years,
                                                default=unique_years,
                                                key="selected_years")
        # Filtering data based on date/month/year
        start_dt, end_dt = refined_date_range
        selected_indices = [i for i, d in enumerate(DATES)
                            if start_dt <= d <= end_dt and d.month in selected_months and d.year in selected_years]
        if not selected_indices:
            st.error("No data for the selected date/month/year combination.")
            st.stop()

        stack_filtered = STACK[selected_indices, :, :]
        days_filtered = np.array(DAYS)[selected_indices]
        filtered_dates = np.array(DATES)[selected_indices]

        lower_thresh, upper_thresh = threshold_range
        in_range = np.logical_and(stack_filtered >= lower_thresh, stack_filtered <= upper_thresh)
        days_in_range = np.nansum(in_range, axis=0)

        days_array = days_filtered.reshape((-1, 1, 1))
        sum_days = np.nansum(days_array * in_range, axis=0)
        count_in_range = np.nansum(in_range, axis=0)
        mean_day = np.divide(sum_days, count_in_range, out=np.full(sum_days.shape, np.nan), where=(count_in_range != 0))

        # Create Plotly figures
        fig_days = px.imshow(days_in_range, color_continuous_scale="plasma",
                             title="Days In Range Map", labels={"color": "Days In Range"})
        fig_days.update_layout(width=800, height=600)
        fig_mean = px.imshow(mean_day, color_continuous_scale="RdBu",
                             title="Mean Day Map", labels={"color": "Mean Day"})
        fig_mean.update_layout(width=800, height=600)

        if display_option.lower() == "thresholded":
            filtered_stack = np.where(in_range, stack_filtered, np.nan)
        else:
            filtered_stack = stack_filtered

        average_sample_img = np.nanmean(filtered_stack, axis=0)
        if np.all(np.isnan(average_sample_img)):
            avg_min, avg_max = 0, 0
        else:
            avg_min = float(np.nanmin(average_sample_img))
            avg_max = float(np.nanmax(average_sample_img))

        sample_img_fig = px.imshow(average_sample_img, color_continuous_scale="jet",
                                   range_color=[avg_min, avg_max],
                                   title=("Filtered Average Image" if display_option.lower() == "thresholded"
                                          else "Original Average Image"),
                                   labels={"color": "Pixel Value"})
        sample_img_fig.update_layout(width=800, height=600)

        st.header("Analysis Maps")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_days, use_container_width=True)
        with col2:
            st.plotly_chart(fig_mean, use_container_width=True)

        st.header("Sample Image Analysis")
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(sample_img_fig, use_container_width=True)
        with col4:
            st.info("Additional sample image details could be shown here.")

        st.info("Τέλος του Lake Processing.")
        st.markdown('</div>', unsafe_allow_html=True)

def run_water_processing(waterbody: str, index: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Water Processing ({waterbody} - {index}) [Placeholder]")
        st.info("No data or functionality yet for Water Processing.")
        st.markdown('</div>', unsafe_allow_html=True)

def run_water_quality_dashboard(waterbody: str, index: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Water Quality Dashboard ({waterbody} - {index})")
        data_folder = get_data_folder(waterbody, index)
        if data_folder is None:
            st.error("Δεν υπάρχει φάκελος δεδομένων για το επιλεγμένο υδάτινο σώμα / δείκτη.")
            st.stop()
        # (Dashboard code remains largely similar to the original implementation.)
        st.info("Η λειτουργία του Dashboard βρίσκεται σε εξέλιξη.")
        st.markdown('</div>', unsafe_allow_html=True)

def run_burned_areas():
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title("Burned Areas around reservoir (Γαδουρά Only)")
        st.info("No data or functionality yet for burned-area analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

def run_water_level_profiles(waterbody: str, index: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Water level Height Profiles ({waterbody}) [Placeholder]")
        st.info("No data or functionality yet for water-level height profiles.")
        st.markdown('</div>', unsafe_allow_html=True)

def run_pattern_analysis(waterbody: str, index: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Pattern Analysis ({waterbody} - {index})")
        data_folder = get_data_folder(waterbody, index)
        if data_folder is None:
            st.error("Δεν υπάρχει φάκελος δεδομένων για το επιλεγμένο υδάτινο σώμα / δείκτη.")
            st.stop()
        input_folder = os.path.join(data_folder, "GeoTIFFs")
        try:
            STACK, DAYS, DATES = load_data(input_folder)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()
        st.sidebar.header("Pattern Analysis Controls")
        unique_years = sorted({d.year for d in DATES})
        selected_years_pattern = st.sidebar.multiselect("Έτη", options=unique_years, default=unique_years, key="pattern_years")
        selected_months_pattern = st.sidebar.multiselect("Μήνες", options=list(range(1, 13)),
                                                         default=list(range(1, 13)),
                                                         key="pattern_months",
                                                         format_func=lambda m: datetime(2000, m, 1).strftime('%B'))
        threshold_range = st.sidebar.slider("Pixel Value Range", 0, 255, (0, 255), key="pattern_threshold")
        lower_thresh, upper_thresh = threshold_range
        if not selected_years_pattern or not selected_months_pattern:
            st.error("Please select at least one year and one month.")
            st.stop()
        indices = [i for i, d in enumerate(DATES)
                   if d.year in selected_years_pattern and d.month in selected_months_pattern]
        if not indices:
            st.error("No data for the selected criteria in pattern analysis.")
            st.stop()
        STACK_filtered = STACK[indices, :, :]
        stack_full_in_range = (STACK_filtered >= lower_thresh) & (STACK_filtered <= upper_thresh)
        temporal_data = []
        for m in sorted(selected_months_pattern):
            month_indices = [i for i, d in enumerate(np.array(DATES)[indices]) if d.month == m]
            if month_indices:
                avg_days = np.nanmean(stack_full_in_range[month_indices, :, :])
                temporal_data.append((m, avg_days))
        if temporal_data:
            months, means = zip(*temporal_data)
            month_names = [datetime(2000, mm, 1).strftime('%B') for mm in months]
            fig_temporal = px.bar(x=month_names, y=means,
                                  labels={'x': 'Month', 'y': 'Average Fraction In Range'},
                                  title="Temporal Pattern")
        else:
            fig_temporal = go.Figure()
        st.header("Pattern Analysis")
        st.plotly_chart(fig_temporal, use_container_width=True)
        st.info("Τέλος Pattern Analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 7) Main – Navigation & Execution
# -----------------------------------------------------------------------------
def main():
    run_intro_page()
    # Sidebar configuration (waterbody, index, analysis type)
    waterbody, index, analysis = run_configuration()
    st.markdown("---")
    st.header("Εκτέλεση Ανάλυσης")
    # Call the appropriate analysis function based on the sidebar selection
    if analysis == "Lake Processing":
        run_lake_processing_app(waterbody, index)
    elif analysis == "Water Processing":
        run_water_processing(waterbody, index)
    elif analysis == "Water Quality Dashboard":
        run_water_quality_dashboard(waterbody, index)
    elif analysis == "Burned Areas":
        if waterbody in ["Γαδουρά", "Κορώνεια"]:
            run_burned_areas()
        else:
            st.warning("Τα Burned Areas είναι διαθέσιμα μόνο για Γαδουρά (ή Κορώνεια, εάν υπάρχουν δεδομένα).")
    elif analysis == "Water level":
        run_water_level_profiles(waterbody, index)
    elif analysis == "Pattern Analysis":
        run_pattern_analysis(waterbody, index)
    else:
        st.warning("Παρακαλώ επιλέξτε ένα είδος ανάλυσης.")

# -----------------------------------------------------------------------------
# 8) Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
