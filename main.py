#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Water Quality App (Enterprise-Grade UI)
-----------------------------------------
This version hides debugging messages by default and features a highly polished,
enterprise-level, user-friendly interface.
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
    """Helper function for conditional debugging prints."""
    if DEBUG:
        st.write(*args, **kwargs)

# -------------------------------------------------------------------------
# Streamlit page configuration
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Ποιοτικά χαρακτηριστικά Επιφανειακού Ύδατος",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom CSS for an Enterprise Look and Feel
# -----------------------------------------------------------------------------
def inject_custom_css():
    custom_css = """
    <link href="https://fonts.googleapis.com/css?family=Roboto:400,500,700&display=swap" rel="stylesheet">
    <style>
        /* Global styling */
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif;
        }
        .block-container {
            background: #0d0d0d;
            color: #e0e0e0;
            padding: 1rem;
        }
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background: #1b1b1b;
            border: none;
        }
        /* Card styling */
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
        /* Navigation section inside sidebar */
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
        /* Button styling */
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
        /* Plotly figure container */
        .plotly-graph-div {
            border: 1px solid #333;
            border-radius: 8px;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

inject_custom_css()

# -----------------------------------------------------------------------------
# Data Folder Helper
# -----------------------------------------------------------------------------
def get_data_folder(waterbody: str, index: str) -> str:
    """
    Maps the chosen waterbody and index to the correct data folder on disk.
    Returns None if folder doesn't exist.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    debug("DEBUG: Current base directory:", base_dir)

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

    debug("DEBUG: Data folder resolved to:", data_folder)
    if data_folder is not None and not os.path.exists(data_folder):
        st.error(f"Folder does NOT exist on disk: {data_folder}")
        return None

    return data_folder

# -----------------------------------------------------------------------------
# Helper Functions for Data Extraction & Image Processing
# -----------------------------------------------------------------------------
def extract_date_from_filename(filename: str):
    """
    Extracts a date (YYYY-MM-DD) from the filename using a regex.
    Returns (day_of_year, datetime_obj) or (None, None) if no match.
    """
    basename = os.path.basename(filename)
    debug("DEBUG: Extracting date from filename:", basename)
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
    Loads a lake shape from a custom XML file with <point x=.. y=..>.
    Optionally transforms to geocoordinates if bounds is provided.
    """
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
            st.warning("No points found in the shapefile XML/TXT:", xml_file)
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
        st.error(f"Error reading lake shape from file {xml_file}: {e}")
        return None

def read_image(file_path: str, lake_shape: dict = None):
    """
    Reads a single-band GeoTIFF file and optionally applies a polygon mask (lake_shape).
    Returns (image_array, profile).
    """
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
    """
    Loads all TIF files from input_folder, applies optional shapefile mask, and extracts date info.
    Returns (stack, days_array, date_list).
    """
    debug("DEBUG: load_data called with input_folder =", input_folder)
    if not os.path.exists(input_folder):
        raise Exception(f"Folder does not exist: {input_folder}")

    shapefile_path_xml = os.path.join(input_folder, shapefile_name)
    shapefile_path_txt = os.path.join(input_folder, "shapefile.txt")
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
        raise Exception("No valid images were loaded after applying the shapefile/date filter.")

    stack = np.stack(images, axis=0)
    return stack, np.array(days), date_list

# -----------------------------------------------------------------------------
# Introductory Page
# -----------------------------------------------------------------------------
def run_intro_page():
    """Renders an introductory card with a logo and title."""
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col_logo, col_text = st.columns([1, 3])
        with col_logo:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            logo_path = os.path.join(base_dir, "logo.jpg")
            if os.path.exists(logo_path):
                st.image(logo_path, width=250)
            else:
                debug("DEBUG: Logo not found.")
        with col_text:
            st.markdown(
                "<h2 class='header-title'>Ποιοτικά χαρακτηριστικά Επιφανειακού Ύδατος</h2>",
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='text-align: center; font-size: 1.1rem;'>"
                "Εφαρμογή ανάλυσης με χρήση εργαλείων δορυφορικής τηλεπισκόπησης. "
                "Επιλέξτε τις ρυθμίσεις στην πλαϊνή μπάρα και εξερευνήστε τα δεδομένα.</p>",
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Navigation (Custom UI)
# -----------------------------------------------------------------------------
def run_custom_ui():
    """Creates the sidebar for waterbody, index, and analysis choice."""
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
# Lake Processing (Full Analysis) with Monthly and Yearly Analysis
# -----------------------------------------------------------------------------
def run_lake_processing_app(waterbody: str, index: str):
    """Main function for Lake Processing with monthly & yearly analyses."""
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Lake Processing ({waterbody} - {index})")

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

        if not DATES:
            st.error("No date information available.")
            st.stop()

        # Basic filters in the sidebar
        min_date = min(DATES)
        max_date = max(DATES)
        unique_years = sorted({d.year for d in DATES if d is not None})

        st.sidebar.header(f"Filters (Lake Processing: {waterbody})")
        threshold_range = st.sidebar.slider("Pixel value threshold range", 0, 255, (0, 255))
        broad_date_range = st.sidebar.slider("Broad date range", min_value=min_date, max_value=max_date,
                                             value=(min_date, max_date))
        refined_date_range = st.sidebar.slider("Refine date range", min_value=min_date, max_value=max_date,
                                               value=(min_date, max_date))
        display_option = st.sidebar.radio("Display mode", options=["Thresholded", "Original"], index=0)

        # Month/Year selection
        st.sidebar.markdown("### Επιλογή Μηνών")
        month_options = {i: datetime(2000, i, 1).strftime('%B') for i in range(1, 13)}
        if "selected_months" not in st.session_state:
            st.session_state.selected_months = list(month_options.keys())
        selected_months = st.sidebar.multiselect("Μήνες",
                                                 options=list(month_options.keys()),
                                                 format_func=lambda x: month_options[x],
                                                 default=st.session_state.selected_months,
                                                 key="selected_months")

        st.session_state.selected_years = unique_years
        selected_years = st.sidebar.multiselect("Έτη", options=unique_years,
                                                default=unique_years,
                                                key="selected_years")

        start_dt, end_dt = refined_date_range
        selected_indices = [i for i, d in enumerate(DATES)
                            if start_dt <= d <= end_dt
                            and d.month in selected_months
                            and d.year in selected_years]

        if not selected_indices:
            st.error("No data for the selected date range/month/year combination.")
            st.stop()

        # Filter the main stack
        stack_filtered = STACK[selected_indices, :, :]
        days_filtered = np.array(DAYS)[selected_indices]
        filtered_dates = np.array(DATES)[selected_indices]

        # Create a boolean mask for in-range pixels
        lower_thresh, upper_thresh = threshold_range
        in_range = np.logical_and(stack_filtered >= lower_thresh, stack_filtered <= upper_thresh)

        # 1) Days In Range
        days_in_range = np.nansum(in_range, axis=0)

        # 2) Mean Day (of In Range)
        days_array = days_filtered.reshape((-1, 1, 1))
        sum_days = np.nansum(days_array * in_range, axis=0)
        count_in_range = np.nansum(in_range, axis=0)
        mean_day = np.divide(sum_days, count_in_range,
                             out=np.full(sum_days.shape, np.nan),
                             where=(count_in_range != 0))

        # For display
        fig_days = px.imshow(days_in_range, color_continuous_scale="plasma",
                             title="Days In Range Map", labels={"color": "Days In Range"})
        fig_days.update_layout(width=800, height=600)

        fig_mean = px.imshow(mean_day, color_continuous_scale="RdBu",
                             title="Mean Day of In-Range Exceedance", labels={"color": "Mean Day"})
        fig_mean.update_layout(width=800, height=600)
        tick_vals = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365]
        tick_text = ["1 (Jan)", "32 (Feb)", "60 (Mar)", "91 (Apr)",
                     "121 (May)", "152 (Jun)", "182 (Jul)", "213 (Aug)",
                     "244 (Sep)", "274 (Oct)", "305 (Nov)", "335 (Dec)", "365 (Dec)"]
        fig_mean.update_layout(coloraxis_colorbar=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text))

        # Thresholded vs Original
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

        # 3) Time-of-Max analysis
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

        sample_title = "Average Sample Image (Filtered)" if display_option.lower() == "thresholded" \
                       else "Original Average Sample Image"
        time_title = "Time-of-Maximum Map (Day-of-Year)" if display_option.lower() == "thresholded" \
                     else "Original Time-of-Maximum Map (Day-of-Year)"

        sample_img_fig = px.imshow(average_sample_img, color_continuous_scale="jet",
                                   range_color=[avg_min, avg_max],
                                   title=sample_title, labels={"color": "Pixel Value"})
        sample_img_fig.update_layout(width=800, height=600)

        time_max_fig = px.imshow(time_max, color_continuous_scale="RdBu",
                                 range_color=[1, 365],
                                 title=time_title, labels={"color": "Day-of-Year"})
        time_max_fig.update_layout(width=800, height=600)
        time_max_fig.update_layout(coloraxis_colorbar=dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text))

        # Display results
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
            st.plotly_chart(time_max_fig, use_container_width=True)

        # ------------------------------
        # Additional Annual Analysis: Monthly Days In Range
        # ------------------------------
        st.header("Additional Annual Analysis: Monthly Days In Range Analysis")

        # Use the full STACK (unfiltered by date) for monthly analysis
        stack_full_in_range = (STACK >= lower_thresh) & (STACK <= upper_thresh)
        monthly_days_in_range = {}
        for m in range(1, 13):
            indices_m = [i for i, d in enumerate(DATES) if d is not None and d.month == m]
            if indices_m:
                monthly_days_in_range[m] = np.sum(stack_full_in_range[indices_m, :, :], axis=0)
            else:
                monthly_days_in_range[m] = None

        fig_monthly = make_subplots(
            rows=3, cols=4,
            subplot_titles=[datetime(2000, m, 1).strftime('%B') for m in range(1, 13)],
            horizontal_spacing=0.05,
            vertical_spacing=0.04  # Keep this small enough for multiple rows
        )
        trace_count = 0
        for m in range(1, 13):
            row = (m - 1) // 4 + 1
            col = (m - 1) % 4 + 1
            img = monthly_days_in_range[m]
            if img is not None:
                showscale = True if trace_count == 0 else False
                fig_monthly.add_trace(
                    go.Heatmap(
                        z=np.flipud(img),
                        colorscale="plasma",
                        showscale=showscale,
                        colorbar=dict(title="Days In Range") if showscale else None
                    ),
                    row=row, col=col
                )
                trace_count += 1
            else:
                fig_monthly.add_annotation(
                    text="No data",
                    showarrow=False, row=row, col=col
                )
        fig_monthly.update_layout(height=1400)
        st.plotly_chart(fig_monthly, use_container_width=True)

        # ------------------------------
        # Additional Annual Analysis: Yearly Days In Range
        # ------------------------------
        st.header("Additional Annual Analysis: Yearly Days In Range Analysis")

        unique_years_full = sorted({d.year for d in DATES if d is not None})
        if not unique_years_full:
            st.error("No valid years found in the data.")
            st.stop()

        # Reuse the in-range mask for the entire dataset
        stack_full_in_range = (STACK >= lower_thresh) & (STACK <= upper_thresh)

        yearly_days_in_range = {}
        for year in unique_years_full:
            indices_y = [i for i, d in enumerate(DATES) if d.year == year]
            if indices_y:
                yearly_days_in_range[year] = np.sum(stack_full_in_range[indices_y, :, :], axis=0)
            else:
                yearly_days_in_range[year] = None

        # Each year in its own row
        n_years = len(unique_years_full)
        # Keep vertical_spacing small enough for multiple rows
        vertical_spacing = 0.92 if n_years == 1 else min(0.92, 0.3 / n_years)

        fig_yearly = make_subplots(
            rows=n_years, cols=1,
            subplot_titles=[f"Year: {year}" for year in unique_years_full],
            vertical_spacing=vertical_spacing
        )
        for idx, year in enumerate(unique_years_full, start=1):
            img = yearly_days_in_range[year]
            if img is not None:
                fig_yearly.add_trace(
                    go.Heatmap(
                        z=np.flipud(img),
                        colorscale="plasma",
                        colorbar=dict(title="Days In Range", len=0.5) if idx == 1 else dict(showticklabels=False)
                    ),
                    row=idx, col=1
                )
            else:
                fig_yearly.add_annotation(
                    text="No data",
                    row=idx, col=1,
                    showarrow=False
                )

        # 300 px per year is a decent baseline
        fig_yearly.update_layout(
            height=300 * n_years,
            title_text="Yearly Days In Range Analysis"
        )
        st.plotly_chart(fig_yearly, use_container_width=True)

        st.info("End of Lake Processing section.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Water Processing (Placeholder)
# -----------------------------------------------------------------------------
def run_water_processing(waterbody: str, index: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Water Processing ({waterbody} - {index}) [Placeholder]")
        st.info("No data or functionality yet for Water Processing.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Water Quality Dashboard
# -----------------------------------------------------------------------------
def run_water_quality_dashboard(waterbody: str, index: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Water Quality Dashboard ({waterbody} - {index})")

        data_folder = get_data_folder(waterbody, index)
        if data_folder is None:
            st.error("Δεν υπάρχει φάκελος δεδομένων για το επιλεγμένο υδάτινο σώμα / δείκτη.")
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

        st.sidebar.header(f"Ρυθμίσεις Ανάλυσης ({waterbody} - Dashboard)")
        x_start = st.sidebar.date_input("Έναρξη", date(2015, 1, 1))
        x_end = st.sidebar.date_input("Λήξη", date(2026, 12, 31))
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
                    debug("DEBUG: Error parsing date from", filename, ":", e)
                    continue

        if available_dates:
            sorted_dates = sorted(available_dates.keys())
            selected_bg_date = st.selectbox("Επιλέξτε ημερομηνία για το background", sorted_dates)
        else:
            selected_bg_date = None
            st.warning("Δεν βρέθηκαν GeoTIFF εικόνες με ημερομηνία στον τίτλο.")

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
                st.error(f"GeoTIFF background file not found: {bg_path}")
                st.stop()
        else:
            st.error("Δεν έχει επιλεγεί έγκυρη ημερομηνία για το background.")
            st.stop()

        # KML parsing and analysis functions (similar to before)
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
                st.error("Error parsing sampling KML:", e)
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

            # Build the base image figure with sampling points
            rgb_image = first_image_data.transpose((1, 2, 0)) / 255.0
            fig_geo = px.imshow(rgb_image, title='GeoTIFF Image with Sampling Points')
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
                st.error(f"Error reading lake height file: {e}")
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
                    name='Lake Height', mode='lines', line=dict(color='blue', width=2)
                )
                fig_colors.add_trace(trace_height, secondary_y=True)

            fig_colors.update_layout(title='Pixel Colors and Lake Height Over Time',
                                     xaxis_title='Date',
                                     yaxis_title='Sampling Points',
                                     showlegend=True)
            fig_colors.update_yaxes(title_text="Lake Height", secondary_y=True)

            # Compute an average mg for each date (all sampling points)
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
                name='Average mg/m³'
            ))
            fig_mg.update_layout(title='Average mg/m³ Over Time',
                                 xaxis_title='Date', yaxis_title='mg/m³',
                                 showlegend=False)

            fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
            if not lake_data.empty:
                fig_dual.add_trace(go.Scatter(
                    x=lake_data['Date'],
                    y=lake_data[lake_data.columns[1]],
                    name='Lake Height', mode='lines'
                ), secondary_y=False)

            fig_dual.add_trace(go.Scatter(
                x=sorted_dates,
                y=avg_mg,
                name='Average mg/m³',
                mode='markers',
                marker=dict(color=avg_mg, colorscale='Viridis', reversescale=True,
                            colorbar=dict(title='mg/m³'), size=10)
            ), secondary_y=True)
            fig_dual.update_layout(title='Lake Height and Average mg/m³ Over Time',
                                   xaxis_title='Date', showlegend=True)
            fig_dual.update_yaxes(title_text="Lake Height", secondary_y=False)
            fig_dual.update_yaxes(title_text="mg/m³", secondary_y=True)

            return fig_geo, fig_dual, fig_colors, fig_mg, results_colors, results_mg, lake_data

        # Two main sampling tabs
        if "default_results" not in st.session_state:
            st.session_state.default_results = None
        if "upload_results" not in st.session_state:
            st.session_state.upload_results = None

        tab_names = ["Δειγματοληψία 1 (Default)", "Δειγματοληψία 2 (Upload)"]
        tabs = st.tabs(tab_names)

        # -- Tab 1 (Default)
        with tabs[0]:
            st.header("Ανάλυση για Δειγματοληψία 1 (Default)")
            default_sampling_points = []
            if os.path.exists(sampling_kml_path):
                default_sampling_points = parse_sampling_kml(sampling_kml_path)
            else:
                st.warning("Default sampling.kml not found at:", sampling_kml_path)

            point_names = [name for name, _, _ in default_sampling_points]
            selected_points = st.multiselect("Select points for mg/m³ concentrations",
                                             options=point_names,
                                             default=point_names)
            if st.button("Run Analysis (Default)"):
                with st.spinner("Running analysis..."):
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
                    st.error("Analysis result format error. Please run the analysis again.")
                    st.stop()

                nested_tabs = st.tabs(["GeoTIFF", "Video/GIF", "Pixel Colors", "Average mg", "Dual Plots", "Detail mg"])
                with nested_tabs[0]:
                    st.plotly_chart(fig_geo, use_container_width=True)
                with nested_tabs[1]:
                    if video_path is not None:
                        if video_path.endswith(".mp4"):
                            st.video(video_path)
                        else:
                            st.image(video_path)
                    else:
                        st.info("No timelapse file found.")
                with nested_tabs[2]:
                    st.plotly_chart(fig_colors, use_container_width=True)
                with nested_tabs[3]:
                    st.plotly_chart(fig_mg, use_container_width=True)
                with nested_tabs[4]:
                    st.plotly_chart(fig_dual, use_container_width=True)
                with nested_tabs[5]:
                    selected_detail_point = st.selectbox("Select a point for detailed mg analysis",
                                                         options=list(results_mg.keys()))
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
                            fig_detail.update_layout(title=f"Detailed mg analysis for {selected_detail_point}",
                                                     xaxis_title="Date", yaxis_title="mg/m³")
                            st.plotly_chart(fig_detail, use_container_width=True)
                        else:
                            st.info("No mg data for this point.")

        # -- Tab 2 (Upload)
        with tabs[1]:
            st.header("Analysis for Uploaded Sampling")
            uploaded_file = st.file_uploader("Upload a KML file for new sampling points", type="kml", key="upload_tab")
            if uploaded_file is not None:
                try:
                    new_sampling_points = parse_sampling_kml(uploaded_file)
                except Exception as e:
                    st.error(f"Error processing the uploaded file: {e}")
                    new_sampling_points = []

                point_names = [name for name, _, _ in new_sampling_points]
                selected_points = st.multiselect("Select points for mg/m³ concentrations",
                                                 options=point_names,
                                                 default=point_names)

                if st.button("Run Analysis (Upload)"):
                    with st.spinner("Running analysis..."):
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
                        st.error("Analysis result format error (Upload). Please run the analysis again.")
                        st.stop()

                    nested_tabs = st.tabs(["GeoTIFF", "Video/GIF", "Pixel Colors", "Average mg", "Dual Plots", "Detail mg"])
                    with nested_tabs[0]:
                        st.plotly_chart(fig_geo, use_container_width=True)
                    with nested_tabs[1]:
                        if video_path is not None:
                            if video_path.endswith(".mp4"):
                                st.video(video_path)
                            else:
                                st.image(video_path)
                        else:
                            st.info("Video/GIF file not found.")
                    with nested_tabs[2]:
                        st.plotly_chart(fig_colors, use_container_width=True)
                    with nested_tabs[3]:
                        st.plotly_chart(fig_mg, use_container_width=True)
                    with nested_tabs[4]:
                        st.plotly_chart(fig_dual, use_container_width=True)
                    with nested_tabs[5]:
                        selected_detail_point = st.selectbox("Select a point for detailed mg analysis",
                                                             options=list(results_mg.keys()), key="detail_upload")
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
                                fig_detail.update_layout(title=f"Detailed mg analysis for {selected_detail_point}",
                                                         xaxis_title="Date", yaxis_title="mg/m³")
                                st.plotly_chart(fig_detail, use_container_width=True)
                            else:
                                st.info("No mg data for this point.")
            else:
                st.info("Please upload a KML file for new sampling points.")

        st.info("End of Water Quality Dashboard section.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Burned Areas (Placeholder)
# -----------------------------------------------------------------------------
def run_burned_areas():
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title("Burned Areas around reservoir (Γαδουρά Only)")
        st.info("No data or functionality yet for burned-area analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Water Level Profiles (Placeholder)
# -----------------------------------------------------------------------------
def run_water_level_profiles(waterbody: str, index: str):
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.title(f"Water Level Profiles ({waterbody}) [Placeholder]")
        st.info("No data or functionality yet for water-level height profiles.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Pattern Analysis
# -----------------------------------------------------------------------------
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
        unique_years = sorted({d.year for d in DATES if d is not None})
        selected_years_pattern = st.sidebar.multiselect(
            "Select Years",
            options=unique_years,
            default=unique_years,
            key="pattern_years"
        )
        selected_months_pattern = st.sidebar.multiselect(
            "Select Months",
            options=list(range(1, 13)),
            default=list(range(1, 13)),
            key="pattern_months",
            format_func=lambda m: datetime(2000, m, 1).strftime('%B')
        )
        threshold_range = st.sidebar.slider("Pixel value threshold range", 0, 255, (0, 255), key="pattern_threshold")
        lower_thresh, upper_thresh = threshold_range

        if not selected_years_pattern or not selected_months_pattern:
            st.error("Please select at least one year and one month.")
            st.stop()

        # Filter by chosen months and years
        indices = [i for i, d in enumerate(DATES)
                   if d.year in selected_years_pattern and d.month in selected_months_pattern]
        if not indices:
            st.error("No data for the selected years/months in pattern analysis.")
            st.stop()

        STACK_filtered = STACK[indices, :, :]
        stack_full_in_range = (STACK_filtered >= lower_thresh) & (STACK_filtered <= upper_thresh)
        filtered_dates = [DATES[i] for i in indices]

        # Monthly average in-range fraction
        monthly_avg = {}
        for m in selected_months_pattern:
            month_indices = [i for i, dd in enumerate(filtered_dates) if dd.month == m]
            if month_indices:
                avg_days = np.nanmean(stack_full_in_range[month_indices, :, :], axis=0)
                monthly_avg[m] = avg_days
            else:
                monthly_avg[m] = None

        # Aggregate across chosen months
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

        # Prepare temporal data for a bar chart
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
                labels={'x': 'Month', 'y': 'Average Fraction In Range'},
                title="Temporal Pattern per Month"
            )
        else:
            fig_temporal = go.Figure()

        # Spatial classification map
        if overall_avg is not None:
            classification = np.full(overall_avg.shape, "Unclassified", dtype=object)
            valid_mask = ~np.isnan(overall_avg)
            classification[valid_mask & (overall_avg < 0.3)] = "Low"
            classification[valid_mask & (overall_avg >= 0.3) & (overall_avg < 0.7)] = "Medium"
            classification[valid_mask & (overall_avg >= 0.7)] = "High"

            mapping_dict = {"Low": 0, "Medium": 1, "High": 2, "Unclassified": 3}
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
                title="Spatial Pattern Classification"
            )
            fig_class.update_traces(
                colorbar=dict(tickvals=[0, 1, 2, 3],
                              ticktext=["Low", "Medium", "High", "Unclassified"])
            )
        else:
            fig_class = go.Figure()

        st.header("Pattern Analysis")
        st.markdown(
            "Analysis of monthly days-in-range data and spatial classification of persistent in-range fractions."
        )
        st.subheader("Temporal Pattern")
        st.plotly_chart(fig_temporal, use_container_width=True)

        st.subheader("Spatial Classification")
        st.plotly_chart(fig_class, use_container_width=True)

        if temporal_data:
            df_temporal = pd.DataFrame(temporal_data, columns=["Month", "Average Fraction In Range"])
            csv = df_temporal.to_csv(index=False).encode('utf-8')
            st.download_button("Download Analysis CSV", data=csv,
                               file_name="temporal_analysis.csv", mime="text/csv")

        st.info("End of Pattern Analysis section.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    debug("DEBUG: Entered main()")
    run_intro_page()
    run_custom_ui()

    wb = st.session_state.get("waterbody_choice", None)
    idx = st.session_state.get("index_choice", None)
    analysis = st.session_state.get("analysis_choice", None)

    debug("DEBUG: User selected waterbody =", wb, "index =", idx, "analysis =", analysis)

    # Routing to the appropriate function
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
            st.warning("Burned Areas section is under development.")
        else:
            st.warning("Το 'Burned Areas' είναι διαθέσιμο μόνο για το υδάτινο σώμα 'Γαδουρά'.")
    else:
        st.warning(
            "Δεν υπάρχουν διαθέσιμα δεδομένα για αυτόν τον συνδυασμό δείκτη / υδάτινου σώματος. "
            "Για παράδειγμα, η Χλωροφύλλη είναι διαθέσιμη μόνο για (Κορώνεια, Πολυφύτου, Γαδουρά, Αξιός)."
        )

if __name__ == "__main__":
    main()
