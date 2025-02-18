#!/usr/bin/env python
"""
Εφαρμογή Ποιότητας Νερού Ταμιευτήρων και Πίνακας Ελέγχου Ποιότητας Νερού

Αυτή η εφαρμογή Streamlit περιλαμβάνει:

1) Lake Processing (Eπεξεργασία Λίμνης) με λεπτομερή χρονοσειρά, ημερομηνίες, 
   thresholding, monthly/yearly analysis κ.λπ.
2) Water Processing (Placeholder)
3) Water Quality Dashboard (με default/upload sampling, timelapse, διαδραστικά γραφήματα)
4) Burned Areas around reservoir (Placeholder)
5) Water level Height Profiles (Placeholder)
6) Pattern Analysis (πλήρης κώδικας, αλλά ΔΕΝ συνδέεται από το νέο 3-row UI — 
   μπορείτε να το συνδέσετε εάν θέλετε).

+ Νέα Διάταξη: 3 Σειρές Κουμπιών
  Row 1: "Επιλογή υδάτινου σώματος" → [Κορώνεια, Πολυφύτου, Γαδουρά, Αξιός]
  Row 2: "Επιλογή Δείκτη" → [Πραγματικό, Χλωροφύλλη, CDOM, Colour]
  Row 3: [Lake processing, Water Processing, Water Quality Dashboard,
          Burned Areas around reservoir, Water level Height Profiles]

Σημαντική Σημείωση:
- Αυτή τη στιγμή, τα δεδομένα (GeoTIFF, κλπ.) αφορούν μόνο (Κορώνεια + Χλωροφύλλη).
  Για οποιοδήποτε άλλο συνδυασμό, εμφανίζεται προειδοποίηση "no data".
- Βεβαιωθείτε ότι τα GeoTIFF αρχεία, shapefile.xml, KML, Excel, κλπ. βρίσκονται 
  στους σωστούς φακέλους.

Για να τρέξετε:
  streamlit run app.py
"""

# Disable Streamlit's file watchdog to avoid inotify watch limit issues.
import os
os.environ["STREAMLIT_SERVER_ENABLEWATCHDOG"] = "false"

import glob
import re
from datetime import datetime, date
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import rasterio
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional: For geometry operations (if needed)
# from shapely import wkt
# from shapely.geometry import mapping

# Suppress not-georeferenced warnings from rasterio
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# -----------------------------------------------------------------------------
# Page configuration and Global Settings
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ποιοτικά χαρακτηριστικά Επιφανειακού Ύδατος",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def extract_date_from_filename(filename: str):
    """
    Extract the day-of-year and date object from a filename
    containing a date in YYYY_MM_DD or YYYY-MM-DD format.
    """
    basename = os.path.basename(filename)
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
    Load the lake boundary polygon from an XML file (PlotDigitizer format).
    Optionally transform to GeoTIFF coordinates using provided bounds.
    """
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
            st.warning("No points found in the XML file.")
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

        return {"type": "Polygon", "coordinates": [points]}
    except Exception as e:
        st.error(f"Error reading lake shape from XML file {xml_file}: {e}")
        return None

def read_image(file_path: str, lake_shape: dict = None):
    """
    Read a GeoTIFF image file (1 band) and optionally apply a lake shape mask.
    """
    with rasterio.open(file_path) as src:
        img = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        profile.update(dtype="float32")

        no_data_value = src.nodata
        if no_data_value is not None:
            img = np.where(img == no_data_value, np.nan, img)

        # Optionally treat 0 as no-data
        img = np.where(img == 0, np.nan, img)

        if lake_shape is not None:
            from rasterio.features import geometry_mask
            poly_mask = geometry_mask(
                [lake_shape],
                transform=src.transform,
                invert=False,
                out_shape=img.shape
            )
            img = np.where(~poly_mask, img, np.nan)

    return img, profile

def load_data(input_folder: str):
    """
    Load GeoTIFF images from 'input_folder' plus date info from filenames.
    If 'shapefile.xml' is present, use it to mask the images.
    Returns: (stack, days, date_list)
    """
    shapefile_path = os.path.join(input_folder, "shapefile.xml")
    lake_shape = None

    all_tif_files = sorted(glob.glob(os.path.join(input_folder, "*.tif")))
    tif_files = [fp for fp in all_tif_files if os.path.basename(fp).lower() != "mask.tif"]
    if not tif_files:
        raise Exception("No GeoTIFF files found in the specified folder.")

    with rasterio.open(tif_files[0]) as src:
        bounds = src.bounds

    if os.path.exists(shapefile_path):
        st.write(f"Found lake shape file at {shapefile_path}.")
        lake_shape = load_lake_shape_from_xml(shapefile_path, bounds=bounds)
    else:
        st.write("Lake shape file not found. Skipping lake shape processing.")

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
# Introductory Page
# -----------------------------------------------------------------------------
def run_intro_page():
    """
    Shows the main headline, optional logo, and a short introduction.
    """
    st.markdown(
        "<h2 style='text-align: center;'>"
        "Ποιοτικά χαρακτηριστικά επιφανειακού Ύδατος με χρήση Εργαλείων Δορυφορικής Τηλεπισκόπησης"
        "</h2>",
        unsafe_allow_html=True
    )

    base_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_dir, "logo.jpg")
    if os.path.exists(logo_path):
        st.image(logo_path, width=300)
    else:
        st.write("Logo not found.")

    st.markdown(
        """
        ### Introduction
        
        This application processes and analyzes satellite-based water quality data for various lakes.
        Currently, real data exist for **Λίμνη Κορώνεια** with the **Χλωροφύλλη** index. 
        Other water bodies/indices are placeholders until their data are added.

        **Pages**:
        - **Lake Processing:**  
          Spatiotemporal analysis (sliders, thresholds, monthly/yearly maps).
        - **Water Quality Dashboard:**  
          Interactive sampling points, timelapse videos, etc.
        - **Burned Areas** & **Water level**: Placeholders.
        - **Pattern Analysis**: Full code is included but not linked to the new UI. 
          You can link it if needed.

        **Folders**:
        - Place your GeoTIFFs in "GeoTIFFs".
        - "shapefile.xml" in "GeoTIFFs" if you want to mask the lake.
        - "lake height.xlsx" for height data, "sampling.kml" for sampling points, etc.
        
        Use the 3-row matrix of buttons below to pick:
          1) Water body
          2) Index
          3) Analysis type
        Then see the results.
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# Lake Processing (full monthly/yearly analysis)
# -----------------------------------------------------------------------------
def run_lake_processing_app():
    """
    Lake Processing: thresholding, date filters, monthly/yearly analysis, etc.
    This is your large spatiotemporal analysis code.
    """
    st.title("Lake Processing (Κορώνεια - Χλωροφύλλη)")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "GeoTIFFs")

    # ~~~~~ Load data ~~~~~
    try:
        STACK, DAYS, DATES = load_data(input_folder)
        st.success("Data loaded successfully.")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    if not DATES:
        st.error("No date information available.")
        st.stop()

    # ~~~~~ Define min/max date, years, etc. ~~~~~
    min_date = min(DATES)
    max_date = max(DATES)
    unique_years = sorted({d.year for d in DATES})

    # --- Sidebar filters ---
    st.sidebar.header("Filters (Lake Processing)")
    threshold_range = st.sidebar.slider("Select pixel value threshold range", 0, 255, (0, 255))
    broad_date_range = st.sidebar.slider("Select a broad date range",
                                         min_value=min_date,
                                         max_value=max_date,
                                         value=(min_date, max_date))
    st.sidebar.write("Broad Date Range:",
                     broad_date_range[0].strftime("%Y-%m-%d"),
                     "to",
                     broad_date_range[1].strftime("%Y-%m-%d"))
    refined_date_range = st.sidebar.slider("Refine the date range",
                                           min_value=min_date,
                                           max_value=max_date,
                                           value=(min_date, max_date))
    st.sidebar.write("Refined Date Range:",
                     refined_date_range[0].strftime("%Y-%m-%d"),
                     "to",
                     refined_date_range[1].strftime("%Y-%m-%d"))
    display_option = st.sidebar.radio("Sample Image Display Mode", options=["Thresholded", "Original"], index=0)

    st.sidebar.markdown("### Select Months")
    month_options = {i: datetime(2000, i, 1).strftime('%B') for i in range(1, 13)}
    if "selected_months" not in st.session_state:
        st.session_state.selected_months = list(month_options.keys())
    selected_months = st.sidebar.multiselect(
        "Select Months",
        options=list(month_options.keys()),
        format_func=lambda x: month_options[x],
        default=st.session_state.selected_months,
        key="selected_months"
    )
    if st.sidebar.button("Deselect All Months"):
        st.session_state.selected_months = []
        selected_months = []

    st.sidebar.markdown("### Select Years")
    if "selected_years" not in st.session_state:
        st.session_state.selected_years = unique_years
    selected_years = st.sidebar.multiselect(
        "Select Years",
        options=unique_years,
        default=st.session_state.selected_years,
        key="selected_years"
    )
    if st.sidebar.button("Deselect All Years"):
        st.session_state.selected_years = []
        selected_years = []

    # ~~~~~ Filter data based on date & month/year picks ~~~~~
    start_dt, end_dt = refined_date_range
    selected_indices = [
        i for i, d in enumerate(DATES)
        if start_dt <= d <= end_dt and d.month in selected_months and d.year in selected_years
    ]
    if not selected_indices:
        st.error("No data for the selected date range and month/year combination.")
        st.stop()

    stack_filtered = STACK[selected_indices, :, :]
    days_filtered = np.array(DAYS)[selected_indices]
    filtered_dates = np.array(DATES)[selected_indices]

    # ~~~~~ Compute analysis maps ~~~~~
    lower_thresh, upper_thresh = threshold_range
    in_range = np.logical_and(stack_filtered >= lower_thresh, stack_filtered <= upper_thresh)
    days_in_range = np.nansum(in_range, axis=0)
    days_array = days_filtered.reshape((-1, 1, 1))
    sum_days = np.nansum(days_array * in_range, axis=0)
    count_in_range = np.nansum(in_range, axis=0)
    mean_day = np.divide(sum_days, count_in_range,
                         out=np.full(sum_days.shape, np.nan),
                         where=(count_in_range != 0))

    # ~~~~~ Days In Range Map ~~~~~
    fig_days = px.imshow(
        days_in_range,
        color_continuous_scale="plasma",
        title="Days In Range Map",
        labels={"color": "Days In Range"}
    )
    fig_days.update_layout(width=2000, height=1600)
    fig_days.update_traces(colorbar=dict(len=0.4))

    # ~~~~~ Mean Day Map ~~~~~
    fig_mean = px.imshow(
        mean_day,
        color_continuous_scale="RdBu",
        title="Mean Day of In-Range Exceedance Map",
        labels={"color": "Mean Day"}
    )
    fig_mean.update_layout(width=2000, height=1600)
    fig_mean.update_traces(colorbar=dict(len=0.4))
    tick_vals = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365]
    tick_text = [
        "1 (Jan)", "32 (Feb)", "60 (Mar)", "91 (Apr)",
        "121 (May)", "152 (Jun)", "182 (Jul)", "213 (Aug)",
        "244 (Sep)", "274 (Oct)", "305 (Nov)", "335 (Dec)", "365 (Dec)"
    ]
    fig_mean.update_layout(coloraxis_colorbar=dict(
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_text,
        len=0.4
    ))

    # ~~~~~ Sample Image Analysis ~~~~~
    if display_option.lower() == "thresholded":
        filtered_stack = np.where(in_range, stack_filtered, np.nan)
    else:
        filtered_stack = stack_filtered

    average_sample_img = np.nanmean(filtered_stack, axis=0)
    if np.all(np.isnan(average_sample_img)):
        avg_min, avg_max = 0, 0
    else:
        avg_min, avg_max = float(np.nanmin(average_sample_img)), float(np.nanmax(average_sample_img))

    filtered_day_of_year = np.array([d.timetuple().tm_yday for d in filtered_dates])
    def nanargmax_or_nan(arr):
        return np.nan if np.all(np.isnan(arr)) else np.nanargmax(arr)
    max_index = np.apply_along_axis(nanargmax_or_nan, 0, filtered_stack)
    time_max = np.full(max_index.shape, np.nan, dtype=float)
    valid_mask = ~np.isnan(max_index)
    max_index_int = np.zeros_like(max_index, dtype=int)
    max_index_int[valid_mask] = max_index[valid_mask].astype(int)
    time_max[valid_mask] = filtered_day_of_year[max_index_int[valid_mask]]

    sample_title = ("Average Sample Image (Filtered)"
                    if display_option.lower() == "thresholded"
                    else "Original Average Sample Image")
    time_title = ("Time-of-Maximum Map (Day-of-Year)"
                  if display_option.lower() == "thresholded"
                  else "Original Time-of-Maximum Map (Day-of-Year)")

    sample_img_fig = px.imshow(
        average_sample_img,
        color_continuous_scale="jet",
        range_color=[avg_min, avg_max],
        title=sample_title,
        labels={"color": "Pixel Value"}
    )
    sample_img_fig.update_layout(width=2000, height=1600)
    sample_img_fig.update_traces(colorbar=dict(len=0.4))

    time_max_fig = px.imshow(
        time_max,
        color_continuous_scale="RdBu",
        range_color=[1, 365],
        title=time_title,
        labels={"color": "Day-of-Year"}
    )
    time_max_fig.update_layout(width=2000, height=1600)
    time_max_fig.update_traces(colorbar=dict(len=0.4))
    time_max_fig.update_layout(coloraxis_colorbar=dict(
        tickmode='array',
        tickvals=tick_vals,
        ticktext=tick_text,
        len=0.4
    ))

    st.write(
        f"Threshold range: {lower_thresh} to {upper_thresh} | "
        f"Refined date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} "
        f"({len(selected_indices)} images matched)"
    )

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

    # ~~~~~ Additional Analysis (Monthly/Yearly) ~~~~~
    unique_years_full = sorted({d.year for d in DATES})
    if not unique_years_full:
        st.error("No valid years found in the data.")
        st.stop()
    min_year = unique_years_full[0]
    max_year = unique_years_full[-1]

    st.sidebar.header("Additional Analysis Controls")
    selected_years_analysis = st.sidebar.multiselect(
        "Select Years for Days in Range Analysis",
        options=unique_years_full,
        default=unique_years_full,
        key="additional_years"
    )
    if selected_years_analysis:
        monthly_year_range = (min(selected_years_analysis), max(selected_years_analysis))
    else:
        monthly_year_range = (min_year, max_year)
    st.sidebar.write("Monthly Analysis Year Range is set to:", monthly_year_range)

    # Group 1: Monthly Days in Range
    st.header("Monthly Days in Range Analysis")
    st.write("Number of days each pixel is in range for each month over the selected years.")
    stack_full_in_range = (STACK >= lower_thresh) & (STACK <= upper_thresh)
    monthly_days_in_range = {}
    for m in range(1, 13):
        indices_m = [i for i, d in enumerate(DATES)
                     if monthly_year_range[0] <= d.year <= monthly_year_range[1] and d.month == m]
        if indices_m:
            monthly_days_in_range[m] = np.sum(stack_full_in_range[indices_m, :, :], axis=0)
        else:
            monthly_days_in_range[m] = None

    fig_monthly = make_subplots(
        rows=3, cols=4,
        subplot_titles=[datetime(2000, m, 1).strftime('%B') for m in range(1, 13)],
        horizontal_spacing=0.05, vertical_spacing=0.1
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

    # Group 2: Yearly Days in Range
    st.header("Yearly Days in Range Analysis")
    st.write("Number of days each pixel is in range for selected months in the selected years.")
    selected_months_yearly = st.sidebar.multiselect(
        "Select Months for Yearly Days in Range Analysis",
        options=list(range(1, 13)),
        default=list(range(1, 13)),
        format_func=lambda m: datetime(2000, m, 1).strftime('%B'),
        key="yearly_days_months"
    )
    if not selected_years_analysis:
        st.warning("Please select at least one year for yearly analysis.")
    elif not selected_months_yearly:
        st.warning("Please select at least one month for yearly analysis.")
    else:
        n_rows = len(selected_years_analysis)
        n_cols = len(selected_months_yearly)
        subplot_titles = [
            f"{year} - {datetime(2000, m, 1).strftime('%B')}"
            for year in selected_years_analysis for m in selected_months_yearly
        ]
        fig_yearly = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.03, vertical_spacing=0.08
        )
        yearly_days_in_range = {}
        for i, year in enumerate(selected_years_analysis):
            for j, m in enumerate(selected_months_yearly):
                indices_ym = [k for k, d in enumerate(DATES) if d.year == year and d.month == m]
                if indices_ym:
                    count_img = np.sum(stack_full_in_range[indices_ym, :, :], axis=0)
                    yearly_days_in_range[(year, m)] = count_img
                    fig_yearly.add_trace(
                        go.Heatmap(
                            z=np.flipud(count_img),
                            colorscale="plasma",
                            coloraxis="coloraxis",
                            showscale=False
                        ),
                        row=i+1, col=j+1
                    )
                else:
                    yearly_days_in_range[(year, m)] = None
                    fig_yearly.add_annotation(
                        text="No data",
                        showarrow=False, row=i+1, col=j+1
                    )
        fig_yearly.update_layout(
            coloraxis=dict(
                colorscale="plasma",
                colorbar=dict(title="Days In Range", len=0.75)
            ),
            height=300 * n_rows,
            width=1200,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig_yearly, use_container_width=True)

        # Enlarged View
        available_pairs = [(y, m) for (y, m), data in yearly_days_in_range.items() if data is not None]
        if available_pairs:
            pair_labels = [f"{y} - {datetime(2000, m, 1).strftime('%B')}" for y, m in available_pairs]
            selected_pair_label = st.selectbox("Select a Year-Month pair for larger view", options=pair_labels)
            selected_index = pair_labels.index(selected_pair_label)
            selected_pair = available_pairs[selected_index]
            large_img = yearly_days_in_range[selected_pair]
            fig_large = go.Figure(
                data=go.Heatmap(
                    z=np.flipud(large_img),
                    colorscale="plasma",
                    colorbar=dict(title="Days In Range")
                )
            )
            fig_large.update_layout(
                title=f"Larger View: {selected_pair[0]} - {datetime(2000, selected_pair[1], 1).strftime('%B')}",
                width=800,
                height=800,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            st.plotly_chart(fig_large, use_container_width=False)
        else:
            st.info("No valid yearly data available for enlarged view.")

    st.info("End of Lake Processing section.")

# -----------------------------------------------------------------------------
# Water Processing (Placeholder)
# -----------------------------------------------------------------------------
def run_water_processing():
    """
    Placeholder for future water processing code.
    """
    st.title("Water Processing (Placeholder)")
    st.info("No data or functionality yet for Water Processing.")

# -----------------------------------------------------------------------------
# Water Quality Dashboard (with default/upload sampling)
# -----------------------------------------------------------------------------
def run_water_quality_dashboard():
    """
    Full interactive water quality dashboard:
      - Timelapse videos
      - Default sampling points (sampling.kml)
      - Upload sampling points
      - Lake height data (lake height.xlsx)
      - mg/m³ conversions
      - etc.
    """
    st.title("Water Quality Dashboard (Κορώνεια - Χλωροφύλλη)")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Example custom CSS
    st.markdown(
        """
        <style>
            body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
            .main-header {
                background-color: #4CAF50;
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 5px;
                margin-bottom: 20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .css-1d391kg {
                background-color: #f0f2f6;
                border-radius: 10px;
                padding: 20px;
                margin-top: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Optional top logo & heading
    with st.container():
        col_logo, col_title = st.columns([1, 4])
        with col_logo:
            logo_path = os.path.join(base_dir, "logo.jpg")
            if os.path.exists(logo_path):
                st.image(logo_path, width=200)
            else:
                st.write("Logo not found.")
        with col_title:
            st.markdown(
                '<div class="main-header"><h1>Ποιοτικά χαρακτηριστικά Επιφανειακού Ύδατος Λίμνης Κορώνεια</h1></div>',
                unsafe_allow_html=True
            )

    # Define default paths
    lake_coordinates_path = os.path.join(base_dir, "lake coordinates.txt")
    sampling_kml_path = os.path.join(base_dir, "sampling.kml")
    images_folder = os.path.join(base_dir, "GeoTIFFs")

    # Sidebar: date range, background image selection
    with st.sidebar:
        st.header("Ρυθμίσεις Ανάλυσης (Dashboard)")
        x_start = st.date_input("Έναρξη", date(2015, 1, 1))
        x_end = st.date_input("Λήξη", date(2026, 12, 31))
        x_start_dt = datetime.combine(x_start, datetime.min.time())
        x_end_dt = datetime.combine(x_end, datetime.min.time())

        tif_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.tif', '.tiff'))]
        available_dates = {}
        for filename in tif_files:
            match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
            if match:
                date_str = match.group(1)
                try:
                    date_obj = datetime.strptime(date_str, '%Y_%m_%d').date()
                    available_dates[str(date_obj)] = filename
                except Exception:
                    continue
        if available_dates:
            sorted_dates = sorted(available_dates.keys())
            selected_bg_date = st.selectbox(
                "Επιλέξτε ημερομηνία για το background της GeoTIFF εικόνας",
                sorted_dates
            )
        else:
            selected_bg_date = None
            st.warning("Δεν βρέθηκαν GeoTIFF εικόνες με ημερομηνία στον τίτλο.")

    if selected_bg_date is not None:
        bg_filename = available_dates[selected_bg_date]
        with rasterio.open(os.path.join(images_folder, bg_filename)) as src:
            if src.count >= 3:
                first_image_data = src.read([1, 2, 3])
                first_transform = src.transform
            else:
                st.error("Το επιλεγμένο GeoTIFF δεν περιέχει τουλάχιστον 3 κανάλια.")
                st.stop()
    else:
        st.error("Δεν έχει επιλεγεί έγκυρη ημερομηνία για το background.")
        st.stop()

    # Optional MP4 video
    video_file = "Sentinel-2_L1C-372961813061153-timelapse.mp4"
    video_path = os.path.join(base_dir, video_file)
    if not os.path.exists(video_path):
        st.info("Το αρχείο βίντεο (timelapse) δεν βρέθηκε στο φάκελο της εφαρμογής.")

    # Helper functions for sampling analysis
    def parse_sampling_kml(kml_file) -> list:
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

    def geographic_to_pixel(lon: float, lat: float, transform) -> tuple:
        inverse_transform = ~transform
        col, row = inverse_transform * (lon, lat)
        return int(col), int(row)

    def map_rgb_to_mg(r: float, g: float, b: float, mg_factor: float = 2.0) -> float:
        """
        Example formula: mg_value ~ (g / 255) * mg_factor
        """
        return (g / 255.0) * mg_factor

    def mg_to_color(mg: float) -> str:
        """
        Example color scale from 0.00 mg to 2.00 mg.
        """
        scale = [
            (0.00, "#0000ff"), (0.02, "#0007f2"), (0.04, "#0011de"), (0.06, "#0017d0"),
            (1.98, "#80007d"), (2.00, "#800080")
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
        """
        Reads each GeoTIFF in 'images_folder', extracts pixel color for each sampling point,
        converts to mg, merges with lake height data from 'lake_height_path'.
        Returns multiple figures + color data arrays.
        """
        results_colors = {name: [] for name, _, _ in sampling_points}
        results_mg = {name: [] for name, _, _ in sampling_points}

        # For each TIF, read R/G/B at sampling coords
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
                        # We need at least 3 bands for R/G/B
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

        # Plot the chosen TIF as background with sampling points
        rgb_image = first_image_data.transpose((1, 2, 0)) / 255.0
        fig_geo = px.imshow(rgb_image, title='GeoTIFF Image with Sampling Points')
        for name, lon, lat in sampling_points:
            col, row = geographic_to_pixel(lon, lat, first_transform)
            fig_geo.add_trace(go.Scatter(
                x=[col], y=[row],
                mode='markers',
                marker=dict(color='red', size=8),
                name=name
            ))
        fig_geo.update_xaxes(visible=False)
        fig_geo.update_yaxes(visible=False)
        fig_geo.update_layout(width=1200, height=600, showlegend=False)

        # Try to load lake height data
        try:
            lake_data = pd.read_excel(lake_height_path)
            lake_data['Date'] = pd.to_datetime(lake_data.iloc[:, 0])
            lake_data.sort_values('Date', inplace=True)
        except Exception as e:
            st.error(f"Error reading lake height file: {e}")
            lake_data = pd.DataFrame()

        # Build color timeline plot (colors over time + lake height)
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
            scatter_traces.append(go.Scatter(
                x=dates,
                y=[idx] * len(dates),
                mode='markers',
                marker=dict(color=colors, size=10),
                name=name
            ))

        fig_colors = make_subplots(specs=[[{"secondary_y": True}]])
        for trace in scatter_traces:
            fig_colors.add_trace(trace, secondary_y=False)

        if not lake_data.empty:
            trace_height = go.Scatter(
                x=lake_data['Date'],
                y=lake_data[lake_data.columns[1]],
                name='Lake Height',
                mode='lines',
                line=dict(color='blue', width=2)
            )
            fig_colors.add_trace(trace_height, secondary_y=True)

        fig_colors.update_layout(
            title='Pixel Colors and Lake Height Over Time',
            xaxis_title='Date',
            yaxis_title='Sampling Points',
            showlegend=True
        )
        fig_colors.update_yaxes(title_text="Lake Height", secondary_y=True)

        # Compute average mg across all sampling points
        all_dates = {}
        for data_list in results_mg.values():
            for date_obj, mg_val in data_list:
                all_dates.setdefault(date_obj, []).append(mg_val)
        sorted_dates = sorted(all_dates.keys())
        avg_mg = [np.mean(all_dates[d]) for d in sorted_dates]

        fig_mg = go.Figure()
        fig_mg.add_trace(go.Scatter(
            x=sorted_dates,
            y=avg_mg,
            mode='markers',
            marker=dict(
                color=avg_mg,
                colorscale='Viridis',
                reversescale=True,
                colorbar=dict(title='mg/m³'),
                size=10
            ),
            name='Average mg/m³'
        ))
        fig_mg.update_layout(
            title='Average mg/m³ Over Time',
            xaxis_title='Date',
            yaxis_title='mg/m³',
            showlegend=False
        )

        # Dual plot: lake height + average mg
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        if not lake_data.empty:
            fig_dual.add_trace(
                go.Scatter(
                    x=lake_data['Date'],
                    y=lake_data[lake_data.columns[1]],
                    name='Lake Height',
                    mode='lines'
                ),
                secondary_y=False
            )
        fig_dual.add_trace(
            go.Scatter(
                x=sorted_dates,
                y=avg_mg,
                name='Average mg/m³',
                mode='markers',
                marker=dict(
                    color=avg_mg,
                    colorscale='Viridis',
                    reversescale=True,
                    colorbar=dict(title='mg/m³'),
                    size=10
                )
            ),
            secondary_y=True
        )
        fig_dual.update_layout(
            title='Lake Height and Average mg/m³ Over Time',
            xaxis_title='Date',
            showlegend=False
        )
        fig_dual.update_yaxes(title_text="Lake Height", secondary_y=False)
        fig_dual.update_yaxes(title_text="mg/m³", secondary_y=True)

        return fig_geo, fig_dual, fig_colors, fig_mg, results_colors, results_mg, lake_data

    # ~~~~~ Session state for results ~~~~~
    if "default_results" not in st.session_state:
        st.session_state.default_results = None
    if "upload_results" not in st.session_state:
        st.session_state.upload_results = None

    # ~~~~~ Tabs for default vs. uploaded sampling ~~~~~
    tab_names = ["Δειγματοληψία 1 (Default)", "Δειγματοληψία 2 (Upload)"]
    tabs = st.tabs(tab_names)

    # ----- Tab 1: Default Sampling -----
    with tabs[0]:
        st.header("Ανάλυση για Δειγματοληψία 1 (Default)")
        sampling_kml_path = os.path.join(base_dir, "sampling.kml")
        try:
            with open(sampling_kml_path, "r", encoding="utf-8") as kml_file:
                default_sampling_points = parse_sampling_kml(kml_file)
        except Exception as e:
            st.error(f"Error opening 'sampling.kml': {e}")
            default_sampling_points = []

        st.write("Using default sampling points:")
        st.write(default_sampling_points)
        point_names = [name for name, _, _ in default_sampling_points]
        selected_points = st.multiselect(
            "Select points to display mg/m³ concentrations",
            options=point_names,
            default=point_names
        )

        if st.button("Run Analysis (Default)"):
            with st.spinner("Running analysis, please wait..."):
                st.session_state.default_results = analyze_sampling(
                    default_sampling_points,
                    first_image_data,
                    first_transform,
                    images_folder,
                    os.path.join(base_dir, "lake height.xlsx"),
                    selected_points
                )

        if st.session_state.default_results is not None:
            results = st.session_state.default_results
            if isinstance(results, tuple) and len(results) == 7:
                fig_geo, fig_dual, fig_colors, fig_mg, results_colors, results_mg, lake_data = results
            else:
                st.error("Analysis result format error. Please run the analysis again.")
                st.stop()

            nested_tabs = st.tabs(["GeoTIFF", "Video", "Pixel Colors", "Average mg", "Dual Plots", "Detail mg", "Yearly Charts"])
            with nested_tabs[0]:
                st.plotly_chart(fig_geo, use_container_width=True, config={'scrollZoom': True})
            with nested_tabs[1]:
                if os.path.exists(video_path):
                    st.video(video_path)
                else:
                    st.info("Video file not found.")
            with nested_tabs[2]:
                st.plotly_chart(fig_colors, use_container_width=True, config={'scrollZoom': True})
            with nested_tabs[3]:
                st.plotly_chart(fig_mg, use_container_width=True, config={'scrollZoom': True})
            with nested_tabs[4]:
                st.plotly_chart(fig_dual, use_container_width=True, config={'scrollZoom': True})
            with nested_tabs[5]:
                selected_detail_point = st.selectbox(
                    "Select a point for detailed mg analysis",
                    options=list(results_mg.keys())
                )
                if selected_detail_point:
                    mg_data = results_mg[selected_detail_point]
                    if mg_data:
                        mg_data_sorted = sorted(mg_data, key=lambda x: x[0])
                        dates_mg = [d for d, _ in mg_data_sorted]
                        mg_values = [val for _, val in mg_data_sorted]
                        detail_colors = [mg_to_color(val) for val in mg_values]
                        fig_detail = go.Figure()
                        fig_detail.add_trace(go.Scatter(
                            x=dates_mg,
                            y=mg_values,
                            mode='lines+markers',
                            marker=dict(color=detail_colors, size=10),
                            line=dict(color="gray"),
                            name=selected_detail_point
                        ))
                        fig_detail.update_layout(
                            title=f"Detailed mg analysis for {selected_detail_point}",
                            xaxis_title="Date",
                            yaxis_title="mg/m³"
                        )
                        st.plotly_chart(fig_detail, use_container_width=True)
                    else:
                        st.info("No mg data for this point.")
            with nested_tabs[6]:
                # Yearly charts by color + lake height
                if not lake_data.empty:
                    if not np.issubdtype(lake_data['Date'].dtype, np.datetime64):
                        lake_data['Date'] = pd.to_datetime(lake_data['Date'])
                    years = list(range(2015, 2027))
                    st.markdown("### Yearly Charts")
                    for idx in range(0, len(years), 3):
                        cols = st.columns(3)
                        for j, year in enumerate(years[idx:idx+3]):
                            fig_year = make_subplots(specs=[[{"secondary_y": True}]])
                            for pt_idx, name in enumerate(selected_points):
                                data_list = results_colors.get(name, [])
                                filtered_data = [(d, color) for d, color in data_list if d.year == year]
                                if not filtered_data:
                                    continue
                                dates_year = [d for d, _ in filtered_data]
                                colors_year = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for _, c in filtered_data]
                                fig_year.add_trace(go.Scatter(
                                    x=dates_year,
                                    y=[pt_idx] * len(dates_year),
                                    mode='markers',
                                    marker=dict(color=colors_year, size=10),
                                    name=name
                                ), secondary_y=False)
                            lake_year = lake_data[lake_data['Date'].dt.year == year]
                            if not lake_year.empty:
                                fig_year.add_trace(go.Scatter(
                                    x=lake_year['Date'],
                                    y=lake_year[lake_data.columns[1]],
                                    mode='lines',
                                    line=dict(color='blue', width=2),
                                    name='Lake Height'
                                ), secondary_y=True)
                            fig_year.update_layout(
                                title=f'Pixel Colors and Lake Height for {year}',
                                xaxis_title='Date',
                                showlegend=True,
                                xaxis_range=[datetime(year, 1, 1), datetime(year, 12, 31)]
                            )
                            cols[j].plotly_chart(fig_year, use_container_width=True, config={'scrollZoom': True})

    # ----- Tab 2: Uploaded Sampling -----
    with tabs[1]:
        st.header("Analysis for Uploaded Sampling")
        uploaded_file = st.file_uploader("Upload a KML file for new sampling points", type="kml", key="upload_tab")
        if uploaded_file is not None:
            try:
                new_sampling_points = parse_sampling_kml(uploaded_file)
            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")
                new_sampling_points = []
            st.write("Using the following new sampling points:")
            st.write(new_sampling_points)
            point_names = [name for name, _, _ in new_sampling_points]
            selected_points = st.multiselect(
                "Select points to display mg/m³ concentrations",
                options=point_names, default=point_names
            )
            if st.button("Run Analysis (Upload)"):
                with st.spinner("Running analysis, please wait..."):
                    st.session_state.upload_results = analyze_sampling(
                        new_sampling_points,
                        first_image_data,
                        first_transform,
                        images_folder,
                        os.path.join(base_dir, "lake height.xlsx"),
                        selected_points
                    )
            if st.session_state.upload_results is not None:
                results = st.session_state.upload_results
                if isinstance(results, tuple) and len(results) == 7:
                    fig_geo, fig_dual, fig_colors, fig_mg, results_colors, results_mg, lake_data = results
                else:
                    st.error("Analysis result format error (Upload). Please run the analysis again.")
                    st.stop()

                nested_tabs = st.tabs(["GeoTIFF", "Video", "Pixel Colors", "Average mg", "Dual Plots", "Detail mg", "Yearly Charts"])
                with nested_tabs[0]:
                    st.plotly_chart(fig_geo, use_container_width=True, config={'scrollZoom': True})
                with nested_tabs[1]:
                    if os.path.exists(video_path):
                        st.video(video_path)
                    else:
                        st.info("Video file not found.")
                with nested_tabs[2]:
                    st.plotly_chart(fig_colors, use_container_width=True, config={'scrollZoom': True})
                with nested_tabs[3]:
                    st.plotly_chart(fig_mg, use_container_width=True, config={'scrollZoom': True})
                with nested_tabs[4]:
                    st.plotly_chart(fig_dual, use_container_width=True, config={'scrollZoom': True})
                with nested_tabs[5]:
                    selected_detail_point = st.selectbox(
                        "Select a point for detailed mg analysis",
                        options=list(results_mg.keys()), key="detail_upload"
                    )
                    if selected_detail_point:
                        mg_data = results_mg[selected_detail_point]
                        if mg_data:
                            mg_data_sorted = sorted(mg_data, key=lambda x: x[0])
                            dates_mg = [d for d, _ in mg_data_sorted]
                            mg_values = [val for _, val in mg_data_sorted]
                            detail_colors = [mg_to_color(val) for val in mg_values]
                            fig_detail = go.Figure()
                            fig_detail.add_trace(go.Scatter(
                                x=dates_mg,
                                y=mg_values,
                                mode='lines+markers',
                                marker=dict(color=detail_colors, size=10),
                                line=dict(color="gray"),
                                name=selected_detail_point
                            ))
                            fig_detail.update_layout(
                                title=f"Detailed mg analysis for {selected_detail_point}",
                                xaxis_title="Date",
                                yaxis_title="mg/m³"
                            )
                            st.plotly_chart(fig_detail, use_container_width=True)
                        else:
                            st.info("No mg data for this point.")
                with nested_tabs[6]:
                    if not lake_data.empty:
                        if not np.issubdtype(lake_data['Date'].dtype, np.datetime64):
                            lake_data['Date'] = pd.to_datetime(lake_data['Date'])
                        years = list(range(2015, 2027))
                        st.markdown("### Yearly Charts")
                        for idx in range(0, len(years), 3):
                            cols = st.columns(3)
                            for j, year in enumerate(years[idx:idx+3]):
                                fig_year = make_subplots(specs=[[{"secondary_y": True}]])
                                for pt_idx, name in enumerate(selected_points):
                                    data_list = results_colors.get(name, [])
                                    filtered_data = [(d, color) for d, color in data_list if d.year == year]
                                    if not filtered_data:
                                        continue
                                    dates_year = [d for d, _ in filtered_data]
                                    colors_year = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for _, c in filtered_data]
                                    fig_year.add_trace(go.Scatter(
                                        x=dates_year,
                                        y=[pt_idx] * len(dates_year),
                                        mode='markers',
                                        marker=dict(color=colors_year, size=10),
                                        name=name
                                    ), secondary_y=False)
                                lake_year = lake_data[lake_data['Date'].dt.year == year]
                                if not lake_year.empty:
                                    fig_year.add_trace(go.Scatter(
                                        x=lake_year['Date'],
                                        y=lake_year[lake_data.columns[1]],
                                        mode='lines',
                                        line=dict(color='blue', width=2),
                                        name='Lake Height'
                                    ), secondary_y=True)
                                fig_year.update_layout(
                                    title=f'Pixel Colors and Lake Height for {year}',
                                    xaxis_title='Date',
                                    showlegend=True,
                                    xaxis_range=[datetime(year, 1, 1), datetime(year, 12, 31)]
                                )
                                cols[j].plotly_chart(fig_year, use_container_width=True, config={'scrollZoom': True})
        else:
            st.info("Please upload a KML file for new sampling points.")

    st.info("End of Water Quality Dashboard section.")

# -----------------------------------------------------------------------------
# Burned Areas (Placeholder)
# -----------------------------------------------------------------------------
def run_burned_areas():
    st.title("Burned Areas around reservoir (Placeholder)")
    st.info("No data or functionality yet for burned-area analysis.")

# -----------------------------------------------------------------------------
# Water Level (Placeholder)
# -----------------------------------------------------------------------------
def run_water_level_profiles():
    st.title("Water level Height Profiles (Placeholder)")
    st.info("No data or functionality yet for water-level height profiles.")

# -----------------------------------------------------------------------------
# Pattern Analysis (Not linked to new UI, but included in full)
# -----------------------------------------------------------------------------
def run_pattern_analysis():
    """
    Full pattern analysis code: monthly days in range, spatial classification, etc.
    Currently not linked from the new 3-row UI. 
    You can call run_pattern_analysis() from main() if you want a separate page for it.
    """
    st.title("Pattern Analysis - Spatial and Temporal Reports")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "GeoTIFFs")

    try:
        STACK, DAYS, DATES = load_data(input_folder)
        st.success("Data loaded successfully for pattern analysis.")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    st.sidebar.header("Pattern Analysis Controls")
    unique_years = sorted({d.year for d in DATES})
    selected_years_pattern = st.sidebar.multiselect("Select Years for Pattern Analysis", options=unique_years, default=unique_years, key="pattern_years")
    selected_months_pattern = st.sidebar.multiselect("Select Months for Pattern Analysis", options=list(range(1, 13)), default=list(range(1, 13)), key="pattern_months", format_func=lambda m: datetime(2000, m, 1).strftime('%B'))
    threshold_range = st.sidebar.slider("Select pixel value threshold range", 0, 255, (0, 255), key="pattern_threshold")
    lower_thresh, upper_thresh = threshold_range

    if not selected_years_pattern or not selected_months_pattern:
        st.error("Please select at least one year and one month.")
        st.stop()

    indices = [i for i, d in enumerate(DATES) if d.year in selected_years_pattern and d.month in selected_months_pattern]
    if not indices:
        st.error("No data for the selected years/months.")
        st.stop()

    STACK_filtered = STACK[indices, :, :]
    stack_full_in_range = (STACK_filtered >= lower_thresh) & (STACK_filtered <= upper_thresh)
    filtered_dates = [DATES[i] for i in indices]

    # Compute monthly fraction
    monthly_avg = {}
    for m in selected_months_pattern:
        month_indices = [i for i, dd in enumerate(filtered_dates) if dd.month == m]
        if month_indices:
            avg_days = np.nanmean(stack_full_in_range[month_indices, :, :], axis=0)
            monthly_avg[m] = avg_days
        else:
            monthly_avg[m] = None

    # Overall average
    agg_avg = None
    count = 0
    for m in monthly_avg:
        if monthly_avg[m] is not None:
            if agg_avg is None:
                agg_avg = monthly_avg[m]
            else:
                agg_avg += monthly_avg[m]
            count += 1
    if agg_avg is not None and count > 0:
        overall_avg = agg_avg / count
    else:
        overall_avg = None

    # Temporal pattern
    temporal_data = []
    for m in sorted(monthly_avg.keys()):
        if monthly_avg[m] is not None:
            spatial_avg = np.nanmean(monthly_avg[m])
            temporal_data.append((m, spatial_avg))
    if temporal_data:
        months, means = zip(*temporal_data)
        month_names = [datetime(2000, mm, 1).strftime('%B') for mm in months]
        fig_temporal = px.bar(
            x=month_names,
            y=means,
            labels={'x': 'Month', 'y': 'Average Fraction In Range'},
            title="Temporal Pattern: Average Fraction In Range per Month"
        )
    else:
        fig_temporal = go.Figure()

    # Spatial classification
    if overall_avg is not None:
        classification = np.full(overall_avg.shape, "Unclassified", dtype=object)
        valid_mask = ~np.isnan(overall_avg)
        classification[valid_mask & (overall_avg < 0.3)] = "Low"
        classification[valid_mask & (overall_avg >= 0.3) & (overall_avg < 0.7)] = "Medium"
        classification[valid_mask & (overall_avg >= 0.7)] = "High"
        mapping_dict = {"Low": 0, "Medium": 1, "High": 2, "Unclassified": 3}
        numeric_class = np.vectorize(lambda x: mapping_dict[x])(classification)
        discrete_colorscale = [[0, "blue"], [0.33, "yellow"], [0.66, "red"], [1.0, "gray"]]
        fig_class = px.imshow(
            numeric_class,
            color_continuous_scale=discrete_colorscale,
            title="Spatial Pattern Classification"
        )
        fig_class.update_traces(
            colorbar=dict(tickvals=[0,1,2,3], ticktext=["Low", "Medium", "High", "Unclassified"])
        )
    else:
        fig_class = go.Figure()

    st.header("Pattern Analysis")
    st.markdown("Analyzes monthly days in range data, plus a spatial classification of persistent in-range fractions.")
    st.subheader("Temporal Pattern")
    st.plotly_chart(fig_temporal, use_container_width=True)
    st.subheader("Spatial Pattern Classification")
    st.plotly_chart(fig_class, use_container_width=True)

    if temporal_data:
        df_temporal = pd.DataFrame(temporal_data, columns=["Month", "Average Fraction In Range"])
        csv = df_temporal.to_csv(index=False).encode('utf-8')
        st.download_button("Download Temporal Analysis CSV", data=csv, file_name="temporal_analysis.csv", mime="text/csv")

    st.info("End of Pattern Analysis section.")

# -----------------------------------------------------------------------------
# 3-Row UI
# -----------------------------------------------------------------------------
def run_custom_ui():
    """
    Row 1: Επιλογή υδάτινου σώματος → [Κορώνεια, Πολυφύτου, Γαδουρά, Αξιός]
    Row 2: Επιλογή Δείκτη → [Πραγματικό, Χλωροφύλλη, CDOM, Colour]
    Row 3: [Lake processing, Water Processing, Water Quality Dashboard,
            Burned Areas around reservoir, Water level Height Profiles]
    """
    if "waterbody_choice" not in st.session_state:
        st.session_state["waterbody_choice"] = None
    if "index_choice" not in st.session_state:
        st.session_state["index_choice"] = None
    if "analysis_choice" not in st.session_state:
        st.session_state["analysis_choice"] = None

    # --- Row 1 ---
    row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns(5)
    with row1_col1:
        st.subheader("Επιλογή υδάτινου σώματος")
    with row1_col2:
        if st.button("Κορώνεια"):
            st.session_state["waterbody_choice"] = "Κορώνεια"
    with row1_col3:
        if st.button("Πολυφύτου"):
            st.session_state["waterbody_choice"] = "Πολυφύτου"
    with row1_col4:
        if st.button("Γαδουρά"):
            st.session_state["waterbody_choice"] = "Γαδουρά"
    with row1_col5:
        if st.button("Αξιός"):
            st.session_state["waterbody_choice"] = "Αξιός"

    st.write("")

    # --- Row 2 ---
    row2_col1, row2_col2, row2_col3, row2_col4, row2_col5 = st.columns(5)
    with row2_col1:
        st.subheader("Επιλογή Δείκτη")
    with row2_col2:
        if st.button("Πραγματικό"):
            st.session_state["index_choice"] = "Πραγματικό"
    with row2_col3:
        if st.button("Χλωροφύλλη"):
            st.session_state["index_choice"] = "Χλωροφύλλη"
    with row2_col4:
        if st.button("CDOM"):
            st.session_state["index_choice"] = "CDOM"
    with row2_col5:
        if st.button("Colour"):
            st.session_state["index_choice"] = "Colour"

    st.write("")

    # --- Row 3 ---
    row3_col1, row3_col2, row3_col3, row3_col4, row3_col5 = st.columns(5)
    with row3_col1:
        if st.button("Lake processing"):
            st.session_state["analysis_choice"] = "Lake Processing"
    with row3_col2:
        if st.button("Water Processing"):
            st.session_state["analysis_choice"] = "Water Processing"
    with row3_col3:
        if st.button("Water Quality Dashboard"):
            st.session_state["analysis_choice"] = "Water Quality Dashboard"
    with row3_col4:
        if st.button("Burned Areas around reservoir"):
            st.session_state["analysis_choice"] = "Burned Areas"
    with row3_col5:
        if st.button("Water level Height Profiles"):
            st.session_state["analysis_choice"] = "Water level"

    st.write("---")
    st.write(f"**Επιλεγμένο υδάτινο σώμα:** {st.session_state.get('waterbody_choice', 'None')}")
    st.write(f"**Επιλεγμένος Δείκτης:** {st.session_state.get('index_choice', 'None')}")
    st.write(f"**Επιλεγμένο Είδος Ανάλυσης:** {st.session_state.get('analysis_choice', 'None')}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    """
    1) Show Intro page
    2) Show 3-row UI
    3) If user picks (Κορώνεια + Χλωροφύλλη), route to chosen analysis
    4) Otherwise, show placeholders or warnings
    """
    # 1) Intro
    # We do the intro page to show the headline & some text
    run_intro_page()

    # 2) Show the custom UI
    run_custom_ui()

    # 3) Decide which page to run
    wb = st.session_state.get("waterbody_choice", None)
    idx = st.session_state.get("index_choice", None)
    analysis = st.session_state.get("analysis_choice", None)

    # Real data only for (Κορώνεια + Χλωροφύλλη)
    if wb == "Κορώνεια" and idx == "Χλωροφύλλη":
        if analysis == "Lake Processing":
            run_lake_processing_app()
        elif analysis == "Water Processing":
            run_water_processing()
        elif analysis == "Water Quality Dashboard":
            run_water_quality_dashboard()
        elif analysis == "Burned Areas":
            run_burned_areas()
        elif analysis == "Water level":
            run_water_level_profiles()
        else:
            st.info("Please select an Είδος Ανάλυσης (third row).")
    else:
        if analysis is not None:
            st.warning(
                "Currently, data are only available for (Κορώνεια + Χλωροφύλλη). "
                "Other combinations are placeholders until data are added."
            )

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
