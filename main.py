#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Water Quality Analysis Platform
-------------------------------
An enterprise-grade application for analyzing surface water quality using satellite data.
Features a professional, user-friendly interface with advanced visualization tools.
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

# Global debug flag (hidden by default)
DEBUG = False

def debug(*args, **kwargs):
    """Display debug messages if DEBUG is enabled."""
    if DEBUG:
        st.write(*args, **kwargs)

# -------------------------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Water Quality Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom CSS for Professional Appearance
# -----------------------------------------------------------------------------
def inject_custom_css():
    """Inject custom CSS for a modern, professional UI."""
    custom_css = """
    <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Open Sans', sans-serif;
            background: #f5f6f5;
            color: #333333;
        }
        .stApp {
            max-width: 1400px;
            margin: 0 auto;
        }
        .block-container {
            padding: 2rem;
        }
        .sidebar .sidebar-content {
            background: #ffffff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            padding: 1.5rem;
        }
        .card {
            background: #ffffff;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        .header-title {
            color: #1e88e5;
            font-size: 2rem;
            font-weight: 600;
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .nav-section {
            padding: 1rem;
            background: #f9f9f9;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .nav-section h4 {
            color: #1e88e5;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .stButton button {
            background-color: #1e88e5;
            color: white;
            border-radius: 6px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: background-color 0.3s ease;
        }
        .stButton button:hover {
            background-color: #1565c0;
        }
        .plotly-graph-div {
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

inject_custom_css()

# -----------------------------------------------------------------------------
# Utility Functions (Unchanged Core Logic)
# -----------------------------------------------------------------------------
def get_data_folder(waterbody: str, index: str) -> str:
    """Map selected waterbody and index to the correct data folder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    waterbody_map = {"Κορώνεια": "Koroneia", "Πολυφύτου": "polyphytou", "Γαδουρά": "Gadoura", "Αξιός": "Axios"}
    waterbody_folder = waterbody_map.get(waterbody)
    if not waterbody_folder:
        return None
    folder_map = {"Χλωροφύλλη": "Chlorophyll", "Burned Areas": "Burned Areas"}
    folder = folder_map.get(index, index)
    data_folder = os.path.join(base_dir, waterbody_folder, folder)
    return data_folder if os.path.exists(data_folder) else None

def extract_date_from_filename(filename: str):
    """Extract date (YYYY-MM-DD) from filename using regex."""
    match = re.search(r'(\d{4})[_-](\d{2})[_-](\d{2})', os.path.basename(filename))
    if match:
        year, month, day = map(int, match.groups())
        date_obj = datetime(year, month, day)
        return date_obj.timetuple().tm_yday, date_obj
    return None, None

def load_lake_shape_from_xml(xml_file: str, bounds: tuple = None, xml_width: float = 518.0, xml_height: float = 505.0):
    """Load lake outline from XML file, optionally transforming coordinates."""
    try:
        tree = ET.parse(xml_file)
        points = [[float(p.get("x")), float(p.get("y"))] for p in tree.findall("point") if p.get("x") and p.get("y")]
        if not points:
            return None
        if bounds:
            minx, miny, maxx, maxy = bounds
            points = [[minx + (x / xml_width) * (maxx - minx), maxy - (y / xml_height) * (maxy - miny)] for x, y in points]
        points.append(points[0]) if points[0] != points[-1] else None
        return {"type": "Polygon", "coordinates": [points]}
    except Exception as e:
        st.error(f"Failed to load lake shape from {xml_file}: {e}")
        return None

def read_image(file_path: str, lake_shape: dict = None):
    """Read a GeoTIFF file and apply a mask if provided."""
    with rasterio.open(file_path) as src:
        img = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        profile.update(dtype="float32")
        img = np.where(img == src.nodata, np.nan, img) if src.nodata is not None else img
        img = np.where(img == 0, np.nan, img)
        if lake_shape:
            from rasterio.features import geometry_mask
            mask = geometry_mask([lake_shape], transform=src.transform, invert=False, out_shape=img.shape)
            img = np.where(~mask, img, np.nan)
    return img, profile

def load_data(input_folder: str, shapefile_name="shapefile.xml"):
    """Load all TIFF files from a folder, applying masks and extracting dates."""
    if not os.path.exists(input_folder):
        raise ValueError(f"Folder not found: {input_folder}")
    shape_file = next((os.path.join(input_folder, ext) for ext in [shapefile_name, "shapefile.txt"] if os.path.exists(os.path.join(input_folder, ext))), None)
    tif_files = [f for f in sorted(glob.glob(os.path.join(input_folder, "*.tif"))) if "mask.tif" not in f.lower()]
    if not tif_files:
        raise ValueError("No GeoTIFF files found.")
    with rasterio.open(tif_files[0]) as src:
        bounds = src.bounds
    lake_shape = load_lake_shape_from_xml(shape_file, bounds) if shape_file else None
    images, days, dates = [], [], []
    for file in tif_files:
        day, date_obj = extract_date_from_filename(file)
        if day:
            img, _ = read_image(file, lake_shape)
            images.append(img)
            days.append(day)
            dates.append(date_obj)
    if not images:
        raise ValueError("No valid images found.")
    return np.stack(images), np.array(days), dates

# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------
def display_intro():
    """Display a professional introduction card."""
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3], gap="medium")
        with col1:
            logo_path = os.path.join(os.path.dirname(__file__), "logo.jpg")
            if os.path.exists(logo_path):
                st.image(logo_path, width=200)
        with col2:
            st.markdown('<h2 class="header-title">Water Quality Analysis Platform</h2>', unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align: center; color: #666;'>"
                "Analyze surface water quality using advanced satellite remote sensing tools. "
                "Configure your analysis via the sidebar to explore detailed insights.</p>",
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

def render_sidebar():
    """Render a structured sidebar for analysis configuration."""
    with st.sidebar:
        st.markdown('<div class="nav-section"><h4>Analysis Configuration</h4></div>', unsafe_allow_html=True)
        waterbody = st.selectbox("Waterbody", ["Κορώνεια", "Πολυφύτου", "Γαδουρά", "Αξιός"], key="waterbody_choice")
        index = st.selectbox("Index", ["Πραγματικό", "Χλωροφύλλη", "CDOM", "Colour", "Burned Areas"], key="index_choice")
        analysis = st.selectbox("Analysis Type", ["Lake Processing", "Water Processing", "Water Quality Dashboard",
                                                  "Burned Areas", "Water Level", "Pattern Analysis"], key="analysis_choice")
        st.markdown(
            f"<div style='padding: 1rem; background: #f0f0f0; border-radius: 6px; font-size: 0.9rem;'>"
            f"<strong>Waterbody:</strong> {waterbody}<br>"
            f"<strong>Index:</strong> {index}<br>"
            f"<strong>Analysis:</strong> {analysis}"
            f"</div>", unsafe_allow_html=True
        )
    return waterbody, index, analysis

def render_analysis_filters(waterbody, analysis_type, dates):
    """Render analysis-specific filters in the sidebar."""
    with st.sidebar:
        st.markdown(f'<div class="nav-section"><h4>Filters ({analysis_type})</h4></div>', unsafe_allow_html=True)
        min_date, max_date = min(dates), max(dates)
        date_range = st.slider("Date Range", min_date, max_date, (min_date, max_date), key=f"date_{analysis_type}")
        threshold = st.slider("Pixel Value Range", 0, 255, (0, 255), key=f"thresh_{analysis_type}")
        unique_years = sorted({d.year for d in dates})
        years = st.multiselect("Years", unique_years, default=unique_years, key=f"years_{analysis_type}")
        months = st.multiselect("Months", list(range(1, 13)), default=list(range(1, 13)), 
                                format_func=lambda x: datetime(2000, x, 1).strftime('%B'), key=f"months_{analysis_type}")
        return date_range, threshold, years, months

# -----------------------------------------------------------------------------
# Analysis Functions (Simplified for Example)
# -----------------------------------------------------------------------------
def run_lake_processing(waterbody: str, index: str):
    """Run lake processing analysis with enhanced visualizations."""
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<h2 class="header-title">Lake Processing: {waterbody} - {index}</h2>', unsafe_allow_html=True)
        
        data_folder = get_data_folder(waterbody, index)
        if not data_folder:
            st.error("Data folder not found for the selected waterbody/index.")
            return
        
        with st.spinner("Loading data..."):
            stack, days, dates = load_data(os.path.join(data_folder, "GeoTIFFs"))
        
        date_range, threshold, years, months = render_analysis_filters(waterbody, "Lake Processing", dates)
        lower, upper = threshold
        filtered_indices = [i for i, d in enumerate(dates) if date_range[0] <= d <= date_range[1] 
                            and d.year in years and d.month in months]
        
        if not filtered_indices:
            st.warning("No data available for the selected filters.")
            return
        
        stack_filtered = stack[filtered_indices]
        in_range = np.logical_and(stack_filtered >= lower, stack_filtered <= upper)
        
        # Example Visualization
        days_in_range = np.nansum(in_range, axis=0)
        fig = px.imshow(days_in_range, color_continuous_scale="viridis", title="Days Within Range")
        fig.update_layout(width=800, height=600, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Placeholder for other analysis functions (implement similarly)
def run_water_quality_dashboard(waterbody: str, index: str):
    st.warning("Water Quality Dashboard is under development.")

def run_pattern_analysis(waterbody: str, index: str):
    st.warning("Pattern Analysis is under development.")

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    display_intro()
    waterbody, index, analysis = render_sidebar()
    
    analysis_map = {
        "Lake Processing": run_lake_processing,
        "Water Quality Dashboard": run_water_quality_dashboard,
        "Pattern Analysis": run_pattern_analysis,
        # Add other analysis types as needed
    }
    
    if analysis in analysis_map:
        with st.spinner(f"Running {analysis}..."):
            analysis_map[analysis](waterbody, index)
    else:
        st.info("Please select an analysis type to proceed.")

if __name__ == "__main__":
    main()
