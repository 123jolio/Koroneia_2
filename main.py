import streamlit as st
import json
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, date
import rasterio
import xml.etree.ElementTree as ET
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

# ========================================
# SET WORKING DIRECTORY & PAGE CONFIGURATION
# ========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

st.set_page_config(
    page_title="Ποιοτικά χαρακτηριστικά Επιφανειακού Ύδατος Ταμιευτήρα Γαδουρά Ρόδου", 
    layout="wide"
)

# ========================================
# CUSTOM CSS FOR A MODERN, RESPONSIVE UI
# ========================================
st.markdown(
    """
    <style>
    /* Global typography */
    body {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Header styling */
    .main-header {
         background-color: #4CAF50;
         color: white;
         padding: 20px;
         text-align: center;
         border-radius: 5px;
         margin-bottom: 20px;
         box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Sidebar styling */
    .css-1d391kg {  
         background-color: #f0f2f6;
         border-radius: 10px;
         padding: 20px;
         margin-top: 10px;
    }
    /* Video container: fixed container so that we can control its height via Streamlit */
    .responsive-video {
        position: relative;
        width: 100%;
        height: 100%;
        border: 2px solid #ddd;
        border-radius: 5px;
    }
    .responsive-video video {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    /* Plotly chart container styling */
    .plotly-graph-div {
        margin: 20px auto;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header"><h1>Ποιοτικά χαρακτηριστικά Επιφανειακού Ύδατος Ταμιευτήρα Γαδουρά Ρόδου</h1></div>', unsafe_allow_html=True)
st.write(f"Τρέχων φάκελος εκτέλεσης: {os.getcwd()}")

# ========================================
# DEFINE DEFAULT FILE PATHS FROM THE REPOSITORY
# ========================================
lake_coordinates_path = os.path.join(base_dir, "lake coordinates.txt")
sampling_kml_path = os.path.join(base_dir, "sampling.kml")  # Default sampling file

# ========================================
# SIDEBAR: SETTINGS
# ========================================
st.sidebar.header("Ρυθμίσεις Ανάλυσης")
st.sidebar.subheader("Ορισμός Χρονικού Διαστήματος (x-axis)")
x_start = st.sidebar.date_input("Έναρξη", date(2015, 1, 1))
x_end = st.sidebar.date_input("Λήξη", date(2026, 12, 31))
x_start_dt = datetime.combine(x_start, datetime.min.time())
x_end_dt = datetime.combine(x_end, datetime.min.time())

# Sidebar: Select GeoTIFF Background Date
images_folder = os.path.join(base_dir, "GeoTIFFs")
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
    selected_bg_date = st.sidebar.selectbox("Επιλέξτε ημερομηνία για το background της GeoTIFF εικόνας", sorted_dates)
else:
    selected_bg_date = None
    st.sidebar.warning("Δεν βρέθηκαν GeoTIFF εικόνες με ημερομηνία στον τίτλο.")

# ====================================================
# LOAD THE GEO-TIFF IMAGE (first_image_data and first_transform)
# ====================================================
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

# ========================================
# HELPER FUNCTIONS
# ========================================
def parse_sampling_kml(kml_file):
    """
    Parses a KML file (from disk or an uploaded file-like object) and returns a list of sampling points.
    Each point is a tuple: (name, longitude, latitude)
    """
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

def geographic_to_pixel(lon, lat, transform):
    inverse_transform = ~transform
    col, row = inverse_transform * (lon, lat)
    return int(col), int(row)

def map_rgb_to_mg(r, g, b, mg_factor=2.0):
    mg_value = (g / 255.0) * mg_factor
    return mg_value

def mg_to_color(mg):
    """
    Maps a given mg value (ranging from 0 to 2.0) to a color using the provided scale.
    The scale is defined by a list of (mg_value, hex_color) pairs.
    For mg values between two scale points, linear interpolation is used.
    """
    scale = [
        (0.00, "#0000ff"),
        (0.11, "#002ca8"),
        (0.21, "#005752"),
        (0.32, "#128900"),
        (0.42, "#93c900"),
        (0.53, "#ffef00"),
        (0.63, "#ffae00"),
        (0.74, "#ff6800"),
        (0.84, "#ff2700"),
        (0.95, "#f10000"),
        (1.05, "#d30000"),
        (1.16, "#b30000"),
        (1.26, "#960000"),
        (1.37, "#8a000d"),
        (1.47, "#880020"),
        (1.58, "#870034"),
        (1.68, "#850047"),
        (1.79, "#83005b"),
        (1.89, "#82006d"),
        (2.00, "#800080")
    ]
    # Clamp mg value to the scale range
    if mg <= scale[0][0]:
        low_color = scale[0][1]
        low_r = int(low_color[1:3], 16)
        low_g = int(low_color[3:5], 16)
        low_b = int(low_color[5:7], 16)
        return f"rgb({low_r},{low_g},{low_b})"
    if mg >= scale[-1][0]:
        high_color = scale[-1][1]
        high_r = int(high_color[1:3], 16)
        high_g = int(high_color[3:5], 16)
        high_b = int(high_color[5:7], 16)
        return f"rgb({high_r},{high_g},{high_b})"
    # Interpolate between the two bounding scale points
    for i in range(len(scale) - 1):
        low_mg, low_color = scale[i]
        high_mg, high_color = scale[i+1]
        if low_mg <= mg <= high_mg:
            t = (mg - low_mg) / (high_mg - low_mg)
            low_r = int(low_color[1:3], 16)
            low_g = int(low_color[3:5], 16)
            low_b = int(low_color[5:7], 16)
            high_r = int(high_color[1:3], 16)
            high_g = int(high_color[3:5], 16)
            high_b = int(high_color[5:7], 16)
            r = int(low_r + (high_r - low_r) * t)
            g = int(low_g + (high_g - low_g) * t)
            b = int(low_b + (high_b - low_b) * t)
            return f"rgb({r},{g},{b})"

def analyze_sampling(sampling_points, first_image_data, first_transform, images_folder, lake_height_path, selected_points=None):
    """
    Performs analysis for a given set of sampling points and returns:
      (fig_geo, fig_dual, fig_colors, fig_mg, results_colors, results_mg, lake_data)
    """
    results_colors = {name: [] for name, lon, lat in sampling_points}
    results_mg = {name: [] for name, lon, lat in sampling_points}
    
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
                for name, lon, lat in sampling_points:
                    col, row = geographic_to_pixel(lon, lat, transform)
                    if 0 <= col < width and 0 <= row < height:
                        window = rasterio.windows.Window(col, row, 1, 1)
                        r = src.read(1, window=window)[0, 0]
                        g = src.read(2, window=window)[0, 0]
                        b = src.read(3, window=window)[0, 0]
                        mg_value = map_rgb_to_mg(r, g, b, mg_factor=2.0)
                        results_mg[name].append((date_obj, mg_value))
                        pixel_color = (r / 255, g / 255, b / 255)
                        results_colors[name].append((date_obj, pixel_color))
    
    # Build the GeoTIFF image with sampling points overlaid.
    rgb_image = first_image_data.transpose((1, 2, 0)) / 255.0
    fig_geo = px.imshow(rgb_image, title='GeoTIFF Image with Sampling Points')
    for name, lon, lat in sampling_points:
        col, row = geographic_to_pixel(lon, lat, first_transform)
        fig_geo.add_trace(go.Scatter(x=[col], y=[row], mode='markers',
                                     marker=dict(color='red', size=8), name=name))
    fig_geo.update_xaxes(visible=False)
    fig_geo.update_yaxes(visible=False)
    fig_geo.update_layout(width=1200, height=600, showlegend=False)
    
    # Lake height plot (used in dual plot)
    try:
        lake_data = pd.read_excel(lake_height_path)
        lake_data['Date'] = pd.to_datetime(lake_data.iloc[:, 0])
        lake_data.sort_values('Date', inplace=True)
        fig_lake = px.line(lake_data, x='Date', y=lake_data.columns[1], title='Lake Height Over Time')
        fig_lake.update_layout(xaxis_range=[datetime(2015, 1, 1), datetime(2026, 12, 31)], showlegend=False)
    except Exception as e:
        st.error(f"Error reading lake height file: {str(e)}")
        fig_lake = go.Figure()
    
    # Pixel colors over time plot
    scatter_traces = []
    point_names = list(results_colors.keys())
    if selected_points is not None:
        point_names = [name for name in point_names if name in selected_points]
    for idx, name in enumerate(point_names):
        data_list = results_colors[name]
        if not data_list:
            continue
        data_list.sort(key=lambda x: x[0])
        dates = [d for d, _ in data_list]
        colors = [f'rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})' for _, c in data_list]
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
    try:
        lake_data = pd.read_excel(lake_height_path)
        lake_data['Date'] = pd.to_datetime(lake_data.iloc[:, 0])
        lake_data.sort_values('Date', inplace=True)
        trace_height = go.Scatter(
            x=lake_data['Date'],
            y=lake_data[lake_data.columns[1]],
            name='Lake Height',
            mode='lines',
            line=dict(color='blue', width=2)
        )
        fig_colors.add_trace(trace_height, secondary_y=True)
    except Exception as e:
        st.error(f"Error reading lake height file for Plot 3: {str(e)}")
    
    fig_colors.update_layout(
        title='Pixel Colors and Lake Height Over Time',
        xaxis_title='Date',
        yaxis_title='Sampling Points',
        xaxis_range=[x_start_dt, x_end_dt],
        showlegend=True
    )
    fig_colors.update_yaxes(title_text="Lake Height", secondary_y=True)
    
    # Average mg plot
    all_dates = {}
    for data_list in results_mg.values():
        for date_obj, mg in data_list:
            if date_obj in all_dates:
                all_dates[date_obj].append(mg)
            else:
                all_dates[date_obj] = [mg]
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
        xaxis_range=[x_start_dt, x_end_dt],
        showlegend=False
    )
    
    # Dual plot combining lake height and average mg
    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
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
        xaxis_range=[x_start_dt, x_end_dt],
        showlegend=False
    )
    fig_dual.update_yaxes(title_text="Lake Height", secondary_y=False)
    fig_dual.update_yaxes(title_text="mg/m³", secondary_y=True)
    
    return fig_geo, fig_dual, fig_colors, fig_mg, results_colors, results_mg, lake_data

# ========================================
# SESSION STATE HANDLING for each tab
# ========================================
if "default_results" not in st.session_state:
    st.session_state.default_results = None
if "upload_results" not in st.session_state:
    st.session_state.upload_results = None

# ----------------------------------------
# LOAD MP4 VIDEO FOR DEMO (if available)
# ----------------------------------------
mp4_files = [f for f in os.listdir(base_dir) if f.lower().endswith(".mp4")]
if mp4_files:
    mp4_file = mp4_files[0]
    with open(mp4_file, "rb") as f:
        video_bytes = f.read()
    b64_video = base64.b64encode(video_bytes).decode('utf-8')
    video_html = f"""
    <div class="responsive-video">
      <video id="myVideo" controls loop>
          <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
          Your browser does not support the video tag.
      </video>
    </div>
    <br>
    <div style="text-align:center;">
      <button onclick="document.getElementById('myVideo').play()">Play</button>
      <button onclick="document.getElementById('myVideo').pause()">Pause</button>
      <br>
      <input type="range" id="slider" min="0" max="100" value="0" step="0.1" style="width:90%;"
             oninput="var vid = document.getElementById('myVideo'); if(vid.duration){{vid.currentTime = (this.value/100)*vid.duration;}}">
    </div>
    """
else:
    video_html = "<p>Δεν βρέθηκε κανένα αρχείο MP4 στο φάκελο της εφαρμογής.</p>"

# ----------------------------------------
# Create two separate tabs with their own analysis buttons.
# ----------------------------------------
tab_names = ["Δειγματοληψία 1 (Default)", "Δειγματοληψία 2 (Upload)"]
tabs = st.tabs(tab_names)

# --------------- Tab 1: Default Sampling ---------------
with tabs[0]:
    st.header("Ανάλυση για Δειγματοληψία 1 (Default)")
    try:
        with open(sampling_kml_path, "r", encoding="utf-8") as sampling_kml_file:
            default_sampling_points = parse_sampling_kml(sampling_kml_file)
    except Exception as e:
        st.error(f"Σφάλμα στο άνοιγμα του 'sampling.kml': {str(e)}")
        default_sampling_points = []
    
    st.write("Χρησιμοποιούνται τα προεπιλεγμένα σημεία δειγματοληψίας:")
    st.write(default_sampling_points)
    
    point_names = [name for name, lon, lat in default_sampling_points]
    selected_points = st.multiselect("Επιλέξτε σημεία για εμφάνιση συγκεντρώσεων mg/m³",
                                     options=point_names, default=point_names)
    
    if st.button("Εκτέλεση Ανάλυσης (Default)"):
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
            (fig_geo, fig_dual, fig_colors, fig_mg, results_colors, results_mg, lake_data) = results
        else:
            st.error("Το αποτέλεσμα της ανάλυσης δεν έχει τη σωστή μορφή. Πατήστε ξανά το κουμπί 'Εκτέλεση Ανάλυσης (Default)'.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_geo, use_container_width=True, config={'scrollZoom': True})
        with col2:
            st.components.v1.html(video_html, height=600)
        st.plotly_chart(fig_dual, use_container_width=True, config={'scrollZoom': True})
        st.plotly_chart(fig_colors, use_container_width=True, config={'scrollZoom': True})
        st.plotly_chart(fig_mg, use_container_width=True, config={'scrollZoom': True})
        
        # PER-YEAR PLOTS
        if not np.issubdtype(lake_data['Date'].dtype, np.datetime64):
            lake_data['Date'] = pd.to_datetime(lake_data['Date'])
        years = list(range(2015, 2027))
        st.markdown("### Ετήσια διαγράμματα")
        for idx in range(0, len(years), 3):
            cols = st.columns(3)
            for j, year in enumerate(years[idx:idx+3]):
                fig_year = make_subplots(specs=[[{"secondary_y": True}]])
                for pt_idx, name in enumerate(selected_points):
                    data_list = results_colors.get(name, [])
                    filtered_data = [(d, color) for d, color in data_list if d.year == year]
                    if not filtered_data:
                        continue
                    dates_year = [d for d, c in filtered_data]
                    colors_year = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for d, c in filtered_data]
                    fig_year.add_trace(go.Scatter(
                        x=dates_year,
                        y=[pt_idx]*len(dates_year),
                        mode='markers',
                        marker=dict(color=colors_year, size=10),
                        name=name
                    ), secondary_y=False)
                lake_year = lake_data[lake_data['Date'].dt.year == year]
                if not lake_year.empty:
                    fig_year.add_trace(go.Scatter(
                        x=lake_year['Date'],
                        y=lake_year[lake_year.columns[1]],
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
        
        # NEW: Detailed mg time-series plot for a selected sampling point.
        # Here, each mg value is mapped to a color based on the provided mg-color scale.
        selected_detail_point = st.selectbox("Επιλέξτε σημείο για λεπτομερή ανάλυση mg τιμών", options=list(results_mg.keys()))
        if selected_detail_point:
            mg_data = results_mg[selected_detail_point]
            # We now ignore the original pixel colors for this plot
            if mg_data:
                mg_data_sorted = sorted(mg_data, key=lambda x: x[0])
                dates_mg = [d for d, mg in mg_data_sorted]
                mg_values = [mg for d, mg in mg_data_sorted]
                
                # Map each mg value to a color using the provided scale
                detail_colors = [mg_to_color(mg) for mg in mg_values]
                
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
                    title=f"Επεξεργασία mg τιμών για το {selected_detail_point}",
                    xaxis_title="Ημερομηνία",
                    yaxis_title="mg/m³",
                )
                st.plotly_chart(fig_detail, use_container_width=True)
            else:
                st.info("Δεν υπάρχουν δεδομένα mg για αυτό το σημείο.")

# --------------- Tab 2: Uploaded Sampling ---------------
with tabs[1]:
    st.header("Ανάλυση για Δειγματοληψία 2 (Upload)")
    uploaded_file = st.file_uploader("Ανεβάστε ένα αρχείο KML για νέα σημεία δειγματοληψίας", type="kml", key="upload_tab")
    if uploaded_file is not None:
        try:
            new_sampling_points = parse_sampling_kml(uploaded_file)
        except Exception as e:
            st.error(f"Σφάλμα στην επεξεργασία του ανεβασμένου αρχείου: {str(e)}")
            new_sampling_points = []
        st.write("Χρησιμοποιούνται τα νέα σημεία δειγματοληψίας:")
        st.write(new_sampling_points)
        
        point_names = [name for name, lon, lat in new_sampling_points]
        selected_points = st.multiselect("Επιλέξτε σημεία για εμφάνιση συγκεντρώσεων mg/m³",
                                         options=point_names, default=point_names)
        
        if st.button("Εκτέλεση Ανάλυσης (Upload)"):
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
                (fig_geo, fig_dual, fig_colors, fig_mg, results_colors, results_mg, lake_data) = results
            else:
                st.error("Το αποτέλεσμα της ανάλυσης (Upload) δεν έχει τη σωστή μορφή. Πατήστε ξανά το κουμπί.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_geo, use_container_width=True, config={'scrollZoom': True})
            with col2:
                st.components.v1.html(video_html, height=600)
            st.plotly_chart(fig_dual, use_container_width=True, config={'scrollZoom': True})
            st.plotly_chart(fig_colors, use_container_width=True, config={'scrollZoom': True})
            st.plotly_chart(fig_mg, use_container_width=True, config={'scrollZoom': True})
            
            # PER-YEAR PLOTS for uploaded sampling points
            if not np.issubdtype(lake_data['Date'].dtype, np.datetime64):
                lake_data['Date'] = pd.to_datetime(lake_data['Date'])
            years = list(range(2015, 2027))
            st.markdown("### Ετήσια διαγράμματα")
            for idx in range(0, len(years), 3):
                cols = st.columns(3)
                for j, year in enumerate(years[idx:idx+3]):
                    fig_year = make_subplots(specs=[[{"secondary_y": True}]])
                    for pt_idx, name in enumerate(selected_points):
                        data_list = results_colors.get(name, [])
                        filtered_data = [(d, color) for d, color in data_list if d.year == year]
                        if not filtered_data:
                            continue
                        dates_year = [d for d, c in filtered_data]
                        colors_year = [f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})" for d, c in filtered_data]
                        fig_year.add_trace(go.Scatter(
                            x=dates_year,
                            y=[pt_idx]*len(dates_year),
                            mode='markers',
                            marker=dict(color=colors_year, size=10),
                            name=name
                        ), secondary_y=False)
                    lake_year = lake_data[lake_data['Date'].dt.year == year]
                    if not lake_year.empty:
                        fig_year.add_trace(go.Scatter(
                            x=lake_year['Date'],
                            y=lake_year[lake_year.columns[1]],
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
            
            # NEW: Detailed mg time-series for a selected sampling point in Tab 2.
            # Each mg value is mapped to its color via the mg_to_color function.
            selected_detail_point = st.selectbox("Επιλέξτε σημείο για λεπτομερή ανάλυση mg τιμών", options=list(results_mg.keys()), key="detail_upload")
            if selected_detail_point:
                mg_data = results_mg[selected_detail_point]
                if mg_data:
                    mg_data_sorted = sorted(mg_data, key=lambda x: x[0])
                    dates_mg = [d for d, mg in mg_data_sorted]
                    mg_values = [mg for d, mg in mg_data_sorted]
                    
                    detail_colors = [mg_to_color(mg) for mg in mg_values]
                    
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
                        title=f"Επεξεργασία mg τιμών για το {selected_detail_point}",
                        xaxis_title="Ημερομηνία",
                        yaxis_title="mg/m³",
                    )
                    st.plotly_chart(fig_detail, use_container_width=True)
                else:
                    st.info("Δεν υπάρχουν δεδομένα mg για αυτό το σημείο.")
    else:
        st.info("Παρακαλώ ανεβάστε ένα αρχείο KML για τα νέα σημεία δειγματοληψίας.")

# ----------------------------------------
# Global Info (if any analysis was run)
# ----------------------------------------
if st.session_state.default_results or st.session_state.upload_results:
    st.info("Τα διαγράμματα παραμένουν ορατά μέχρι να πατήσετε ξανά το κουμπί 'Εκτέλεση Ανάλυσης'.")