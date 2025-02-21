import streamlit as st
import geopandas as gpd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from affine import Affine
from pysheds.grid import Grid
import tempfile

# Page configuration
st.set_page_config(page_title="Hydrological Analysis", layout="wide")
st.title("Hydrological Analysis from KML (Using PySheds)")
st.markdown("""
This application creates a DEM from contour lines contained in a KML file,
performs hydrological processing (sink filling, flow direction, accumulation),
extracts the stream network, and delineates the watershed draining to a lake.
""")

# Sidebar: file upload and parameters
uploaded_file = st.sidebar.file_uploader("Upload your KML file", type=["kml"])
grid_res = st.sidebar.number_input("Grid resolution (points per dimension)", min_value=100, max_value=1000, value=500, step=50)
threshold = st.sidebar.number_input("Flow accumulation threshold", min_value=10, max_value=1000, value=100, step=10)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp_file:
        tmp_file.write(uploaded_file.read())
        kml_path = tmp_file.name

    try:
        gdf = gpd.read_file(kml_path, driver='KML')
    except Exception as e:
        st.error(f"Error reading KML file: {e}")
        st.stop()

    # Separate contour lines and lake polygons
    contours = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
    lakes = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    
    if lakes.empty:
        st.error("No lake polygon found in the KML.")
        st.stop()

    # Extract contour points and elevations
    points = []
    values = []
    for idx, row in contours.iterrows():
        geom = row.geometry
        # Handle both MultiLineString and LineString
        if geom.type == 'MultiLineString':
            lines = list(geom)
        else:
            lines = [geom]
        for line in lines:
            for coord in line.coords:
                # Prefer 3D coordinates; if not available, try an attribute
                if len(coord) >= 3:
                    x, y, z = coord[:3]
                else:
                    x, y = coord
                    z = row.get('elevation', None)
                    if z is None:
                        continue
                points.append((x, y))
                values.append(z)
    points = np.array(points)
    values = np.array(values)
    
    if len(points) == 0:
        st.error("No elevation data found in the contours.")
        st.stop()

    st.success("Contour data extracted successfully!")
    
    # Create a grid for DEM interpolation
    minx, miny = points.min(axis=0)
    maxx, maxy = points.max(axis=0)
    grid_x, grid_y = np.mgrid[minx:maxx:complex(grid_res), miny:maxy:complex(grid_res)]
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

    # Create an affine transformation for the DEM grid
    dx = (maxx - minx) / grid_res
    dy = (maxy - miny) / grid_res
    affine_transform = Affine(dx, 0, minx, 0, -dy, maxy)

    # Initialize a pysheds Grid and add the DEM
    grid_obj = Grid()
    grid_obj.add_gridded_data(grid_z, data_name='dem', affine=affine_transform, nodata=-9999)

    # Fill depressions in the DEM
    filled_dem = grid_obj.fill_depressions(data='dem', out_name='filled_dem')

    # Compute flow direction using the D8 algorithm (dirmap 'D8' is the default)
    grid_obj.flowdir(data='filled_dem', out_name='flowdir')

    # Compute flow accumulation
    grid_obj.accumulation(data='flowdir', out_name='acc')

    # Extract streams by applying the threshold
    acc_array = grid_obj.view('acc')
    streams = np.where(acc_array > threshold, 1, 0)

    # Use the lake polygon's centroid as the pour point for watershed delineation
    lake = lakes.iloc[0]
    centroid = lake.geometry.centroid
    pour_point_x = centroid.x
    pour_point_y = centroid.y

    # Convert the pour point coordinates to grid indices using the inverse affine transform
    col, row = ~affine_transform * (pour_point_x, pour_point_y)
    col = int(col)
    row = int(row)

    # Delineate the watershed (catchment) draining to the pour point
    catchment = grid_obj.catchment(data='flowdir', x=pour_point_x, y=pour_point_y,
                                   dirmap='D8', recursionlimit=25000, out_name='watershed')

    st.success("Hydrological analysis completed!")

    # Create tabs for visualizing each result
    tab1, tab2, tab3, tab4 = st.tabs(["Interpolated DEM", "Filled DEM", "Stream Network", "Watershed"])

    with tab1:
        st.subheader("Interpolated DEM")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(grid_z, cmap='terrain')
        ax.set_title("Interpolated DEM")
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

    with tab2:
        st.subheader("Filled DEM")
        filled = grid_obj.view('filled_dem')
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(filled, cmap='terrain')
        ax.set_title("Filled DEM")
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

    with tab3:
        st.subheader("Stream Network")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(streams, cmap='Blues')
        ax.set_title("Extracted Streams (Threshold = {})".format(threshold))
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

    with tab4:
        st.subheader("Watershed")
        ws = grid_obj.view('watershed')
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(ws, cmap='viridis')
        ax.set_title("Watershed Delineation")
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
else:
    st.info("Please upload a KML file from the sidebar to begin the analysis.")
