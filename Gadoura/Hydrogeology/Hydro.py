import streamlit as st
import geopandas as gpd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import richdem as rd
import tempfile

# Set up the page configuration
st.set_page_config(
    page_title="Hydrological Analysis from KML",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a polished look
st.markdown("""
    <style>
    .main {background-color: #f8f9fa; }
    .sidebar .sidebar-content { background-color: #e9ecef; }
    h1 { color: #343a40; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar - Upload and Parameters
st.sidebar.header("Input & Parameters")
uploaded_file = st.sidebar.file_uploader("Upload your KML file", type=["kml"])

st.sidebar.markdown("---")
st.sidebar.subheader("DEM & Stream Extraction Settings")
grid_res = st.sidebar.number_input("Grid resolution (points per dimension)", min_value=100, max_value=1000, value=500, step=50)
threshold = st.sidebar.number_input("Flow accumulation threshold", min_value=10, max_value=1000, value=100, step=10)

st.sidebar.markdown("---")
st.sidebar.markdown("This tool creates a DEM from contour lines in the KML, computes flow direction/accumulation, extracts streams, and delineates the lake watershed based on the lake polygon.")

# Main Title and Instructions
st.title("Hydrological Analysis from KML")
st.markdown("""
This application performs a hydrological analysis on your uploaded KML file containing contour lines and a lake polygon.
Follow the steps below:
1. Upload a KML file containing 3D contour lines and a lake polygon.
2. Adjust the grid resolution and flow accumulation threshold if needed.
3. The app will create a DEM, compute flow direction/accumulation, extract streams, and delineate the watershed that drains into the lake.
""")

if uploaded_file is not None:
    with st.spinner("Reading and processing KML file..."):
        # Write the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp_file:
            tmp_file.write(uploaded_file.read())
            kml_path = tmp_file.name

        try:
            # Read the KML file
            gdf = gpd.read_file(kml_path, driver='KML')
        except Exception as e:
            st.error("Error reading KML file: " + str(e))
            st.stop()

        # Separate features: contours and lake polygons
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
            # Handle MultiLineString and LineString
            if geom.type == 'MultiLineString':
                lines = list(geom)
            else:
                lines = [geom]
            for line in lines:
                for coord in line.coords:
                    # Use 3D coordinates if available
                    if len(coord) == 3:
                        x, y, z = coord
                    else:
                        x, y = coord
                        # Try to get elevation from an attribute, otherwise skip
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

    st.success("KML file successfully processed!")

    # Interpolate the DEM
    with st.spinner("Generating DEM..."):
        minx, miny = points.min(axis=0)
        maxx, maxy = points.max(axis=0)
        grid_x, grid_y = np.mgrid[minx:maxx:complex(grid_res), miny:maxy:complex(grid_res)]
        grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')

        # Create DEM for hydrological analysis with RichDEM
        dem = rd.rdarray(grid_z, no_data=-9999)
        dem.geotransform = [minx, (maxx - minx) / grid_res, 0, maxy, 0, -(maxy - miny) / grid_res]

    # Hydrological Analysis
    with st.spinner("Performing hydrological analysis..."):
        dem_filled = rd.FillDepressions(dem, in_place=False)
        flow_directions = rd.FlowDirection(dem_filled, method='D8')
        flow_acc = rd.FlowAccumulation(flow_directions, method='D8')
        streams = (flow_acc > threshold).astype(np.int16)

        # Use lake polygon's centroid as pour point for watershed delineation
        lake = lakes.iloc[0]
        pour_point_geom = lake.geometry.centroid
        pour_point_x = pour_point_geom.x
        pour_point_y = pour_point_geom.y

        # Convert pour point to grid coordinates
        col_idx = int((pour_point_x - minx) / ((maxx - minx) / grid_res))
        row_idx = int((maxy - pour_point_y) / ((maxy - miny) / grid_res))
        pour_point = (row_idx, col_idx)
        watershed = rd.Watershed(flow_directions, pour_point, dirmap='D8')

    st.success("Hydrological analysis completed!")

    # Create tabbed views for visualization
    tab1, tab2, tab3, tab4 = st.tabs(["DEM (Interpolated)", "DEM (Filled)", "Streams", "Watershed"])

    with tab1:
        st.subheader("DEM (Interpolated)")
        fig1, ax1 = plt.subplots(figsize=(6, 5))
        im1 = ax1.imshow(dem, cmap='terrain')
        ax1.set_title('Interpolated DEM')
        plt.colorbar(im1, ax=ax1)
        st.pyplot(fig1)

    with tab2:
        st.subheader("DEM (Filled)")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        im2 = ax2.imshow(dem_filled, cmap='terrain')
        ax2.set_title('Filled DEM')
        plt.colorbar(im2, ax=ax2)
        st.pyplot(fig2)

    with tab3:
        st.subheader("Extracted Streams")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        im3 = ax3.imshow(streams, cmap='Blues')
        ax3.set_title('Stream Network (Threshold = {})'.format(threshold))
        plt.colorbar(im3, ax=ax3)
        st.pyplot(fig3)

    with tab4:
        st.subheader("Lake Watershed")
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        im4 = ax4.imshow(watershed, cmap='viridis')
        ax4.set_title('Watershed Delineation')
        plt.colorbar(im4, ax=ax4)
        st.pyplot(fig4)

    st.markdown("### Analysis Complete")
    st.info("Review the tabs above to see the DEM, filled DEM, stream network, and watershed delineation.")
else:
    st.info("Please upload a KML file from the sidebar to begin the analysis.")
