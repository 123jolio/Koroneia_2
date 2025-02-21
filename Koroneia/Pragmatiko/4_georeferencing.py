import os
import json

# Get the current directory
current_dir = os.getcwd()

# Path to the folder containing images (assumed to be in a folder named "jpgs" in the current directory)
folder_path = os.path.join(current_dir, "jpgs")

# Path to the lake coordinates file in the current directory
coordinates_file_path = os.path.join(current_dir, "lake coordinates.txt")

# Read and parse the GeoJSON content from the file
with open(coordinates_file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract the coordinates (assuming standard GeoJSON Polygon format)
ring = data["coordinates"][0]

# Separate longitudes and latitudes (GeoJSON: [longitude, latitude])
lons = [pt[0] for pt in ring]
lats = [pt[1] for pt in ring]

# Calculate the bounding box from the polygon
polygon_coords = {
    "north": max(lats),  # highest latitude
    "south": min(lats),  # lowest latitude
    "east": max(lons),   # highest longitude
    "west": min(lons)    # lowest longitude
}

print("Polygon Coordinates:", polygon_coords)

# Create the header and footer for the KML file
kml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>All Images Overlay</name>
'''

kml_footer = '''
  </Document>
</kml>
'''

# Build the GroundOverlay elements for each JPG in the folder
ground_overlays = ""
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".jpg"):
        image_name = os.path.splitext(filename)[0]
        ground_overlay = f'''
    <GroundOverlay>
      <name>{image_name}</name>
      <Icon>
        <href>{filename}</href>
      </Icon>
      <LatLonBox>
        <north>{polygon_coords["north"]}</north>
        <south>{polygon_coords["south"]}</south>
        <east>{polygon_coords["east"]}</east>
        <west>{polygon_coords["west"]}</west>
        <rotation>0</rotation>
      </LatLonBox>
    </GroundOverlay>
'''
        ground_overlays += ground_overlay

# Combine header, overlays, and footer to form the complete KML content
kml_content = kml_header + ground_overlays + kml_footer

# Save the KML file to the same folder as the images
kml_file_path = os.path.join(folder_path, "All_Images_Overlay.kml")
with open(kml_file_path, 'w', encoding="utf-8") as file:
    file.write(kml_content)

print(f"KML file created: {kml_file_path}")
