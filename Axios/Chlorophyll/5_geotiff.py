import json
import os
from rasterio.transform import from_bounds
from PIL import Image
import rasterio
import numpy as np

# Determine the current directory
current_dir = os.getcwd()

# Load lake coordinates from the text file
coords_file_path = os.path.join(current_dir, "lake coordinates.txt")
with open(coords_file_path, "r") as f:
    data = json.load(f)

# Assuming the coordinates are in the first element of the "coordinates" array
polygon_coords = data["coordinates"][0]
# Separate the longitude and latitude values
longitudes = [point[0] for point in polygon_coords]
latitudes = [point[1] for point in polygon_coords]

# Compute bounds
min_x = min(longitudes)
max_x = max(longitudes)
min_y = min(latitudes)
max_y = max(latitudes)

print("Calculated bounds:")
print("min_x:", min_x, "min_y:", min_y)
print("max_x:", max_x, "max_y:", max_y)

# Define input and output folders
input_folder = os.path.join(current_dir, "jpgs")
output_folder = os.path.join(current_dir, "GeoTIFFs")
os.makedirs(output_folder, exist_ok=True)

# Process each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".jpg"):
        img_path = os.path.join(input_folder, filename)
        tiff_path = os.path.join(output_folder, filename.replace('.jpg', '.tif'))
        
        # Open the image and convert to numpy array
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img_array = np.array(img)
        
        # Get image dimensions
        height, width = img_array.shape[:2]
        
        # Create transform for georeferencing using the calculated bounds
        transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
        
        # Write GeoTIFF with georeferencing
        with rasterio.open(
            tiff_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,  # RGB has 3 bands
            dtype=img_array.dtype,
            crs='EPSG:4326',  # WGS84
            transform=transform
        ) as dst:
            dst.write(img_array[:, :, 0], 1)  # Red band
            dst.write(img_array[:, :, 1], 2)  # Green band
            dst.write(img_array[:, :, 2], 3)  # Blue band
        
        print(f"GeoTIFF created: {tiff_path}")

print("GeoTIFF creation process completed.")
