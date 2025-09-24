import base64
import os
from math import asin, cos, radians, sin, sqrt

import geopandas as gpd
import numpy as np
import openeo
import rasterio
from PIL import Image
from pyproj import Proj, Transformer
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box
from shapely.ops import transform as geom_transform


def get_download_link(filepath):
    """Generate a download link allowing the direct download of a file from Streamlit."""
    try:
        with open(filepath, "rb") as f:
            # Read the file data
            bytes_data = f.read()
        # Encode file data into base64
        b64 = base64.b64encode(bytes_data).decode()
        # Create and return the HTML link
        href = f'<a href="data:file/octet-stream;base64,{b64}" download="{os.path.basename(filepath)}">Download {os.path.basename(filepath)}</a>'
        return href
    except Exception as e:
        return f"Error creating download link: {e}"


def convert_to_rgb(image_array, output_path=None):
    # Ensure the array is the expected shape and type
    if (
        image_array.dtype != np.int16
        or image_array.ndim != 3
        or image_array.shape[2] < 3
    ):
        raise ValueError("Expected a 3D int16 array with at least 3 bands")

    # Normalize the bands to 0-255 (8-bit format)
    def normalize(array):
        # Adjust these percentiles based on your specific data characteristics
        percentiles = np.percentile(array, (2, 98))
        array_clipped = np.clip(array, percentiles[0], percentiles[1])
        array_normalized = (array_clipped - array_clipped.min()) / (
            array_clipped.max() - array_clipped.min()
        )
        return (array_normalized * 255).astype(np.uint8)

    # Assuming the first three bands are used for RGB
    red = image_array[:, :, 0]  # First band
    green = image_array[:, :, 1]  # Second band
    blue = image_array[:, :, 2]  # Third band

    # Normalize each band
    red_norm = normalize(red)
    green_norm = normalize(green)
    blue_norm = normalize(blue)

    # Stack bands into an RGB image
    rgb = np.dstack((red_norm, green_norm, blue_norm))

    # Optionally save the result
    if output_path:
        img = Image.fromarray(rgb, "RGB")
        img.save(output_path)

    return rgb


def reproject_bbox_to_utm(lon_min, lon_max, lat_min, lat_max):
    # Find the UTM zone for the center of the bounding box and setup UTM projection
    utm_zone = int((lon_min + lon_max) / 2 / 6) + 31
    proj_utm = Proj(
        f"+proj=utm +zone={utm_zone} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    )
    proj_latlon = Proj(proj="latlong", datum="WGS84")

    # Create polygon
    geo = box(lon_min, lat_min, lon_max, lat_max)

    # Setup transformer
    transformer = Transformer.from_proj(proj_latlon, proj_utm)

    # Reproject the polygon to UTM
    geo_utm = geom_transform(lambda x, y: transformer.transform(x, y), geo)
    return geo_utm, proj_utm, proj_latlon


def crop_tiff_to_bbox(tiff_path, output_path, bbox):
    """
    Crop a TIFF image to the specified bounding box.

    Args:
    tiff_path (str): Path to the input TIFF file.
    output_path (str): Path where the cropped TIFF will be saved.
    bbox (list): Bounding box with format [[lon_min, lon_max], [lat_min, lat_max]].

    Returns:
    None: Saves the cropped image to the specified path.
    """
    # Convert the bounding box to a GeoDataFrame
    # Unpack the bbox list to minx, miny, maxx, maxy
    print(tiff_path)

    print_bounds_and_bbox(tiff_path, bbox)
    minx, maxx = bbox[0]
    miny, maxy = bbox[1]

    # Convert the bounding box to a shapely polygon
    bbox_polygon = box(minx, miny, maxx, maxy)
    print(bbox_polygon)
    geo_df = gpd.GeoDataFrame({"geometry": [bbox_polygon]}, crs="EPSG:4326")

    # Open the source TIFF file
    with rasterio.open(tiff_path) as src:
        # Reproject GeoDataFrame to the same CRS as the raster
        geo_df = geo_df.to_crs(src.crs.data)

        try:
            # Perform cropping
            out_image, out_transform = mask(src, geo_df.geometry, crop=True)
            out_meta = src.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                }
            )

            # Write the cropped raster
            with rasterio.open(output_path, "w", **out_meta) as dst:
                dst.write(out_image)
            print("Cropping successful, saved to:", output_path)
        except ValueError as e:
            print("Error:", e)
            print("No overlap between the bounding box and the raster.")


def extend_bbox_for_scaling(bbox, scale_factor=3, resolution=10):
    lon_min, lon_max = bbox[0]
    lat_min, lat_max = bbox[1]

    # Reproject bbox to UTM
    bbox_utm, proj_utm, proj_latlon = reproject_bbox_to_utm(
        lon_min, lon_max, lat_min, lat_max
    )

    # Get dimensions in meters from the UTM-projected bbox
    minx, miny, maxx, maxy = bbox_utm.bounds
    width_meters = maxx - minx
    height_meters = maxy - miny

    # Calculate expected dimensions in pixels after scaling
    expected_width_pixels = (width_meters / resolution) * scale_factor
    expected_height_pixels = (height_meters / resolution) * scale_factor
    # print("Expected width in pixels " + str(expected_width_pixels))
    # print("Expected width in meters " + str(width_meters))
    # print("Expected height in meters " + str(height_meters))
    # print("Expected width in resolution " + str(resolution))
    # Check if adjustment is necessary
    if expected_width_pixels < 512 or expected_height_pixels < 512:
        # Calculate the required dimensions in meters
        required_width_meters = 512 / scale_factor * resolution
        required_height_meters = 512 / scale_factor * resolution

        # Adjust dimensions
        new_minx = minx - (required_width_meters - width_meters) / 2
        new_maxx = maxx + (required_width_meters - width_meters) / 2
        new_miny = miny - (required_height_meters - height_meters) / 2
        new_maxy = maxy + (required_height_meters - height_meters) / 2

        # Create adjusted bbox in UTM and transform back to lat/lon
        adjusted_bbox_utm = box(new_minx, new_miny, new_maxx, new_maxy)
        transformer_back = Transformer.from_proj(proj_utm, proj_latlon)
        adjusted_bbox_latlon = geom_transform(
            lambda x, y: transformer_back.transform(x, y), adjusted_bbox_utm
        )
        bbox = polygon_to_bbox(adjusted_bbox_latlon)

        message = f"BBOX extended to meet minimum size requirement of 512x512 pixels after scaling: Initial dimensions {int(expected_width_pixels)}x{int(expected_height_pixels)} pixels."
    else:
        message = f"The processed image will be approximately {int(expected_width_pixels*4)}x{int(expected_height_pixels*4)} pixels, which exceeds the maximum allowed size of 512x512 pixels."
    # If no adjustment is needed
    return bbox, message


def polygon_to_bbox(polygon):
    """
    Convert a Shapely polygon into a bounding box format [[lon_min, lon_max], [lat_min, lat_max]].
    """
    minx, miny, maxx, maxy = polygon.bounds
    return [[minx, maxx], [miny, maxy]]


# Function to fetch Sentinel-2 patches from Copernicus Open Access Hub
def fetch_sentinel2_patches(lonlat, start_date, end_date, output_dir):
    # Connect to the Copernicus Open Access Hub
    conn = openeo.connect("openeo.dataspace.copernicus.eu")
    conn.authenticate_oidc()

    # Define the spatial and temporal extent for the Sentinel-2 data using CRS information
    spatial_extent = {
        "west": lonlat[0][0],
        "south": lonlat[1][0],
        "east": lonlat[0][1],
        "north": lonlat[1][1],
        "crs": "EPSG:4326",
    }
    sentinel2 = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=[start_date, end_date],
        bands=["B04", "B03", "B02", "B08"],
        max_cloud_cover=10,
    )

    # Download the data
    sentinel2.download(output_dir + "/Sentinel2_raw_patch.tif")
    reproject_to_4326(
        output_dir + "/Sentinel2_raw_patch.tif", output_dir + "/Sentinel2_raw_patch.tif"
    )
    return output_dir + "/Sentinel2_raw_patch.tif"


def stack_bands(band_files, output_path):
    with rasterio.open(band_files[0]) as src0:
        meta = src0.meta.copy()
    meta.update(count=len(band_files))
    # print(meta)
    with rasterio.open(output_path, "w", **meta) as dst:
        for id, band_file in enumerate(band_files, start=1):
            with rasterio.open(band_file) as src:
                dst.write(src.read(1), id)
    return output_path


def print_bounds_and_bbox(raster_path, bbox):
    with rasterio.open(raster_path) as src:
        print("Raster bounds:", src.bounds)
        show(src, title="Raster")
        print("Bounding box:", bbox)


def process_and_save_images(
    original_path, processed_image, model_name, output_dir, bbox
):
    """Process images, calculate NDVI, and save outputs."""
    results = []

    processed_path = f"{output_dir}/{model_name}_output.tif"
    ndvi_path = f"{output_dir}/{model_name}_ndvi.tif"

    # Crop and save the processed image
    save_image(processed_image, processed_path)
    crop_tiff_to_bbox(processed_path, processed_path, bbox)

    # Calculate and save NDVI for processed image
    ndvi = calculate_ndvi(processed_path)
    save_ndvi(ndvi, processed_path.replace(".tif", "_ndvi.tif"))

    results.append((processed_path, ndvi_path))

    # Process the original image
    original_ndvi_path = f"{output_dir}/original_ndvi.tif"

    # Crop and save the original image
    crop_tiff_to_bbox(original_path, original_path, bbox)

    # Calculate and save NDVI for the original image
    original_ndvi = calculate_ndvi(original_path)
    save_ndvi(original_ndvi, original_ndvi_path)

    results.append((original_path, original_ndvi_path))
    return results


def calculate_ndvi(image_path):
    """Calculate NDVI from a single-band image path."""
    with rasterio.open(image_path) as src:
        red = src.read(3)  # Assuming red is band 3
        nir = src.read(4)  # Assuming NIR is band 4
        ndvi = (nir - red) / (nir + red + 0.00001)  # Avoid division by zero

    return ndvi


def save_ndvi(ndvi, output_path):
    """Save NDVI data to a file."""
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=ndvi.shape[0],
        width=ndvi.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
    ) as dst:
        dst.write(ndvi, 1)


def save_image(data, filename, profile=None):
    """Save an image using a default or existing profile."""
    # Determine the number of bands based on the shape of the data
    if data.ndim == 2:
        # Single band image
        height, width = data.shape
        count = 1
    elif data.ndim == 3:
        # Multi-band image
        height, width, count = data.shape
    else:
        raise ValueError("Unsupported image data shape")

    # Use a provided profile or define a default
    if profile is None:
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": count,
            "dtype": data.dtype,
            "crs": "EPSG:4326",
            "transform": rasterio.transform.from_origin(
                -180, 90, 0.1, 0.1
            ),  # Example transform
        }

    with rasterio.open(filename, "w", **profile) as dst:
        if count == 1:
            dst.write(data, 1)  # Write single band data
        else:
            for i in range(count):
                dst.write(data[:, :, i], i + 1)  # Write each band


def reproject_to_4326(input_path, output_path):
    """
    Reproject a raster image to EPSG:4326.

    Args:
    input_path (str): Path to the input raster.
    output_path (str): Path to save the reprojected raster.
    """
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": "EPSG:4326",
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):  # Loop through each band
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.nearest,
                )

        print(f"Reprojected image saved to {output_path}")


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in meters between two points
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371000  # Radius of Earth in meters
    return c * r


def calculate_image_size(bbox, resolution=10):
    """
    Calculate the estimated size of the image based on the bounding box.
    bbox should be [[lon_min, lon_max], [lat_min, lat_max]]
    resolution is the size of each pixel in meters.
    """
    lon_min, lon_max = bbox[0]
    lat_min, lat_max = bbox[1]

    width_meters = haversine(lon_min, lat_min, lon_max, lat_min)
    height_meters = haversine(lon_min, lat_min, lon_min, lat_max)

    # Calculate width and height in pixels
    width_pixels = width_meters / resolution
    height_pixels = height_meters / resolution

    return int(width_pixels), int(height_pixels)


def load_tiff_as_array(file_path):
    # Open the TIFF file
    with rasterio.open(file_path) as src:
        # Read all bands at once into a numpy array
        # src.read() reads all bands in the order they are in the file
        img_array = src.read()

        # Transpose the array dimensions from (bands, height, width) to (height, width, bands)
        img_array = np.transpose(img_array, (1, 2, 0))

        # Optional: print image properties
        # print(f"Loaded image shape: {img_array.shape}")
        # print(f"Coordinate reference system: {src.crs}")
        # print(f"Transform: {src.transform}")

        return img_array, src.crs, src.transform
