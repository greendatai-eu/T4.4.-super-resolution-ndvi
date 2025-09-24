![GreenDataI Logo](./assets/greendatailogo.png)

# Super-Resolution NDVI Application

This repository contains the complete Super-Resolution NDVI application, including both the source code and deployment configuration. Users can clone this repository and deploy the application locally using Docker Compose.

## About the Application

The application provides an end-to-end workflow for generating upscaled Sentinel-2 imagery and corresponding NDVI outputs using the state-of-the-art Swin2-MoSE super-resolution model. Users must have a Copernicus Data Space Ecosystem account (free registration) to access Sentinel-2 satellite data.

### Key Features

- **Interactive User Interface**: Built with Python using Streamlit-Folium for web UI and OpenStreetMap for map-based bounding box selection
- **Complete Processing Pipeline**:
  - User selects a bounding box on the interactive map
  - Sentinel-2 patch is automatically fetched from Copernicus Data Space
  - Image is super-resolved using the Swin2-MoSE model
  - Three outputs are generated:
    1. Upscaled image
    2. Side-by-side comparison (original vs. upscaled)
    3. NDVI image derived from super-resolved Red (B4) and NIR (B8) bands

### Technical Implementation

- **Core Libraries**: PyTorch and timm for model inference, NumPy for array processing
- **Geospatial Processing**: tifffile and rasterio for GeoTIFF I/O operations
- **Visualization**: Pillow and matplotlib for image processing and display
- **Band Preservation**: Maintains proper Sentinel-2 channel ordering (B2, B3, B4, B8) for accurate NDVI calculation
- **Organized Structure**: Reproducible folder structure with `input/` for source tiles and `output/` for processed results

The modular architecture encapsulates data acquisition, super-resolution inference, and vegetation index computation into a repeatable, user-friendly workflow.

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Git LFS enabled (for model weights)

### Deployment

1. Clone the repository:

```bash
git clone https://github.com/greendatai-eu/T4.4.-super-resolution-ndvi.git
cd T4.4.-super-resolution-ndvi
```

2. Start the application:

```bash
docker-compose up --build
```

3. Access the application at `http://localhost:8501`

The application will automatically build the Docker image locally and start the Streamlit interface.
