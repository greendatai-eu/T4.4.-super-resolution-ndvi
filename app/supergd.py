import datetime
import os
import shutil
import time

import folium
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

from configLogging import logger
from models import swin2_mose_model
from utils import (
    calculate_image_size,
    extend_bbox_for_scaling,
    fetch_sentinel2_patches,
    load_tiff_as_array,
)

# --- Streamlit setup
st.set_page_config(layout="wide")

INPUT_DIR = "./input"
OUTPUT_DIR = "./output"

# --- Session state initialization
if "bounding_box_selected" not in st.session_state:
    st.session_state.bounding_box_selected = False
if "lonlat" not in st.session_state:
    st.session_state.lonlat = None
if "image_path" not in st.session_state:
    st.session_state.image_path = ""
if "start_date" not in st.session_state or "end_date" not in st.session_state:
    st.session_state["start_date"] = datetime.date.today() - datetime.timedelta(days=30)
    st.session_state["end_date"] = datetime.date.today()

# --- Sidebar
st.sidebar.title("Super Resolution\nModel: Swin2Mose")

# --- Map selection
st.header("Select a Bounding Box on the Map")
m = folium.Map(location=[46.0569, 14.5058], zoom_start=10)  # Ljubljana, Slovenia
draw = folium.plugins.Draw(export=True)
draw.add_to(m)
map_data = st_folium(m, width=1600, height=500)

if map_data and "all_drawings" in map_data and map_data["all_drawings"]:
    last_drawing = map_data["all_drawings"][-1]
    if last_drawing["geometry"]["type"] == "Polygon":
        coordinates = last_drawing["geometry"]["coordinates"][0]
        st.session_state.lonlat = [
            [
                min(coord[0] for coord in coordinates),
                max(coord[0] for coord in coordinates),
            ],
            [
                min(coord[1] for coord in coordinates),
                max(coord[1] for coord in coordinates),
            ],
        ]
        st.write(f"Selected Bounding Box Coordinates: {st.session_state.lonlat}")
        st.session_state.bounding_box_selected = True

# --- If bounding box chosen, fetch patch and process
if st.session_state.bounding_box_selected:
    start_date = st.sidebar.date_input(
        "Start Date", value=st.session_state["start_date"]
    )
    end_date = st.sidebar.date_input("End Date", value=st.session_state["end_date"])
    st.session_state["start_date"] = start_date
    st.session_state["end_date"] = end_date

    if st.button("Fetch and Process Sentinel-2 Patches"):
        scale_factor = 3
        estimated_width, estimated_height = calculate_image_size(
            st.session_state.lonlat
        )
        original_bbox = st.session_state.lonlat.copy()

        extended_bbox, message = extend_bbox_for_scaling(
            st.session_state.lonlat, scale_factor, 10
        )
        st.write(f"Original Bounding Box: {original_bbox}")
        st.write(f"Extended Bounding Box: {extended_bbox}")
        st.write(message)

        if estimated_width > 513 or estimated_height > 513:
            st.warning("Please adjust the bounding box.")
        else:
            if extended_bbox:
                patch_path = fetch_sentinel2_patches(
                    extended_bbox,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    INPUT_DIR,
                )
            else:
                patch_path = fetch_sentinel2_patches(
                    original_bbox,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    INPUT_DIR,
                )

            if patch_path:
                image_array, crs, transform = load_tiff_as_array(patch_path)
                st.write(
                    f"Sentinel-2 patches retrieved for {st.session_state.lonlat} from {start_date} to {end_date}."
                )

                # --- Save input patch in input_dir with unique name
                base_name = os.path.basename(patch_path)
                name, ext = os.path.splitext(base_name)
                unique_name = f"{int(time.time())}{ext}"
                stored_input_path = os.path.join(INPUT_DIR, unique_name)
                shutil.copy(patch_path, stored_input_path)
                st.success(f"Input file saved as {stored_input_path}")
                st.session_state.image_path = stored_input_path

                # Show original
                logger.info(
                    "basename: %s", os.path.basename(st.session_state.image_path)
                )
                logger.info("dirname: %s", os.path.dirname(st.session_state.image_path))

                # --- Run Swin2Mose
                sr_results = swin2_mose_model(
                    [os.path.basename(st.session_state.image_path)],
                    os.path.dirname(st.session_state.image_path),
                    OUTPUT_DIR,
                )

                if sr_results["status"] == "success":
                    logger.info("STATUS SUCCESS")
                    for res in sr_results["results"]:
                        if res["status"] == "success":
                            comp_path = os.path.join(
                                OUTPUT_DIR, f"comparison_{res['filename']}"
                            )
                            logger.debug("COMP_PATH: %s", comp_path)

                            if os.path.exists(comp_path):
                                comp_img = Image.open(comp_path)
                                st.image(
                                    comp_img,
                                    caption="Original vs Super-resolved Comparison",
                                    width="stretch",
                                )
                else:
                    st.error(f"Model failed: {sr_results['message']}")

                # --- Clean up: delete specific file
                file_to_delete = os.path.join(INPUT_DIR, "Sentinel2_raw_patch.tif")
                if os.path.exists(file_to_delete):
                    try:
                        os.remove(file_to_delete)
                    except Exception as e:
                        logger.error(f"Failed to delete {file_to_delete}: {e}")
