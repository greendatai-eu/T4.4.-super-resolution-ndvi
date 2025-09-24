import os

import numpy as np
import tifffile as tiff
import torch

from configLogging import logger
from strechminmax8bit import (
    add_black_for_no_data,
    plot_images_side_by_side,
    stretch_to_min_max_8bit,
)
from superIX.swin2_mose.utils import load_config, load_swin2_mose, run_swin2_mose


def swin2_mose_model(filenames_to_process, input_image_path, output_dir):
    logger.info("Running the SWIN 2 Mose model...")
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device set to: {device}")

    if not filenames_to_process:
        return {"status": "info", "message": "No files were provided for processing."}

    # Paths for model and config
    path = "./superIX/swin2_mose/weights/config-70.yml"
    model_weights_path = "./superIX/swin2_mose/weights/model-70.pt"
    logger.info(f"Model weights path: {model_weights_path}, Config path: {path}")

    # Load config once
    try:
        cfg = load_config(path)
        logger.info("Configuration loaded successfully.")
    except Exception as e:
        logger.error("Error loading config file: %s", e)
        return {"status": "error", "message": f"Failed to load config file: {str(e)}"}

    # Load model once
    try:
        model = load_swin2_mose(model_weights_path, cfg)
        model.to(device)
        model.eval()
        logger.info("Model loaded and set to eval mode.")
    except Exception as e:
        logger.error("Error loading model: %s", e)
        return {
            "status": "error",
            "message": f"Failed to load SWIN 2 Mose model: {str(e)}",
        }

    for filename in filenames_to_process:
        try:
            input_image_path = os.path.join(input_image_path, filename)
            output_filepath = os.path.join(output_dir, f"swin_processed_{filename}")

            # Skip if file already exists in output folder
            if os.path.exists(output_filepath):
                results.append(
                    {
                        "filename": filename,
                        "status": "found",
                        "message": f"Processed file '{output_filepath}' already exists.",
                    }
                )
                continue

            # Step 1: Load the image as a NumPy array
            image_np = tiff.imread(input_image_path)
            image_original_stretched = stretch_to_min_max_8bit(image_np)

            # Step 2: Transpose the image to (C, H, W) as the first step
            image_np_transposed = np.transpose(image_np, (2, 0, 1))

            # Step 3: Apply the add_black_for_no_data function to the transposed array
            image_np_processed = add_black_for_no_data(
                image_np_transposed, image_np_transposed, no_data_value=-32768
            )

            # Step 4: Run the model with the prepared NumPy array.
            # The run_swin2_mose function will handle the conversion to a tensor.
            swin_model_results = run_swin2_mose(
                model, image_np_processed, image_np_processed, device=device
            )

            # Step 5: Extract and transpose output
            super_resolved_img_array = swin_model_results["sr"]

            # Transpose the output back to (H, W, C) for saving
            image_np_final = np.transpose(super_resolved_img_array, (1, 2, 0))

            # Calculate NDVI from super-resolved image
            # Bands are in order: B2(Blue), B3(Green), B4(Red), B8(NIR)
            red_band = image_np_final[:, :, 2]  # B4 (Red) - index 2
            nir_band = image_np_final[:, :, 3]  # B8 (NIR) - index 3

            # Calculate NDVI: (NIR - Red) / (NIR + Red)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-8
            ndvi = (nir_band.astype(np.float64) - red_band.astype(np.float64)) / (
                nir_band.astype(np.float64) + red_band.astype(np.float64) + epsilon
            )

            # Save NDVI as separate file
            ndvi_filepath = os.path.join(output_dir, f"ndvi_{filename}")
            tiff.imwrite(
                ndvi_filepath,
                ndvi.astype(np.float32),
                photometric="minisblack",
            )

            logger.info("NDVI calculated and saved to: %s", ndvi_filepath)

            # Stretch and save the final image
            image_stretched_swin2_mose = stretch_to_min_max_8bit(image_np_final)
            tiff.imwrite(
                output_filepath,
                image_stretched_swin2_mose.astype(np.float32),
                photometric="minisblack",
            )

            results.append(
                {
                    "filename": filename,
                    "status": "success",
                    "message": f"Processed and saved '{output_filepath}'.",
                }
            )
            output_plot_path = os.path.join(output_dir, f"comparison_{filename}")
            plot_images_side_by_side(
                image_original_stretched,
                image_stretched_swin2_mose,
                output_path=output_plot_path,
            )
        except Exception as e:
            logger.error("Error processing %s with Swin2 Mose model: %s", filename, e)
            results.append(
                {
                    "filename": filename,
                    "status": "error",
                    "message": f"An error occurred: {str(e)}",
                }
            )

    logger.info("result: %s", results)
    return {
        "status": "success",
        "results": results,
        "message": "SWIN 2 Mose model finished processing files.",
    }
