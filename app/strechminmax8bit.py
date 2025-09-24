import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt


def stretch_to_min_max_8bit(image_np, min_vals=None, max_vals=None, no_data_value=-32768, output_filename='output_image.tiff'):
    """
    Process an RGBNIR image by removing the NIR channel, applying a valid mask, 
    and stretching the remaining RGB channels to the 8-bit range.

    Args:
        image_np (numpy.ndarray): The input image array (can be 2D or 3D).
        min_vals (list or None): Minimum values for each band (R, G, B) for stretching.
                                 If None, defaults to 0 for each channel.
        max_vals (list or None): Maximum values for each band (R, G, B) for stretching.
                                 If None, defaults to 4095 (12-bit max value) for each channel.
        no_data_value (int): The value representing no-data in the image.
        output_filename (str): The filename for saving the processed image.

    Returns:
        numpy.ndarray: The processed and stretched 8-bit RGB image.
    """

    # Ensure the image has at least 3 dimensions
    if image_np.ndim == 2:
        image_np = image_np[..., np.newaxis]

    # Remove the 4th channel if present (assuming it's the last one)
    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]  # Keep only the first three channels (RGB)

    # Handle specific no-data values if they exist
    valid_mask = image_np != no_data_value

    # Default min and max values for 12-bit Sentinel-2 data if not provided
    if min_vals is None:
        min_vals = [0] * image_np.shape[-1]  # Defaults to 0 for each channel
    if max_vals is None:
        max_vals = [4095] * image_np.shape[-1]  # Defaults to 4095 (12-bit max) for each channel

    # Check if the number of bands matches the min_vals and max_vals
    num_bands = image_np.shape[-1]
    if num_bands != len(min_vals) or num_bands != len(max_vals):
        raise ValueError("Number of bands in the image does not match the number of min/max values provided")

    # Initialize the output array
    image_stretched = np.zeros_like(image_np, dtype=np.uint8)

    # Loop through each band and apply the stretching
    for i in range(num_bands):
        # Apply the valid mask to this band
        valid_band_mask = valid_mask[..., i]

        # Clip the data to avoid outliers
        clipped_band = np.clip(image_np[..., i], min_vals[i], max_vals[i])

        # Stretch the valid data for this band to the full 8-bit range [0, 255]
        image_stretched[..., i][valid_band_mask] = (
            ((clipped_band[valid_band_mask] - min_vals[i]) / (max_vals[i] - min_vals[i]) * 255)
            .astype(np.uint8)
        )

    # Optionally save the stretched image
    # if output_filename:
    #     tiff.imwrite(output_filename, image_stretched)

    return image_stretched

# Example usage:
# image_np = np.array(...)  # Load your image here
# processed_image = strech_to_min_max_8bit(image_np, min_vals=[0, 0, 0], max_vals=[8000, 8000, 8000])


def plot_images_side_by_side(original, swin_mose_processed,  
                             titles=['Original', 'Swin MOSE'],
                             output_path='plot.png'):
    """
    Plots three images side by side for comparison.
    
    Args:
        original (numpy.ndarray): The original image.
        swin_mose_processed (numpy.ndarray): The image processed with the Swin MOSE model.
        si_processed (numpy.ndarray): The image processed with the SI model.
        titles (list): Titles for the subplots.
    """
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot each image in a subplot
    axes[0].imshow(original)
    axes[0].set_title(titles[0])
    axes[0].axis('off')  # Hide axes

    axes[1].imshow(swin_mose_processed)
    axes[1].set_title(titles[1])
    axes[1].axis('off')  # Hide axes

    # Adjust layout to avoid overlapping titles
    plt.tight_layout()
    plt.savefig(output_path) # Save the plot to a file
    plt.close(fig) # Close the figure to free up memory

# Example usage:
# original_image = np.array(...)  # Load your original image here
# swin_mose_image = np.array(...)  # Load the Swin MOSE processed image
# si_image = np.array(...)  # Load the SI model processed image

# plot_images_side_by_side(original_image, swin_mose_image, si_image)


def add_black_for_no_data(original_image, processed_image, no_data_value=-32768):
    """
    Replaces the no-data values in the processed image with black pixels (0 value).
    
    Args:
        original_image (numpy.ndarray): The original image with no-data values.
        processed_image (numpy.ndarray): The processed image where no-data values will be replaced with black.
        no_data_value (int): The value representing no-data in the original image.

    Returns:
        numpy.ndarray: The processed image with no-data areas replaced by black.
    """
    # Ensure the original and processed images have the same shape
    if original_image.shape != processed_image.shape:
        raise ValueError("Original and processed images must have the same shape.")
    
    # Create a mask where the original image has the no-data value
    no_data_mask = (original_image == no_data_value)
    
    # Set the no-data regions in the processed image to black (0)
    processed_image[no_data_mask] = 0
    
    return processed_image

# Example usage:
# original_image = np.array(...)  # Load your original image here
# processed_image = np.array(...)  # Load your processed image here

# result_image = add_black_for_no_data(original_image, processed_image, no_data_value=-32768)
