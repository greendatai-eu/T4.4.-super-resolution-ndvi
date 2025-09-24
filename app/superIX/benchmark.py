import rasterio as rio
import pathlib
import opensr_test
import matplotlib.pyplot as plt

from typing import Callable, Union


def create_geotiff(
    model: Callable,
    fn: Callable,
    datasets: Union[str, list],
    output_path: str,
    force: bool = False,
    **kwargs
) -> None:
    """Create all the GeoTIFFs for a specific dataset snippet 

    Args:
        model (Callable): The model to use to run the fn function.
        fn (Callable): A function that return a dictionary with the following keys:
            - "lr": Low resolution image
            - "sr": Super resolution image
            - "hr": High resolution image
        datasets (list): A list of dataset snippets to use to run the fn function.
        output_path (str): The output path to save the GeoTIFFs.
        force (bool, optional): If True, the dataset is redownloaded. Defaults 
            to False.
    """
    
    if datasets == "all":
        datasets = opensr_test.datasets 

    for snippet in datasets:
        create_geotiff_batch(
            model=model,
            fn=fn,
            snippet=snippet,
            output_path=output_path,
            force=force,
            **kwargs
        )    

    return None

def create_geotiff_batch(
    model: Callable,
    fn: Callable,
    snippet: str,
    output_path: str,
    force: bool = False,
    **kwargs
) -> pathlib.Path:
    """Create all the GeoTIFFs for a specific dataset snippet 

    Args:
        model (Callable): The model to use to run the fn function.
        fn (Callable): A function that return a dictionary with the following keys:
            - "lr": Low resolution image
            - "sr": Super resolution image
            - "hr": High resolution image
        snippet (str): The dataset snippet to use to run the fn function.
        output_path (str): The output path to save the GeoTIFFs.
        force (bool, optional): If True, the dataset is redownloaded. Defaults 
            to False.

    Returns:
        pathlib.Path: The output path where the GeoTIFFs are saved.
    """
    
    # Create folders to save results
    output_path = pathlib.Path(output_path)  / "results" / "SR"
    output_path.mkdir(parents=True, exist_ok=True)

    output_path_dataset_geotiff = output_path / snippet / "geotiff"
    output_path_dataset_geotiff.mkdir(parents=True, exist_ok=True)

    output_path_dataset_png = output_path / snippet / "png"
    output_path_dataset_png.mkdir(parents=True, exist_ok=True)

    # Load the dataset 
    dataset = opensr_test.load(snippet, force=force)
    lr_dataset, hr_dataset, metadata = dataset["L2A"], dataset["HRharm"], dataset["metadata"]
    for index in range(len(lr_dataset)):
        print(f"Processing {index}/{len(lr_dataset)}")

        # Run the model    
        results = fn(
            model=model,
            lr=lr_dataset[index],
            hr=hr_dataset[index],
            **kwargs
        )

        # Get the image name
        image_name = metadata.iloc[index]["hr_file"]

        # Get the CRS and transform
        crs = metadata.iloc[index]["crs"]
        transform_str = metadata.iloc[index]["affine"]
        transform_list = [float(x) for x in transform_str.split(",")]
        transform_rio = rio.transform.from_origin(
            transform_list[2],
            transform_list[5],
            transform_list[0],
            transform_list[4] * -1
        )

        # Create rio dict
        meta_img = {
            "driver": "GTiff",
            "count": 3,
            "dtype": "uint16",
            "height": results["hr"].shape[1],
            "width": results["hr"].shape[2],
            "crs": crs,
            "transform": transform_rio,
            "compress": "deflate",
            "predictor": 2,
            "tiled": True
        }

        # Save the GeoTIFF
        with rio.open(output_path_dataset_geotiff / (image_name + ".tif"), "w", **meta_img) as dst:
            dst.write(results["sr"])

        # Save the PNG
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow((results["lr"].transpose(1, 2, 0) / 3000).clip(0, 1))
        ax[0].set_title("LR")
        ax[0].axis("off")
        ax[1].imshow((results["sr"].transpose(1, 2, 0) / 3000).clip(0, 1))
        ax[1].set_title("SR")
        ax[1].axis("off")
        ax[2].imshow((results["hr"].transpose(1, 2, 0) / 3000).clip(0, 1))
        ax[2].set_title("HR")
        # remove whitespace around the image
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis("off")
        plt.savefig(output_path_dataset_png / (image_name + ".png"))
        plt.close()
        plt.clf()

    return output_path_dataset_geotiff




def run(
    model_path: str
) -> pathlib.Path:
    """Run the all metrics for a specific model.

    Args:
        model_path (str): The path to the model folder.
    
    Returns:
        pathlib.Path: The output path where the metrics are 
        saved as a pickle file.
    """
    pass


def plot(
    model_path: str
) -> pathlib.Path:
    """Generate the plots and tables for a specific model.

    Args:
        model_path (str): The path to the model folder.
    
    Returns:
        pathlib.Path: The output path where the plots and tables are 
        saved.
    """
    pass


