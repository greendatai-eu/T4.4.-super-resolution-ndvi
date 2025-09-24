import benchmark
import matplotlib.pyplot as plt
import opensr_test

from superIX.swin2_mose.utils import load_swin2_mose, load_config, run_swin2_mose

device = "cuda"
path = 'swin2_mose/weights/config-70.yml'
model_weights = "swin2_mose/weights/model-70.pt"

# load config
cfg = load_config(path)

# load model
model = load_swin2_mose(model_weights, cfg)
model.to(device)
model.eval()

# load the dataset
index = 2
dataset = opensr_test.load("venus")
lr_dataset, hr_dataset = dataset["L2A"], dataset["HRharm"]
results = run_swin2_mose(model, lr_dataset[index], hr_dataset[index], device=device)

# Display the results
#fig, ax = plt.subplots(1, 3, figsize=(10, 5))
#ax[0].imshow(results['lr'].numpy().transpose(1, 2, 0)/3000)
#ax[0].set_title("LR")
#ax[0].axis("off")
#ax[1].imshow(results["sr"].detach().numpy().transpose(1, 2, 0)/3000)
#ax[1].set_title("SR")
#ax[1].axis("off")
#ax[2].imshow(results['hr'].numpy().transpose(1, 2, 0) / 3000)
#ax[2].set_title("HR")
#plt.show()

# Run the experiment
benchmark.create_geotiff(model, run_swin2_mose, "all", "swin2_mose/")
