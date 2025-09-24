---
license: apache-2.0
datasets:
- isp-uv-es/opensr-test
language:
- en
pipeline_tag: image-to-image
tags:
- Sentinel-2
- sentinel2
- S2
- super-resolution
---

# **SUPERIX: Super-Resolution Intercomparison Exercise**



## **Introduction**

Super-resolution (SR) techniques are becoming more popular in improving the spatial resolution of freely 
available satellite imagery, such as Sentinel-2 and Landsat. SR could significantly 
improve the accuracy of various remote sensing downstream tasks, including road detection, crop delineation, 
and object recognition. However, some researchers argue that the benefits of SR are primarily aesthetic, 
suggesting that its main value lies in creating more visually appealing maps or aiding in visual interpretation.

Another criticism of SR is that it can degrade the original input data, potentially leading to incorrect conclusions.
However, some SR methods appear more conservative than others in preserving reflectance integrity. Given this, 
a reliable benchmark is essential for providing quantitative assessments of the current state-of-the-art. Without 
such benchmarks, it remains difficult to conclusively determine the true impact of SR techniques on remote sensing data.

To establish a reliable framework, we propose the creation of a dedicated working group aimed at intercomparing super-resolution
algorithms for Sentinel-2 data (SUPERIX). SR algorithms developed by teams from universities, research centers, industry, 
and space agencies are encouraged to participate in SUPERIX. This initiative will use OpenSR-test datasets and proposed metrics 
to evaluate the consistency with the original input data and the reliability of the high-frequency details introduced by the 
SR models.

Summarizing, multiple methods have been developed to address the problem of super-resolution in satellite imagery, 
but very few studies were carried out to quantitatively inter-compare state-of-the-art methods in this domain. 

- SUPERIX aims at inter-comparing SR algorithms for ESA Sentinel-2 mission. 
- SUPERIX will involve defining reference datasets, metrics and an analysis framework.
- SUPERIX should allow to identify strengths and weaknesses of existing algorithms and potential areas of improvements. 


## **Teams and SR Algorithms**

Are you interested? Contact us!


## **Validation Datasets**

Accurate validation datasets will allow a detailed analysis of SR strengths and weaknesses. 

Validation datasets might vary in the way they are sampled and generated: 
- cross-sensor or synthetic 
- spatial scale factor
- geographical distribution

Performance of SR algorithms will vary also depending on the reference dataset, which can be attributed to differences in 
radiometry, spectral response, spatial alignment, effective spatial resolution, considered landscapes, etc.

About the high-resolution (HR) reference, we are considering:

- **naip:** A set of 62 RGBNIR orthophotos mainly from agricultural and forest regions in the USA.
- **spot:** A set of 10 SPOT images obtained from Worldstrat. 
- **spain_urban:** A set of 20  RGBNIR orthophotos, primarily from urban areas of Spain, including roads.
- **spain_crops:** A set of 20  RGBNIR orthophotos, primarily taken from agricultural areas near cities in Spain.
- **venus:** A set of 60 VENµS images obtained from SEN2VENµS.

Each HR reference includes the corresponding Sentinel-2 imagery preprocessed at 1C and 2A levels. Here is an example of how 
to load each dataset.

```{python}
import opensr_test

dataset = opensr_test.load("naip")
lr, hr = dataset["L2A"], dataset["HRharm"]
```


## **Quality Metrics**

We propose the following metrics to assess the consistency of SR models:

- **Reflectance:** This metric evaluates how SR affects the reflectance of the LR image, utilizing the Mean Absolute
  Error (MAE) distance by default. Lower values indicate better reflectance consistency. The SR image is downsampled to LR
  resolution using a triangular anti-aliasing filter and downsampling by the scale factor (bilinear interpolation).

- **Spectral:** This metric measures how SR impacts the spectral signature of the LR image, employing the Spectral Angle
Distance (SAM) by default. Lower values indicate better spectral consistency, with angles measured in degrees. The SR image
is downsampled to LR resolution using a triangular anti-aliasing filter and downsampling by the scale factor (bilinear interpolation).


- **Spatial:** This metric assesses the spatial alignment between SR and LR images, utilizing the Phase Correlation
Coefficient (PCC) by default. Some SR models introduce spatial shifts, which this metric detects. The SR image is downsampled
to LR resolution using a triangular anti-aliasing filter and downsampling by the scale factor (bilinear interpolation).

We propose three metrics to evaluate the high-frequency details introduced by SR models. The sum of these metrics always equals 1:

- **Improvements (im_score):** This metric quantifies the similarity between the SR and HR images.
A value closer to 1 indicates that the SR model closely corresponds to the HR image (i.e. improves the high-frequency details).

- **Omissions (om_score):** This metric measures the similarity between the SR and LR images. A value closer to 1 suggests that the SR model
closely compares the LR image downsampled with bilinear interpolation (i.e. omits high-frequency details present in HR but not in LR).

- **Halucinations (ha_score):** This metric evaluates the similarity between SR and the HR and LR images. A value closer to 1 indicates that the
SR model deviates significantly from both references (i.e. hallucinates introducing high-frequency details not present in HR).



## **Proposed Experiments**

We are planning two experiments for both x4 and x2 scale factors. Participants are encouraged to submit their SR models 
for both scales. Additionally, models designed solely for the x4 scale will be assessed at the x2 scale by downsampling
the SR image by a factor of 2.

In each experiment, we will employ two distinct approaches to evaluate the high-frequency details introduced by SR models.
The first approach utilizes the Mean Absolute Error (MAE) as the distance metric for assessing high-frequency details. 
Alternatively, the second approach employs LPIPS. While MAE is sensitive to the intensity of high-frequency details, 
LPIPS is more sensitized to their structural differences. Contrasting the outcomes of these two metrics can offer a comprehensive
understanding of the high-frequency details introduced by SR models. LPIPS metrics are consistently run on 32x32 patches 
of the HR image, while MAE is computed on 2x2 patches for x2 scale and 4x4 patches for x4 scale evaluations.


## **Proposed Protocol**


- The SUPERIX working group should first agree on the validation datasets appropriate for SR, the definition of best quality metrics, and how quantify hallucinations. 

- Each team will submit their SR models up to the deadline. 

- We will have two different types of models: **open-source** and **closed-source**.
To be considered open-source, the code must be available in this repository within a folder named as the model name.
Keep the code as simple as possible. See examples using torch, diffuser, and tensorflow libraries [here](), [here](), and [here]().
The closed-source models are required to **only provide the results in GeoTIFF format**. See an example [here]().

- The submission will be made through a [pull request](https://huggingface.co/docs/hub/en/repositories-pull-requests-discussions) to this repository. The pull request **MUST** include the `metadata.json` file and the results in GeoTIFF format. The results must be in the same resolution as the HR image.
We expect the following information in the metadata.json file:

```{json}
{
  "name": "model_name",
  "authors": ["author1", "author2"],
  "affiliations": ["affiliation1", "affiliation2"],
  "description": "A brief description of the model",
  "code": "open-source" or "closed-source",
  "scale": "x2" or "x4",
  "url": "[OPTIONAL] URL to the model repository if it is open-source",
  "license": "license of the model"
}
```

- The SUPERIX working group will evaluate the SR models after the deadline using the metrics discussed above.

- After the metrics estimation, we will first independently contact the teams providing the results. If there are any issues with
the submission, we will ask for clarification, and the team will have up to two weeks to provide the necessary corrections.

- Questions and discussions will be held in the discussion section of this [repository](https://huggingface.co/isp-uv-es/superIX/discussions).
The progress of the SUPERIX working group will be informed through the discussion section and by email. 

- After all the participants have provided the necessary corrections, the results will be published in the discussion section of this repository.


## **Expected Outcomes**

- No clear superiority of any methodology in all metrics is expected. 

- Analysis on validation scenes with major discrepancies between algorithms will be carried out.

- A dedicated website and a technical report will be prepared to present the results and recommendations. 

- A research publication will be submitted to a remote sensing journal.

- The paper will be prepared in overleaf, and all the participants will be invited to contribute to it.


