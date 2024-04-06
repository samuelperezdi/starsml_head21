# Exploring Astronomical Catalog Crossmatching with Machine Learning

- Multi-wavelength approaches are crucial for understanding astrophysical X-ray sources, beyond traditional methods.
- Existing cross-matching algorithms, using Bayesian statistics and spatial data, may overlook key physical characteristics of astronomical objects, allowing for misidentification (See Figure 1).
- We analyze property distributions for optical match candidates of X-ray sources, finding significant differences beyond certain spatial thresholds. We highlight this as a limit of spatial crossmatching.
- We introduce a random forest model to crossmatch the Chandra Source Catalog 2.1 (CSC2.1) with Gaia DR3. This model produces match likelihoods.
- We achieve ~86% recall scores for positive optical counterparts to X-ray sources. This demonstrates the potential of machine learning for astronomical catalog crossmatching. In the future, we will explore contrastive learning techniques in order to leverage representations of catalog data for cross-matching.

## Getting Started
- Install the requirements using the following command:
```pip install -r requirements.txt```

- Get the data.

  ```wget https://zenodo.org/records/10937089/files/out_data.zip```

  ```mkdir -p ./out_data```

  ```unzip out_data.zip -d ./out_data```

Now you have the data under the right directories. You can run the notebook to reproduce the visualizations in the poster.

- Reproduce the results in ```results_posterhead21.ipynb```.