# Testing Phylogenetic Signal with Categorical Traits and Tree Uncertainty

This repository contains the code for calculating the Delta statistic, which is designed to test the phylogenetic signal of categorical traits in the presence of tree uncertainty. The code is organized into two main sections:

## Delta-Python
This folder contains the Python implementation of the Delta statistic.

- **input/**: A folder containing multiple example input files.
- **delta_functs.py**: Python file containing the Delta functions.
- **requirements.txt**: A text file listing the necessary Python libraries to install.
- **delta_example.py**: A Python file with a runnable example demonstrating the Delta statistic.
- **MANUAL.(ipynb/pdf)**: A Jupyter Notebook or PDF file providing step-by-step instructions for running the code.

## Delta-Webapp
This folder contains the Django web application for running the Delta statistic locally.

- **App/**: Folder with the Django web application code that can be deployed locally.
- **requirements.txt**: A text file listing the necessary Python libraries to install.
- **runtime.txt**: A text file specifying the Python version used in testing.
- **LocalApp_MANUAL.(ipynb/pdf)**: Guide for implementing the Django web application using a Jupyter Notebook or PDF file.
- **MANUAL.(mp4/pdf)**: A video (MP4) and PDF file providing step-by-step instructions for calculating Delta in the web application.

---

## Live Deployment

The web application is also available through free hosting services. You can access them at the following URLs:

- [delta-statistic.onrender.com](https://delta-statistic.onrender.com)
- ~~delta-statistic.up.railway.app~~ (unavailable)

---

## Citation

If you use this repository or the Delta statistic in your work, please cite the following papers:

**Rui Borges, João Paulo Machado, Cidália Gomes, Ana Paula Rocha, Agostinho Antunes.**  
*"Measuring phylogenetic signal between categorical traits and phylogenies."*  
*Bioinformatics*, Volume 35, Issue 11, June 2019, Pages 1862–1869, [https://doi.org/10.1093/bioinformatics/bty800](https://doi.org/10.1093/bioinformatics/bty800).

**Diogo Ribeiro, Rui Borges, Ana Paula Rocha, Agostinho Antunes.**  
*"Testing phylogenetic signal with categorical traits and tree uncertainty."*  
*Bioinformatics*, Volume 39, Issue 7, July 2023, btad433, [https://doi.org/10.1093/bioinformatics/btad433](https://doi.org/10.1093/bioinformatics/btad433).
