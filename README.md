# Adversarial Perturbations Improve Generalization of Confidence Prediction in Medical Image Segmentation

We introduce a straightforward adversarial training strategy that enhances the reliability of direct confidence prediction in medical image segmentation under domain shifts.

[MIDL 2025 Conference Paper](https://openreview.net/pdf?id=0BQ6JPGwZa)

<!-- ## Table of Contents
- [Adversarial Perturbations Improve Generalization of Confidence Prediction in Medical Image Segmentation](#adversarial-perturbations-improve-generalization-of-confidence-prediction-in-medical-image-segmentation)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Data](#data)
  - [Experiments \& Results](#experiments--results)
  - [Citation \& License](#citation--license)
  - [Contact](#contact) -->

## Installation
1. Clone the deployment branch of this repo (no code, only docker utils)
```bash
git clone --branch deploy --single-branch git@github.com:MedVisBonn/midl25.git
```
1. Build the image 
```bash
cd midl25/docker
bash build.sh
```
1. Create shared direotories for data other files, adapt `docker/run.sh` accordingly and create a container
```bash
bash run.sh
```
- (optional) in the container, navigate to `/root/workplace/repos/midl25/` and create `results/`, `pre-trained/monai-unets/` and `pre-trained/score-predictor/` directories.

## Usage
All applications can be run from bash files in `src/apps`.
- To train a U-Net, adapt `src/apps/train_unet.sh` and run it. To see all options, adapt `src/configs/unet/monai_unet.yaml` and `src/configs/trainer/unet_trainer.yaml`.
- To train a score predictor, adapt `src/apps/trai_score_predictor.sh` and run it. Again, to see all options, take a look at `src/configs/model/score_predictor.yaml` and `src/configs/trainer/score_predictor_trainer.yaml` respectively.

## Data
We evaluate our approach using two datasets: the [SAML Dataset](https://liuquande.github.io/SAML/) and the [MNMS-2 Dataset](https://www.ub.edu/mnms-2/). To work with these datasets, adapt the paths in the configuration files in `src/configs/data` to match your local environment. Any pre-processing is handled by the respective classes in `src/dataset`.

## Citation & License
TBA

## Contact
For questions, reach out to: lennartz (ät) cs.uni-bonn.de