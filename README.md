d# CaloDiffusion official repository

Repo for diffusion based calorimeter generation.

Pytorch v1.9  was used to implement all models. 

Name: Grey Singh
Email: singh8@uchicago.edu

Carina's Version 

# Data

Results are presented using the [Fast Calorimeter Data Challenge dataset](https://calochallenge.github.io/homepage/) and are available for download on zenodo:
* [Dataset 1](https://zenodo.org/record/6368338)
* [Dataset 2](https://zenodo.org/record/6366271)
* [Dataset 3](https://zenodo.org/record/6366324)

# Run the training scripts with

```bash
cd scripts
python train_diffu.py  --data_folder DIRECTORY_OF_INPUTS --config CONFIG
```
* Example configs used to produce the results in the paper are in `configs` folder
* Trained models used to produce the results in the paper are in the `trained_models` folder

# Sampling with the learned model

The `plot.py` both samples from the models (if the `--sample` flag is included) and creates the summary plots. 

```bash
python plot.py --data_folder DIRECTORY_OF_INPUTS -g GENERATED_SHOWERS.h5 --nevts N  --sample  --config CONFIG --sample_steps N_DIFFUSION_STEPS --sample_offset N_OFFSET
```
There are additional options in `plot.py`. 
400 diffusion steps were used for the baseline results in the paper. 
A sampling offset of 2 was used for datasets 1 and 2 (necessary to prevent instabilities). 

An example commands to generate showers for dataset 1 photons would be as follows:

```bash
python plot.py --config ../configs/config_dataset1_photon.json --model_loc ../trained_models/dataset1_photon.pth --sample --sample_steps 400 --sample_offset 2 --nevts 1000 -g test_ds1.h5
```

# Creating the plots shown in the paper

```bash
python plot.py --data_folder DIRECTORY_OF_INPUTS -g GENERATED_SHOWERS.h5 --config CONFIG --plot_folder PLOTTING_DIRECTORY
```
An example to plot the showers generated for dataset 1 would be as follows:

```bash
python plot.py --config ../configs/config_dataset1_photon.json --plot_folder test_phot/ -g test_ds1.h5
```

Some of the quantitative metrics and plotting for dataset 1 are done based on the [CaloChallenge evaluation code](https://github.com/OzAmram/CaloChallenge)
The evaluation for the CaloChallenge code proceeds can be performed as:

```
python evaluate.py -i GENERATED_DATASET.h5 -r REFERENCE_DATASET.h5 -d DATASET_NUMBER --output_dir DIRECTORY_FOR_OUTPUTS/ -m all --ratio
```
The `-m all` flag runs all the evaluations of the CaloChallenge (plots, chi2 metric, classifier metric, KPD/FPD metric). 
See `evaluate.py` for more options


