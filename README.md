# CaloDiffusion official repository

Repo for diffusion based calorimeter generation.

Pytorch v1.9  was used to implement all models. 

# Data

Results are presented using the [Fast Calorimeter Data Challenge dataset](https://calochallenge.github.io/homepage/) and are available for download on zenodo:
* [Dataset 1](https://zenodo.org/record/6368338)
* [Dataset 2](https://zenodo.org/record/6366271)
* [Dataset 3](https://zenodo.org/record/6366324)

# Run the training scripts with

```bash
cd scripts
python train_diffu.py  --config CONFIG
```
* Example configs used to produce the results in the paper are in `configs` folder
* Trained models used to produce the results in the paper are in the `trained_models` folder

# Sampling with the learned model

```bash
python plot.py  --nevts N  --sample  --config CONFIG --sample_steps
N_DIFFUSION_STEPS 
```
# Creating the plots shown in the paper

```bash
python plot.py  --config CONFIG
```

Some of the quantitative metrics are computed based on the [CaloChallenge evaluation code](https://github.com/OzAmram/CaloChallenge) 



