# XR-VLM

Official Code for "XR-VLM: Cross-Relationship Modeling with Multi-part Prompts and Visual Features for Fine-Grained Recognition".

# Usage 

## Install

`pip install -r requirement.txt`

Note that all experiments are conducted on vGPU-32GB in autodl.

## Download Datasets

We use CUB-200-2011, Stanford-Cars, Stanford-Dogs, FGVC-Aircraft, NABirds for experiments, and please download them and put into a folder.

Then, change the data path in `libcore/config/default.py`.

## Run Training Script
`bash run scripts/run_exp.sh`

The configures can be modified in `cfg/single_xpr.yml`

## Evaluate

If the trained model is saved in `saved_path`, in which there is a `train_param.yml` file and the used dataset is CUB-200-2011, then the evaluation is performed by running:

`python main.py --data cub --evaluate --config saved_path/train_params.yml`

## Pre-trained models.
We are working on the pre-trained models. 

## Citation
If you find this paper or our code useful in your research, please consider citing:
```bib
@inproceedings{wcm_xr-vlm2025,
  author    = {Chuanming Wang and
               Hengming Mao and
               Huanhuan Zhang and
               Huiyuan Fu and
               Huadong Ma},
  title     = {XR-VLM: Cross-Relationship Modeling with Multi-part Prompts and Visual Features for Fine-Grained Recognition},
  year      = {2025},
}
```