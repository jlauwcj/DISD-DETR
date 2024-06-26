# DISD DETR

## Abstract.
This paper proposes a multitask model (DISD-DETR) that combines the global modeling capabilities of Transformers, the local processing power of Convolutional Neural Networks (CNNs), and the advantages of positional encoding for end-to-end 2D object detection and instance segmentation. First, an improved ResNet is introduced into the model to enhance the feature extraction capability of the backbone network. Additionally, an improved attention mechanism tailored for instance segmentation tasks is incorporated into the Transformer encoder stage, and a reference window updating mechanism is added to the Transformer decoder to optimize attention intervals. These enhancements boost the expressiveness and discriminative power of the model’s queries and accelerate model convergence. Building on this, a new instance segmentation head is constructed to address occlusion issues, resulting in finer segmentation of leaves and spots and the extraction of more tiny spots. To improve classification performance, focal loss is employed as the classification loss and binary loss as the segmentation loss. Experimental results demonstrate that this study provides an efficient and accurate method for crop leaf disease segmentation, offering a solid foundation for precise analysis of crop leaf diseases. Our model achieves an AP of 73.9% for detection and an AP of 68.2% for segmentation on a self-constructed dataset.

## Installation

### Requirements

- Linux, CUDA>=11, GCC>=5.4
- Python>=3.8

  We recommend you to use Anaconda to create a conda environment:

  ```bash
  conda create -n DISD python=3.8
  ```

  Then, activate the environment:

  ```bash
  conda activate DISD
  ```

- PyTorch>=1.10.1, torchvision>=0.11.2 (following instructions [here](https://pytorch.org/))

  For example, you could install pytorch and torchvision as following:

  ```bash
  conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
  ```

- Other requirements & Compilation

  ```bash
  python -m pip install -e DISD
  ```
	
  If you are inside the ```DISD``` folder, then run:
  ```bash
  python -m pip install -e ./
  ```

  You can test the CUDA operators (box and instance attention) by running

  ```bash
  python tests/box_attn_test.py
  python tests/instance_attn_test.py
  ```

## Usage

### Dataset preparation

The datasets are assumed to exist in a directory specified by the environment variable $E2E_DATASETS.
If the environment variable is not specified, it will be set to be `.data`.
Under this directory, detectron2 will look for datasets in the structure described below.

```
$E2E_DATASETS/
├── coco/
```

For COCO Detection and Instance Segmentation, please download [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:

```
$E2E_DATASETS/
└── coco/
	├── annotation/
		├── instances_train2017.json
		├── instances_val2017.json
		└── image_info_test-dev2017.json
	├── image/
		├── train2017/
		├── val2017/
		└── test2017/
	└── vocabs/
		└── coco_categories.txt - the mapping from coco categories to indices.
```
You can generate data files for our training and evaluation from raw data by running `create_gt_database.py` and `create_imdb` in `tools/preprocess`.

### Training

Our script is able to automatically detect the number of available gpus on a single node.
It works best with Slurm system when it can auto-detect the number of available gpus along with nodes.
The command for training DISD is simple as following:

```bash
python tools/run.py --config ${CONFIG_PATH} --model ${MODEL_TYPE} --task ${TASK_TYPE}
```

For example,

- COCO Detection

```bash
python tools/run.py --config e2edet/config/COCO-Detection/DISD2d_R_50_3x.yaml --model DISD2d --task detection
```

- COCO Instance Segmentation

```bash
python tools/run.py --config e2edet/config/COCO-InstanceSegmentation/DISD2d_R_50_3x.yaml --model DISD2d --task detection
```

#### Some tips to speed-up training

- If your file system is slow to read images but your memory is huge, you may consider enabling 'cache_mode' option to load whole dataset into memory at the beginning of training:

```bash
python tools/run.py --config ${CONFIG_PATH} --model ${MODEL_TYPE} --task ${TASK_TYPE} dataset_config.${TASK_TYPE}.cache_mode=True
```

- If your GPU memory does not fit the batch size, you may consider to use 'iter_per_update' to perform gradient accumulation:

```bash
python tools/run.py --config ${CONFIG_PATH} --model ${MODEL_TYPE} --task ${TASK_TYPE} training.iter_per_update=2
```

- Our code also supports mixed precision training. It is recommended to use when you GPUs architecture can perform fast FP16 operations:

```bash
python tools/run.py --config ${CONFIG_PATH} --model ${MODEL_TYPE} --task ${TASK_TYPE} training.use_fp16=(float16 or bfloat16)
```

### Evaluation

You can get the config file and pretrained model of DISD, then run following command to evaluate it on COCO 2017 validation/test set:

```bash
python tools/run.py --config ${CONFIG_PATH} --model ${MODEL_TYPE} --task ${TASK_TYPE} training.run_type=(val or test or val_test)
```

### Analysis and Visualization

You can get the statistics of DISD (fps, flops, \# parameters) by running `tools/analyze.py` from the root folder.

```bash
python tools/analyze.py --config-path save/COCO-InstanceSegmentation/DISD2d_R_50_3x.yaml --model-path save/COCO-InstanceSegmentation/DISD2d_final.pth --tasks speed flop parameter
```

The notebook for DISD-2D visualization is provided in `tools/visualization/DISD_2d_segmentation.ipynb`.

## Citation
```
@article{nguyen2021boxer,
  title={BoxeR: Box-Attention for 2D and 3D Transformers},
  author={Duy{-}Kien Nguyen and Jihong Ju and Olaf Booij and Martin R. Oswald and Cees G. M. Snoek},
  journal={arXiv preprint arXiv:2111.13087},
  year={2021}
}

@inproceedings{
  liu2022dabdetr,
  title={{DAB}-{DETR}: Dynamic Anchor Boxes are Better Queries for {DETR}},
  author={Shilong Liu and Feng Li and Hao Zhang and Xiao Yang and Xianbiao Qi and Hang Su and Jun Zhu and Lei Zhang},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=oMI9PjOb9Jl}
}
```
