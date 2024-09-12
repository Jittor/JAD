# Jittor-MTR

![teaser](https://github.com/sshaoshuai/MTR/blob/master/docs/mtr_demo.png)


## Introduction

Jittor-MTR is an open-source [Jittor](https://github.com/Jittor/jittor) implementation of [MTR](https://arxiv.org/abs/2209.13508) focusing on Transformer-based motion prediction.

## Environment Configuration

Jittor-MTR is developed and tested under the environment:

- **Operating System:** Linux
- **Python Version:** 3.8
- **Compiler:** G++ 9.4
- **CUDA Version**: 11.3
- **MPI Version:** 5.0.3 (for multi-processing support)

## Installation

Follow these steps to set up the Jittor-MTR environment:

1. **Clone the repository:**

   ```shell
   git clone ***
   cd Motion-Prediction/MTR
   ```

2. **Create a conda virtual environment and activate it:**

   ```shell
   conda create -n jittor_mtr python=3.8 -y
   conda activate jittor_mtr
   ```

3. **Install Jittor:**

   Before installing Jittor, make sure you have the necessary system libraries.

   ```shell
   sudo apt install libomp-dev

   pip install git+https://github.com/Jittor/jittor.git # make sure use the latest version, after commit da45615
   ```

4. **Install CUDA support for Jittor:**

   If you have a GPU and want to enable CUDA acceleration, install CUDA to the Jittor cache:

   ```shell
   # Remeber to Set Correct $CUDA_HOME
   python -m jittor_utils.install_cuda 
   ```

5. **Install dependencies:**

   Navigate to the project directory and install the required Python packages:

   ```shell
   # Consider using mirror site of pip if the downloading is slow
   pip install -r requirements.txt
   ```

## Getting Started

### Dataset

1. **Prepare the nuScenes dataset:**

Download nuScenes dataset and organize the data as follows: 
   ```
   MTR
   ├── data
   |   ├── nuscenes
   |   |   ├── cluster_64_center_dict.pkl
   │   |   ├── nuscenes_raw
   │   │   |   ├── maps
   │   │   |   ├── samples
   │   │   |   ├── v1.0-mini (or v1.0)
   ├── mtr_jittor
   ├── tools_jittor
   ```

2. **Preprocess:**
   ```
   # The first argument is the input directory and the second one is the output directory
   python mtr_jittor/datasets/nuscenes/data_preprocess.py data/nuscenes/nuscenes_raw data/nuscenes
   ```
   It could take hours. You may adjust the number of workers to speed up according to your device.

3. **Prepare the checkpoint:**

   Navigate to the appropriate directory and create a folder for the checkpoints:

   ```shell
   mkdir ckpts
   cd ckpts
   ```

   Download the checkpoints trained on nuScenes ([GoogleDrive](https://drive.google.com/file/d/1uPc-2MVAuWjCkSoiNysiKUeBLtCm-VNn/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1mbFxxchxemRrV02OgnYV0Q?pwd=j57w))  and place them in the `ckpts` directory as shown below:

   ```
   ├── ckpts
       └── mtr_nuscenes.pth
   ```

## Inference

To evaluate the performance of the model, you can run inference as follows:


```shell
# Under the directory Motion-Prediction/MTR/tools_jittor. The first step could be slow since Jittor backend is compiling all operators.
python test.py --cfg_file cfgs/nuscenes/nuscenes_mini.yaml --ckpt ../ckpts/mtr_nuscenes.pth
```


## Results
We compare the performance of Jittor and PyTorch implementations trained on the nuScenes v1.0-mini dataset. The results are summarized below:


### Stage 2

| Framework  | Vehicle mAP | Vehicle minADE | Vehicle minFDE | Vehicle MR |
| :--------: | -------------: | ---------------: | ------------: | ---------------: | 
| **PyTorch**|          **0.7751** |           0.9513 |       1.3305 |             **0.1617** | 
| **Jittor** |          0.7741 |            **0.9510** |        **1.3280** |            0.1606 |

These results demonstrate the capability of Jittor to achieve comparable, if not better, performance than PyTorch.

### Reference
1. [Jittor](https://github.com/Jittor/jittor)
2. [MTR](https://github.com/sshaoshuai/MTR)
