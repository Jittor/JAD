# Jittor-UniAD

<div><video controls src="https://github.com/OpenDriveLab/UniAD/assets/48089846/bcf685e4-2471-450e-8b77-e028a46bd0f7" muted="false"></video></div>

<br><br>

![teaser](https://github.com/OpenDriveLab/UniAD/blob/main/sources/pipeline.png)

## Introduction

Jittor-UniAD is an open-source [Jittor](https://github.com/Jittor/jittor) implementation of [UniAD - CVPR 2023 Best Paer](https://arxiv.org/abs/2212.10156) focusing on end-to-end autonomous driving.

## Environment Configuration

Jittor-UniAD is developed and tested under the environment:

- **Operating System:** Linux
- **Python Version:** 3.8
- **Compiler:** G++ 9.4
- **CUDA Version**: 11.3
- **MPI Version:** 5.0.3 (for multi-processing support)

## Installation

Follow these steps to set up the Jittor-UniAD environment (It could share the same conda environment with Jittor-BEVFormer):

1. **Clone the repository:**

   ```shell
   git clone ***
   cd E2E-AD/UniAD
   ```

2. **Create a conda virtual environment and activate it:**

   ```shell
   conda create -n jittor_uniad python=3.8 -y
   conda activate jittor_uniad
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

   Additionally, manually install the following libraries for specific functionalities:

   ```shell
   pip install spconv-cu113
   pip install cupy-cuda113
   ```

   Finally, setup the project:

   ```shell
   python setup.py develop
   ```

6. **Troubleshooting:**

   If you encounter the error `AttributeError: module 'numpy.typing' has no attribute 'NDArray'`, this is likely due to an incompatibility with the `pillow` package. Resolve this by installing a specific version of `pillow`:

   ```shell
   pip install pillow==9.2.0
   ```

## Getting Started

### Dataset

1. **Prepare the nuScenes dataset:**

   Follow the instructions provided in the [OpenDriveLab/UniAD documentation](https://github.com/OpenDriveLab/UniAD/blob/main/docs/DATA_PREP.md) to prepare the nuScenes dataset.

   Note that the parent directory should be ./adzoo/uniad, as shown in the example below:
   ```
   ├── adzoo
      ├── uniad
         ├── data
            ├── infos
            ├── nuscenes
            └── others
   ```

2. **Download the model checkpoints:**

   Navigate to the appropriate directory and create a folder for the checkpoints:

   ```shell
   cd adzoo/uniad
   mkdir ckpts
   ```

   Download the checkpoints from ([GoogleDrive](https://drive.google.com/drive/folders/1kL9Qzkr_5Qrwg4B0VG7SsLyKzc9SpLMx?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1B48cS4RcI-J2YBvYTehI-Q?pwd=k7xj)) from [UniAD official repo](https://github.com/OpenDriveLab/UniAD) and place them in the `ckpts` directory as shown below:

   ```
   ├── ckpts
       ├── uniad_base_e2e.pth
       └── uniad_base_track_map.pth
   ```

## Inference

To evaluate the performance of the models, you can run inference for Stage 1 and Stage 2 as follows:

- **Stage 1:**
  ```shell
  # Under the highest directory E2E-AD/UniAD. The first step could be slow since Jittor backend is compiling all operators.
   ./adzoo/uniad/uniad_jittor_test.sh adzoo/uniad/configs/stage1_track_map/base_track_map.py adzoo/uniad/ckpts/uniad_base_track_map.pth 1
  ```

- **Stage 2 - End-to-End Testing:**

  ```shell
  # Under the highest directory E2E-AD/UniAD. The first step could be slow since Jittor backend is compiling all operators.
  ./adzoo/uniad/uniad_jittor_test.sh adzoo/uniad/configs/stage2_e2e/base_e2e.py adzoo/uniad/ckpts/uniad_base_e2e.pth 1
  ```

## Results
We compare the performance of Jittor and PyTorch implementations on the nuScenes v1.0-mini dataset. The results are summarized below:

### Stage 1

| Framework  | AMOTA | AMOTP | IoU-lane |
| :--------: | :---: | :---: | :------: |
| **PyTorch** | 0.547 | 1.136 |  0.222   |
| **Jittor** | 0.565 | 1.122 |  0.222   |


### Stage 2

| Framework  | Tracking AMOTA | Mapping IoU-lane | Motion minADE | Occupancy IoU-n. | Planning avg.Col. | Planning L2(3s) |
| :--------: | -------------: | ---------------: | ------------: | ---------------: | ----------------: | --------------- |
| **PyTorch**|          0.591 |            0.248 |        0.4448 |             66.9 |                 0 | 1.6950          |
| **Jittor** |          0.610 |            0.248 |        0.4558 |             73.9 |                 0 | 1.6969          |

These results demonstrate the capability of Jittor to achieve comparable, if not better, performance than PyTorch.

### Reference
1. [Jittor](https://github.com/Jittor/jittor)
2. [mmcv](https://github.com/open-mmlab/mmcv)
3. [UniAD](https://github.com/OpenDriveLab/UniAD)
