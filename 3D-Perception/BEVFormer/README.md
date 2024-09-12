# Jittor-BEVFormer

<div><video controls src="https://user-images.githubusercontent.com/27915819/161392594-fc0082f7-5c37-4919-830a-2dd423c1d025.mp4" muted="false"></video></div>



## Introduction


Jittor-BEVFormer is an open-source [Jittor](https://github.com/Jittor/jittor) implementation of [BEVFormer](https://arxiv.org/abs/2203.17270) - a state-of-the-art method for camera-only multi-view representation.


## Environment Configuration

Jittor-BEVFormer is developed and tested under the environment:

- **Operating System:** Linux
- **Python Version:** 3.8
- **Compiler:** G++ 9.4
- **CUDA Version**: 11.3
- **MPI Version:** 5.0.3 (for multi-processing support)

## Installation

Follow these steps to set up the Jittor-BEVFormer environment (It could share the same conda environment with Jittor-UniAD):

1. **Clone the repository:**

   ```shell
   git clone ***
   cd 3D-Perception/BEVFormer
   ```

2. **Create a conda virtual environment and activate it:**

   ```shell
   conda create -n jittor_bevformer python=3.8 -y
   conda activate jittor_bevformer 
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

   ```shell
   cd adzoo/bevformer
   mkdir ckpts
   ```

   Download the checkpoints from ([GoogleDrive](https://drive.google.com/file/d/1AKs6Xp3GT_x04Z1fBaDzOO61dCBAwAlG/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1doRXhunjWwAi3U3MWH_FJg?pwd=yfrc)) or from [BEVFormer official repo](https://github.com/fundamentalvision/BEVFormer) and place them in the `ckpts` directory as shown below:
   ```
   ├── ckpts
       ├── bevformer_r101_dcn_24ep.pth
   ```

## Inference

To evaluate the performance of the models, you can run inference for 3D object detection as follows:

```shell
# Under the highest directory 3D-Perception/BEVFormer. The first step could be slow since Jittor backend is compiling all operators.
./adzoo/bevformer/bevformer_jittor_test.sh adzoo/bevformer/configs/bevformer/bevformer_base.py adzoo/bevformer/ckpts/bevformer_r101_dcn_24ep.pth 1
```



## Results


We compared the performance of Jittor and PyTorch implementations on the nuScenes v1.0-mini dataset. The results are summarized below:

|        | mAP    | mATE   | mASE   | mAOE   | mAVE   | mAAE   | NDS    |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| PyTorch  | 0.3767 | 0.7225 | 0.4625 | 0.6084 | 0.5343 | 0.2994 | 0.4256 |
| Jittor | 0.3766| 0.7229 | 0.4626 | 0.6096 | 0.5337 | 0.2994 | 0.4255 |

These results demonstrate the capability of Jittor to achieve comparable, if not better, performance than PyTorch.

### Reference
1. [Jittor](https://github.com/Jittor/jittor)
2. [mmcv](https://github.com/open-mmlab/mmcv)
3. [BEVFormer](https://github.com/fundamentalvision/BEVFormer)