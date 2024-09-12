from setuptools import setup,find_packages
import os
path = os.path.join(os.path.dirname(__file__))

with open(os.path.join(path, "jtmmcv/__init__.py"), "r", encoding='utf8') as fh:
    for line in fh:
        if line.startswith('__version__'):
            version = line.split("'")[1]
            break
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name="jtmmcv",
    version=version,
    author="Jittor Group",
    author_email="jittor@qq.com",
    description="mmcv in Jittor ",
    url="http://jittor.com",
    python_requires='>=3.7',
    packages=find_packages(path),
    package_dir={'./': 'jtmmcv'},
    install_requires=[
        "numpy",
        "pillow",
        "jittor",
        "tensorboardX",
        "opencv-python",
        "terminaltables",
    ],
)