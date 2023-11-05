# EfficientPose

Signle person pose estimation using webcam.

## Installation
```conda create --name efficientpose python=3.8 -y```

```conda activate efficientpose```

```git clone https://github.com/rahul-t-p/EfficientPose.git```

```cd EfficientPose/```

```pip3 install -r requirements.txt```

## Usage
```
usage: demo_webcam.py [-h] [--model_version MODEL_VERSION]

optional arguments:
  -h, --help            show this help message and exit
  --model_version MODEL_VERSION
                        The version of the EfficientPose to use -> RT_LITE (default) / I_LITE / II_LITE
```
Example usage;
```python3 demo_webcam.py --model_version I_LITE```

## References
1. https://github.com/daniegr/EfficientPose
2. https://github.com/PINTO0309/PINTO_model_zoo/tree/main/084_EfficientPose
