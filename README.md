# selfdrive
Perception

## Setup [MaskRCNN](https://github.com/matterport/Mask_RCNN#4k-video-demo-by-karol-majek)
I installed MaskRCNN with Python 3.6, tensorflow-gpu 1.14.0, CUDA 10.0, keras 2.3.1 on Ubuntu 16.04.

Below is a detailed instruction of installing MaskRCNN on my local desktop,
1. Create conda environment for this project
```
conda create -n perception python=3.6
conda activate perception
```

2. Clone MaskRCNN repository
```
git clone https://github.com/matterport/Mask_RCNN
cd Mask_RCNN
```

3. Install prerequistes for tensorflow-gpu by following https://www.tensorflow.org/install/gpu#linux_setup.
In my case, I have installed CUDA 10 on Ubuntu 16.04.

**Note: please make sure your cuda version and tensorflow version are compatible. 
To check compatibility, see: https://www.tensorflow.org/install/source#tested_build_configurations**

4. Install dependencies inside `perception` environment

First, replace `requirements.txt` with following file
```
numpy
scipy
Pillow
cython
matplotlib
scikit-image
tensorflow==1.14.0
keras==2.3.1
opencv-python
h5py
imgaug
IPython[all]
```
Then run `pip` inside `perception` environment
```
pip install -r requirements.txt
```

5. Run setup.py from the MaskRCNN root directory
```
python3 setup.py install
```

6. To train or test on MS COCO dataset, install `pycocotools` from https://github.com/waleedka/coco.

```
cd ..
conda activate perception
git clone https://github.com/waleedka/coco.git
cd coco/PythonAPI
make
make install
```

7. Then test your installation by running `samples/demo.ipynb` and `samples/shapes/train_shapes.ipynb`. 
They both should be run without error. The `train_shape.ipynb` should use GPU to train by default. 

8. Note that every time you want to modify installed code, run step 5 again to update installed package.

