## NTrack: A Multiple-Object Tracker and Dataset for Infield Cotton Boll Counting

### Overview

In agriculture, automating the accurate tracking of fruits, vegetables, and
fiber is a very tough problem. The issue becomes extremely challenging in
dynamic field environments. Yet, this information is critical for making
day-to-day agricultural decisions, assisting breeding programs, and much more.

<p align="center">
  <img src="images/overview.png" alt="overview" width="400"/>
</p>

This repository provides source code for our 2023 IEEE Transactions on
Automation Science and Engineering article titled "[NTrack: A Multiple-Object
Tracker and Dataset for Infield Cotton Boll Counting]()." NTrack, is a multiple
object tracking framework based on the linear relationship between the
locations of neighboring tracks. It computes dense optical flow and utilizes
particle filtering to guide each tracker. Correspondences between detections
and tracks are found through data association via direct observations and
indirect cues, which are then combined to obtain an updated observation. NTrack
is independent of the underlying detection method, thus allowing for the
interchangeable use of any off-the-shelf object detector.

NTrack was created for the task of tracking and counting infield cotton bolls.
To develop and test NTrack, we created TexCot22, an infield cotton boll video
dataset. Each tracking sequence was collected from unique rows of an outdoor
cotton crop research plot located in the High Plains region of Texas. More
information on the TexCot22 dataset can be found on the [NTrack
website](https://robotic-vision-lab.github.io/ntrack).

### NTrack Pipeline 

<p align="center">
  <img src="images/ntrack_pipeline.png" alt="model_architecture" width="800"/>
</p>

### Setup

Python 3 dependencies:
    
    * argparse
    * lap
    * pycocotolls
    * opencv
    * tqdm
    * tensorpack
    * scikit-image
    * scikit-learn
    * filterpy
    * pysyaml
    * motmetrics

### Setup environment

To setup a conda environment:
```
conda create --name ntrack python=3.8.
conda activate ntrack
pip install -r requirements.txt
```
### Setup data directory

You can download the TexCot22 dataset from [here](). Unzip the data. The data folder should
have the following structure:

```
cot22_base/
    - train
        - vid14_02
        - vid23_04
        - ...
        - ...
    - test
        - vid09_01
            - det
            - gt
            - img1
            - seqinfo.ini
        - vid09_02
        - ...
        - ...
```

## Running code

Here we show how to run our code on test data split.
```
python ntrack.py --data_base_dir {your/data/dir/cot22_base} --data_split test --use_pf True
```
