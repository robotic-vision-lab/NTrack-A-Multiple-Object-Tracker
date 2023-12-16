## NTrack: A Multiple-Object Tracker and Dataset for Infield Cotton Boll Counting

### Overview

### [Project Page]( https://robotic-vision-lab.github.io/ntrack/) | [Presentation](https://www.youtube.com/watch?v=VTUNa2EoG0U) | [Data]()
In the paper we introduced NTrack, a new multiple object tracking framework that relies on
the linear relationship between neighboring tracks. To guide each tracker, NTrack uses dense
optical flow computation and particle filtering. NTrack establishs connections between object
detections and existing tracks through data association using both direct observations and
indirect cues. These associations are then combined to create an updated observation.
<div align=center>
    <img src='images/ntrack_pipeline.png'/>
</div>


## Setup

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

## Setup environment

To setup a conda environment:
```
conda create --name ntrack python=3.8.
conda activate ntrack
pip install -r requirements.txt
```
## Setup data directory
You can download the cot22 dataset from [here](). Unzip the data. The data folder should
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
## Acknowledgement
This repo is highly based on [ByteTrack](https://github.com/ifzhang/ByteTrack), thanks for their excellent work.

## Citation

```
@article{ahmed2023ntrack,
  author    = {Md Ahmed Al Muzaddid, William Beksi},
  title     = {NTrack: A Multiple-Object Tracker and Dataset for Infield Cotton Boll Counting},
  journal   = {T-ASE},
  year      = {2023}
```
