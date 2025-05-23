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
Tracker and Dataset for Infield Cotton Boll
Counting](https://arxiv.org/pdf/2312.10922.pdf)." NTrack, is a multiple object
tracking framework based on the linear relationship between the locations of
neighboring tracks. It computes dense optical flow and utilizes particle
filtering to guide each tracker. Correspondences between detections and tracks
are found through data association via direct observations and indirect cues,
which are then combined to obtain an updated observation. NTrack is independent
of the underlying detection method, thus allowing for the interchangeable use
of any off-the-shelf object detector. NTrack was created for the task of
tracking and counting infield cotton bolls. To develop and test NTrack, we
created TexCot22, an infield cotton boll video dataset. Each tracking sequence
was collected from unique rows of an outdoor cotton crop research plot located
in the High Plains region of Texas. 

More information on the project can be found on the 
[NTrack website](https://robotic-vision-lab.github.io/ntrack).

### Citation

If you find this project useful, then please consider citing both our paper and
dataset.

```bibitex
@article{muzaddid2024ntrack,
  title={NTrack: A Multiple-Object Tracker and Dataset for Infield Cotton Boll Counting},
  author={Muzaddid, Md Ahmed Al and Beksi, William J},
  journal={IEEE Transactions on Automation Science and Engineering},
  volume={21},
  number={9},
  pages={7452--7464},
  doi={10.1109/TASE.2023.3342791},
  year={2024}
}

@data{T8/5M9NCI_2024,
  title={{TexCot22}},
  author={Muzaddid, Md Ahmed Al and Beksi, William J},
  publisher={Texas Data Repository},
  version={V2},
  url={https://doi.org/10.18738/T8/5M9NCI},
  doi={10.18738/T8/5M9NCI},
  year={2024}
}
```

### NTrack Pipeline 

<p align="center">
  <img src="images/ntrack_pipeline.png" alt="model_architecture" width="800"/>
</p>

### Installation 

First, begin by cloning the project:

    $ git clone https://github.com/robotic-vision-lab/NTrack-A-Multiple-Object-Tracker.git
    $ cd NTrack-A-Multiple-Object-Tracker

Next, create an environment and install the dependencies:

    $ conda create --name ntrack python=3.8.
    $ conda activate ntrack
    $ pip install -r requirements.txt

### Dataset 

Download the [TexCot22](https://doi.org/10.18738/T8/5M9NCI) dataset files.
TexCot22 consists of five ZIP files for tracking (TexCot22-[1-5]). Each ZIP file
contains training and testing sequences, ground-truth bounding boxes, and the
detections. The dataset directories have the following structure:

```
TexCot22-X/
  - train
    - vid
      - img1
        - 0001.jpg
        - 0002.jpg
        - ...
      - gt
        - gt.txt
    - ... 
  - test
    - vid
      - img1
        - 0001.jpg
        - 0002.jpg
        - ...
      - det
        - det.txt
      - gt
        - gt.txt
    - ...
```

In addition, there are four ZIP files (TexCot22_Detection-1_[1-2],
TexCot22_Detection-2_[1-2]), which can be used to train a detection model.  To
train an off-the-shelf object detector, unzip TexCot22_Detection-X_1.zip and
TexCot22_Detection-X_2.zip into the same directory and then merge the two img
directories into one. The file frameid_to_imgfile.npy contains the mapping from
the frameid (1st column in detection.csv) to the image name in the img folder. 

### Usage 

To run on a test data split, invoke the following command: 

    $ python ntrack.py --data_base_dir {your/data/dir/TexCot22-X} --data_split test --use_pf True


### NTrack Source Code License

[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/robotic-vision-lab/NTrack-A-Multiple-Object-Tracker/blob/main/LICENSE)

### TexCot22 Dataset License

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
