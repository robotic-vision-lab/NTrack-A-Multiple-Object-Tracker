import os
img_base_dir = "/home/ahmed/Dropbox/deep_learning/code/python/PhD/RVL/cotton/Tracking/project_cottonboll_counting/paper_writing/TASE/figures/qualitative_comp"
for filename in os.listdir(img_base_dir):
    os.system(f"ffmpeg -y -i {os.path.join(img_base_dir,filename)} -qscale:v 25 {os.path.join(img_base_dir,filename)}")
