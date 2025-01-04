1) Read article:  https://www.digitalocean.com/community/tutorials/how-to-generate-and-add-subtitles-to-videos-using-python-openai-whisper-and-ffmpeg

2) Perform preliminary steps described in article (install python, setup virtual env, run virtual env, create requirements.txt file)

sudo apt install python3.12-venv
mkdir generate-subtitle
cd generate-subtitle/
python3 -m venv env
source env/bin/activate
pip3 install faster-whisper ffmpeg-python

3) Install NVIDIA CUDA dev tools and deep learning toolkit:

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions

https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit
sudo apt-get -y install cudnn9-cuda-12
sudo apt install nvidia-cuda-toolkit

4) Perform NVIDIA post-installation steps:

https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html

export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}

5) Install Python wheels for CUDA:

https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html#python-wheels-linux-installation

python3 -m pip install --upgrade pip wheel
python3 -m pip install nvidia-cudnn-cu12

6) Create requirments.tx ile fopython dependencies:

pip3 freeze > requirements.txt

7) Setup environment vars:

export LD_LIBRARY_PATH=${PWD}/env/lib64/python3.12/site-packages/nvidia/cublas/lib:${PWD}/env/lib64/python3.12/site-packages/nvidia/cudnn/lib

8) Copy the video file into this projet's directory

9) Run the program:

python3 main.py input-filename.mp4 output-filename.mkv

10) Translate english subtitles into italian, uploading the ".srt" file into this web app:

https://www.syedgakbar.com/projects/dst

11) Add italian subtitle to other subtitles in a new video file (mkv):

ffmpeg -i input-with-english-subtitles.mp4 -f srt -sub_charenc UTF-8 -i subtitle-it.srt -map 0:v -map 0:a -map 0:s -map 1 -metadata:s:s:0 language=English -metadata:s:s:1 language=Italian -c:v copy -c:a copy -c:s srt video_NEW.mkv

