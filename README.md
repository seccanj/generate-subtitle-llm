# Generate subtitles for video, using Whisper OpenAI LLM and ffmpeg

1) Clone this repository and enter into the main directory

2) Read article that formed the source of this tool:
https://www.digitalocean.com/community/tutorials/how-to-generate-and-add-subtitles-to-videos-using-python-openai-whisper-and-ffmpeg

3) Perform preliminary steps described in article (install python, setup virtual env, run virtual env, create requirements.txt file)

```
# Summary
sudo apt install python3.12-venv
python3 -m venv env
source env/bin/activate
pip3 install faster-whisper ffmpeg-python
```

3) Install NVIDIA CUDA dev tools and deep learning toolkit:

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions

https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html

```
# Summary
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit
sudo apt-get -y install cudnn9-cuda-12
sudo apt install nvidia-cuda-toolkit
```

4) Perform NVIDIA post-installation steps:

https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html

```
# Summary
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
```

5) Install Python wheels for CUDA:
https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html#python-wheels-linux-installation

```
# Summary
python3 -m pip install --upgrade pip wheel
python3 -m pip install nvidia-cudnn-cu12
```

6) Create requirements.txt file for python dependencies:

```
pip3 freeze > requirements.txt
```

7) Setup environment vars:

```
export LD_LIBRARY_PATH=${PWD}/env/lib64/python3.12/site-packages/nvidia/cublas/lib:${PWD}/env/lib64/python3.12/site-packages/nvidia/cudnn/lib
```

Most environment setup is in the file "source-me.sh". 
To activate it, use this command:
```
source source-me.sh
```

8) Copy the input video file into this project's directory

9) Run the program:
```
python3 main.py input-filename.mp4 output-filename.mkv
```

All input video formats are supported.
Supported output video formats are "mp4" and "mkv". 

10) Optional: Translate english subtitles into your language, uploading the ".srt" file into this web app:

https://www.syedgakbar.com/projects/dst

11) Add your own-language subtitle to the other subtitles into a new video file (mkv):

```
ffmpeg -i input-with-english-subtitles.mp4 -f srt -sub_charenc UTF-8 -i subtitle-your-language.srt -map 0:v -map 0:a -map 0:s -map 1 -metadata:s:s:0 language=English -metadata:s:s:1 language=Yourlanguage -c:v copy -c:a copy -c:s srt video_NEW.mkv
```

