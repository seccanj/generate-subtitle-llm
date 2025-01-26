# Subtitle tool
A tool that can:
 * Extract audio from a video, transcribe it and generate a subtitle file.
 * Extract existing soft subtitles from a video and save them to an .srt file.
 * Translate subtitles from an srt file to another language and save them into another .srt file.
 * Add subtitles from an .srt file to a video as soft subtitles.
 * One, some or all the above steps in any combination.

Typical use cases:
 1. Extract audio from a video that has no subtitles, generate the original language subtitles, translate them into another language, add the additional subtitles to a copy of the video.
 2. Extract existing subtitles from a video, translate them into another language, add the additional subtitles to a copy of the video.
 3. You have manually downloaded subtitles for some video and want to add them as soft subtitles to the video file.

The tool uses the following Large Language Models (LLM) from Hugging Face:

 * For speech-to-text, Whisper LLM:  https://huggingface.co/openai/whisper-large-v3
 * For translation, Mistral LLM:  https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF

The tool runs much better with an NVidia graphic card that supports CUDA.

# Setup for Linux

1) Clone this repository and enter into the main directory

2) Perform preliminary steps: install python, setup virtual env, run virtual env, install the Whisper LLM for speech-to-text transcription:
```
# Summary
sudo apt install python3.12-venv
python3 -m venv env
source env/bin/activate
python3 -m pip install --upgrade pip wheel
pip install faster-whisper python-ffmpeg
```

3) Assuming you have an NVIDIA graphical card with CUDA support, install NVIDIA CUDA dev tools and deep learning toolkit (update the package versions in these commands with the latest available ones):

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

4) Perform NVIDIA post-installation steps (update the CUDA version in these commands with the latest available ones):
https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html

```
# Summary
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
```

5) Install Python wheels for CUDA:
https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html#python-wheels-linux-installation

```
# Summary
pip install nvidia-cudnn-cu12
```

6) Install Python integration with Hugging Face:
https://huggingface.co/docs/huggingface_hub/en/guides/cli

```
# Summary
pip install -U "huggingface_hub[cli]"
```

7) Install llama-cpp-python with CUDA support:
https://github.com/abetlen/llama-cpp-python

```
# Summary for Linux
CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

8) Create requirements.txt file for python dependencies:

```
pip3 freeze > requirements.txt
```

9) Setup environment vars (update the python version in these vars with your specific version and "${PWD}/env" with the path to your python virtual environment):

```
export LD_LIBRARY_PATH=${PWD}/env/lib64/python3.12/site-packages/nvidia/cublas/lib:${PWD}/env/lib64/python3.12/site-packages/nvidia/cudnn/lib
export LLAMA_CPP_LIB=${PWD}/env/lib/python3.12/site-packages/llama_cpp/lib/libllama.so
```

Most environment setup is in the file "source-me.sh". 
To activate it, use this command (after checking the paths in the script):
```
source source-me.sh
```

10) Run the program to display usage information:
```
python3 subtitle_tool.py --help

usage: subtitle_tool.py [-h] [-iv <input_video>] [-g] [-e] [-t] [-is <input_subtitle_file>] [-il <input_language>] [--ilname <input_language_full_name>] [-ol <output_language>]
                        [--olname <output_language_full_name>] [-ov <output_video_file>] [-oos <output_original_subtitle_file>] [-ots <output_translated_subtitle_file>]

Extract or generate, translate, add subtitles

options:
  -h, --help            show this help message and exit
  -iv <input_video>     input video file. It may contain subtitles to extract if the -extract language option is used.
  -g, --generate        generate subtitles from the <input_video> speech (audio), assuming they are speaking <input_language>.
  -e, --extract         extract subtitle of <input_language> from the <input_video>.
  -t, --translate       translate subtitles.
  -is <input_subtitle_file>
                        take the specified subtitle file as the one to be translated
  -il <input_language>  specifies the language of the input subtitles (three letters, e.g. "eng", "ita", ...).
  --ilname <input_language_full_name>
                        provides the full name of the input (original) language. E.g. if the -il value is "eng", specify "English" here.
  -ol <output_language>
                        translate the subtitle into the specified language (three letters, e.g. "eng", "ita", ...).
  --olname <output_language_full_name>
                        provides the full name of the output language. E.g. if the -ol value is "eng", specify "English" here.
  -ov <output_video_file>
                        video file that will contain the original and the translated subtitles
  -oos <output_original_subtitle_file>
                        text file that contains or will contain the original subtitles
  -ots <output_translated_subtitle_file>
                        text file that contains or will contain the translated subtitles
```

## Note
All input video formats are supported.
Supported output video formats are "mp4" and "mkv". 

# Sample usages

## Video with no subtitles: generate from audio and translate into Italian
```
python3 subtitle_tool.py -iv input_video.mkv -il eng --ilname English --generate --translate -ol ita --olname Italian -ov 'video-sub-ita.mkv'
```

## Video with subtitles: extract and translate into Italian
```
python3 subtitle_tool.py -iv input_video.mkv -il eng --ilname English --extract --translate -ol ita --olname Italian -ov 'video-sub-ita.mkv'
```

## Add existing subtitle file to video
```
python3 subtitle_tool.py -iv input_video.mkv -is subtitles.srt -il ita --ilname Italian -ol ita --olname Italian -ov 'video-sub-ita.mkv'
```

# Useful commands

## Monitor the NVidia GPU usage
```
watch -d nvidia-smi
```

## Delete AI models previously downloaded from Hugginface to local cache
This command allows for an interactive selection of the models to be deleted:

```
huggingface-cli delete-cache

```

# Credits

Thanks to Carlos Samito Francisco Mucuho and Anish Singh Walia for this article:
    https://www.digitalocean.com/community/tutorials/how-to-generate-and-add-subtitles-to-videos-using-python-openai-whisper-and-ffmpeg

Thanks to Muideen (mdauthentic on GitHub) for the basis of the Subtitle class:
    https://github.com/mdauthentic/soustitle-py

