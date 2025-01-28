#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
#import torch

from datetime import datetime
from ffmpeg import FFmpeg, Progress
from pathlib import Path

from faster_whisper import WhisperModel
#from transformers import AutoTokenizer
#from transformers import AutoModelForCausalLM

#from transformers import LlamaTokenizer, LlamaForCausalLM

#from translang import TranslationService

from llama_cpp import Llama

input_video = "test.mp4"
input_video_name = "test"
output_video = "testnew.mkv"
output_video_extension = "mkv"


class Subtitle:
    def __init__(self, srt_file=None, string_stream=None):
        self.file_path = srt_file
        self.string_stream = string_stream
        self.main_lang_subs = None
        self.translated_subs = []

    @staticmethod
    def format_time_delta(time_delta):
        return time_delta.strip().replace(",", ":")

    def parse(self, file_stream=None):
        subtitles = []
        try:
            if file_stream is None:
                print(f"Parsing subtitles from input string")

                subtitle_line = [
                    line.split("\r\n") for line in self.string_stream.split("\n\n")
                ]
            else:
                print(f"Parsing subtitles from input file")

                subtitle_line = [line.split("\r\n") for line in file_stream.split("\n\n")]

            for item in subtitle_line:
                subtitle_list = [sub_list.split("\n") for sub_list in item]
                each_subtitle = [
                    sub for each_subtitle in subtitle_list for sub in each_subtitle
                ]

                if len(each_subtitle) > 1:
                    split_time = each_subtitle[1].split(" --> ")
                    subtitles.append(
                        {
                            "start_time": self.format_time_delta(split_time[0].strip()),
                            "end_time": self.format_time_delta(split_time[1].strip()),
                            "subtitle_text": " ".join(
                                subs.strip()
                                for subs in each_subtitle[2 : len(each_subtitle)]
                            ),
                        }
                    )

            return subtitles
        
        except ValueError as verr:
            logging.error(f"File format error {verr}")

    def open(self):
        print(f"Opening subtitles file {self.file_path}")

        if self.file_path.endswith(".srt"):
            try:
                p = Path(self.file_path)
                with p.open(mode="r") as f:
                    file_stream = f.read()
                    self.main_lang_subs = self.parse(file_stream)

            except FileNotFoundError as f_error:
                logging.error(f_error)
        else:
            print("Subtitles file must have '.srt' extension.")
            exit(1)

    def translate(self, output_language_name):
        num_subs = len(self.main_lang_subs)

        print("Loading model MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF quantized 8 bit: Mistral-7B-Instruct-v0.3.Q8_0.gguf\n")

        # Set the device (replace 'cuda:0' with the appropriate GPU if you have multiple GPUs)
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #####device = torch.device('cuda:0')
        # Set the device for PyTorch
        #torch.cuda.set_device(device)

        #torch.cuda.empty_cache()

        llm = Llama.from_pretrained(
            repo_id="MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF",
            #filename="Mistral-7B-Instruct-v0.3.Q4_K_M.gguf",
            filename="Mistral-7B-Instruct-v0.3.Q8_0.gguf",
            verbose=False,
            n_ctx=4096,
            n_gpu_layers=28, n_threads=12, n_batch=521,
        )

        vocabulary = {}
        reused_translations = 0
        reused_translations_percentage = 0.0

        print(f"Starting translation. Number of subtitles to translate: {num_subs}.")

        # record start time (in datetime format)
        start = datetime.now()
        
        sub_idx = 0
        for item in self.main_lang_subs:
            
            subtitle_text = item["subtitle_text"].replace('\n', ' ').replace('"', "").replace('♪', "").replace('🎵', "").replace('🎶', "").strip()
            
            if len(subtitle_text) > 0:
                if subtitle_text in vocabulary and vocabulary[subtitle_text] != None:

                    #print(f"  Reused translation: \"{subtitle_text}\" => \"{vocabulary[subtitle_text] }\"")

                    self.translated_subs.append(
                        {
                            "start_time": item["start_time"],
                            "end_time": item["end_time"],
                            "subtitle_text": vocabulary[subtitle_text],
                        }
                    )

                    reused_translations += 1
                    reused_translations_percentage = reused_translations / num_subs

                else:
                    try:
                        translation = llm.create_completion(
                            f"Translate into {output_language_name}: \"" + subtitle_text + 
                                f"\". Respond only with the phrase translated into {output_language_name}. No explainations, comments, requests, appraisals or notes, no pronounciation notes. Do not translate proper nouns. Please.", 
                            #f"Translate the following text from English into {output_language_name}.\nEnglish: " + item["subtitle_text"].replace('"', "").strip() + f".\n{output_language_name}:",
                            max_tokens=0,
                            temperature=0.0,
                            #suffix=None, max_tokens=16, temperature=0.8, top_p=0.95, min_p=0.05, typical_p=1.0, logprobs=None, echo=False, stop=[], frequency_penalty=0.0, presence_penalty=0.0, repeat_penalty=1.0, top_k=40, stream=False, seed=None, tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0, mirostat_eta=0.1, model=None, stopping_criteria=None, logits_processor=None, grammar=None, logit_bias=None
                        )

                        translation = translation["choices"][0]["text"].replace("\"", "").replace('🇮🇹', "").replace('♪', "").replace('🎵', "").replace('🎶', "").strip().rstrip().lstrip().split("\n")[0]

                        self.translated_subs.append(
                            {
                                "start_time": item["start_time"],
                                "end_time": item["end_time"],
                                "subtitle_text": translation,
                            }
                        )

                        vocabulary[subtitle_text] = translation

                    except:
                        print(f"\nignoring exception on subtitle {sub_idx}\n")

            # record end time
            end = datetime.now()
            elapsed = (end - start).total_seconds()

            sub_idx += 1
            percentage = sub_idx / num_subs * 100
            eta = elapsed / sub_idx * num_subs - elapsed

            print(f"\r   Translated {sub_idx}/{num_subs} - {percentage:.2f}% - ETA: {format_time(eta, millisecs = False)} - Reused {reused_translations} ({reused_translations_percentage:.2f})", end='\r', flush=True)

        print(f"\nTranslated {num_subs} subtitles into {output_language_name} in {format_time(elapsed, millisecs = False)}.\n")


    def write_translated_subtitle_file(self, output_file_name):

        print(f"Writing subtitles file {output_file_name}")

        text = ""
        for index, item in enumerate(self.translated_subs):
            text += f"{str(index+1)} \n"
            text += f"{item["start_time"]} --> {item["end_time"]} \n"
            text += f"{item["subtitle_text"]} \n"
            text += "\n"
            
        f = open(output_file_name, "w")
        f.write(text)
        f.close()


def extract_audio(input_video_file, input_language, output_audio_file):
    print(f"Extracting {input_language} audio from file {input_video_file} into {output_audio_file}")

    # record start time (in datetime format)
    start = datetime.now()

    ffmpeg = (
    FFmpeg()
        .option("y")
        .input(input_video_file)
        .output(
            output_audio_file,
            #{"c:a": "copy"},
            #map=[f"0:s:m:language:{input_language}"]
        )
    )

    @ffmpeg.on("progress")
    def on_progress(progress: Progress):
        print(progress, end='\r', flush=True)

    ffmpeg.execute()

    # record end time
    end = datetime.now()
    elapsed = (end - start).total_seconds()

    print(f"\nExtracted audio in {format_time(elapsed, millisecs = False)}.\n")


def transcribe(audio, original_language):
    print("Loading model openai/whisper-large-v3")
    #print("Using model openai/whisper-large-v3-turbo")

    #model = WhisperModel("small")
    #model = WhisperModel("large-v2")
    model = WhisperModel("large-v3")
    #model = WhisperModel("large-v3-turbo")

    #model = WhisperModel("large-v3", 
                         #compute_type="translate",
                         #return_timestamps=True,
                         #cpu_threads=12,
                         #num_workers=12,
    #                     )


    print(f"Starting transcription.\n")

    # record start time (in datetime format)
    start = datetime.now()

    segments, info = model.transcribe(
        audio, 
        #language=original_language[0:2].lower(),
        #temperature=0.0,
        )

    # record end time
    end = datetime.now()
    elapsed = (end - start).total_seconds()

    language = info.language

    segments = list(segments)
    #for segment in segments:
    #    print("[%.2fs -> %.2fs] %s" %
    #          (segment.start, segment.end, segment.text))
    
    print(f"\nTranscribed {len(segments)} phrases from {language} in {format_time(elapsed, millisecs = False)}.\n")

    return language, segments

def format_time(seconds, millisecs = True):

    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d}"
    
    if millisecs:
        milliseconds = round((seconds - math.floor(seconds)) * 1000)
        formatted_time += f",{milliseconds:03d}"

    return formatted_time

def generate_subtitle_file(segments, output_subtitle_file):
    print(f"Writing subtitles to file {output_subtitle_file}")

    text = ""
    for index, segment in enumerate(segments):
        segment_start = format_time(segment.start)
        segment_end = format_time(segment.end)
        text += f"{str(index+1)} \n"
        text += f"{segment_start} --> {segment_end} \n"
        text += f"{segment.text} \n"
        text += "\n"
        
    f = open(output_subtitle_file, "w")
    f.write(text)
    f.close()

def extract_subtitle(input_video, output_srt_file, input_language):
    print(f"Extracting subtitles from {input_video} to {output_srt_file} for language {input_language}")

    ffmpeg = (
        FFmpeg()
            .option("y")
            .input(input_video)
            .output(
                output_srt_file,
                {"c": "copy"},
                map=[f"0:s:m:language:{input_language}"]
            )
        )

    @ffmpeg.on("progress")
    def on_progress(progress: Progress):
        print(progress, end='\r', flush=True)

    ffmpeg.execute()


def add_subtitle_to_video(input_video, original_subtitle_file, original_language, original_language_name,
            output_video_file, translated_subtitle_file, translated_language, translated_language_name):

    output_file_split = split_path(output_video_file)
    if output_file_split[1] == '.mp4':
        sub_encoder = "mov_text"
    else:
        sub_encoder = "srt"

    # record start time (in datetime format)
    start = datetime.now()

    if original_subtitle_file != None and translated_subtitle_file != None:
        print(f"adding '{original_language_name}' and '{translated_language_name}' subtitles from '{original_subtitle_file}' and '{translated_subtitle_file}' to '{input_video}' into '{output_video_file}'")

        ffmpeg = (
            FFmpeg()
                .option("y")
                .input(input_video)
                .input(original_subtitle_file)
                .input(translated_subtitle_file)
                .output(
                    output_video_file,
                    {"c:v": "copy", "c:a": "copy", "c:s": sub_encoder, 
                     "metadata:s:s:0": f"language={original_language}", 
                     "metadata:s:s:1": f"language={translated_language}"
                    },
                    map=["0:v", "0:a?", "1", "2"],
                )
            )

    else:
        if original_subtitle_file != None:
            print(f"adding '{original_language_name}' subtitles from '{original_subtitle_file}' to '{input_video}' into '{output_video_file}'")

            ffmpeg = (
                FFmpeg()
                    .option("y")
                    .input(input_video)
                    .input(original_subtitle_file)
                    .output(
                        output_video_file,
                        {"c:v": "copy", "c:a": "copy", "c:s": sub_encoder, 
                         "metadata:s:s:0": f"language={original_language}"
                        },
                        map=["0:v", "0:a?", "1"],
                    )
                )
        
        elif translated_subtitle_file != None:
            print(f"adding '{translated_language_name}' subtitles from '{translated_subtitle_file}' to '{input_video}' into '{output_video_file}'")

            ffmpeg = (
                FFmpeg()
                    .option("y")
                    .input(input_video)
                    .input(translated_subtitle_file)
                    .output(
                        output_video_file,
                        {"c:v": "copy", "c:a": "copy", "c:s": sub_encoder, 
                         "metadata:s:s:0": f"language={translated_language}"
                        },
                        map=["0:v", "0:a?", "1"],
                    )
                )

        else:
            print("At least one between the original or the translated subtitle files must be provided")
            exit(1)

    @ffmpeg.on("progress")
    def on_progress(progress: Progress):
        print(progress, end='\r', flush=True)

    ffmpeg.execute()

    # record end time
    end = datetime.now()
    elapsed = (end - start).total_seconds()
    
    print(f"\nAdded subtitles in {format_time(elapsed, millisecs = False)}.\n")


def parse_args():
    parser = argparse.ArgumentParser(description ='Extract or generate, translate, add subtitles')
    
    parser.add_argument('-iv', metavar ='<input_video>', dest ='input_video', 
                        help ='input video file. It may contain subtitles to extract if the -extract language option is used.')

    parser.add_argument('-g', '--generate', dest ='generate', action='store_true',
                        help ='generate subtitles from the <input_video> speech (audio), assuming they are speaking <input_language>.')

    parser.add_argument('-e', '--extract', dest ='extract', action='store_true',
                        help ='extract subtitle of <input_language> from the <input_video>.')

    parser.add_argument('-t', '--translate', dest ='translate', action='store_true',
                        help ='translate subtitles.')

    parser.add_argument('-is', metavar ='<input_subtitle_file>', dest ='input_subtitle_file', 
                        help ='take the specified subtitle file as the one to be translated')

    parser.add_argument('-il', metavar ='<input_language>', dest ='input_language', 
                        help ='specifies the language of the input subtitles (three letters, e.g. "eng", "ita", ...).')

    parser.add_argument('--ilname', metavar ='<input_language_full_name>', dest ='input_language_name', 
                        help ='provides the full name of the input (original) language. E.g. if the -il value is "eng", specify "English" here.')

    parser.add_argument('-ol', metavar ='<output_language>', dest ='output_language', 
                        help ='translate the subtitle into the specified language (three letters, e.g. "eng", "ita", ...).')

    parser.add_argument('--olname', metavar ='<output_language_full_name>', dest ='output_language_name', 
                        help ='provides the full name of the output language. E.g. if the -ol value is "eng", specify "English" here.')

    parser.add_argument('-ov', metavar ='<output_video_file>', dest ='output_video_file', 
                        help ='video file that will contain the original and the translated subtitles')

    parser.add_argument('-oos', metavar ='<output_original_subtitle_file>', dest ='output_original_subtitle_file', 
                        help ='text file that contains or will contain the original subtitles')

    parser.add_argument('-ots', metavar ='<output_translated_subtitle_file>', dest ='output_translated_subtitle_file', 
                        help ='text file that contains or will contain the translated subtitles')

    args = parser.parse_args()

    return args

def split_path(s):
    p = os.path.splitext(s)
    
    return (p[0].split(os.pathsep)[-1], p[1])

def run():
    args = parse_args()

    if args.input_video  == None and args.input_subtitle_file == None:
        print("An input video or an input original subtitle file is required.")
        exit(1)

    if args.generate:
        if args.input_video == None:
            print("--generate requires to specify the input video")
            exit(1)

        if args.input_language == None or args.input_language_name == None:
            print("--generate requires to specify the input language and language name")
            exit(1)

        if args.extract:
            print("--generate is alternative to --extract. If the video already contains the desired subtitles, just use --extract")
            exit(1)

        output_audio_file = f"audio-{args.input_video}.wav"

        extract_audio(args.input_video, args.input_language, output_audio_file)

        if args.output_original_subtitle_file == None:
            input_video_split = split_path(args.input_video)
            args.output_original_subtitle_file = input_video_split[0] + ".srt"

        language, segments = transcribe(output_audio_file, args.input_language)

        generate_subtitle_file(segments, args.output_original_subtitle_file)

        args.input_subtitle_file = args.output_original_subtitle_file

    elif args.extract:
        if args.input_video == None:
            print("--extract requires to specify the input video")
            exit(1)
    
        if args.input_language == None or args.input_language_name == None:
            print("--extract requires to specify the input language and language name")
            exit(1)

        if args.output_original_subtitle_file == None:
            input_video_split = split_path(args.input_video)

            args.output_original_subtitle_file = input_video_split[0] + ".srt"

        if args.input_subtitle_file == None:
            args.input_subtitle_file = args.output_original_subtitle_file

        extract_subtitle(args.input_video, args.output_original_subtitle_file, args.input_language)

    if args.input_subtitle_file == None:
        print("no original subtitles file could be found or generated.")
        exit(1)

    if args.translate:
        if args.input_language == None:
            print("translation requires to specify the input language")
            exit(1)

        if args.output_language == None or args.output_language_name == None:
            print("The output language and output language name are required")
            exit(1)

        if args.output_translated_subtitle_file == None:
            input_subtitle_file_split = split_path(args.input_subtitle_file)
            args.output_translated_subtitle_file = input_subtitle_file_split[0] + "-" + args.output_language + ".srt"

        original_srt = Subtitle(args.input_subtitle_file)
        original_srt.open()

        #print(srt.main_lang_subs)

        original_srt.translate(args.output_language_name)
        original_srt.write_translated_subtitle_file(args.output_translated_subtitle_file)

        #print(srt.translated_subs)

    if args.output_video_file != None:
        if args.input_video == None:
            print("An input video is required.")
            exit(1)

        if args.input_language == None or args.input_language_name == None:
            print("The input language and input language name are required")
            exit(1)

        if args.output_language  == None or args.output_language_name == None:
            print("The output language and output language name are required")
            exit(1)

        if args.output_original_subtitle_file  == None and args.input_subtitle_file == None:
            print("The subtitles file for the original language is required")
            exit(1)

        add_subtitle_to_video(
            args.input_video, args.input_subtitle_file, args.input_language, args.input_language_name, 
            args.output_video_file, args.output_translated_subtitle_file, args.output_language, args.output_language_name)

run()
