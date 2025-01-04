import time
import math
import ffmpeg
import os
import sys

from faster_whisper import WhisperModel

input_video = "test.mp4"
input_video_name = "test"
output_video = "testnew.mkv"
output_video_extension = "mkv"

def extract_audio():
    extracted_audio = f"audio-{input_video_name}.wav"
    stream = ffmpeg.input(input_video)
    stream = ffmpeg.output(stream, extracted_audio)
    ffmpeg.run(stream, overwrite_output=True)
    return extracted_audio

def transcribe(audio):
    model = WhisperModel("small")
    segments, info = model.transcribe(audio)
    language = info.language
    print("Transcription language", language)
    segments = list(segments)
    for segment in segments:
        # print(segment)
        print("[%.2fs -> %.2fs] %s" %
              (segment.start, segment.end, segment.text))
    return language, segments


def format_time(seconds):

    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"

    return formatted_time

def generate_subtitle_file(language, segments):

    subtitle_file = f"sub-{input_video_name}.{language}.srt"
    text = ""
    for index, segment in enumerate(segments):
        segment_start = format_time(segment.start)
        segment_end = format_time(segment.end)
        text += f"{str(index+1)} \n"
        text += f"{segment_start} --> {segment_end} \n"
        text += f"{segment.text} \n"
        text += "\n"
        
    f = open(subtitle_file, "w")
    f.write(text)
    f.close()

    return subtitle_file


def add_subtitle_to_video(soft_subtitle, subtitle_file,  subtitle_language):

    video_input_stream = ffmpeg.input(input_video)
    subtitle_input_stream = ffmpeg.input(subtitle_file)
    #output_video = f"output-{input_video_name}.mp4"
    #subtitle_track_title = subtitle_file.replace(".srt", "")

    if output_extension == 'mp4':
        sub_encoder = "mov_text"
    else:
        sub_encoder = "srt"

    if subtitle_language == "en":
        subtitle_language = "English"

    if soft_subtitle:
        stream = ffmpeg.output(
            video_input_stream, subtitle_input_stream, output_video, **{"c": "copy", "c:s": sub_encoder},
            **{"metadata:s:s:0": f"language={subtitle_language}"}
        )
        ffmpeg.run(stream, overwrite_output=True)
    else:
        stream = ffmpeg.output(video_input_stream, output_video,
            vf=f"subtitles={subtitle_file}")

        ffmpeg.run(stream, overwrite_output=True)


def run():
    # total arguments
    n = len(sys.argv)

    if n < 3:
        print("\nInsufficient number of arguments: required 2, have " + str(n-1))
        print("\nUsage: python3 " + sys.argv[0] + " inputfilename.ext outputfilename.ext\n")

        exit(-1)

    # Rewrite global varible values, do not redefine local variables with same name
    global input_video
    global input_video_name
    global output_video
    global output_extension

    input_video = sys.argv[1]

    input_base, extension = os.path.splitext(input_video)

    input_video_name = input_base

    output_video = sys.argv[2]

    output_base, output_extension = os.path.splitext(output_video)

    print(f"\nExtracting audio from {input_video} and adding subtitles into {output_video}\n\n")

    extracted_audio = extract_audio()
    language, segments = transcribe(audio=extracted_audio)
    subtitle_file = generate_subtitle_file(
            language=language,
            segments=segments
        )
    
    add_subtitle_to_video(
            soft_subtitle=True,
            subtitle_file=subtitle_file,
            subtitle_language=language
        )

run()
