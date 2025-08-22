# VibeClipper
VibeClipper is a simple AI clip detecting tool that utilizes Whisper transcription and LLMs.

It works as follows.
1. The user provides an MP4 or SRT.
2. If the file is an MP4 it transcribes the file. If the user sets the -srt flag and uses an SRT file, skip this step.
3. Then the script breaks the SRT down into the individual lines.
4. The LLM is first asked to answer a simple true/false question about the 'vibe' of the line.
5. The LLM is then asked to gauge the vibe from 0 to 100% as an integer.
6. Based upon the user's setting the line is saved in the SRT output.
7. The script outputs an SRT in the same directory as the input after it is done checking each line.

## Options
*--is_srt*/*-srt*, This flag allows the user to pass in an SRT instead of a video file, skipping the whisper transcription.

*--whisper_model*/*-wm*, Set the whisper model size, default 'medium'.

*--llm_repo*/*-repo*, Set the repo of the LLM on Huggingface.

*--llm_model*/*-llm*, Set the filename of the LLM in the repo in Huggingface.

*--ctx*/*-ctx*, Set the amount of context to give the LLM for phase 2 (step 5 above).

*--vibe*/*-vibe*, Set the vibe to scan for, default 'humorous'.

*--verbose*/*-v*, Enables the LLM output to be shown in the console.

*--confidence_pct*/*-percent*, Set the confidence percentage for phase 2 (step 5), default 40.

*--repeats*/*-repeat*, Sets how many times to repeat the LLM check (steps 4 and 5).

*--add_fake*/*-fake*, Adds a fake subtitle at the start of the SRT to make dragging the SRT onto NLE timelines easier.

## Example usage
py vibeclipper.py -v "P:/ATH/TO/FILE.mp4"

The -v flag will cause the LLM to output it's "thought process"

The output will be saved as "P:/ATH/TO/FILE.mp4.out.srt"

## Models
The models used for this project are Jan v1 ( https://huggingface.co/janhq/Jan-v1-4B-GGUF ) and Whisper medium ( https://huggingface.co/openai/whisper-medium ) by default.

Whisper uses Pytorch, so if you can install the CUDA version. Otherwise it will take some time to transcribe a video.

Jan v1 is ran via llama-cpp-python, so I recommend also trying to get that up and running with CUDA enabled. 

## Installation
I will set up an installation guide later, probably. Hopefully, I'll be able to make it just work out of the box. Maybe make a little GUI wrapper.

Libraries used were llama-cpp-python, pytorch, whisper, click, huggingface-hub
