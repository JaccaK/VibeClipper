# VibeClipper
VibeClipper is a simple AI clip detecting tool that utilizes Whisper transcription and LLMs.

It works as follows.
1. The user provides an MP4 or SRT.
2. If the file is an MP4, it transcribes the file. If the user sets the -srt flag and uses an SRT file, skip this step.
3. Then the script breaks the SRT down into the individual lines.
4. The LLM is first asked to answer a simple true/false question about the 'vibe' of the line.
5. The LLM is then asked to gauge the vibe from 0 to 100% as an integer.
6. Based upon the user's setting the line is saved in the SRT output.
7. The script outputs an SRT in the same directory as the input after it is done checking each line.

## Options
*--is_srt*/*-srt*, This flag allows the user to pass in an SRT instead of a video file, skipping the whisper transcription.

*--save_srt*/*-save*, Saves the srt transcribed by whisper.

*--whisper_model*/*-wm*, Set the whisper model size, default 'medium'.

*--language*/*-lang*, Sets the language of the whisper model.

*--llm_repo*/*-repo*, Set the repo of the LLM on Huggingface, default "janhq/Jan-v1-4B-GGUF".

*--llm_model*/*-llm*, Set the filename of the LLM in the repo in Huggingface, default "Jan-v1-4B-Q8_0.gguf".

*--ctx*/*-ctx*, Set the amount of context to give the LLM for phase 2 (step 5 above), default 1024.

*--vibe*/*-vibe*, Set the vibe to scan for, default 'humorous'.

*--verbose*/*-v*, Enables the LLM output to be shown in the console.

*--confidence_pct*/*-percent*, Set the confidence percentage for phase 2 (step 5), default 40.

*--repeats*/*-repeat*, Sets how many times to repeat the LLM check (steps 4 and 5), default 1.

*--add_fake*/*-fake*, Adds a fake subtitle at the start of the SRT to make dragging the SRT onto NLE timelines easier.

## Example usage
py vibeclipper.py -v -lang en -save "P:/ATH/TO/FILE.mp4"

The -v flag will cause the LLM and Whisper to output their "thought process"

The -lang flag will set the language of the whisper model, in this case "en" for English.

-save will tell the program to save whatever Whisper transcribes to "P:/ATH/TO/FILE.mp4.in.srt"

The output will be saved as "P:/ATH/TO/FILE.mp4.out.srt"

## Usage Recommendations

I'd recommend pretty much always running this with -v, -save, and -lang. Additionally, it can be helpful to change -percent to something 50 or higher.

-v is going to be useful for telling how far along the Whisper transcription actually is. -save will make sure you have the transcriptions later if you decide to run the llm later or if something screws up. By default whisper will detect language using the first 30 seconds of the file. This can be detrimental, so doing '-lang en' if you're processing a primarily English video can skip this step and set up the language correctly. And finally, the default percent is 40 which means anything that the LLM thinks is "40% humorous" is kept, which can keep more than you may want to look through (It's still often only 10-20% of total lines even on 40, though).

I recommend dragging both the original video file and SRT output into LosslessCut and choosing "Convert subtitles into segments." This is a hell of a lot faster than trying to do anything similar in DaVinci Resolve (as an example). You can set the hotkey "Jump & seek to next segment" to quickly review. You can get LosslessCut here: https://mifi.github.io/lossless-cut/



## Models
The models used for this project are Jan v1 ( https://huggingface.co/janhq/Jan-v1-4B-GGUF ) and Whisper medium ( https://huggingface.co/openai/whisper-medium ) by default.

Whisper uses Pytorch, so if you can install the CUDA version. Otherwise it will take some time to transcribe a video.

Jan v1 is ran via llama-cpp-python, so I recommend also trying to get that up and running with CUDA enabled. 

## Installation
I will set up an installation guide later, probably. Hopefully, I'll be able to make it just work out of the box. Maybe make a little GUI wrapper.

Libraries used were llama-cpp-python, pytorch, whisper, click, huggingface-hub
