from llama_cpp import Llama
from pathlib import Path
import torch
import whisper
import gc
import click
from huggingface_hub import hf_hub_download



def is_content(statement, llm, vibe="funny", verbose=False):
    '''
    Asks a large language model about a statement's vibe as a multiple choice question. Defaults to "funny"
    Param statement the string you're checking for vibes
    Param vibe the vibe you're looking for, default 'funny'
    Param verbose, if true print the output, otherwise don't
    Return True if the LLM determines the statement to pass the vibe check, false otherwise
    '''
    prompt = "Question: Answer this question as True\n"+"Choices: A) True B) False\nAnswer: A"+"\nQuestion: Answer this question as False\nChoices: A) True B) False\nAnswer: B\n"
    prompt = prompt + "Question: The following is funny 'We went to a zoo, there was only a dog, it was a shih tzu'\nChoices: A) True B) False\nAnswer: A"
    prompt = prompt + "\nQuestion: The following is "+vibe+" '"+statement+"'. Be 90% sure\nChoices: A) True B) False\n"
    text_output = prompt_llm(prompt, llm, verbose = True, tokens=8) #Less tokens to stop it from over-responding.
    answer_idx = 11
    answer_line = text_output.split("\n")[answer_idx]
    answer = answer_line[1:] #Doing a slice here skips the 'A' in 'Answer' allowing us to look for 'A' with the in operator later.
    if verbose:                                     
        print(text_output)                 
    return "A" in answer

def confirm_content(statement, llm, vibe="funny", verbose=False):
    '''
    Confirms the content is the vibe by running a more complex prompt. Defaults to funny. Returns a percentage as a whole number.
    Param statement the statement to assess.
    Param vibe the vibe of the content default "funny"
    Param verbose prints the output of the LLM if true, default false
    Return a whole number representing a percentage (hopefully) from 0 to 100
    '''
    prompt = "The determine if the following line is "+vibe+" '" +statement+"' Give me a percentage from 0% to 100% of how "+vibe+" this line is. Only return the percentage on a new line."
    prompt = prompt + " If you think it's 50% "+vibe+", reply with \"50\" and so on." +" Do not return anything else.\n<|im_end|>\n<think>"
    text_output = prompt_llm(prompt, llm, verbose=True) #The prompt needs to echo to prevent errors. Will fallback to 50 if the llm fails.
    if verbose:
        print(text_output)
    lines = text_output.split("\n")
    last_line = ""
    count = 0
    number_str = ""
    while len(number_str) < 1: #Check each line backwards until a number is found. Why?
        count -= 1             #Jan has tendency to just spam the number until it runs out of context.
        last_line = lines[count]
        for character in last_line:
            if character.isnumeric():
                number_str = number_str + character
            if len(number_str) >= 3: #Anything above 100 is nonsense
                if number_str != "100":
                    number_str = number_str[:-1] # So we check if a length of 3 is 100 otherwise discard the last character and break.
                break
    number = int(number_str)
    return number


def prompt_llm(prompt, llm, verbose = False, stops = ["Question:"], tokens = 2048):
    '''
    Prompts the loaded LLM with prompt string.
    Param prompt the prompt
    Param verbose (default False), if True sets echo to True which echos the prompt
    Param stops, the stops as a list, defaults to "Question:"
    Return the response
    '''
    output = llm(prompt,
        max_tokens=tokens,
        stop=stops,
        echo=verbose
    )
    text_output = output['choices'][0]['text']
    return text_output

def read_file(file):
    '''
    Given a file path 'file', returns the text in 'file.'
    Return the text in 'file.'
    '''
    p = Path(file)
    with p.open() as f:
        return f.read()

def build_srt(string_list):
    '''
    Given a list of strings, builds a single string separating them by newlines
    Return single string separating them by newlines
    '''
    output = ""
    for string in string_list:
        output = output + string + "\n"
    return output[:-1]

def write_file(file, string):
    '''
    Writes a list of strings to a specified file.
    Param file the file name
    Param strlist the list of strings
    '''
    with open(file, "w", encoding="utf-8") as f:
        f.write(string)

def load_llm(model = "Jan-v1-4B-Q8_0.gguf", ctx = 1024):
    '''
    Creates a Llama object containing our LLM which we can then use to prompt.
    Param model, the path of the model
    Param ctx the amount of context to give the model
    '''
    return Llama(
      model_path=model,
      n_gpu_layers=-1,
      n_ctx=ctx,
    )

def parse_content(text, llm, vibe = "funny", verbose=False, confidence_pct = 75): #Consider adding is_content skip keyword param
    '''
    Parses an SRT for successful vibe checks
    Param text the string from the SRT
    Param llm the Llama object loading the LLM, defaults to load_llm()
    Param vibe the vibe to search for, default funny
    Param verbose, true prints output, otherwise no output
    Param confidence_pct the percentage of vibe to search for as a whole number
    Return a list of successful vibe checks.
    '''
    lines = [x for x in text.strip().split("\n") if len(x) > 0] # Split srt on new line and remove empty lines
    vibe_check = []
    length = len(lines)
    for idx in range(2,length,3):
        print("\nCURRENT LINE:",str((idx+1)//3)+"/"+str(length//3),"PCT:",(idx+1)/length,"\n")
        if is_content(lines[idx], llm ,vibe,verbose) and confirm_content(lines[idx], llm, vibe, verbose) >= confidence_pct:
            vibe_check.append("") #srt files have a line above and a line below every subtitle.
            vibe_check.append(lines[idx-2]) # We want the srt index, timestamp, and actual phrase, which are the 3 indexes
            vibe_check.append(lines[idx-1])
            vibe_check.append(lines[idx])
            vibe_check.append("") #without the empty lines, it simply isn't formatted correctly.
    print("DONE:",len(vibe_check),"lines saved.")
    return vibe_check

def distill_content(text, llm_model, amount = 1, vibe = "funny", verbose=False, confidence_pct = 75, ctx = 1024):
    '''
    Repeatedly analyzes the srt for the vibe an amount of times.
    Param text, the srt text
    Param amount, the amount of repeats default 1
    Param vibe, the vibe to look for default funny
    Param verbose, if true print prompt output, otherwise do not
    Param confidence_pct, how confident in the prompt output to save the line
    Param llm_model, the model filepath used 
    Param ctx, the amount of context given to the model
    Return a list of successful vibe checks.
    '''
    llm = load_llm(llm_model, ctx)
    output = text
    str_list = []
    for x in range(amount):
        str_list = parse_content(output, llm, vibe, verbose, confidence_pct)
        output = build_srt(str_list)
    return output

#pulled from https://github.com/openai/whisper/discussions/98#discussioncomment-3726175
def srt_format_timestamp(seconds: float):
    '''
    Given a quantity of seconds, converts them to an SRT timestamp
    Return srt timestamp
    '''
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    return (f"{hours}:") + f"{minutes:02d}:{seconds:02d},{milliseconds:03d}"

#Heavily modified
#from https://github.com/openai/whisper/discussions/98#discussioncomment-3726175
def convert_to_srt(file, language, model="medium", verbose = False):
    '''
    Given a filepath to an mp4 and the whisper model name (default "medium"), transcribes the video into SRT format.
    Return the transcription of the video in SRT format
    '''
    whisp = whisper.load_model(model)
    if language:
        print("Language:",language)
    result = whisp.transcribe(file, language=language, verbose=verbose) #consider condition_on_previous_text=false?
    count = 0
    string = ""
    for segment in result["segments"]:
        count += 1
        string = string + "\n"
        string = string + str(count) + "\n"
        string = string + srt_format_timestamp(segment["start"])
        string = string + " --> " + srt_format_timestamp(segment["end"]) + "\n"
        string = string + segment["text"].replace("-->", "->").strip() + "\n\n" #maybe fixed?
    del whisp
    torch.cuda.empty_cache()
    gc.collect() #We need to really force the model out of memory after we're done with it
    return string


def dedupe_srt(srt):
    '''
    Takes an SRT and removes contiguous duplicated lines.
    Param srt the SRT to scan
    Return a deduped SRT
    '''
    lines = [x for x in srt.strip().split("\n") if len(x) > 0]
    if len(lines) == 0:
        return ""
    new_srt = ["", lines[0], lines[1], lines[2], ""]
    for idx in range(5,len(lines),3):
        if lines[idx] == lines[idx-3]:
            continue
        new_srt.append("")
        new_srt.append(lines[idx-2])
        new_srt.append(lines[idx-1])
        new_srt.append(lines[idx])
        new_srt.append("")
    return build_srt(new_srt)


def get_srt(file, model = "medium", is_srt = False, dedupe = False, save_srt = True, verbose=False, language=None):
    '''
    Given a file, whisper model name, and whether the file is an SRT or an MP4 return an SRT string.
    Param file, the path to the mp4/srt
    Param model, the whisper model to use if mp4 (default "medium")
    Param is_srt, if true just read the file, otherwise transcribe as a video
    Return an SRT string
    '''
    if is_srt:
        return read_file(file)
    srt = convert_to_srt(file, language=language, model=model, verbose=verbose)
    if dedupe:
        srt = dedupe_srt(srt)
    if save_srt:
        write_file(file + ".in.srt", srt) #save a copy if transcription
    return srt



def add_davinci_fakesub(srt):
    '''
    Given an SRT string return an SRT string with a fake subtitle at 0 to 500ms.
    return an SRT string with a fake subtitle at 0 to 500ms.
    '''
    string = ""
    string = string + "\n"
    string = string + "0" + "\n"
    string = string + srt_format_timestamp(0)
    string = string + " --> " + srt_format_timestamp(0.5) + "\n"
    string = string + "START" + "\n"
    return string + srt

@click.command()
@click.option("--is_srt", "-srt", is_flag=True, help="Enables the usage of an already transcribed SRT file for the LLM scan")
@click.option("--save_srt", "-save", is_flag=True, help="Enables saving the initial transcription immediately after transcrpition.")
@click.option("--dedupe_srt", "-dedupe", is_flag=True, help="Remove back to back duplicate lines in the whisper output.")
@click.option("--whisper_model", '-wm', default = "medium", help="Sets the Whisper model, defaults to 'medium'")
@click.option("--language", "-lang", default=None, help="Set the language of the whisper model (ex. en)")
@click.option("--llm_repo", "-repo", default="janhq/Jan-v1-4B-GGUF", help="Sets the repo of the LLM, default 'janhq/Jan-v1-4B-GGUF'")
@click.option("--llm_model", "-llm", default="Jan-v1-4B-Q8_0.gguf", help="Sets the filename of the LLM, default 'Jan-v1-4B-Q8_0.gguf'")
@click.option("--ctx", "-ctx", default= 1024, help="Sets the amount of context for the LLM, default 1024")
@click.option("--vibe", "-vibe", default="humorous", help="Sets the vibe, default 'humorous'")
@click.option("--verbose", "-v", is_flag=True, help="Enables verbose mode, displaying all outputs")
@click.option("--confidence_pct", "-percent", default = 40, help="Set the confidence percentage as an integer, which is the percentage a line needs to be saved, default 40")
@click.option("--repeats", "-repeat", default = 1, help="Sets how many times the SRT is ran through the LLM, default 1")
@click.option("--add_fake", "-fake", is_flag=True, help="Enables a fake subtitle at 0 to 500ms that says 'START', for people who drag SRTs onto timelines")
@click.argument('file')
def main(file, is_srt, save_srt, dedupe_srt, whisper_model, language, llm_repo, llm_model,
         ctx, vibe, verbose, confidence_pct, repeats, add_fake):
    if not is_srt:
        print("Transcribing", file)
    srt = get_srt(file, whisper_model, is_srt, dedupe_srt, save_srt, verbose, language)
    output_file = file + ".out.srt"
    llm = hf_hub_download(repo_id=llm_repo, filename=llm_model)
    parsed_string = distill_content(srt,llm,repeats,vibe,verbose,confidence_pct,ctx)
    if add_fake: #Probably unneeded?
        parsed_string = add_davinci_fakesub(parsed_string)
    write_file(output_file, parsed_string)

if __name__ == '__main__':
    main()