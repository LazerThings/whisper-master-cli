# Whisper Master CLI
A whisper CLI  that uses [`whisper-small`](https://huggingface.co/openai/whisper-small) for all systems, built with Python, made with Claude 3.5 Sonnet (Oct. 2024 updated ver.)
## Setup
### Step 0
You will need a working python enviroment and a good enough computer to run `whisper-small`.
You will also need to install `torch transformers librosa numpy soundfile tqdm`
### Step 1
Put the `whisper` folder in your home directory (the script is made for Mac, you can change where to look for the python file in the shell script).
### Step 2
Set up the symbolic link with these two commands:
```shell
mkdir -p ~/bin # If ~/bin exists, this won't do anything
ln -s ~/whisper/whisper ~/bin/whisper
```
### Step 3
Make sure the script is executable:
```shell
chmod +x ~/whisper/whisper
```
### Step 4
Make sure it works:
```shell
which whisper
```
## Usage
When set up correctly, you can use `whisper` to use whisper on-device.
Argument lists and examples written by AI.
### Required Arguments
```shell
--input PATH, -i PATH        Path to input audio file
```
### Optinal Arguments
```shell
--output PATH, -o PATH       Path to output directory (defaults to '.')
--chunk-length N, -cln N     Length of audio chunks in seconds (defaults to 30)
--chunkless, -cls           Process the entire audio file as one chunk
--language LANG, -l LANG    Language of the audio (e.g., english, french, spanish)
--default-language LANG     Set default language for future transcriptions
--help, -h                  Show help message and exit
```
### Examples
```shell
# Basic usage (defaults to English)
whisper -i audio.mp3

# Specify language
whisper -i audio.mp3 -l french
whisper -i audio.mp3 --language spanish

# Set default language (saves to ~/.whisper_default_language)
whisper --default-language french

# Custom chunk length (two ways)
whisper -i audio.mp3 --chunk-length 45
whisper -i audio.mp3 -cln 45

# Combine options
whisper -i audio.mp3 -o ~/transcripts -l german -cln 45
whisper -i audio.mp3 -o ~/transcripts -l french -cls

# Show help with full language list
whisper --help
```
