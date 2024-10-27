# Whisper Master CLI
A whisper CLI  that uses any whisper model for all systems, built with Python, made with Claude 3.5 Sonnet (Oct. 2024 updated ver.)
![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)
## Setup
### Step 0
You will need a working python enviroment and a good enough computer to run whatever whisper model you're running.
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
--input PATH, -i PATH        Path to input audio/video file
```
### Optinal Arguments
```shell
--output PATH, -o PATH       Path to output directory (defaults to '.')
--chunk-length N, -cln N     Length of audio chunks in seconds (defaults to 30)
--chunkless, -cls           Process the entire file as one chunk
--language LANG, -l LANG    Language of the audio (e.g., english, french, spanish)
--model MODEL, -m MODEL     Model to use for transcription (e.g., whisper-large-v3)
--default-language LANG     Set default language for future transcriptions
--default-model MODEL       Set default model for future transcriptions
--help, -h                  Show help message and exit
```
### Examples
```shell
# Basic usage with defaults (whisper-small model)
whisper -i audio.mp3

# Process video file
whisper -i video.mp4

# Specify output directory
whisper -i audio.mp3 -o ~/transcripts

# Use specific model
whisper -i audio.mp3 -m whisper-large-v3

# Specify language
whisper -i audio.mp3 -l french

# Custom chunk length (two ways)
whisper -i audio.mp3 --chunk-length 45
whisper -i audio.mp3 -cln 45

# Process as single chunk
whisper -i audio.mp3 --chunkless
whisper -i audio.mp3 -cls

# Set default preferences
whisper --default-language french
whisper --default-model whisper-large-v3

# Combine options
whisper -i video.mp4 -o ~/transcripts -l german -m whisper-large-v3 -cln 45
```
### Supported Languages
```shell
albanian, arabic, armenian, azerbaijani, basque, bengali, bosnian,
  breton, bulgarian, catalan, chinese, croatian, czech, danish, dutch,
  english, estonian, finnish, french, german, greek, hebrew, hindi,
  hungarian, icelandic, indonesian, italian, japanese, kannada,
  kazakh, korean, latin, latvian, lithuanian, macedonian, malay,
  malayalam, maori, mongolian, nepali, norwegian, persian, polish,
  portuguese, romanian, russian, serbian, slovak, slovenian, spanish,
  swahili, swedish, tamil, telugu, thai, turkish, ukrainian, urdu,
  vietnamese, welsh
```
### Supported File Formats
```shell
Video: .mp4, .avi, .mov, .mkv, .webm
Audio: .mp3, .wav, .m4a, .wma, .ogg, .flac
```
### Available Models
```shell
whisper-tiny, whisper-base, whisper-small (default), whisper-medium, whisper-large, 
whisper-large-v2, whisper-large-v3, whisper-large-v3-turbo
```
