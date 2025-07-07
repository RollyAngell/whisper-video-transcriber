# 🎬 Local Video Transcriber with Whisper

A complete command-line application to transcribe videos to text using OpenAI's Whisper running locally, optimized for Chilean Spanish, English variants, and international multi-accent scenarios.

## ✨ Features

- 🎤 **Local transcription**: No internet needed, completely private
- 🧠 **Multiple models**: From fast to high accuracy
- 🌍 **Automatic language detection**: Supports over 90 languages
- 🇨🇱 **Chilean Spanish optimization**: Enhanced recognition for Chilean dialect
- 🇺🇸🇬🇧🇦🇺 **English variants**: Optimized for American, British, Australian accents
- 🌐 **Multi-accent support**: Handles international speakers and mixed accents
- ⏰ **Optional timestamps**: Includes timestamps in transcription
- 📊 **Detailed metrics**: Processing time, file size, speed factor analysis
- 🖥️ **Command-line interface**: Powerful and efficient CLI tool
- 📁 **Multiple formats**: MP4, AVI, MOV, MKV, WMV, FLV, WebM
- 🔧 **Robust error handling**: Automatic fallback methods and cleanup

## 🚀 Installation

### 1. Prerequisites

You need to have **ffmpeg** installed on your system:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from: https://ffmpeg.org/download.html
```

### 2. Create virtual environment (recommended)

```bash
# Create virtual environment
python -m venv whisper_env

# Activate virtual environment
source whisper_env/bin/activate  # macOS/Linux
# or
whisper_env\Scripts\activate     # Windows

# Update pip
pip install --upgrade pip
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify installation

```bash
python transcriptor_whisper.py --help
```

## 🎯 Usage

### Command Line

```bash
# Basic usage (auto-detection)
python transcriptor_whisper.py my_video.mp4

# With specific model
python transcriptor_whisper.py my_video.mp4 -m medium

# Chilean Spanish optimization
python transcriptor_whisper.py my_video.mp4 -l es-cl

# American English optimization
python transcriptor_whisper.py my_video.mp4 -l en-us

# British English optimization
python transcriptor_whisper.py my_video.mp4 -l en-uk

# Multi-accent international content
python transcriptor_whisper.py my_video.mp4 -l multi -m medium

# With timestamps
python transcriptor_whisper.py my_video.mp4 -t

# Complete example with Chilean Spanish
python transcriptor_whisper.py my_video.mp4 -m medium -l es-cl -t -v
```

## 🧠 Available Models

| Model    | Size     | Speed     | Accuracy  | RAM required | Best for     |
|----------|----------|-----------|-----------|--------------|--------------|
| `tiny`   | ~39 MB   | Very fast | Basic     | ~1 GB        | Quick tests  |
| `base`   | ~74 MB   | Fast      | Good      | ~1 GB        | General use  |
| `small`  | ~244 MB  | Medium    | Better    | ~2 GB        | Balanced     |
| `medium` | ~769 MB  | Slow      | Very good | ~5 GB        | High quality |
| `large`  | ~1550 MB | Very slow | Excellent | ~10 GB       | Best results |

**Recommendation**: Use `base` for general use, `medium` for better quality, or `large` for international/multi-accent content.

## 🌍 Language Support

### General Language Codes
- `es` - Spanish (General)
- `en` - English (General)
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `ko` - Korean
- `zh` - Chinese
- And many more...

### 🇨🇱 Spanish Variants with Optimization
- `es-cl` - **Chilean Spanish** (optimized with context)
- `es-ar` - Argentinian Spanish
- `es-mx` - Mexican Spanish
- `es-co` - Colombian Spanish
- `es-pe` - Peruvian Spanish
- `es-es` - Spanish (Spain)

### 🇺🇸🇬🇧🇦🇺 English Variants with Optimization
- `en-us` - **American English** (optimized)
- `en-uk` - **British English** (optimized)
- `en-au` - Australian English
- `en-ca` - Canadian English
- `en-nz` - New Zealand English
- `en-za` - South African English
- `en-in` - Indian English

### 🌐 Multi-Language/Accent Support
- `multi` - **Multi-language/accent optimization**
- `en` - **International English** (handles multiple accents)
- `es` - **International Spanish** (handles multiple dialects)

## 🌐 Multi-Accent & International Content

### Handling International Speakers

Perfect for videos with speakers from different countries:

```bash
# 🌍 Best for mixed accents (RECOMMENDED)
python transcriptor_whisper.py conference.mp4 -m medium

# 🌐 Multi-language optimization
python transcriptor_whisper.py meeting.mp4 -l multi -m medium

# 🇺🇸🇬🇧🇦🇺 English speakers from multiple countries
python transcriptor_whisper.py video.mp4 -l en -m medium

# 🇪🇸🇲🇽🇨🇱 Spanish speakers from multiple countries
python transcriptor_whisper.py video.mp4 -l es -m medium
```

### 📊 Scenario-Based Recommendations

| Scenario | Best Command | Reason |
|----------|-------------|--------|
| **🌍 Unknown/Mixed languages** | `python transcriptor_whisper.py video.mp4 -m medium` | Auto-detection works best |
| **🇺🇸🇬🇧🇦🇺 English multi-accent** | `python transcriptor_whisper.py video.mp4 -l en -m medium` | General English handles all accents |
| **🇪🇸🇲🇽🇨🇱 Spanish multi-dialect** | `python transcriptor_whisper.py video.mp4 -l es -m medium` | General Spanish handles all dialects |
| **🎤 International conference** | `python transcriptor_whisper.py video.mp4 -l multi -m large` | Maximum flexibility |
| **🏢 Business meeting (mixed)** | `python transcriptor_whisper.py video.mp4 -m medium -t` | Auto-detect with timestamps |

### 🌟 Examples of International Content

```bash
# Conference with speakers from India, Canada, Australia
python transcriptor_whisper.py conference.mp4 -l en -m medium

# Business meeting: US, UK, Australia participants
python transcriptor_whisper.py meeting.mp4 -l en -m medium -t

# Spanish podcast: Chile, Mexico, Argentina speakers
python transcriptor_whisper.py podcast.mp4 -l es -m medium

# International webinar (multiple languages)
python transcriptor_whisper.py webinar.mp4 -l multi -m large -t -v
```

## 🇨🇱 Chilean Spanish Optimization

### Enhanced Recognition
When using `-l es-cl`, the system applies:

- **Contextual prompts**: Includes Chilean expressions
- **Vocabulary boost**: Better recognition of Chilean terms
- **Common expressions**: Po, cachai, fome, bacán, pololo, pega, lucas, weón, etc.

### Usage Examples
```bash
# Basic Chilean Spanish
python transcriptor_whisper.py video.mp4 -l es-cl

# High-quality Chilean Spanish with timestamps
python transcriptor_whisper.py video.mp4 -m medium -l es-cl -t

# Verbose output for debugging
python transcriptor_whisper.py video.mp4 -l es-cl -v
```

## 🇺🇸🇬🇧 English Optimization

### American English (`en-us`)
- **Optimized for**: American accent, slang, and expressions
- **Recognizes**: "awesome", "dude", "gonna", "wanna", "totally"
- **Uses**: American spelling conventions

### British English (`en-uk`)
- **Optimized for**: British accent, slang, and expressions
- **Recognizes**: "brilliant", "mate", "cheers", "bloke", "quite"
- **Uses**: British spelling conventions

### Usage Examples
```bash
# American English optimization
python transcriptor_whisper.py video.mp4 -l en-us -m medium

# British English optimization
python transcriptor_whisper.py video.mp4 -l en-uk -m medium

# General English (handles all accents)
python transcriptor_whisper.py video.mp4 -l en -m medium
```

## 📊 Performance Metrics

The application provides detailed performance analysis:

### Video Information
- **File size**: MB and bytes
- **Duration**: HH:MM:SS format
- **Format**: Video container format
- **Resolution**: Width x Height
- **FPS**: Frames per second

### Processing Statistics
- **Audio extraction time**: Time to extract audio
- **Transcription time**: Time for Whisper processing
- **Total processing time**: Complete workflow time
- **Speed factor**: How many times faster/slower than real-time

### Example Output

```