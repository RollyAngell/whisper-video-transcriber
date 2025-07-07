# 🎬 Local Video Transcriber with Whisper

A complete command-line application to transcribe videos to text using OpenAI's Whisper running locally, optimized for Chilean Spanish, English variants, and international multi-accent scenarios.

**✅ CROSS-PLATFORM COMPATIBLE** - Works perfectly on Windows and Mac operating systems.

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
- 💻 **Cross-platform**: Works on Windows, macOS, and Linux

## 🚀 Installation

### 1. Prerequisites

You need to have **ffmpeg** installed on your system:

#### **Windows** (Choose one method):

**Method A: Manual Download (Recommended)**
```bash
# 1. Download from: https://www.gyan.dev/ffmpeg/builds/
# 2. Extract to C:\ffmpeg\
# 3. Add C:\ffmpeg\bin to your PATH
# 4. Test with: ffmpeg -version
```

**Method B: Using winget (Windows 10/11)**
```bash
winget install ffmpeg
```

**Method C: Using Scoop**
```bash
# Install Scoop first:
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# Then install ffmpeg:
scoop install ffmpeg
```

#### **macOS**:
```bash
brew install ffmpeg
```

#### **Linux (Ubuntu/Debian)**:
```bash
sudo apt install ffmpeg
```

### 2. Create virtual environment (recommended)

#### **Windows (PowerShell)**:
```bash
# Create virtual environment
python -m venv whisper_env

# Activate virtual environment
whisper_env\Scripts\activate

# Update pip
python -m pip install --upgrade pip
```

#### **macOS/Linux**:
```bash
# Create virtual environment
python -m venv whisper_env

# Activate virtual environment
source whisper_env/bin/activate

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

### 🪟 Windows-Specific Examples

```bash
# Basic usage in Windows
python transcriptor_whisper.py "C:\Users\Username\Desktop\video.mp4"

# With quotes for spaces in path
python transcriptor_whisper.py ".\2025-01-08 15-30-06 - Meeting.mp4" -l en-us -m medium

# Chilean Spanish with timestamps
python transcriptor_whisper.py "my_video.mp4" -l es-cl -t -v
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

## 🔧 Troubleshooting

### Windows Issues

**Error: "ffmpeg not found"**
```bash
# Verify ffmpeg is installed:
ffmpeg -version

# If not working, add to PATH or reinstall ffmpeg
```

**Error: "No module named 'whisper'"**
```bash
# Make sure virtual environment is activated:
whisper_env\Scripts\activate

# Reinstall dependencies:
pip install -r requirements.txt
```

**Permission errors**
```bash
# Run PowerShell as Administrator
# Or use --user flag:
pip install --user -r requirements.txt
```

**Unicode/Character encoding errors**
```bash
# The application now automatically handles Windows encoding
# If you still see issues, try:
chcp 65001  # Set console to UTF-8
```

### macOS/Linux Issues

**MoviePy not available**
```bash
# This is normal, the app will use ffmpeg instead
# No action needed
```

**Permission denied**
```bash
# Make sure you have write permissions in the directory
chmod +x transcriptor_whisper.py
```

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
Starting transcription of: my_video.mp4
Analyzing video file...
Video Info Video Information:
   File: my_video.mp4
   Size: 156.78 MB (164,435,792 bytes)
   Duration: 00:15:23
   Format: .mp4
   Resolution: 1920x1080
   FPS: 30.00

Extracting audio audio...
Success Audio extraction completed in 3.45s
Transcribing Starting transcription...
American English optimization activated
Transcription completed in 89.23s
Saving Saving transcription...
Completed Transcription completed successfully!
Processing Stats Processing Statistics:
   Audio extraction: 3.45s
   Transcription: 89.23s
   Total time: 92.68s
   Speed factor: 9.95x faster than real-time
Output file Output file: transcriptions/my_video.txt
```

## 📁 File Structure

After installation and first use:

```
whisper-video-transcriber/
├── README.md
├── requirements.txt
├── transcriptor_whisper.py
├── whisper_env/              # Virtual environment
├── transcriptions/           # Generated transcriptions
├── temp/                     # Temporary files
└── transcriptor.log          # Log file
```

## 🎯 Quick Start Examples

### For Chilean Spanish Content
```bash
# Basic Chilean Spanish
python transcriptor_whisper.py video.mp4 -l es-cl

# High quality with timestamps
python transcriptor_whisper.py video.mp4 -m medium -l es-cl -t
```

### For International English Content
```bash
# Auto-detection (best for mixed accents)
python transcriptor_whisper.py video.mp4 -m medium

# Specific American English
python transcriptor_whisper.py video.mp4 -l en-us -m medium
```

### For Multi-Language Content
```bash
# Best for international conferences
python transcriptor_whisper.py video.mp4 -l multi -m large -t
```

## 🔍 Advanced Usage

### Batch Processing
```bash
# Process multiple files
for file in *.mp4; do
    python transcriptor_whisper.py "$file" -l es-cl -m medium
done
```

### Custom Output Directory
```bash
python transcriptor_whisper.py video.mp4 -o custom_output_folder
```

### Debug Mode
```bash
python transcriptor_whisper.py video.mp4 -v
```

## 🤝 Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Submitting pull requests
- Improving documentation

## 📄 License

This project is open source and available under the MIT License.

---

**Made with ❤️ for the global community - Works perfectly on Windows and Mac!**