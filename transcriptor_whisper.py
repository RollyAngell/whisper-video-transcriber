#!/usr/bin/env python3
"""
Optimized Video Transcriber with Local Whisper
Author: AI Assistant
Description: Transcribes videos to text using OpenAI's Whisper running locally
Optimized for Chilean Spanish and English variants
"""

import os
import sys
import argparse
import logging
import whisper
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any
import time
import subprocess
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('transcriptor.log')
    ]
)
logger = logging.getLogger(__name__)

# Robust import for moviepy
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    logger.warning("MoviePy not available, falling back to ffmpeg")
    MOVIEPY_AVAILABLE = False

class VideoInfo:
    """Container for video information"""
    def __init__(self, path: Path):
        self.path = path
        self.size_bytes = path.stat().st_size
        self.size_mb = self.size_bytes / (1024 * 1024)
        self.duration_seconds = 0.0
        self.format = path.suffix.lower()
        self.fps = 0.0
        self.resolution = "Unknown"
        
    def __str__(self) -> str:
        duration_str = self._format_duration(self.duration_seconds)
        return (f"File: {self.path.name}\n"
                f"Size: {self.size_mb:.2f} MB ({self.size_bytes:,} bytes)\n"
                f"Duration: {duration_str}\n"
                f"Format: {self.format}\n"
                f"Resolution: {self.resolution}\n"
                f"FPS: {self.fps:.2f}")
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

class ProcessingStats:
    """Container for processing statistics"""
    def __init__(self):
        self.start_time = time.time()
        self.audio_extraction_time = 0.0
        self.transcription_time = 0.0
        self.total_time = 0.0
        self.video_duration = 0.0
        
    def finish(self):
        """Mark processing as finished"""
        self.total_time = time.time() - self.start_time
    
    def get_speed_factor(self) -> float:
        """Calculate processing speed factor (video_duration / processing_time)"""
        if self.total_time > 0:
            return self.video_duration / self.total_time
        return 0.0
    
    def __str__(self) -> str:
        speed_factor = self.get_speed_factor()
        return (f"⏱️  Processing Statistics:\n"
                f"   Audio extraction: {self.audio_extraction_time:.2f}s\n"
                f"   Transcription: {self.transcription_time:.2f}s\n"
                f"   Total time: {self.total_time:.2f}s\n"
                f"   Speed factor: {speed_factor:.2f}x "
                f"({'faster' if speed_factor > 1 else 'slower'} than real-time)")

class TranscriptorWhisper:
    """Optimized video transcriber using Whisper"""
    
    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    # Extended support for language variants
    LANGUAGE_VARIANTS = {
        # Spanish variants
        "es": "Spanish (General)",
        "es-cl": "Spanish (Chile)",
        "es-ar": "Spanish (Argentina)",
        "es-mx": "Spanish (Mexico)",
        "es-co": "Spanish (Colombia)",
        "es-pe": "Spanish (Peru)",
        "es-es": "Spanish (Spain)",
        
        # English variants
        "en": "English (General)",
        "en-us": "English (United States)",
        "en-uk": "English (United Kingdom)",
        "en-au": "English (Australia)",
        "en-ca": "English (Canada)",
        "en-nz": "English (New Zealand)",
        "en-za": "English (South Africa)",
        "en-in": "English (India)",
    }
    
    def __init__(self, model: str = "base", output_dir: str = "transcripciones"):
        """
        Initialize the transcriber with specified Whisper model
        
        Args:
            model: Whisper model to use (tiny, base, small, medium, large)
            output_dir: Directory to save transcriptions
        """
        self.model_name = model
        self.output_dir = Path(output_dir)
        self.temp_dir = Path("temp")
        
        self._validate_model()
        self._setup_directories()
        self._load_model()
        
        # Context prompts for better recognition
        self.chilean_spanish_prompt = (
            "Transcripción en español de Chile. "
            "Incluye chilenismos, modismos y expresiones típicas como: "
            "po, cachai, fome, bacán, pololo, pega, lucas, luca, "
            "weón, compadre, al tiro, piola, terrible, brigido, "
            "cuático, raja, choro, macanudo, filete."
        )
        
        self.english_us_prompt = (
            "Transcription in American English. "
            "Include common American expressions, slang, and terminology. "
            "Use American spelling and punctuation conventions. "
            "Common terms: awesome, dude, yeah, gonna, wanna, gotta, "
            "like, totally, basically, literally, actually."
        )
        
        self.english_uk_prompt = (
            "Transcription in British English. "
            "Include British expressions, slang, and terminology. "
            "Use British spelling and punctuation conventions. "
            "Common terms: brilliant, mate, cheers, bloke, quite, "
            "rather, bloody, proper, innit, lovely, brilliant."
        )
        
        self.english_general_prompt = (
            "Clear English transcription with proper grammar and punctuation. "
            "Include common expressions and natural speech patterns. "
            "Handle contractions, filler words, and conversational language appropriately."
        )
    
    def _validate_model(self) -> None:
        """Validate the specified model"""
        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model '{self.model_name}'. Available: {', '.join(self.AVAILABLE_MODELS)}")
    
    def _setup_directories(self) -> None:
        """Create necessary directories"""
        for directory in [self.output_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
            logger.debug(f"Directory created/verified: {directory}")
    
    def _load_model(self) -> None:
        """Load the Whisper model"""
        logger.info(f"Loading Whisper model '{self.model_name}'...")
        start_time = time.time()
        
        try:
            self.whisper_model = whisper.load_model(self.model_name)
            load_time = time.time() - start_time
            logger.info(f"Model '{self.model_name}' loaded successfully in {load_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_language_config(self, language: str) -> Dict[str, Any]:
        """Get optimized configuration for specific language variants"""
        if not language:
            return {
                "language": None, 
                "initial_prompt": None,
                "log_message": "🌍 Auto-detection mode - Best for multi-accent/multi-language videos"
            }
        
        lang_lower = language.lower()
        
        # Multi-language/accent optimization
        if lang_lower in ["multi", "mixed", "international"]:
            return {
                "language": None,
                "initial_prompt": self._get_multi_language_prompt(),
                "log_message": "🌐 Multi-language/accent optimization activated"
            }
        
        # English - general for multi-accent scenarios
        elif lang_lower == "en":
            return {
                "language": "en",
                "initial_prompt": self._get_international_english_prompt(),
                "log_message": "🌍 International English optimization (handles multiple accents)"
            }
        
        # Spanish - general for multi-dialect scenarios  
        elif lang_lower == "es":
            return {
                "language": "es",
                "initial_prompt": self._get_international_spanish_prompt(),
                "log_message": "🌎 International Spanish optimization (handles multiple dialects)"
            }
        
        # Spanish variants
        elif lang_lower in ["es-cl", "es_cl", "chile", "chileno"]:
            return {
                "language": "es",
                "initial_prompt": self.chilean_spanish_prompt,
                "log_message": "🇨🇱 Chilean Spanish optimization activated"
            }
        elif lang_lower.startswith("es-") or lang_lower.startswith("es_"):
            return {
                "language": "es",
                "initial_prompt": None,
                "log_message": f"🌎 Spanish variant ({language}) activated"
            }
        
        # English variants
        elif lang_lower in ["en-us", "en_us", "american", "usa"]:
            return {
                "language": "en",
                "initial_prompt": self.english_us_prompt,
                "log_message": "🇺🇸 American English optimization activated"
            }
        elif lang_lower in ["en-uk", "en_uk", "british", "uk"]:
            return {
                "language": "en",
                "initial_prompt": self.english_uk_prompt,
                "log_message": "🇬🇧 British English optimization activated"
            }
        elif lang_lower in ["en-au", "en_au", "australian", "australia"]:
            return {
                "language": "en",
                "initial_prompt": self.english_general_prompt,
                "log_message": "🇦🇺 Australian English optimization activated"
            }
        elif lang_lower in ["en-ca", "en_ca", "canadian", "canada"]:
            return {
                "language": "en",
                "initial_prompt": self.english_general_prompt,
                "log_message": "🇨🇦 Canadian English optimization activated"
            }
        elif lang_lower.startswith("en-") or lang_lower.startswith("en_") or lang_lower == "en":
            return {
                "language": "en",
                "initial_prompt": self.english_general_prompt,
                "log_message": f"🌍 English variant ({language}) activated"
            }
        
        # Other languages
        else:
            return {
                "language": language,
                "initial_prompt": None,
                "log_message": f"🌐 Language {language} activated"
            }
    
    def _get_multi_language_prompt(self) -> str:
        """Prompt for mixed language/accent content"""
        return (
            "This content may contain multiple languages, accents, or dialects. "
            "Transcribe accurately regardless of speaker origin. "
            "Handle code-switching, international accents, and mixed terminology appropriately."
        )

    def _get_international_english_prompt(self) -> str:
        """Prompt for international English with multiple accents"""
        return (
            "International English transcription with multiple accents and dialects. "
            "May include speakers from US, UK, Canada, Australia, India, South Africa, etc. "
            "Handle diverse pronunciations, terminology, and speech patterns accurately."
        )

    def _get_international_spanish_prompt(self) -> str:
        """Prompt for international Spanish with multiple dialects"""
        return (
            "Transcripción en español internacional con múltiples dialectos. "
            "Puede incluir hablantes de España, México, Argentina, Chile, Colombia, etc. "
            "Maneja pronunciaciones, terminología y patrones de habla diversos con precisión."
        )
    
    def _get_video_info_ffmpeg(self, video_path: Path) -> VideoInfo:
        """Get video information using ffmpeg"""
        video_info = VideoInfo(video_path)
        
        try:
            # Get video info using ffprobe
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                # Get duration from format
                if 'format' in data and 'duration' in data['format']:
                    video_info.duration_seconds = float(data['format']['duration'])
                
                # Get video stream info
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        fps_str = stream.get('r_frame_rate', '0/1')
                        # Safe evaluation of frame rate
                        try:
                            if '/' in fps_str:
                                num, den = fps_str.split('/')
                                video_info.fps = float(num) / float(den) if float(den) != 0 else 0.0
                            else:
                                video_info.fps = float(fps_str)
                        except (ValueError, ZeroDivisionError):
                            video_info.fps = 0.0
                        
                        width = stream.get('width', 0)
                        height = stream.get('height', 0)
                        if width and height:
                            video_info.resolution = f"{width}x{height}"
                        break
            
        except (subprocess.CalledProcessError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"Could not get detailed video info: {e}")
        
        return video_info
    
    def _get_video_info_moviepy(self, video_path: Path) -> VideoInfo:
        """Get video information using MoviePy"""
        video_info = VideoInfo(video_path)
        
        try:
            with VideoFileClip(str(video_path)) as video:
                video_info.duration_seconds = video.duration or 0.0
                video_info.fps = video.fps or 0.0
                if video.size:
                    video_info.resolution = f"{video.size[0]}x{video.size[1]}"
        except Exception as e:
            logger.warning(f"Could not get video info with MoviePy: {e}")
        
        return video_info
    
    def get_video_info(self, video_path: Path) -> VideoInfo:
        """Get comprehensive video information"""
        # Try ffmpeg first for more detailed info, fallback to moviepy
        try:
            subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
            return self._get_video_info_ffmpeg(video_path)
        except (FileNotFoundError, subprocess.CalledProcessError):
            if MOVIEPY_AVAILABLE:
                return self._get_video_info_moviepy(video_path)
            else:
                return VideoInfo(video_path)  # Basic info only
    
    @contextmanager
    def _temporary_audio_file(self):
        """Context manager for temporary audio files"""
        temp_file = self.temp_dir / f"audio_temp_{int(time.time())}.wav"
        try:
            yield temp_file
        finally:
            if temp_file.exists():
                temp_file.unlink()
                logger.debug(f"Cleaned up temporary file: {temp_file}")
    
    def _extract_audio_moviepy(self, video_path: Path, audio_path: Path) -> bool:
        """Extract audio using MoviePy"""
        try:
            logger.info(f"Extracting audio using MoviePy: {video_path.name}")
            
            video = VideoFileClip(str(video_path))
            if video.audio is None:
                logger.error("No audio track found in video")
                video.close()
                return False
            
            # Extract audio with optimal settings
            audio = video.audio
            audio.write_audiofile(
                str(audio_path),
                verbose=False,
                logger=None,
                temp_audiofile=str(self.temp_dir / "temp_audio.wav")
            )
            
            # Clean up resources
            audio.close()
            video.close()
            
            logger.info(f"Audio extracted successfully to: {audio_path}")
            return True
            
        except Exception as e:
            logger.error(f"MoviePy extraction failed: {e}")
            return False
    
    def _extract_audio_ffmpeg(self, video_path: Path, audio_path: Path) -> bool:
        """Extract audio using ffmpeg directly"""
        try:
            logger.info(f"Extracting audio using ffmpeg: {video_path.name}")
            
            # Check if ffmpeg is available
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            
            # FFmpeg command for audio extraction
            cmd = [
                "ffmpeg", "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # Audio codec
                "-ar", "16000",  # Sample rate optimized for Whisper
                "-ac", "1",  # Mono channel
                "-y",  # Overwrite output file
                str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
            
            logger.info(f"Audio extracted successfully to: {audio_path}")
            return True
            
        except FileNotFoundError:
            logger.error("ffmpeg not found. Install with: brew install ffmpeg")
            return False
        except Exception as e:
            logger.error(f"FFmpeg extraction failed: {e}")
            return False
    
    def extract_audio(self, video_path: Path, stats: ProcessingStats) -> Optional[Path]:
        """
        Extract audio from video using available method
        
        Args:
            video_path: Path to the video file
            stats: Processing statistics object
            
        Returns:
            Path to extracted audio file or None if failed
        """
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        # Check if video format is supported
        if video_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            logger.warning(f"Video format '{video_path.suffix}' may not be supported")
        
        extraction_start = time.time()
        
        with self._temporary_audio_file() as audio_path:
            # Try MoviePy first, then ffmpeg
            if MOVIEPY_AVAILABLE and self._extract_audio_moviepy(video_path, audio_path):
                # Copy to permanent location and return
                permanent_path = self.temp_dir / f"audio_extracted_{int(time.time())}.wav"
                permanent_path.write_bytes(audio_path.read_bytes())
                stats.audio_extraction_time = time.time() - extraction_start
                return permanent_path
            elif self._extract_audio_ffmpeg(video_path, audio_path):
                # Copy to permanent location and return
                permanent_path = self.temp_dir / f"audio_extracted_{int(time.time())}.wav"
                permanent_path.write_bytes(audio_path.read_bytes())
                stats.audio_extraction_time = time.time() - extraction_start
                return permanent_path
            else:
                logger.error("Failed to extract audio with both methods")
                stats.audio_extraction_time = time.time() - extraction_start
                return None
    
    def transcribe_audio(self, audio_path: Path, language: Optional[str], stats: ProcessingStats) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio file using Whisper with language-specific optimization
        """
        try:
            logger.info("Starting audio transcription...")
            transcription_start = time.time()
            
            # Get language-specific configuration
            lang_config = self._get_language_config(language)
            
            # Base transcription arguments
            transcription_args = {"fp16": False}
            
            # Add language and prompt if specified
            if lang_config["language"]:
                transcription_args["language"] = lang_config["language"]
            
            if lang_config["initial_prompt"]:
                transcription_args["initial_prompt"] = lang_config["initial_prompt"]
            
            # Log the optimization being used
            if lang_config.get("log_message"):
                logger.info(lang_config["log_message"])
            
            result = self.whisper_model.transcribe(str(audio_path), **transcription_args)
            
            stats.transcription_time = time.time() - transcription_start
            logger.info(f"Transcription completed in {stats.transcription_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            stats.transcription_time = time.time() - transcription_start if 'transcription_start' in locals() else 0
            return None
    
    def save_transcription(self, result: Dict[str, Any], filename: str, include_timestamps: bool, 
                          video_info: VideoInfo, stats: ProcessingStats) -> Optional[Path]:
        """
        Save transcription to file with metadata
        
        Args:
            result: Whisper transcription result
            filename: Output filename (without extension)
            include_timestamps: Whether to include timestamps
            video_info: Video information
            stats: Processing statistics
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            output_path = self.output_dir / f"{filename}.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header with complete metadata
                f.write(f"=== TRANSCRIPTION OF: {filename} ===\n\n")
                f.write(f"📹 Video Information:\n")
                f.write(f"   File: {video_info.path.name}\n")
                f.write(f"   Size: {video_info.size_mb:.2f} MB\n")
                f.write(f"   Duration: {VideoInfo._format_duration(video_info.duration_seconds)}\n")
                f.write(f"   Format: {video_info.format}\n")
                f.write(f"   Resolution: {video_info.resolution}\n")
                f.write(f"   FPS: {video_info.fps:.2f}\n\n")
                
                f.write(f"🧠 Processing Information:\n")
                f.write(f"   Model used: {self.model_name}\n")
                f.write(f"   Detected language: {result.get('language', 'Unknown')}\n")
                f.write(f"   Audio extraction time: {stats.audio_extraction_time:.2f}s\n")
                f.write(f"   Transcription time: {stats.transcription_time:.2f}s\n")
                f.write(f"   Total processing time: {stats.total_time:.2f}s\n")
                f.write(f"   Processing speed: {stats.get_speed_factor():.2f}x real-time\n\n")
                
                if include_timestamps and 'segments' in result:
                    f.write("=== TRANSCRIPTION WITH TIMESTAMPS ===\n\n")
                    for segment in result['segments']:
                        start_time = self._format_timestamp(segment['start'])
                        end_time = self._format_timestamp(segment['end'])
                        text = segment['text'].strip()
                        f.write(f"[{start_time} --> {end_time}] {text}\n")
                else:
                    f.write("=== COMPLETE TRANSCRIPTION ===\n\n")
                    f.write(result['text'].strip())
            
            logger.info(f"Transcription saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save transcription: {e}")
            return None
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files"""
        try:
            temp_files = list(self.temp_dir.glob("audio_*"))
            for temp_file in temp_files:
                temp_file.unlink()
                logger.debug(f"Removed temporary file: {temp_file}")
            
            if temp_files:
                logger.info(f"Cleaned up {len(temp_files)} temporary files")
        except Exception as e:
            logger.warning(f"Error cleaning temporary files: {e}")
    
    def transcribe_video(self, video_path: str, language: Optional[str] = None, include_timestamps: bool = False) -> bool:
        """
        Complete video transcription process with detailed metrics
        
        Args:
            video_path: Path to video file
            language: Language code (optional)
            include_timestamps: Whether to include timestamps
            
        Returns:
            True if successful, False otherwise
        """
        video_path = Path(video_path)
        filename = video_path.stem
        
        # Initialize processing statistics
        stats = ProcessingStats()
        
        logger.info(f"🚀 Starting transcription of: {video_path.name}")
        
        try:
            # Step 0: Get video information
            logger.info("📊 Analyzing video file...")
            video_info = self.get_video_info(video_path)
            stats.video_duration = video_info.duration_seconds
            
            # Display video information
            logger.info("📹 Video Information:")
            for line in str(video_info).split('\n'):
                logger.info(f"   {line}")
            
            # Step 1: Extract audio
            logger.info("🎵 Extracting audio...")
            audio_path = self.extract_audio(video_path, stats)
            if not audio_path:
                return False
            
            logger.info(f"✅ Audio extraction completed in {stats.audio_extraction_time:.2f}s")
            
            # Step 2: Transcribe
            logger.info("🎤 Starting transcription...")
            result = self.transcribe_audio(audio_path, language, stats)
            if not result:
                return False
            
            # Step 3: Save transcription
            stats.finish()  # Calculate total time
            
            logger.info("💾 Saving transcription...")
            output_path = self.save_transcription(result, filename, include_timestamps, video_info, stats)
            if not output_path:
                return False
            
            # Step 4: Cleanup
            self.cleanup_temp_files()
            
            # Display final statistics
            logger.info("🎉 Transcription completed successfully!")
            logger.info(str(stats))
            logger.info(f"📄 Output file: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Transcription process failed: {e}")
            return False
        finally:
            # Always cleanup temp files
            self.cleanup_temp_files()

def main():
    """Main function with multi-language support"""
    parser = argparse.ArgumentParser(
        description="Video Transcriber with Local Whisper - Optimized for Chilean Spanish and English!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python transcriptor_whisper.py video.mp4
  
  # Spanish variants
  python transcriptor_whisper.py video.mp4 -l es-cl    # Chilean Spanish
  python transcriptor_whisper.py video.mp4 -l es-ar    # Argentinian Spanish
  python transcriptor_whisper.py video.mp4 -l es-mx    # Mexican Spanish
  
  # English variants
  python transcriptor_whisper.py video.mp4 -l en-us    # American English
  python transcriptor_whisper.py video.mp4 -l en-uk    # British English
  python transcriptor_whisper.py video.mp4 -l en-au    # Australian English
  
  # High quality with timestamps
  python transcriptor_whisper.py video.mp4 -m medium -l en-us -t
  python transcriptor_whisper.py video.mp4 -m large -l es-cl -t -v
        """
    )
    
    parser.add_argument(
        "video",
        help="Path to video file"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="base",
        choices=TranscriptorWhisper.AVAILABLE_MODELS,
        help="Whisper model to use (default: base)"
    )
    
    parser.add_argument(
        "-l", "--language",
        help="Language code (es-cl, en-us, en-uk, etc.)"
    )
    
    parser.add_argument(
        "-t", "--timestamps",
        action="store_true",
        help="Include timestamps in transcription"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="transcripciones",
        help="Output directory for transcriptions (default: transcripciones)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create transcriber
        transcriber = TranscriptorWhisper(model=args.model, output_dir=args.output)
        
        # Transcribe video
        success = transcriber.transcribe_video(
            args.video,
            language=args.language,
            include_timestamps=args.timestamps
        )
        
        if success:
            logger.info("✨ Transcription completed successfully")
            return 0
        else:
            logger.error("💥 Transcription failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Transcription interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 