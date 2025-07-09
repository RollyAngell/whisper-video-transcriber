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
from typing import Optional, Dict, Any, List
import time
import subprocess
import json
import platform
import threading
from urllib.error import URLError
import torch
import shutil
import re
from collections import Counter
import math
import asyncio
from typing import Union

# Configure logging for cross-platform compatibility
def setup_logging():
    """Setup logging with cross-platform compatibility"""
    # Check if we're on Windows and configure accordingly
    if platform.system() == 'Windows':
        # Use a simple formatter without emojis for Windows
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        # Set console encoding to UTF-8 if possible
        try:
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except:
            pass
    else:
        # Use the original format for macOS/Linux
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configure logging with the determined format
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('transcriptor.log', encoding='utf-8')
        ]
    )

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Cross-platform emoji/icon handling
class Icons:
    """Cross-platform icons and emojis"""
    def __init__(self):
        self.is_windows = platform.system() == 'Windows'
        
    @property
    def rocket(self):
        return "Starting" if self.is_windows else "🚀"
    
    @property
    def chart(self):
        return "Analyzing" if self.is_windows else "📊"
    
    @property
    def video(self):
        return "Video Info" if self.is_windows else "📹"
    
    @property
    def music(self):
        return "Extracting audio" if self.is_windows else "🎵"
    
    @property
    def check(self):
        return "Success" if self.is_windows else "✅"
    
    @property
    def microphone(self):
        return "Transcribing" if self.is_windows else "🎤"
    
    @property
    def save(self):
        return "Saving" if self.is_windows else "💾"
    
    @property
    def celebrate(self):
        return "Completed" if self.is_windows else "🎉"
    
    @property
    def clock(self):
        return "Processing Stats" if self.is_windows else "⏱️"
    
    @property
    def file(self):
        return "Output file" if self.is_windows else "📄"
    
    @property
    def sparkles(self):
        return "Success" if self.is_windows else "✨"
    
    @property
    def explosion(self):
        return "Error" if self.is_windows else "💥"
    
    # Country flags for language optimization
    def get_language_icon(self, language: str) -> str:
        if self.is_windows:
            lang_map = {
                'es-cl': 'Chilean Spanish',
                'en-us': 'American English',
                'en-uk': 'British English',
                'en-au': 'Australian English',
                'multi': 'Multi-language',
                'en': 'International English',
                'es': 'International Spanish'
            }
            return lang_map.get(language, f'{language} optimization')
        else:
            lang_map = {
                'es-cl': '🇨🇱 Chilean Spanish',
                'en-us': '🇺🇸 American English',
                'en-uk': '🇬🇧 British English',
                'en-au': '🇦🇺 Australian English',
                'multi': '🌐 Multi-language',
                'en': '🌍 International English',
                'es': '🌎 International Spanish'
            }
            return lang_map.get(language, f'🌐 {language}')

# Global icons instance
icons = Icons()

class ProgressBar:
    """Cross-platform progress bar for phase-based processing"""
    
    def __init__(self, use_unicode: bool = None):
        """
        Initialize progress bar
        
        Args:
            use_unicode: Whether to use Unicode characters (auto-detect if None)
        """
        self.is_windows = platform.system() == 'Windows'
        
        # Auto-detect Unicode support
        if use_unicode is None:
            # Use simple characters on Windows for better compatibility
            self.use_unicode = not self.is_windows
        else:
            self.use_unicode = use_unicode
        
        # Progress bar characters
        if self.use_unicode:
            self.filled_char = '█'
            self.empty_char = '▒'
        else:
            self.filled_char = '#'
            self.empty_char = '-'
        
        # Progress bar width
        self.bar_width = 40
        
        # Phase definitions
        self.phases = {
            'analysis': {'name': 'Video Analysis', 'weight': 5},
            'audio_extraction': {'name': 'Audio Extraction', 'weight': 20},
            'transcription': {'name': 'Transcription', 'weight': 70},
            'saving': {'name': 'Saving Results', 'weight': 5}
        }
        
        # Current progress state
        self.current_phase = None
        self.phase_progress = {}
        self.total_progress = 0.0
        
        # Initialize all phases to 0%
        for phase_key in self.phases.keys():
            self.phase_progress[phase_key] = 0.0
    
    def _calculate_total_progress(self) -> float:
        """Calculate overall progress based on weighted phases"""
        total_weight = sum(phase['weight'] for phase in self.phases.values())
        weighted_progress = 0.0
        
        for phase_key, phase_info in self.phases.items():
            phase_completion = self.phase_progress[phase_key] / 100.0
            weighted_progress += phase_completion * phase_info['weight']
        
        return (weighted_progress / total_weight) * 100.0
    
    def _create_progress_bar(self, percentage: float) -> str:
        """Create visual progress bar string"""
        filled_length = int(self.bar_width * percentage / 100)
        empty_length = self.bar_width - filled_length
        
        bar = self.filled_char * filled_length + self.empty_char * empty_length
        return f"[{bar}] {percentage:5.1f}%"
    
    def start_phase(self, phase_key: str):
        """Start a new processing phase"""
        if phase_key not in self.phases:
            raise ValueError(f"Unknown phase: {phase_key}")
        
        self.current_phase = phase_key
        self.phase_progress[phase_key] = 0.0
        self._update_display()
    
    def update_phase_progress(self, phase_key: str, percentage: float):
        """Update progress for a specific phase"""
        if phase_key not in self.phases:
            raise ValueError(f"Unknown phase: {phase_key}")
        
        self.phase_progress[phase_key] = max(0.0, min(100.0, percentage))
        self._update_display()
    
    def complete_phase(self, phase_key: str):
        """Mark a phase as completed"""
        if phase_key not in self.phases:
            raise ValueError(f"Unknown phase: {phase_key}")
        
        self.phase_progress[phase_key] = 100.0
        self._update_display()
    
    def _update_display(self):
        """Update the progress display"""
        # Clear previous lines (move cursor up and clear)
        if hasattr(self, '_last_lines_count'):
            for _ in range(self._last_lines_count):
                print('\033[F\033[K', end='')  # Move up and clear line
        
        lines = []
        
        # Show progress for each phase
        for phase_key, phase_info in self.phases.items():
            progress = self.phase_progress[phase_key]
            bar = self._create_progress_bar(progress)
            
            # Mark current phase
            status_indicator = "→" if phase_key == self.current_phase else " "
            
            # Color coding for completion status
            if progress == 100.0:
                status = "✓" if not self.is_windows else "OK"
            elif progress > 0:
                status = "..." if not self.is_windows else "..."
            else:
                status = "⋅" if not self.is_windows else "."
            
            line = f"{status_indicator} {status} {bar} - {phase_info['name']}"
            lines.append(line)
        
        # Show overall progress
        total_progress = self._calculate_total_progress()
        total_bar = self._create_progress_bar(total_progress)
        lines.append(f"")
        lines.append(f"Overall Progress: {total_bar}")
        
        # Print all lines
        for line in lines:
            print(line)
        
        # Store line count for next update
        self._last_lines_count = len(lines)
        
        # Flush output to ensure immediate display
        sys.stdout.flush()
    
    def finish(self):
        """Complete all phases and show final status"""
        for phase_key in self.phases.keys():
            self.phase_progress[phase_key] = 100.0
        
        self._update_display()
        print()  # Add final newline

# Robust import for moviepy
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    logger.warning("MoviePy not available, falling back to ffmpeg")
    MOVIEPY_AVAILABLE = False

class ModelDownloader:
    """Handles Whisper model downloads with progress tracking"""
    
    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]
    
    def __init__(self):
        """Initialize model downloader"""
        self.whisper_cache_dir = self._get_whisper_cache_dir()
        self.first_run_marker = Path.home() / ".whisper_video_transcriber_init"
    
    def _get_whisper_cache_dir(self) -> Path:
        """Get the Whisper cache directory"""
        # Try to get the cache directory from whisper module
        try:
            import whisper
            # Use the same cache directory as whisper
            cache_dir = Path.home() / ".cache" / "whisper"
            if platform.system() == "Windows":
                cache_dir = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "whisper"
            return cache_dir
        except:
            # Fallback to default location
            return Path.home() / ".cache" / "whisper"
    
    def is_first_run(self) -> bool:
        """Check if this is the first run of the application"""
        return not self.first_run_marker.exists()
    
    def mark_setup_complete(self):
        """Mark the setup as complete"""
        try:
            self.first_run_marker.touch()
            logger.info("First-time setup completed successfully")
        except Exception as e:
            logger.warning(f"Could not create setup marker: {e}")
    
    def check_model_availability(self, model_name: str) -> bool:
        """Check if a specific model is already downloaded"""
        try:
            # Check if model file exists in cache directory
            # Whisper models are typically stored as .pt files
            model_file = self.whisper_cache_dir / f"{model_name}.pt"
            
            if model_file.exists():
                return True
            
            # Fallback: try to load the model to see if it's available
            # This might download the model if it's not cached
            model = whisper.load_model(model_name, download_root=str(self.whisper_cache_dir))
            return True
        except Exception:
            return False
    
    def get_missing_models(self) -> list:
        """Get list of models that are not yet downloaded"""
        missing_models = []
        for model in self.AVAILABLE_MODELS:
            if not self.check_model_availability(model):
                missing_models.append(model)
        return missing_models
    
    def download_all_models(self, progress_bar: ProgressBar = None) -> bool:
        """
        Download all Whisper models with progress tracking
        
        Args:
            progress_bar: Progress bar instance for tracking
            
        Returns:
            True if all models downloaded successfully, False otherwise
        """
        logger.info(f"{icons.rocket} First-time setup: Downloading all Whisper models...")
        
        # Show disk space requirements
        total_size_mb = 39 + 74 + 244 + 769 + 1550  # Approximate sizes
        logger.info(f"{icons.chart} Total download size: ~{total_size_mb} MB (~{total_size_mb/1024:.1f} GB)")
        logger.info(f"{icons.clock} Estimated time: 5-30 minutes depending on your internet connection")
        logger.info("This is a one-time setup that will make future transcriptions faster!")
        
        # Check available disk space
        try:
            free_space = shutil.disk_usage(self.whisper_cache_dir.parent).free
            free_space_mb = free_space / (1024 * 1024)
            
            if free_space_mb < total_size_mb * 1.2:  # 20% buffer
                logger.warning(f"{icons.explosion} Warning: Low disk space!")
                logger.warning(f"Available: {free_space_mb:.0f} MB, Required: {total_size_mb} MB")
                logger.warning("You may need to free up disk space before continuing.")
            else:
                logger.info(f"{icons.check} Available disk space: {free_space_mb:.0f} MB")
        except Exception as e:
            logger.debug(f"Could not check disk space: {e}")
        
        # Give user time to read information
        logger.info("Starting download in 3 seconds... (Press Ctrl+C to cancel)")
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            logger.info("Download cancelled by user")
            return False
        
        # Create a special progress bar for model downloads if none provided
        if progress_bar is None:
            progress_bar = ProgressBar()
            # Override phases for model downloading
            progress_bar.phases = {
                'tiny': {'name': 'Downloading tiny model (~39 MB)', 'weight': 5},
                'base': {'name': 'Downloading base model (~74 MB)', 'weight': 10},
                'small': {'name': 'Downloading small model (~244 MB)', 'weight': 20},
                'medium': {'name': 'Downloading medium model (~769 MB)', 'weight': 30},
                'large': {'name': 'Downloading large model (~1550 MB)', 'weight': 35}
            }
            progress_bar.phase_progress = {key: 0.0 for key in progress_bar.phases.keys()}
        
        total_models = len(self.AVAILABLE_MODELS)
        success_count = 0
        
        for i, model_name in enumerate(self.AVAILABLE_MODELS):
            try:
                logger.info(f"{icons.chart} Downloading {model_name} model ({i+1}/{total_models})...")
                
                # Start this model's phase
                if progress_bar:
                    progress_bar.start_phase(model_name)
                
                # Download the model
                start_time = time.time()
                
                # Use threading to simulate progress during download
                download_thread = None
                download_running = threading.Event()
                
                def simulate_download_progress():
                    """Simulate download progress"""
                    download_running.set()
                    progress = 0.0
                    
                    while download_running.is_set() and progress < 95.0:
                        progress += 2.0  # Increment progress
                        if progress_bar:
                            progress_bar.update_phase_progress(model_name, progress)
                        time.sleep(0.3)  # Update every 300ms
                
                # Start progress simulation
                if progress_bar:
                    download_thread = threading.Thread(target=simulate_download_progress, daemon=True)
                    download_thread.start()
                
                # Actually download the model
                model = whisper.load_model(model_name, download_root=str(self.whisper_cache_dir))
                
                # Stop progress simulation
                if download_thread:
                    download_running.clear()
                    download_thread.join(timeout=1.0)
                
                # Complete the phase
                if progress_bar:
                    progress_bar.complete_phase(model_name)
                
                download_time = time.time() - start_time
                logger.info(f"{icons.check} Model '{model_name}' downloaded in {download_time:.2f}s")
                success_count += 1
                
            except Exception as e:
                logger.error(f"{icons.explosion} Failed to download {model_name} model: {e}")
                if progress_bar:
                    progress_bar.complete_phase(model_name)  # Mark as complete even if failed
                continue
        
        # Finalize progress
        if progress_bar:
            progress_bar.finish()
        
        if success_count == total_models:
            logger.info(f"{icons.celebrate} All {total_models} models downloaded successfully!")
            self.mark_setup_complete()
            return True
        else:
            logger.warning(f"Downloaded {success_count}/{total_models} models. Some downloads failed.")
            return False
    
    def ensure_models_available(self) -> bool:
        """Ensure all models are available, download if needed"""
        missing_models = self.get_missing_models()
        
        if not missing_models:
            logger.info(f"{icons.check} All Whisper models are already available")
            return True
        
        if self.is_first_run():
            logger.info(f"{icons.rocket} First run detected - downloading all models...")
            return self.download_all_models()
        else:
            logger.info(f"Missing models: {', '.join(missing_models)}")
            logger.info("Downloading missing models...")
            # For subsequent runs, just download missing models
            return self.download_missing_models(missing_models)
    
    def download_missing_models(self, missing_models: list) -> bool:
        """Download only the missing models"""
        success_count = 0
        
        for model_name in missing_models:
            try:
                logger.info(f"{icons.chart} Downloading missing {model_name} model...")
                start_time = time.time()
                
                model = whisper.load_model(model_name, download_root=str(self.whisper_cache_dir))
                
                download_time = time.time() - start_time
                logger.info(f"{icons.check} Model '{model_name}' downloaded in {download_time:.2f}s")
                success_count += 1
                
            except Exception as e:
                logger.error(f"{icons.explosion} Failed to download {model_name} model: {e}")
                continue
        
        return success_count == len(missing_models)

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
        return (f"{icons.clock} Processing Statistics:\n"
                f"   Audio extraction: {self.audio_extraction_time:.2f}s\n"
                f"   Transcription: {self.transcription_time:.2f}s\n"
                f"   Total time: {self.total_time:.2f}s\n"
                f"   Speed factor: {speed_factor:.2f}x "
                f"({'faster' if speed_factor > 1 else 'slower'} than real-time)")



class OpenAISummarizer:
    """ChatGPT-powered generative summarizer"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize OpenAI summarizer
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use (gpt-3.5-turbo, gpt-4, gpt-4o)
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        
        # Try to import and initialize OpenAI client
        try:
            import openai
            self.openai = openai
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info(f"OpenAI client initialized with model: {model}")
            else:
                logger.warning("No OpenAI API key provided. ChatGPT summarization disabled.")
        except ImportError:
            logger.warning("OpenAI library not installed. Install with: pip install openai>=1.0.0")
            self.openai = None
    
    def is_available(self) -> bool:
        """Check if OpenAI summarization is available"""
        return self.client is not None and self.openai is not None
    
    def _create_spanish_prompt(self, text: str, max_points: int = 7) -> str:
        """Create Spanish prompt for ChatGPT"""
        return f"""Eres un asistente experto en resumir contenido transcrito. Crea un resumen profesional y estructurado del siguiente texto en español, incluyendo también una traducción completa al inglés.

INSTRUCCIONES:
- Genera un título descriptivo y atractivo que capture la esencia del contenido
- Extrae máximo {max_points} puntos clave más importantes
- Identifica temas principales con su frecuencia/importancia
- Usa un lenguaje claro y profesional en español
- Mantén la coherencia y el flujo narrativo
- Si el contenido es técnico, preserva términos importantes
- Crea oraciones nuevas (no copies textualmente)
- Proporciona una traducción completa y precisa al inglés

FORMATO REQUERIDO:
=== RESUMEN AUTOMÁTICO ===

🎯 TÍTULO: [Título descriptivo y atractivo basado en el contenido principal]

📋 TEMAS PRINCIPALES:
• [tema importante] ([relevancia: alta/media])
• [otro tema] ([relevancia: alta/media])
• [máximo 8 temas]

📝 RESUMEN DEL CONTENIDO:
1. [Punto clave uno - reformulado de manera clara]
2. [Punto clave dos - con contexto relevante]
3. [Continúa hasta máximo {max_points} puntos]

📊 ESTADÍSTICAS DEL RESUMEN:
   • Texto original: {len(text)} caracteres
   • Resumen: [calcular] caracteres
   • Compresión: [calcular]% del texto original
   • Puntos clave: [número de puntos]

=== ENGLISH TRANSLATION ===

🎯 TITLE: [Descriptive and engaging title based on main content - English translation]

📋 MAIN TOPICS:
• [important topic] ([relevance: high/medium])
• [another topic] ([relevance: high/medium])
• [maximum 8 topics]

📝 CONTENT SUMMARY:
1. [Key point one - clearly reformulated]
2. [Key point two - with relevant context]
3. [Continue up to maximum {max_points} points]

TEXTO A RESUMIR:
{text}

Responde ÚNICAMENTE con el resumen en el formato especificado."""

    def _create_english_prompt(self, text: str, max_points: int = 7) -> str:
        """Create English prompt for ChatGPT"""
        return f"""You are an expert assistant specialized in summarizing transcribed content. Create a professional and structured summary of the following English text, including also a complete translation to Spanish.

INSTRUCTIONS:
- Generate a descriptive and engaging title that captures the essence of the content
- Extract maximum {max_points} most important key points
- Identify main topics with their frequency/importance
- Use clear and professional English
- Maintain coherence and narrative flow
- If content is technical, preserve important terminology
- Create new sentences (don't copy verbatim)
- Provide a complete and accurate translation to Spanish

REQUIRED FORMAT:
=== AUTOMATIC SUMMARY ===

🎯 TITLE: [Descriptive and engaging title based on the main content]

📋 MAIN TOPICS:
• [important topic] ([relevance: high/medium])
• [another topic] ([relevance: high/medium])
• [maximum 8 topics]

📝 CONTENT SUMMARY:
1. [Key point one - reformulated clearly]
2. [Key point two - with relevant context]
3. [Continue up to maximum {max_points} points]

📊 SUMMARY STATISTICS:
   • Original text: {len(text)} characters
   • Summary: [calculate] characters
   • Compression: [calculate]% of original
   • Key points: [number of points]

=== TRADUCCIÓN AL ESPAÑOL ===

🎯 TÍTULO: [Título descriptivo y atractivo basado en el contenido principal - traducción al español]

📋 TEMAS PRINCIPALES:
• [tema importante] ([relevancia: alta/media])
• [otro tema] ([relevancia: alta/media])
• [máximo 8 temas]

📝 RESUMEN DEL CONTENIDO:
1. [Punto clave uno - reformulado de manera clara]
2. [Punto clave dos - con contexto relevante]
3. [Continúa hasta máximo {max_points} puntos]

TEXT TO SUMMARIZE:
{text}

Respond ONLY with the summary in the specified format."""

    def _determine_max_points(self, text_length: int) -> int:
        """Determine optimal number of summary points based on content length"""
        if text_length < 500:
            return 3
        elif text_length < 1500:
            return 4
        elif text_length < 3000:
            return 6
        elif text_length < 6000:
            return 8
        else:
            return 10

    def generate_summary(self, text: str, language: str = "auto") -> str:
        """
        Generate summary using ChatGPT
        
        Args:
            text: Text to summarize
            language: Language code (es, en, auto)
            
        Returns:
            Generated summary or error message
        """
        if not self.is_available():
            return self._fallback_message(language)
        
        if not text or len(text.strip()) < 50:
            return self._too_short_message(language)
        
        # Auto-detect language if needed
        if language == "auto":
            language = self._detect_language(text)
        
        # Determine optimal summary length
        max_points = self._determine_max_points(len(text))
        
        # Create appropriate prompt
        if language.startswith('es'):
            prompt = self._create_spanish_prompt(text, max_points)
        else:
            prompt = self._create_english_prompt(text, max_points)
        
        try:
            logger.info(f"🤖 Generating ChatGPT summary using {self.model}...")
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional content summarizer. Always follow the exact format requested and provide high-quality, coherent summaries."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,  # Limit response length
                temperature=0.3,  # Lower temperature for more consistent results
                top_p=0.9
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Validate that we got a proper summary
            if not summary or len(summary) < 100:
                logger.warning("ChatGPT returned unexpectedly short summary, using fallback")
                return self._fallback_message(language)
            
            # Add generation info
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else "unknown"
            summary += f"\n\n🤖 Generated by: {self.model} | Tokens used: {tokens_used}"
            
            logger.info(f"✅ ChatGPT summary generated successfully ({tokens_used} tokens)")
            return summary
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ ChatGPT summarization failed: {error_msg}")
            
            # Provide helpful error messages
            if "insufficient_quota" in error_msg.lower():
                return self._quota_error_message(language)
            elif "invalid_api_key" in error_msg.lower():
                return self._api_key_error_message(language)
            elif "rate_limit" in error_msg.lower():
                return self._rate_limit_message(language)
            else:
                return self._generic_error_message(language, error_msg)
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        spanish_indicators = ['que', 'de', 'la', 'el', 'en', 'y', 'es', 'para', 'con', 'por']
        english_indicators = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that']
        
        words = re.findall(r'\b\w+\b', text.lower())[:200]  # Check first 200 words
        
        spanish_count = sum(1 for word in words if word in spanish_indicators)
        english_count = sum(1 for word in words if word in english_indicators)
        
        if spanish_count > english_count:
            return 'es'
        else:
            return 'en'
    
    def _fallback_message(self, language: str) -> str:
        """Message when ChatGPT is not available"""
        if language.startswith('es'):
            return ("=== RESUMEN AUTOMÁTICO ===\n\n"
                   "⚠️ ChatGPT no disponible. Configure su API key de OpenAI para usar resúmenes generativos.\n"
                   "Use: python transcriptor_whisper.py video.mp4 --openai-key su_api_key\n"
                   "O configure: export OPENAI_API_KEY=su_api_key")
        else:
            return ("=== AUTOMATIC SUMMARY ===\n\n"
                   "⚠️ ChatGPT unavailable. Configure your OpenAI API key for generative summaries.\n"
                   "Use: python transcriptor_whisper.py video.mp4 --openai-key your_api_key\n"
                   "Or set: export OPENAI_API_KEY=your_api_key")
    
    def _too_short_message(self, language: str) -> str:
        """Message for content too short to summarize"""
        if language.startswith('es'):
            return "=== RESUMEN AUTOMÁTICO ===\n\n⚠️ Texto demasiado corto para generar un resumen con ChatGPT."
        else:
            return "=== AUTOMATIC SUMMARY ===\n\n⚠️ Text too short to generate a ChatGPT summary."
    
    def _quota_error_message(self, language: str) -> str:
        """Message for quota exceeded error"""
        if language.startswith('es'):
            return ("=== RESUMEN AUTOMÁTICO ===\n\n"
                   "❌ Error: Cuota de OpenAI agotada.\n"
                   "Verifique su plan de facturación en: https://platform.openai.com/account/billing")
        else:
            return ("=== AUTOMATIC SUMMARY ===\n\n"
                   "❌ Error: OpenAI quota exceeded.\n"
                   "Check your billing plan at: https://platform.openai.com/account/billing")
    
    def _api_key_error_message(self, language: str) -> str:
        """Message for invalid API key error"""
        if language.startswith('es'):
            return ("=== RESUMEN AUTOMÁTICO ===\n\n"
                   "❌ Error: API key de OpenAI inválida.\n"
                   "Obtenga una clave válida en: https://platform.openai.com/api-keys")
        else:
            return ("=== AUTOMATIC SUMMARY ===\n\n"
                   "❌ Error: Invalid OpenAI API key.\n"
                   "Get a valid key at: https://platform.openai.com/api-keys")
    
    def _rate_limit_message(self, language: str) -> str:
        """Message for rate limit error"""
        if language.startswith('es'):
            return ("=== RESUMEN AUTOMÁTICO ===\n\n"
                   "⏳ Error: Límite de velocidad de OpenAI alcanzado.\n"
                   "Espere unos minutos e intente nuevamente.")
        else:
            return ("=== AUTOMATIC SUMMARY ===\n\n"
                   "⏳ Error: OpenAI rate limit reached.\n"
                   "Please wait a few minutes and try again.")
    
    def _generic_error_message(self, language: str, error: str) -> str:
        """Generic error message"""
        if language.startswith('es'):
            return (f"=== RESUMEN AUTOMÁTICO ===\n\n"
                   f"❌ Error al generar resumen con ChatGPT: {error}\n"
                   f"Cayendo de vuelta al resumen local.")
        else:
            return (f"=== AUTOMATIC SUMMARY ===\n\n"
                   f"❌ Error generating ChatGPT summary: {error}\n"
                   f"Falling back to local summary.")



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
    
    def __init__(self, model: str = "medium", output_dir: str = "transcriptions", skip_model_check: bool = False):
        """
        Initialize the transcriber with specified Whisper model
        
        Args:
            model: Whisper model to use (tiny, base, small, medium, large)
            output_dir: Directory to save transcriptions
            skip_model_check: Skip model availability check (for testing)
        """
        self.model_name = model
        self.output_dir = Path(output_dir)
        self.temp_dir = Path("temp")
        
        self._validate_model()
        self._setup_directories()
        
        # Check and download models if needed (unless skipped)
        if not skip_model_check:
            self._ensure_models_available()
        
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
    
    def _ensure_models_available(self) -> None:
        """Ensure all Whisper models are available, download if needed"""
        model_downloader = ModelDownloader()
        
        # Check if this is the first run
        if model_downloader.is_first_run():
            logger.info(f"{icons.rocket} Welcome to Video Transcriber!")
            logger.info("This appears to be your first run. Setting up all Whisper models...")
            logger.info("This is a one-time setup that will download all models for future use.")
            logger.info("After setup, you can use any model (tiny, base, small, medium, large) instantly!")
            
            # Download all models with progress tracking
            success = model_downloader.download_all_models()
            
            if not success:
                logger.warning("Some models failed to download. The application will continue with available models.")
                logger.info("You can retry downloading models later using: --download-models")
        else:
            # For subsequent runs, just ensure the current model is available
            if not model_downloader.check_model_availability(self.model_name):
                logger.info(f"Model '{self.model_name}' not found. Downloading...")
                success = model_downloader.download_missing_models([self.model_name])
                if not success:
                    logger.error(f"Failed to download model '{self.model_name}'. Try using --download-models to set up all models.")
                    raise RuntimeError(f"Required model '{self.model_name}' is not available")
    
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
                "log_message": f"{icons.get_language_icon('en')} Auto-detection mode - Best for multi-accent/multi-language videos"
            }
        
        lang_lower = language.lower()
        
        # Multi-language/accent optimization
        if lang_lower in ["multi", "mixed", "international"]:
            return {
                "language": None,
                "initial_prompt": self._get_multi_language_prompt(),
                "log_message": f"{icons.get_language_icon('multi')} optimization activated"
            }
        
        # English - general for multi-accent scenarios
        elif lang_lower == "en":
            return {
                "language": "en",
                "initial_prompt": self._get_international_english_prompt(),
                "log_message": f"{icons.get_language_icon('en')} optimization (handles multiple accents)"
            }
        
        # Spanish - general for multi-dialect scenarios  
        elif lang_lower == "es":
            return {
                "language": "es",
                "initial_prompt": self._get_international_spanish_prompt(),
                "log_message": f"{icons.get_language_icon('es')} optimization (handles multiple dialects)"
            }
        
        # Spanish variants
        elif lang_lower in ["es-cl", "es_cl", "chile", "chileno"]:
            return {
                "language": "es",
                "initial_prompt": self.chilean_spanish_prompt,
                "log_message": f"{icons.get_language_icon('es-cl')} optimization activated"
            }
        elif lang_lower.startswith("es-") or lang_lower.startswith("es_"):
            return {
                "language": "es",
                "initial_prompt": None,
                "log_message": f"{icons.get_language_icon('es')} variant ({language}) activated"
            }
        
        # English variants
        elif lang_lower in ["en-us", "en_us", "american", "usa"]:
            return {
                "language": "en",
                "initial_prompt": self.english_us_prompt,
                "log_message": f"{icons.get_language_icon('en-us')} optimization activated"
            }
        elif lang_lower in ["en-uk", "en_uk", "british", "uk"]:
            return {
                "language": "en",
                "initial_prompt": self.english_uk_prompt,
                "log_message": f"{icons.get_language_icon('en-uk')} optimization activated"
            }
        elif lang_lower in ["en-au", "en_au", "australian", "australia"]:
            return {
                "language": "en",
                "initial_prompt": self.english_general_prompt,
                "log_message": f"{icons.get_language_icon('en-au')} optimization activated"
            }
        elif lang_lower in ["en-ca", "en_ca", "canadian", "canada"]:
            return {
                "language": "en",
                "initial_prompt": self.english_general_prompt,
                "log_message": "Canadian English optimization activated"
            }
        elif lang_lower.startswith("en-") or lang_lower.startswith("en_") or lang_lower == "en":
            return {
                "language": "en",
                "initial_prompt": self.english_general_prompt,
                "log_message": f"{icons.get_language_icon('en')} variant ({language}) activated"
            }
        
        # Other languages
        else:
            return {
                "language": language,
                "initial_prompt": None,
                "log_message": f"Language {language} activated"
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
    
    def _extract_audio_moviepy(self, video_path: Path, audio_path: Path, progress_bar: ProgressBar = None) -> bool:
        """Extract audio using MoviePy with progress tracking"""
        try:
            logger.info(f"Extracting audio using MoviePy: {video_path.name}")
            
            if progress_bar:
                progress_bar.update_phase_progress('audio_extraction', 40.0)
            
            video = VideoFileClip(str(video_path))
            if video.audio is None:
                logger.error("No audio track found in video")
                video.close()
                return False
            
            if progress_bar:
                progress_bar.update_phase_progress('audio_extraction', 60.0)
            
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
            
            if progress_bar:
                progress_bar.update_phase_progress('audio_extraction', 95.0)
            
            logger.info(f"Audio extracted successfully to: {audio_path}")
            return True
            
        except Exception as e:
            logger.error(f"MoviePy extraction failed: {e}")
            return False
    
    def _extract_audio_ffmpeg(self, video_path: Path, audio_path: Path, progress_bar: ProgressBar = None) -> bool:
        """Extract audio using ffmpeg directly with progress tracking"""
        try:
            logger.info(f"Extracting audio using ffmpeg: {video_path.name}")
            
            if progress_bar:
                progress_bar.update_phase_progress('audio_extraction', 40.0)
            
            # Check if ffmpeg is available
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            
            if progress_bar:
                progress_bar.update_phase_progress('audio_extraction', 50.0)
            
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
            
            if progress_bar:
                progress_bar.update_phase_progress('audio_extraction', 70.0)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
            
            if progress_bar:
                progress_bar.update_phase_progress('audio_extraction', 95.0)
            
            logger.info(f"Audio extracted successfully to: {audio_path}")
            return True
            
        except FileNotFoundError:
            logger.error("ffmpeg not found. Install with: brew install ffmpeg")
            return False
        except Exception as e:
            logger.error(f"FFmpeg extraction failed: {e}")
            return False
    
    def extract_audio(self, video_path: Path, stats: ProcessingStats, progress_bar: ProgressBar = None) -> Optional[Path]:
        """
        Extract audio from video using available method with progress tracking
        
        Args:
            video_path: Path to the video file
            stats: Processing statistics object
            progress_bar: Progress bar instance for tracking
            
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
        
        # Update progress - starting extraction
        if progress_bar:
            progress_bar.update_phase_progress('audio_extraction', 10.0)
        
        with self._temporary_audio_file() as audio_path:
            # Try MoviePy first, then ffmpeg
            if progress_bar:
                progress_bar.update_phase_progress('audio_extraction', 30.0)
            
            success = False
            if MOVIEPY_AVAILABLE and self._extract_audio_moviepy(video_path, audio_path, progress_bar):
                success = True
            elif self._extract_audio_ffmpeg(video_path, audio_path, progress_bar):
                success = True
            
            if success:
                # Copy to permanent location and return
                if progress_bar:
                    progress_bar.update_phase_progress('audio_extraction', 80.0)
                
                permanent_path = self.temp_dir / f"audio_extracted_{int(time.time())}.wav"
                permanent_path.write_bytes(audio_path.read_bytes())
                stats.audio_extraction_time = time.time() - extraction_start
                
                if progress_bar:
                    progress_bar.update_phase_progress('audio_extraction', 100.0)
                
                return permanent_path
            else:
                logger.error("Failed to extract audio with both methods")
                stats.audio_extraction_time = time.time() - extraction_start
                return None
    
    def transcribe_audio(self, audio_path: Path, language: Optional[str], stats: ProcessingStats, progress_bar: ProgressBar = None) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio file using Whisper with language-specific optimization and progress tracking
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
            
            if progress_bar:
                progress_bar.update_phase_progress('transcription', 10.0)
            
            # Setup progress simulation for transcription
            progress_thread = None
            progress_running = threading.Event()
            
            def simulate_progress():
                """Simulate transcription progress in a separate thread"""
                start_time = time.time()
                progress_running.set()
                
                # Estimate transcription time based on audio duration and model
                # These are rough estimates based on typical Whisper performance
                model_speed_factors = {
                    'tiny': 15.0,    # ~15x real-time
                    'base': 8.0,     # ~8x real-time
                    'small': 4.0,    # ~4x real-time
                    'medium': 2.5,   # ~2.5x real-time
                    'large': 1.5     # ~1.5x real-time
                }
                
                speed_factor = model_speed_factors.get(self.model_name, 3.0)
                estimated_duration = stats.video_duration / speed_factor if stats.video_duration > 0 else 30.0
                
                # Simulate progress curve (starts slow, speeds up, then slows down)
                while progress_running.is_set():
                    elapsed = time.time() - start_time
                    progress_ratio = min(elapsed / estimated_duration, 1.0)
                    
                    # Use a sigmoid-like curve for realistic progress
                    # Progress is faster in the middle, slower at start and end
                    if progress_ratio < 0.1:
                        progress_percent = 10.0 + (progress_ratio * 10.0 * 20.0)  # 10-30%
                    elif progress_ratio < 0.9:
                        progress_percent = 30.0 + ((progress_ratio - 0.1) * 1.25 * 55.0)  # 30-85%
                    else:
                        progress_percent = 85.0 + ((progress_ratio - 0.9) * 10.0 * 10.0)  # 85-95%
                    
                    progress_percent = min(95.0, progress_percent)  # Cap at 95% until completion
                    
                    if progress_bar:
                        progress_bar.update_phase_progress('transcription', progress_percent)
                    
                    time.sleep(0.5)  # Update every 500ms
            
            # Start progress simulation thread
            if progress_bar:
                progress_thread = threading.Thread(target=simulate_progress, daemon=True)
                progress_thread.start()
            
            # Run the actual transcription
            result = self.whisper_model.transcribe(str(audio_path), **transcription_args)
            
            # Stop progress simulation
            if progress_thread:
                progress_running.clear()
                progress_thread.join(timeout=1.0)
            
            # Complete the progress
            if progress_bar:
                progress_bar.update_phase_progress('transcription', 100.0)
            
            stats.transcription_time = time.time() - transcription_start
            logger.info(f"Transcription completed in {stats.transcription_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            # Stop progress simulation if it's running
            if 'progress_running' in locals():
                progress_running.clear()
            stats.transcription_time = time.time() - transcription_start if 'transcription_start' in locals() else 0
            return None
    
    def save_transcription(self, result: Dict[str, Any], filename: str, include_timestamps: bool, 
                          video_info: VideoInfo, stats: ProcessingStats, include_summary: bool = True,
                          openai_api_key: Optional[str] = None, 
                          openai_model: str = "gpt-4o") -> Optional[Path]:
        """
        Save transcription to file with metadata and automatic summary
        
        Args:
            result: Whisper transcription result
            filename: Output filename (without extension)
            include_timestamps: Whether to include timestamps
            video_info: Video information
            stats: Processing statistics
            include_summary: Whether to include automatic summary
            openai_api_key: OpenAI API key for ChatGPT
            openai_model: OpenAI model to use
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            output_path = self.output_dir / f"{filename}.txt"
            
            # Generate automatic summary if enabled
            summary = None
            if include_summary:
                detected_language = result.get('language', 'es')
                
                logger.info(f"🤖 Generating ChatGPT summary...")
                
                # Initialize OpenAI summarizer
                summarizer = OpenAISummarizer(
                    api_key=openai_api_key,
                    model=openai_model
                )
                
                # Use the main transcription text for summary
                transcription_text = result['text'].strip()
                
                # Generate summary using ChatGPT
                summary = summarizer.generate_summary(
                    transcription_text, 
                    language=detected_language
                )
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write header with complete metadata
                f.write(f"=== TRANSCRIPTION OF: {filename} ===\n\n")
                f.write(f"{icons.video} Video Information:\n")
                f.write(f"   File: {video_info.path.name}\n")
                f.write(f"   Size: {video_info.size_mb:.2f} MB\n")
                f.write(f"   Duration: {VideoInfo._format_duration(video_info.duration_seconds)}\n")
                f.write(f"   Format: {video_info.format}\n")
                f.write(f"   Resolution: {video_info.resolution}\n")
                f.write(f"   FPS: {video_info.fps:.2f}\n\n")
                
                f.write(f"Processing Information:\n")
                f.write(f"   Model used: {self.model_name}\n")
                f.write(f"   Detected language: {result.get('language', 'Unknown')}\n")
                f.write(f"   Audio extraction time: {stats.audio_extraction_time:.2f}s\n")
                f.write(f"   Transcription time: {stats.transcription_time:.2f}s\n")
                f.write(f"   Total processing time: {stats.total_time:.2f}s\n")
                f.write(f"   Processing speed: {stats.get_speed_factor():.2f}x real-time\n\n")
                
                # Add the automatic summary if generated
                if summary:
                    f.write(f"{summary}\n\n")
                    # Add separator line
                    f.write("=" * 80 + "\n\n")
                
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
            
            if include_summary:
                logger.info(f"{icons.check} Transcription and summary saved to: {output_path}")
            else:
                logger.info(f"{icons.check} Transcription saved to: {output_path}")
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
    
    def transcribe_video(self, video_path: str, language: Optional[str] = None, include_timestamps: bool = False, 
                        include_summary: bool = True, 
                        openai_api_key: Optional[str] = None, openai_model: str = "gpt-4o") -> bool:
        """
        Complete video transcription process with detailed metrics and progress tracking
        
        Args:
            video_path: Path to video file
            language: Language code (optional)
            include_timestamps: Whether to include timestamps
            include_summary: Whether to include automatic summary
            openai_api_key: OpenAI API key for ChatGPT summarization
            openai_model: OpenAI model to use (gpt-4o, gpt-3.5-turbo, gpt-4, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        video_path = Path(video_path)
        filename = video_path.stem
        
        # Initialize processing statistics and progress bar
        stats = ProcessingStats()
        progress_bar = ProgressBar()
        
        logger.info(f"{icons.rocket} Starting transcription of: {video_path.name}")
        
        try:
            # Phase 1: Video Analysis
            progress_bar.start_phase('analysis')
            logger.info(f"{icons.chart} Analyzing video file...")
            
            # Simulate analysis progress (since it's usually fast)
            progress_bar.update_phase_progress('analysis', 50.0)
            video_info = self.get_video_info(video_path)
            stats.video_duration = video_info.duration_seconds
            
            # Complete analysis phase
            progress_bar.complete_phase('analysis')
            
            # Display video information
            logger.info(f"{icons.video} Video Information:")
            for line in str(video_info).split('\n'):
                logger.info(f"   {line}")
            
            # Phase 2: Audio Extraction
            progress_bar.start_phase('audio_extraction')
            logger.info(f"{icons.music} Extracting audio...")
            
            audio_path = self.extract_audio(video_path, stats, progress_bar)
            if not audio_path:
                return False
            
            progress_bar.complete_phase('audio_extraction')
            logger.info(f"{icons.check} Audio extraction completed in {stats.audio_extraction_time:.2f}s")
            
            # Phase 3: Transcription (main phase)
            progress_bar.start_phase('transcription')
            logger.info(f"{icons.microphone} Starting transcription...")
            
            result = self.transcribe_audio(audio_path, language, stats, progress_bar)
            if not result:
                return False
            
            progress_bar.complete_phase('transcription')
            
            # Phase 4: Saving results
            progress_bar.start_phase('saving')
            stats.finish()  # Calculate total time
            
            if include_summary:
                logger.info(f"{icons.save} Saving transcription and generating summary...")
            else:
                logger.info(f"{icons.save} Saving transcription (summary disabled)...")
            progress_bar.update_phase_progress('saving', 30.0)
            
            output_path = self.save_transcription(result, filename, include_timestamps, video_info, stats, include_summary, openai_api_key, openai_model)
            if not output_path:
                return False
            
            progress_bar.update_phase_progress('saving', 90.0)
            progress_bar.complete_phase('saving')
            
            # Finalize progress display
            progress_bar.finish()
            
            # Cleanup
            self.cleanup_temp_files()
            
            # Display final statistics
            logger.info(f"{icons.celebrate} Transcription completed successfully!")
            logger.info(str(stats))
            logger.info(f"{icons.file} Output file: {output_path}")
            
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
        description="Video Transcriber with Local Whisper + ChatGPT - Optimized for Chilean Spanish and English! Features automatic ChatGPT-powered summarization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with ChatGPT summary
  python transcriptor_whisper.py video.mp4 --openai-key sk-...
  
  # Spanish variants
  python transcriptor_whisper.py video.mp4 -l es-cl --openai-key sk-...    # Chilean Spanish
  python transcriptor_whisper.py video.mp4 -l es-ar --openai-key sk-...    # Argentinian Spanish
  python transcriptor_whisper.py video.mp4 -l es-mx --openai-key sk-...    # Mexican Spanish
  
  # English variants
  python transcriptor_whisper.py video.mp4 -l en-us --openai-key sk-...    # American English
  python transcriptor_whisper.py video.mp4 -l en-uk --openai-key sk-...    # British English
  python transcriptor_whisper.py video.mp4 -l en-au --openai-key sk-...    # Australian English
  
  # High quality with timestamps and summary
  python transcriptor_whisper.py video.mp4 -m medium -l en-us -t --openai-key sk-...
  python transcriptor_whisper.py video.mp4 -m large -l es-cl -t -v --openai-key sk-...
  
  # Using different ChatGPT models
  python transcriptor_whisper.py video.mp4 --openai-key sk-... --openai-model gpt-4
  python transcriptor_whisper.py video.mp4 --openai-key sk-... --openai-model gpt-4o
  
  # Environment variable for API key
  export OPENAI_API_KEY=sk-your-api-key
  python transcriptor_whisper.py video.mp4
  
  # Disable automatic summary
  python transcriptor_whisper.py video.mp4 --no-summary
  
  # Setup and maintenance
  python transcriptor_whisper.py --download-models     # Download all models
  python transcriptor_whisper.py --check-models        # Check model status
        """
    )
    
    parser.add_argument(
        "video",
        nargs="?",
        help="Path to video file"
    )
    
    parser.add_argument(
        "-m", "--model",
        default="medium",
        choices=TranscriptorWhisper.AVAILABLE_MODELS,
        help="Whisper model to use (default: medium)"
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
        "--no-summary",
        action="store_true",
        help="Disable automatic ChatGPT summary generation (summary is included by default)"
    )
    
    parser.add_argument(
        "--openai-key",
        help="OpenAI API key for ChatGPT summarization (or set OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--openai-model",
        default="gpt-4o",
        choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o", "gpt-4o-mini"],
        help="OpenAI model to use for summarization (default: gpt-4o)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="transcriptions",
        help="Output directory for transcriptions (default: transcriptions)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download all Whisper models and exit"
    )
    
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Check which models are available and exit"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine if summary should be included
    include_summary = not args.no_summary
    
    # Handle OpenAI API key
    openai_api_key = args.openai_key
    if not openai_api_key:
        # Try to get from environment variable
        openai_api_key = os.environ.get('OPENAI_API_KEY')
    
    # Check if summary is requested without API key
    if include_summary and not openai_api_key:
        logger.warning("⚠️ Summary requested but no OpenAI API key provided")
        logger.warning("   • Use --openai-key your_api_key")
        logger.warning("   • Or set environment variable: export OPENAI_API_KEY=your_api_key")
        logger.warning("   • Get API key at: https://platform.openai.com/api-keys")
        logger.warning("   • Or use --no-summary to disable summarization")
        return 1
    
    try:
        # Handle model management commands
        if args.download_models:
            logger.info(f"{icons.rocket} Downloading all Whisper models...")
            model_downloader = ModelDownloader()
            success = model_downloader.download_all_models()
            if success:
                logger.info(f"{icons.celebrate} All models downloaded successfully!")
                return 0
            else:
                logger.error(f"{icons.explosion} Some models failed to download")
                return 1
        
        if args.check_models:
            logger.info(f"{icons.chart} Checking Whisper model availability...")
            model_downloader = ModelDownloader()
            
            available_models = []
            missing_models = []
            
            for model in ModelDownloader.AVAILABLE_MODELS:
                if model_downloader.check_model_availability(model):
                    available_models.append(model)
                else:
                    missing_models.append(model)
            
            logger.info(f"{icons.check} Available models: {', '.join(available_models) if available_models else 'None'}")
            logger.info(f"{icons.explosion} Missing models: {', '.join(missing_models) if missing_models else 'None'}")
            
            if model_downloader.is_first_run():
                logger.info(f"{icons.rocket} This appears to be your first run. Use --download-models to set up all models.")
            
            return 0
        
        # Require video argument for transcription
        if not args.video:
            parser.error("Video file is required unless using --download-models or --check-models")
        
        # Create transcriber
        transcriber = TranscriptorWhisper(model=args.model, output_dir=args.output)
        
        # Transcribe video
        success = transcriber.transcribe_video(
            args.video,
            language=args.language,
            include_timestamps=args.timestamps,
            include_summary=include_summary,
            openai_api_key=openai_api_key,
            openai_model=args.openai_model
        )
        
        if success:
            logger.info(f"{icons.sparkles} Transcription completed successfully")
            return 0
        else:
            logger.error(f"{icons.explosion} Transcription failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Transcription interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 