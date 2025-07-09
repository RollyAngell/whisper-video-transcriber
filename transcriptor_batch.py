#!/usr/bin/env python3
"""
Batch Video Transcriber - Automated processing for Spanish and English videos
Author: AI Assistant
Description: Automatically processes all videos in Spanish and English folders
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
import json
import time
import argparse
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('transcriptor_batch.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class BatchTranscriptor:
    """Batch processor for video transcription"""
    
    def __init__(self, base_path: str = "/Users/rollyangell/Desktop/RollyAngell/Work/Videos"):
        """
        Initialize batch transcriptor
        
        Args:
            base_path: Base path containing Spanish and English folders
        """
        self.base_path = Path(base_path)
        self.spanish_path = self.base_path / "Spanish"
        self.english_path = self.base_path / "English"
        
        # Supported video formats
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        # Progress tracking
        self.progress_file = Path("transcriptor_progress.json")
        self.processed_files = self.load_progress()
        
        # Drive upload tracking - NEW
        self.drive_uploaded_files = self.load_drive_progress()
        
        # Statistics
        self.stats = {
            'total_videos': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'organized': 0,
            'spanish_videos': 0,
            'english_videos': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Progress tracking variables
        self.current_video_index = 0
        self.total_videos_to_process = 0
    
    def load_progress(self) -> set:
        """Load previously processed files from progress file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return set(data.get('processed_files', []))
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
        return set()
    
    def load_drive_progress(self) -> set:
        """Load previously uploaded files from progress file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return set(data.get('drive_uploaded_files', []))
            except Exception as e:
                logger.warning(f"Could not load drive progress: {e}")
        return set()
    
    def save_progress(self, processed_file: str, drive_uploaded: bool = False):
        """Save progress to file"""
        self.processed_files.add(processed_file)
        
        if drive_uploaded:
            self.drive_uploaded_files.add(processed_file)
        
        try:
            progress_data = {
                'processed_files': list(self.processed_files),
                'drive_uploaded_files': list(self.drive_uploaded_files),
                'last_update': datetime.now().isoformat(),
                'stats': self.stats
            }
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save progress: {e}")
    
    def is_already_uploaded_to_drive(self, video_folder: Path) -> bool:
        """Check if video folder is already uploaded to Drive"""
        return str(video_folder) in self.drive_uploaded_files
    
    def find_videos(self, directory: Path) -> list:
        """Find only loose video files in the root directory (not in subfolders)"""
        videos = []
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return videos
        
        # Only search in the root directory, not in subfolders
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.video_extensions:
                # Check if this video is already processed
                if str(file_path) not in self.processed_files:
                    videos.append(file_path)
                else:
                    logger.debug(f"⏭️  Skipping already processed video: {file_path.name}")
        
        return sorted(videos)
    
    def get_language_for_path(self, video_path: Path) -> str:
        """Determine language based on video path"""
        if "Spanish" in str(video_path):
            return "es-cl"
        elif "English" in str(video_path):
            return "en-us"
        else:
            # Fallback to auto-detection
            return "auto"
    
    def organize_video_into_folder(self, video_path: Path) -> Path:
        """
        Organize video into its own folder structure
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the organized video folder
        """
        # Determine the video stem (filename without extension)
        video_stem = video_path.stem
        
        # Create folder path: Spanish/video_name/ or English/video_name/
        if "Spanish" in str(video_path):
            folder_path = self.spanish_path / video_stem
        elif "English" in str(video_path):
            folder_path = self.english_path / video_stem
        else:
            # Fallback to current directory
            folder_path = video_path.parent / video_stem
        
        try:
            # Create the folder if it doesn't exist
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # Target path for the video
            target_video_path = folder_path / video_path.name
            
            # Only move if it's not already in the right place
            if video_path != target_video_path:
                if target_video_path.exists():
                    logger.warning(f"🔄 Target video already exists: {target_video_path}")
                else:
                    shutil.move(str(video_path), str(target_video_path))
                    logger.info(f"📁 Organized video: {video_path.name} → {folder_path}")
                    self.stats['organized'] += 1
            
            return folder_path
            
        except Exception as e:
            logger.error(f"❌ Failed to organize video {video_path.name}: {e}")
            return video_path.parent
    
    def get_output_path(self, video_path: Path) -> Path:
        """Get expected output path for transcription in the same folder as video"""
        return video_path.parent / f"{video_path.stem}.txt"
    
    def create_progress_bar(self, current: int, total: int, width: int = 40) -> str:
        """Create a visual progress bar"""
        if total == 0:
            return "[" + "=" * width + "]"
        
        filled_length = int(width * current / total)
        bar = "=" * filled_length + "-" * (width - filled_length)
        percentage = (current / total) * 100
        return f"[{bar}] {percentage:6.2f}%"
    
    def log_progress(self, current: int, total: int, video_name: str, directory: str):
        """Log progress with visual indicators"""
        progress_bar = self.create_progress_bar(current, total)
        overall_progress = self.create_progress_bar(self.current_video_index, self.total_videos_to_process)
        
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"📊 OVERALL PROGRESS: {overall_progress} ({self.current_video_index}/{self.total_videos_to_process})")
        logger.info(f"📁 {directory}: {progress_bar} ({current}/{total})")
        logger.info(f"🎬 Processing: {video_name}")
        logger.info(f"{'='*80}")
    
    def transcribe_video(self, video_path: Path, language: str) -> bool:
        """
        Transcribe a single video using the main transcriptor script
        
        Args:
            video_path: Path to video file
            language: Language code (es-cl, en-us, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get output path (in same folder as video)
            output_path = self.get_output_path(video_path)
            
            # Build command
            cmd = [
                sys.executable, "transcriptor_whisper.py",
                str(video_path),
                "-l", language,
                "-m", "medium",  # Use medium model for good quality/speed balance
                "-t",  # Include timestamps
                "-o", str(output_path.parent)  # Output to same folder as video
            ]
            
            # Add OpenAI API key if available
            openai_key = os.environ.get('OPENAI_API_KEY')
            if openai_key:
                cmd.extend(["--openai-key", openai_key])
            else:
                cmd.append("--no-summary")  # Disable summary if no API key
            
            logger.info(f"📍 Language: {language}")
            logger.info(f"📄 Output: {output_path}")
            logger.info(f"🔧 Command: {' '.join(cmd[2:])}")  # Log command without python path
            
            # Execute transcription
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ SUCCESS: {video_path.name} ({processing_time:.2f}s)")
                
                # Check if output file was created
                if output_path.exists():
                    logger.info(f"📄 File created: {output_path}")
                else:
                    logger.warning(f"⚠️ File not found: {output_path}")
                
                return True
            else:
                logger.error(f"❌ FAILED: {video_path.name}")
                logger.error(f"Exit code: {result.returncode}")
                if result.stdout:
                    logger.error(f"STDOUT: {result.stdout}")
                if result.stderr:
                    logger.error(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏱️ TIMEOUT: {video_path.name} (exceeded 1 hour)")
            return False
        except Exception as e:
            logger.error(f"💥 ERROR: {video_path.name} - {e}")
            return False
    
    def process_directory(self, directory: Path, language: str, upload_to_drive: bool = False) -> None:
        """Process all loose videos in a directory"""
        logger.info(f"🔍 Scanning directory: {directory}")
        videos = self.find_videos(directory)
        
        if not videos:
            logger.info(f"📭 No new videos found in: {directory}")
            return
        
        logger.info(f"📹 Found {len(videos)} new video(s) in {directory.name}")
        
        # Initialize Google Drive uploader if needed
        drive_uploader = None
        if upload_to_drive:
            try:
                from google_drive_uploader import GoogleDriveUploader
                drive_uploader = GoogleDriveUploader()
                # Authenticate once at the beginning
                if not drive_uploader.authenticate():
                    logger.error("❌ Google Drive authentication failed")
                    drive_uploader = None
            except ImportError:
                logger.error("❌ Google Drive uploader not available")
                drive_uploader = None
        
        for i, video_path in enumerate(videos, 1):
            # Update progress tracking
            self.current_video_index += 1
            
            # Log progress with visual indicators
            self.log_progress(i, len(videos), video_path.name, directory.name)
            
            # Organize video into its own folder
            video_folder = self.organize_video_into_folder(video_path)
            
            # Update video path to the new location
            organized_video_path = video_folder / video_path.name
            
            # Get output path
            output_path = self.get_output_path(organized_video_path)
            
            # Process the video (since it's not in processed_files)
            logger.info(f"📄 NEW: {video_path.name}")
            logger.info(f"📄 Output: {output_path}")
            
            success = self.transcribe_video(organized_video_path, language)
            
            if success:
                self.stats['processed'] += 1
                
                # Upload to Google Drive immediately after successful transcription
                if drive_uploader:
                    logger.info(f"🌐 Uploading {video_folder.name} to Google Drive...")
                    if self.upload_single_video_folder(drive_uploader, video_folder):
                        logger.info(f"✅ Successfully uploaded to Google Drive: {video_folder.name}")
                        self.save_progress(str(organized_video_path), drive_uploaded=True)
                    else:
                        logger.error(f"❌ Failed to upload to Google Drive: {video_folder.name}")
                        self.save_progress(str(organized_video_path), drive_uploaded=False)
                else:
                    self.save_progress(str(organized_video_path), drive_uploaded=False)
                
                # Calculate and display progress
                overall_percentage = (self.current_video_index / self.total_videos_to_process) * 100
                logger.info(f"📈 Total progress: {self.current_video_index}/{self.total_videos_to_process} ({overall_percentage:.1f}%) completed")
            else:
                self.stats['failed'] += 1
                logger.error(f"📉 Failed to process: {video_path.name}")
            
            # Update language-specific stats
            if language == "es-cl":
                self.stats['spanish_videos'] += 1
            elif language == "en-us":
                self.stats['english_videos'] += 1
            
            # Add a small delay to make progress more visible
            time.sleep(0.5)
    
    def upload_single_video_folder(self, drive_uploader, video_folder: Path) -> bool:
        """Upload a single video folder to Google Drive"""
        try:
            # Ensure we have the main "Recordings" folder
            main_folder_id = drive_uploader.find_or_create_folder("Recordings")
            if not main_folder_id:
                return False
            
            # Upload the video folder
            return drive_uploader.upload_video_folder(video_folder, main_folder_id)
            
        except Exception as e:
            logger.error(f"❌ Error uploading video folder: {e}")
            return False

    def run(self, spanish_only: bool = False, english_only: bool = False, upload_to_drive: bool = False) -> None:
        """
        Run batch transcription process
        
        Args:
            spanish_only: Process only Spanish videos
            english_only: Process only English videos
            upload_to_drive: Upload results to Google Drive after processing each video
        """
        logger.info("🚀 Starting Batch Video Transcription")
        logger.info(f"📂 Base path: {self.base_path}")
        
        self.stats['start_time'] = datetime.now().isoformat()
        
        # Check if transcriptor_whisper.py exists
        if not Path("transcriptor_whisper.py").exists():
            logger.error("❌ transcriptor_whisper.py not found in current directory")
            logger.error("Make sure to run this script from the same directory as transcriptor_whisper.py")
            return
        
        # Check OpenAI API key
        if os.environ.get('OPENAI_API_KEY'):
            logger.info("🔑 OpenAI API key found - summaries will be included")
        else:
            logger.warning("⚠️ OpenAI API key not found - summaries will be disabled")
            logger.warning("Set OPENAI_API_KEY environment variable to enable summaries")
        
        # Count only NEW videos (not already processed)
        spanish_videos = self.find_videos(self.spanish_path) if not english_only else []
        english_videos = self.find_videos(self.english_path) if not spanish_only else []
        
        self.stats['total_videos'] = len(spanish_videos) + len(english_videos)
        self.total_videos_to_process = self.stats['total_videos']  # For progress tracking
        
        if self.stats['total_videos'] == 0:
            logger.info("✅ No new videos found to process")
            logger.info(f"📊 Total videos already processed: {len(self.processed_files)}")
            return
        
        logger.info(f"")
        logger.info(f"📊 INITIAL SUMMARY:")
        logger.info(f"📹 New videos to process: {self.stats['total_videos']}")
        logger.info(f"📹 Already processed: {len(self.processed_files)}")
        logger.info(f"🇪🇸 Spanish videos: {len(spanish_videos)}")
        logger.info(f"🇺🇸 English videos: {len(english_videos)}")
        logger.info(f"🔄 NOTE: Videos will be organized into individual folders")
        logger.info(f"📄 NOTE: Transcriptions will be created in the same folder as each video")
        if upload_to_drive:
            logger.info(f"🌐 NOTE: Results will be uploaded to Google Drive after processing")
        logger.info(f"")
        
        # Process Spanish videos
        if spanish_videos:
            logger.info(f"🇪🇸 Processing Spanish videos...")
            self.process_directory(self.spanish_path, "es-cl", upload_to_drive)
        
        # Process English videos
        if english_videos:
            logger.info(f"🇺🇸 Processing English videos...")
            self.process_directory(self.english_path, "en-us", upload_to_drive)
        
        # Final statistics
        self.stats['end_time'] = datetime.now().isoformat()
        self.print_final_stats()
    
    def print_final_stats(self) -> None:
        """Print final processing statistics"""
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info("🎉 BATCH PROCESSING COMPLETED")
        logger.info(f"{'='*80}")
        
        # Create a final progress bar (should be 100%)
        final_progress = self.create_progress_bar(self.current_video_index, self.total_videos_to_process)
        logger.info(f"📊 Final progress: {final_progress}")
        logger.info(f"")
        
        logger.info(f"📊 FINAL STATISTICS:")
        logger.info(f"   📹 Total videos: {self.stats['total_videos']}")
        logger.info(f"   ✅ Processed: {self.stats['processed']}")
        logger.info(f"   📁 Organized: {self.stats['organized']}")
        logger.info(f"   ⏭️  Skipped: {self.stats['skipped']}")
        logger.info(f"   ❌ Failed: {self.stats['failed']}")
        logger.info(f"   🇪🇸 Spanish: {self.stats['spanish_videos']}")
        logger.info(f"   🇺🇸 English: {self.stats['english_videos']}")
        
        if self.stats['start_time'] and self.stats['end_time']:
            start_time = datetime.fromisoformat(self.stats['start_time'])
            end_time = datetime.fromisoformat(self.stats['end_time'])
            duration = end_time - start_time
            logger.info(f"   ⏱️  Total time: {duration}")
        
        # Success rate
        if self.stats['total_videos'] > 0:
            success_rate = (self.stats['processed'] / self.stats['total_videos']) * 100
            logger.info(f"   📈 Success rate: {success_rate:.1f}%")
        
        logger.info(f"")
        logger.info(f"📄 Check video folders: {self.spanish_path} and {self.english_path}")
        logger.info(f"📋 Progress saved in: {self.progress_file}")
        logger.info(f"📋 Detailed log in: transcriptor_batch.log")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Batch Video Transcriber - Automatically processes Spanish and English videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in both Spanish and English folders
  python transcriptor_batch.py
  
  # Process and upload to Google Drive
  python transcriptor_batch.py --upload-to-drive
  
  # Process Spanish videos only
  python transcriptor_batch.py --spanish-only
  
  # Process English videos only
  python transcriptor_batch.py --english-only
  
  # Custom path with Google Drive upload
  python transcriptor_batch.py --base-path /path/to/videos --upload-to-drive

New Structure:
  Videos/
  ├── Spanish/
  │   ├── video1/
  │   │   ├── video1.mp4
  │   │   └── video1.txt
  │   └── video2/
  │       ├── video2.mp4
  │       └── video2.txt
  └── English/
      ├── video3/
      │   ├── video3.mp4
      │   └── video3.txt
      └── video4/
          ├── video4.mp4
          └── video4.txt

Notes:
  - Videos are automatically organized into individual folders
  - Transcriptions are created in the same folder as each video
  - Progress is automatically saved and can be resumed
  - Existing transcriptions are automatically replaced
  - Use --upload-to-drive to automatically upload to Google Drive
  - Processing log is saved in transcriptor_batch.log
        """
    )
    
    parser.add_argument(
        "--base-path",
        default="/Users/rollyangell/Desktop/RollyAngell/Work/Videos",
        help="Base path containing Spanish and English folders"
    )
    
    parser.add_argument(
        "--spanish-only",
        action="store_true",
        help="Process Spanish videos only"
    )
    
    parser.add_argument(
        "--english-only",
        action="store_true",
        help="Process English videos only"
    )
    
    parser.add_argument(
        "--upload-to-drive",
        action="store_true",
        help="Upload results to Google Drive after processing"
    )
    
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Reset progress and process all videos again"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Reset progress if requested
    if args.reset_progress:
        progress_file = Path("transcriptor_progress.json")
        if progress_file.exists():
            progress_file.unlink()
            logger.info("🔄 Progress reset - all videos will be processed again")
    
    # Validate arguments
    if args.spanish_only and args.english_only:
        logger.error("❌ Cannot use --spanish-only and --english-only at the same time")
        return 1
    
    try:
        # Create and run batch transcriptor
        batch_transcriptor = BatchTranscriptor(base_path=args.base_path)
        batch_transcriptor.run(
            spanish_only=args.spanish_only,
            english_only=args.english_only,
            upload_to_drive=args.upload_to_drive
        )
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Batch processing interrupted by user")
        logger.info("🔄 Progress has been saved. Run again to continue.")
        return 1
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 