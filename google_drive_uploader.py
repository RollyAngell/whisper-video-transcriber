#!/usr/bin/env python3
"""
Google Drive Uploader for Video Transcriptions
Author: AI Assistant
Description: Uploads video folders with transcriptions to Google Drive
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import argparse
from typing import Optional, List, Dict

# Google Drive API imports
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('google_drive_upload.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class GoogleDriveUploader:
    """Upload video folders to Google Drive"""
    
    # Google Drive API scopes
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    
    def __init__(self, credentials_file: str = "credentials.json", token_file: str = "token.json"):
        """
        Initialize Google Drive uploader
        
        Args:
            credentials_file: Path to Google API credentials file
            token_file: Path to store authentication token
        """
        self.credentials_file = Path(credentials_file)
        self.token_file = Path(token_file)
        self.service = None
        self.upload_stats = {
            'folders_created': 0,
            'files_uploaded': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def authenticate(self) -> bool:
        """
        Authenticate with Google Drive API
        
        Returns:
            True if authentication successful, False otherwise
        """
        if not GOOGLE_DRIVE_AVAILABLE:
            logger.error("❌ Google Drive API libraries not installed")
            logger.error("Install with: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            return False
        
        if not self.credentials_file.exists():
            logger.error(f"❌ Credentials file not found: {self.credentials_file}")
            logger.error("Download credentials.json from Google Cloud Console")
            logger.error("Guide: https://developers.google.com/drive/api/quickstart/python")
            return False
        
        creds = None
        
        # Load existing token
        if self.token_file.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(self.token_file), self.SCOPES)
            except Exception as e:
                logger.warning(f"Could not load existing token: {e}")
        
        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.warning(f"Could not refresh token: {e}")
                    creds = None
            
            if not creds:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(self.credentials_file), self.SCOPES)
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    logger.error(f"Authentication failed: {e}")
                    return False
            
            # Save the credentials for next run
            try:
                with open(self.token_file, 'w') as token:
                    token.write(creds.to_json())
            except Exception as e:
                logger.warning(f"Could not save token: {e}")
        
        try:
            self.service = build('drive', 'v3', credentials=creds)
            logger.info("✅ Google Drive authentication successful")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to build Google Drive service: {e}")
            return False
    
    def create_folder(self, folder_name: str, parent_id: Optional[str] = None) -> Optional[str]:
        """
        Create a folder in Google Drive
        
        Args:
            folder_name: Name of the folder to create
            parent_id: Parent folder ID (None for root)
            
        Returns:
            Folder ID if successful, None otherwise
        """
        try:
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_id:
                folder_metadata['parents'] = [parent_id]
            
            folder = self.service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            
            folder_id = folder.get('id')
            logger.info(f"📁 Created folder: {folder_name} (ID: {folder_id})")
            self.upload_stats['folders_created'] += 1
            return folder_id
            
        except HttpError as e:
            logger.error(f"❌ Failed to create folder {folder_name}: {e}")
            self.upload_stats['errors'] += 1
            return None
    
    def file_exists_in_folder(self, file_name: str, folder_id: str) -> bool:
        """
        Check if a file exists in a specific folder
        
        Args:
            file_name: Name of the file to check
            folder_id: ID of the folder to search in
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
            results = self.service.files().list(
                q=query,
                fields='files(id, name)'
            ).execute()
            
            files = results.get('files', [])
            return len(files) > 0
            
        except HttpError as e:
            logger.error(f"❌ Failed to check file existence {file_name}: {e}")
            return False
    
    def upload_file(self, file_path: Path, folder_id: str) -> bool:
        """
        Upload a file to Google Drive (only if it doesn't exist)
        
        Args:
            file_path: Path to the file to upload
            folder_id: ID of the destination folder
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file already exists
            if self.file_exists_in_folder(file_path.name, folder_id):
                logger.info(f"📄 File already exists: {file_path.name} (skipping upload)")
                return True
            
            file_metadata = {
                'name': file_path.name,
                'parents': [folder_id]
            }
            
            # Determine MIME type based on file extension
            mime_type = 'application/octet-stream'  # Default
            if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']:
                mime_type = 'video/mp4'
            elif file_path.suffix.lower() == '.txt':
                mime_type = 'text/plain'
            
            media = MediaFileUpload(
                str(file_path),
                mimetype=mime_type,
                resumable=True
            )
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            logger.info(f"📄 Uploaded file: {file_path.name} (ID: {file_id})")
            self.upload_stats['files_uploaded'] += 1
            return True
            
        except HttpError as e:
            logger.error(f"❌ Failed to upload file {file_path.name}: {e}")
            self.upload_stats['errors'] += 1
            return False
    
    def find_or_create_folder(self, folder_name: str, parent_id: Optional[str] = None) -> Optional[str]:
        """
        Find existing folder or create new one
        
        Args:
            folder_name: Name of the folder
            parent_id: Parent folder ID (None for root)
            
        Returns:
            Folder ID if found/created, None otherwise
        """
        try:
            # Search for existing folder
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            if parent_id:
                query += f" and '{parent_id}' in parents"
            
            results = self.service.files().list(
                q=query,
                fields='files(id, name)'
            ).execute()
            
            files = results.get('files', [])
            
            if files:
                folder_id = files[0]['id']
                logger.info(f"📁 Found existing folder: {folder_name} (ID: {folder_id})")
                return folder_id
            else:
                # Create new folder
                return self.create_folder(folder_name, parent_id)
                
        except HttpError as e:
            logger.error(f"❌ Failed to find/create folder {folder_name}: {e}")
            self.upload_stats['errors'] += 1
            return None
    
    def upload_video_folder(self, video_folder: Path, parent_folder_id: str) -> bool:
        """
        Upload a complete video folder (video + transcription) - only missing files
        
        Args:
            video_folder: Path to the video folder
            parent_folder_id: ID of the parent folder in Google Drive
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create folder for this video
            folder_id = self.find_or_create_folder(video_folder.name, parent_folder_id)
            if not folder_id:
                return False
            
            # Upload all files in the folder (skipping existing ones)
            success = True
            files_to_upload = []
            
            for file_path in video_folder.iterdir():
                if file_path.is_file():
                    files_to_upload.append(file_path)
            
            if not files_to_upload:
                logger.warning(f"📭 No files found in folder: {video_folder.name}")
                return True
            
            # Check if all files already exist
            all_exist = True
            for file_path in files_to_upload:
                if not self.file_exists_in_folder(file_path.name, folder_id):
                    all_exist = False
                    break
            
            if all_exist:
                logger.info(f"📁 All files already exist in Drive: {video_folder.name}")
                return True
            
            # Upload missing files
            for file_path in files_to_upload:
                if not self.upload_file(file_path, folder_id):
                    success = False
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to upload video folder {video_folder.name}: {e}")
            self.upload_stats['errors'] += 1
            return False
    
    def upload_all_videos(self, base_path: Path) -> bool:
        """
        Upload all video folders to Google Drive
        
        Args:
            base_path: Base path containing Spanish and English folders
            
        Returns:
            True if all uploads successful, False otherwise
        """
        if not self.authenticate():
            return False
        
        self.upload_stats['start_time'] = datetime.now().isoformat()
        
        # Create main folder structure - directly in "Recordings"
        main_folder_id = self.find_or_create_folder("Recordings")
        if not main_folder_id:
            return False
        
        # Process Spanish and English folders - upload directly to Recordings
        success = True
        for language_folder in ['Spanish', 'English']:
            lang_path = base_path / language_folder
            
            if not lang_path.exists():
                logger.warning(f"📭 Language folder not found: {lang_path}")
                continue
            
            # Upload all video folders directly to Recordings (no language subfolders)
            logger.info(f"🌐 Uploading {language_folder} videos to Recordings...")
            
            for video_folder in lang_path.iterdir():
                if video_folder.is_dir():
                    logger.info(f"📁 Processing folder: {video_folder.name}")
                    
                    # Upload directly to main Recordings folder
                    if not self.upload_video_folder(video_folder, main_folder_id):
                        success = False
                        logger.error(f"❌ Failed to upload: {video_folder.name}")
                    else:
                        logger.info(f"✅ Successfully uploaded: {video_folder.name}")
        
        self.upload_stats['end_time'] = datetime.now().isoformat()
        self.print_upload_stats()
        
        return success
    
    def print_upload_stats(self):
        """Print upload statistics"""
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info("🎉 GOOGLE DRIVE UPLOAD COMPLETED")
        logger.info(f"{'='*80}")
        
        logger.info(f"📊 UPLOAD STATISTICS:")
        logger.info(f"   📁 Folders created: {self.upload_stats['folders_created']}")
        logger.info(f"   📄 Files uploaded: {self.upload_stats['files_uploaded']}")
        logger.info(f"   ❌ Errors: {self.upload_stats['errors']}")
        
        if self.upload_stats['start_time'] and self.upload_stats['end_time']:
            start_time = datetime.fromisoformat(self.upload_stats['start_time'])
            end_time = datetime.fromisoformat(self.upload_stats['end_time'])
            duration = end_time - start_time
            logger.info(f"   ⏱️  Total time: {duration}")
        
        logger.info(f"")
        logger.info(f"🌐 Check your Google Drive: https://drive.google.com")
        logger.info(f"📋 Upload log saved in: google_drive_upload.log")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Google Drive Uploader for Video Transcriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload all video folders to Google Drive
  python google_drive_uploader.py
  
  # Custom path
  python google_drive_uploader.py --base-path /path/to/videos
  
  # Custom credentials file
  python google_drive_uploader.py --credentials my_credentials.json

Setup:
  1. Go to Google Cloud Console: https://console.cloud.google.com/
  2. Create a new project or select existing one
  3. Enable Google Drive API
  4. Create credentials (Desktop application)
  5. Download credentials.json file
  6. Place credentials.json in the same directory as this script

Notes:
  - First run will open browser for authentication
  - Subsequent runs will use saved token
  - Uploads maintain folder structure: Spanish/English/VideoName/
  - Each video folder contains: video.mp4 and video.txt
  - Files already in Drive will be skipped automatically
        """
    )
    
    parser.add_argument(
        "--base-path",
        default="/Users/rollyangell/Desktop/RollyAngell/Work/Videos",
        help="Base path containing Spanish and English folders"
    )
    
    parser.add_argument(
        "--credentials",
        default="credentials.json",
        help="Path to Google API credentials file"
    )
    
    parser.add_argument(
        "--token",
        default="token.json",
        help="Path to store authentication token"
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
    
    try:
        uploader = GoogleDriveUploader(
            credentials_file=args.credentials,
            token_file=args.token
        )
        
        success = uploader.upload_all_videos(Path(args.base_path))
        
        if success:
            logger.info("✅ All uploads completed successfully")
            return 0
        else:
            logger.error("❌ Some uploads failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Upload interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
