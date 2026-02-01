"""
Tests for FFmpeg audio device failure resilience
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import subprocess

# Ensure project root is in path to import app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '../src/sources'))
sys.path.append(src_path)

from ffmpeg_output_mono_audio import FfmpegOutputMonoAudio


class TestFfmpegAudioFallback(unittest.TestCase):
    """Test audio fallback behavior in FfmpegOutputMonoAudio"""
    
    @patch('ffmpeg_output_mono_audio.subprocess.Popen')
    def test_audio_device_failure_fallback_to_video_only(self, mock_popen):
        """Test that when audio device fails, recording continues with video only"""
        # First call (with audio) fails immediately
        failed_process = Mock()
        failed_process.poll.return_value = 1  # Process exited with error
        failed_process.stderr.read.return_value = b"[alsa @ 0x5556340ba640] cannot open audio device hw:1,0 (No such file or directory)"
        
        # Second call (video only) succeeds
        success_process = Mock()
        success_process.poll.return_value = None  # Process still running
        success_process.stdin = Mock()
        
        # Configure mock to return different processes for each call
        mock_popen.side_effect = [failed_process, success_process]
        
        # Create output with audio enabled
        output = FfmpegOutputMonoAudio("output.mp4", audio=True, audio_device="hw:1,0")
        
        # Start should not raise exception, but should fall back to video-only
        output.start()
        
        # Verify that subprocess.Popen was called twice
        self.assertEqual(mock_popen.call_count, 2)
        
        # Verify that second call doesn't include audio parameters
        second_call_args = mock_popen.call_args_list[1][0][0]
        self.assertNotIn('-f', second_call_args)  # No audio format
        self.assertNotIn('alsa', second_call_args)  # No ALSA
        self.assertNotIn('hw:1,0', second_call_args)  # No audio device
        
        # Verify audio was disabled after fallback
        self.assertFalse(output.audio)
    
    @patch('ffmpeg_output_mono_audio.subprocess.Popen')
    def test_audio_device_success_no_fallback(self, mock_popen):
        """Test that when audio device works, no fallback occurs"""
        # Process starts successfully
        success_process = Mock()
        success_process.poll.return_value = None  # Process still running
        success_process.stdin = Mock()
        
        mock_popen.return_value = success_process
        
        # Create output with audio enabled
        output = FfmpegOutputMonoAudio("output.mp4", audio=True, audio_device="hw:1,0")
        
        # Start should succeed
        output.start()
        
        # Verify that subprocess.Popen was called only once
        self.assertEqual(mock_popen.call_count, 1)
        
        # Verify audio is still enabled
        self.assertTrue(output.audio)
    
    @patch('ffmpeg_output_mono_audio.subprocess.Popen')
    def test_video_only_mode_works(self, mock_popen):
        """Test that video-only mode (audio=False) works correctly"""
        # Process starts successfully
        success_process = Mock()
        success_process.stdin = Mock()
        
        mock_popen.return_value = success_process
        
        # Create output with audio disabled
        output = FfmpegOutputMonoAudio("output.mp4", audio=False)
        
        # Start should succeed
        output.start()
        
        # Verify that subprocess.Popen was called only once
        self.assertEqual(mock_popen.call_count, 1)
        
        # Verify command doesn't include audio parameters
        call_args = mock_popen.call_args[0][0]
        self.assertNotIn('-f', call_args)  # No audio format
        self.assertNotIn('alsa', call_args)  # No ALSA


if __name__ == '__main__':
    unittest.main()
