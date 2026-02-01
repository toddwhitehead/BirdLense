import logging
import os
import subprocess
import cv2
import time


class VideoFileSource:
    """
    Video file source that accurately simulates real camera behavior:
    - Tracks elapsed time between capture() calls
    - Skips frames that would have passed during processing
    - Writes ALL frames to disk (skipped ones too)
    """

    def __init__(self, video_path, main_size=(1280, 720), lores_size=(640, 640)):
        self.logger = logging.getLogger(__name__)
        self.cap = cv2.VideoCapture(video_path)
        self.main_size = main_size
        self.lores_size = lores_size
        self.ffmpeg_process = None
        self.output_path = None
        
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_interval = 1.0 / self.source_fps
        self.last_capture_time = None
        self.frame_count = 0
        
        self.logger.info(f'VideoFileSource: {self.source_fps} FPS')

    def start_recording(self, output):
        self.logger.info(f'Start video recording to {output}')
        self.output_path = output
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output)
        if output_dir and not os.path.exists(output_dir):
            self.logger.info(f'Creating output directory: {output_dir}')
            os.makedirs(output_dir, exist_ok=True)
        
        # Use FFmpeg for reliable video encoding (OpenCV VideoWriter has codec issues in Docker)
        width, height = self.main_size
        command = [
            'ffmpeg',
            '-y',  # Overwrite output
            '-loglevel', 'warning',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(self.source_fps),
            '-i', '-',  # Read from stdin
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-pix_fmt', 'yuv420p',
            output
        ]
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.logger.info(f'FFmpeg process started for {output}')
        except Exception as e:
            self.logger.error(f'Failed to start FFmpeg: {e}')
            self.ffmpeg_process = None
        
        self.frame_count = 0
        self.last_capture_time = None  # Will be set on first capture

    def stop_recording(self):
        self.logger.info(f'Stop video recording, frames written: {self.frame_count}')
        if self.ffmpeg_process is not None:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait(timeout=10)
                stderr_output = self.ffmpeg_process.stderr.read().decode('utf-8', errors='ignore')
                if stderr_output:
                    self.logger.debug(f'FFmpeg stderr: {stderr_output}')
            except subprocess.TimeoutExpired:
                self.logger.warning('FFmpeg did not terminate, killing...')
                self.ffmpeg_process.kill()
            except Exception as e:
                self.logger.error(f'Error closing FFmpeg: {e}')
            finally:
                self.ffmpeg_process = None
        
        # Verify the file was written
        if self.output_path and os.path.exists(self.output_path):
            file_size = os.path.getsize(self.output_path)
            self.logger.info(f'Video file size: {file_size} bytes, frames written: {self.frame_count}')
            if file_size == 0:
                self.logger.error(f'Video file is empty!')
        elif self.output_path:
            self.logger.error(f'Video file was not created: {self.output_path}')

    def capture(self):
        """
        Get next frame for processing.
        Advances video by elapsed real time, writing skipped frames to disk.
        Returns None when video ends.
        """
        if not self.cap.isOpened():
            return None
        
        # Calculate how many frames should have passed since last capture
        now = time.time()
        if self.last_capture_time is None:
            # First capture - start the clock now (syncs with frame_processor)
            frames_to_advance = 1
        else:
            elapsed = now - self.last_capture_time
            frames_to_advance = max(1, int(elapsed / self.frame_interval))
        self.last_capture_time = now
        
        # Read and write frames, return only the last one
        result_frame = None
        for _ in range(frames_to_advance):
            ret, frame = self.cap.read()
            if not ret:
                self.logger.info(f'Video ended after {self.frame_count} frames')
                return None
            
            self.frame_count += 1
            
            # Write ALL frames to FFmpeg
            if self.ffmpeg_process is not None and self.ffmpeg_process.stdin:
                try:
                    frame_main = cv2.resize(frame, self.main_size)
                    self.ffmpeg_process.stdin.write(frame_main.tobytes())
                except BrokenPipeError:
                    self.logger.error('FFmpeg pipe broken, stopping recording')
                    self.ffmpeg_process = None
                except Exception as e:
                    self.logger.error(f'Error writing frame to FFmpeg: {e}')
            
            # Log every 100 frames to confirm writing
            if self.frame_count % 100 == 0:
                self.logger.debug(f'Written {self.frame_count} frames')
            
            result_frame = frame
        
        return cv2.resize(result_frame, self.lores_size) if result_frame is not None else None

    def close(self):
        self.stop_recording()
        if self.cap is not None:
            self.cap.release()
