import logging
import os
import cv2
import numpy as np
import time


class VideoFileSource:
    """
    Video file source that accurately simulates real camera behavior:
    - Tracks elapsed time between capture() calls
    - Skips frames that would have passed during processing
    - Writes ALL frames to disk (skipped ones too)
    """

    # Codecs to try in order of preference (mp4v is most compatible on Raspberry Pi)
    CODEC_FALLBACKS = ['mp4v', 'XVID', 'avc1', 'H264']

    def __init__(self, video_path, main_size=(1280, 720), lores_size=(640, 640)):
        self.logger = logging.getLogger(__name__)
        self.cap = cv2.VideoCapture(video_path)
        self.main_size = main_size
        self.lores_size = lores_size
        self.out = None
        self.output_path = None
        
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_interval = 1.0 / self.source_fps
        self.last_capture_time = None
        self.frame_count = 0
        
        # Log OpenCV build info for debugging codec issues
        self.logger.info(f'VideoFileSource: {self.source_fps} FPS')
        self.logger.info(f'OpenCV version: {cv2.__version__}')
        self.logger.debug(f'OpenCV build info: {cv2.getBuildInformation()}')

    def _try_create_video_writer(self, output, codec):
        """Try to create a VideoWriter with the given codec and verify it works."""
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.logger.debug(f'Trying codec {codec}, fourcc={fourcc}, size={self.main_size}, fps={self.source_fps}')
            
            writer = cv2.VideoWriter(output, fourcc, self.source_fps, self.main_size)
            
            if not writer.isOpened():
                self.logger.debug(f'Codec {codec}: VideoWriter.isOpened() returned False')
                return None
            
            # Write a test frame to verify the codec actually works
            # (some codecs report isOpened=True but fail to write)
            test_frame = np.zeros((self.main_size[1], self.main_size[0], 3), dtype=np.uint8)
            writer.write(test_frame)
            writer.release()
            
            # Check if file was actually written (size > 0)
            if os.path.exists(output):
                file_size = os.path.getsize(output)
                self.logger.debug(f'Codec {codec}: test file size={file_size}')
                if file_size > 0:
                    # Recreate writer to start fresh (without the test frame)
                    os.remove(output)
                    writer = cv2.VideoWriter(output, fourcc, self.source_fps, self.main_size)
                    if writer.isOpened():
                        self.logger.info(f'Codec {codec} works!')
                        return writer
                    else:
                        self.logger.debug(f'Codec {codec}: failed to reopen after test')
                else:
                    self.logger.debug(f'Codec {codec}: test file is empty')
                    os.remove(output)
            else:
                self.logger.debug(f'Codec {codec}: test file not created')
            
            return None
        except Exception as e:
            self.logger.warning(f'Codec {codec} failed with exception: {e}')
            if os.path.exists(output):
                os.remove(output)
            return None

    def start_recording(self, output):
        self.logger.info(f'Start video recording to {output}')
        self.output_path = output
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output)
        if output_dir and not os.path.exists(output_dir):
            self.logger.info(f'Creating output directory: {output_dir}')
            os.makedirs(output_dir, exist_ok=True)
        
        # Log directory permissions and existence
        if output_dir:
            self.logger.debug(f'Output dir exists: {os.path.exists(output_dir)}, writable: {os.access(output_dir, os.W_OK)}')
        
        # Try each codec until one works
        for codec in self.CODEC_FALLBACKS:
            self.out = self._try_create_video_writer(output, codec)
            if self.out is not None:
                self.logger.info(f'VideoWriter opened successfully with codec {codec} for {output}')
                break
        
        if self.out is None:
            self.logger.error(f'Failed to open VideoWriter for {output} with any codec: {self.CODEC_FALLBACKS}')
            # Try one more time with raw AVI as last resort
            self.logger.info('Trying raw AVI fallback...')
            try:
                avi_output = output.replace('.mp4', '.avi')
                self.output_path = avi_output
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.out = cv2.VideoWriter(avi_output, fourcc, self.source_fps, self.main_size)
                if self.out.isOpened():
                    self.logger.info(f'Fallback to MJPG AVI succeeded: {avi_output}')
                else:
                    self.logger.error('MJPG AVI fallback also failed')
                    self.out = None
            except Exception as e:
                self.logger.error(f'MJPG AVI fallback exception: {e}')
        
        self.frame_count = 0
        self.last_capture_time = None  # Will be set on first capture

    def stop_recording(self):
        self.logger.info(f'Stop video recording, frames written: {self.frame_count}')
        if self.out is not None:
            self.out.release()
            self.out = None
        
        # Verify the file was written
        if self.output_path and os.path.exists(self.output_path):
            file_size = os.path.getsize(self.output_path)
            self.logger.info(f'Video file size: {file_size} bytes, frames written: {self.frame_count}')
            if file_size == 0:
                self.logger.error(f'Video file is empty! Codec may not be working correctly.')
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
            
            # Write ALL frames to disk
            if self.out is not None:
                frame_main = cv2.resize(frame, self.main_size)
                self.out.write(frame_main)
                # Log every 100 frames to confirm writing
                if self.frame_count % 100 == 0:
                    self.logger.debug(f'Written {self.frame_count} frames')
            
            result_frame = frame
        
        return cv2.resize(result_frame, self.lores_size) if result_frame is not None else None

    def close(self):
        self.stop_recording()
        if self.cap is not None:
            self.cap.release()
