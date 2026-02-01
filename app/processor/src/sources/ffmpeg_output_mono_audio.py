import gc
import signal
import subprocess
import prctl
import logging
import time
from picamera2.outputs import Output


class FfmpegOutputMonoAudio(Output):
    """
    Picamera2-specific class to handle FFmpeg output with mono audio using ALSA.
    This class is a modified version of the FfmpegOutput class from the Picamera2 library.
    It has been adapted to force mono audio output instead of the default stereo.
    """

    def __init__(self, output_filename, audio=False, audio_device="hw:1,0", audio_sync=-0.3,
                 audio_samplerate=48000, audio_codec="aac", audio_bitrate=128000, pts=None):
        super().__init__(pts=pts)
        self.ffmpeg = None
        self.output_filename = output_filename
        self.audio = audio
        self.audio_device = audio_device
        self.audio_sync = audio_sync
        self.audio_samplerate = audio_samplerate
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        self.timeout = 10  # Give FFmpeg enough time to finalize the video file
        self.error_callback = None
        self.needs_pacing = True
        self.logger = logging.getLogger(__name__)
        self.audio_enabled = audio  # Track if audio was initially requested

    def start(self):
        general_options = ['-loglevel', 'warning', '-y']
        video_input = ['-use_wallclock_as_timestamps', '1',
                       '-thread_queue_size', '64',
                       '-i', '-']
        video_codec = ['-c:v', 'copy']
        audio_input = []
        audio_codec = []

        if self.audio:
            audio_input = [
                '-itsoffset', str(self.audio_sync),
                '-f', 'alsa',
                '-sample_rate', str(self.audio_samplerate),
                '-channels', '1',  # Explicitly set mono
                '-thread_queue_size', '1024',
                '-i', self.audio_device
            ]
            audio_codec = [
                '-b:a', str(self.audio_bitrate),
                '-c:a', self.audio_codec,
                '-ac', '1'  # Force mono output
            ]

        command = ['ffmpeg'] + general_options + audio_input + video_input + \
            audio_codec + video_codec + self.output_filename.split()
        
        self.logger.info(f'Starting FFmpeg for output: {self.output_filename}')
        self.logger.debug(f'FFmpeg command: {" ".join(command)}')

        try:
            self.ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE,
                                           stderr=subprocess.PIPE,
                                           preexec_fn=lambda: prctl.set_pdeathsig(signal.SIGKILL))
            
            # If audio was requested, verify FFmpeg started successfully
            if self.audio:
                # Give FFmpeg a moment to fail if audio device is unavailable
                time.sleep(0.3)
                
                # Check if process has already exited
                poll_result = self.ffmpeg.poll()
                if poll_result is not None:
                    # Process exited, likely due to audio device error
                    stderr_output = self.ffmpeg.stderr.read().decode('utf-8', errors='ignore')
                    self.logger.warning(f"FFmpeg failed with audio device '{self.audio_device}': {stderr_output}")
                    self.logger.warning("Retrying video recording without audio...")
                    
                    # Retry without audio
                    self.audio = False
                    command = ['ffmpeg'] + general_options + video_input + video_codec + self.output_filename.split()
                    self.ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE,
                                                   stderr=subprocess.PIPE,
                                                   preexec_fn=lambda: prctl.set_pdeathsig(signal.SIGKILL))
                    self.logger.info("Successfully started video recording without audio")
        except Exception as e:
            # If audio was requested and Popen itself failed, try without audio
            if self.audio_enabled:
                self.logger.error(f"Failed to start FFmpeg with audio: {e}")
                self.logger.warning("Retrying video recording without audio...")
                try:
                    self.audio = False
                    command = ['ffmpeg'] + general_options + video_input + video_codec + self.output_filename.split()
                    self.ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE,
                                                   stderr=subprocess.PIPE,
                                                   preexec_fn=lambda: prctl.set_pdeathsig(signal.SIGKILL))
                    self.logger.info("Successfully started video recording without audio")
                except Exception as retry_error:
                    self.logger.error(f"Failed to start FFmpeg even without audio: {retry_error}")
                    raise
            else:
                raise
                
        super().start()

    def stop(self):
        self.logger.info(f'Stopping FFmpeg for output: {self.output_filename}')
        super().stop()
        if self.ffmpeg is not None:
            self.ffmpeg.stdin.close()
            try:
                self.ffmpeg.wait(timeout=self.timeout)
                # Log FFmpeg stderr for debugging
                if self.ffmpeg.stderr:
                    stderr_output = self.ffmpeg.stderr.read().decode('utf-8', errors='ignore')
                    if stderr_output:
                        self.logger.debug(f'FFmpeg stderr: {stderr_output}')
                # Verify file was created
                import os
                if os.path.exists(self.output_filename):
                    file_size = os.path.getsize(self.output_filename)
                    self.logger.info(f'Video file created: {self.output_filename}, size: {file_size} bytes')
                else:
                    self.logger.error(f'Video file NOT created: {self.output_filename}')
            except subprocess.TimeoutExpired:
                self.logger.warning(f'FFmpeg timed out after {self.timeout}s, terminating...')
                try:
                    self.ffmpeg.terminate()
                    self.ffmpeg.wait()  # Ensure process cleanup
                    # Still check if file was created
                    import os
                    if os.path.exists(self.output_filename):
                        file_size = os.path.getsize(self.output_filename)
                        self.logger.info(f'Video file created (after timeout): {self.output_filename}, size: {file_size} bytes')
                    else:
                        self.logger.error(f'Video file NOT created: {self.output_filename}')
                except Exception as e:
                    self.logger.error(f'Error during FFmpeg cleanup: {e}')
            self.ffmpeg = None
            gc.collect()

    def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=False):
        if audio:
            raise RuntimeError(
                "FfmpegOutput does not support audio packets from Picamera2")
        if self.recording and self.ffmpeg:
            try:
                self.ffmpeg.stdin.write(frame)
                self.ffmpeg.stdin.flush()
            except Exception as e:
                self.ffmpeg = None
                if self.error_callback:
                    self.error_callback(e)
            else:
                self.outputtimestamp(timestamp)
