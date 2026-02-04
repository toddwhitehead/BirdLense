import warnings
import threading
import time
from datetime import datetime, timezone
import argparse
import logging
import os
import shutil

# Suppress NumPy subnormal warning on ARM platforms (Raspberry Pi)
warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
from frame_processor import FrameProcessor
from detection_strategy import SingleStageStrategy, TwoStageStrategy, GlobalTwoStageStrategy
from motion_detectors.pir import PIRMotionDetector
from motion_detectors.fake import FakeMotionDetector
from decision_maker import DecisionMaker
from fps_tracker import FPSTracker
from api import API
from sources.media_source import MediaSource
from sources.video_file_source import VideoFileSource
from audio_processor import AudioProcessor
from llm_verifier import LLMVerifier
from app_config.app_config import app_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to the console
    ]
)


def get_output_path():
    output_dir = "data/recordings/" + time.strftime("%Y/%m/%d/%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def heartbeat():
    api = API()
    id = None
    retry_delay = 1
    max_retry_delay = 300  # 5 minutes
    
    while True:
        try:
            # keep updating activity_log record until restart
            id = api.activity_log(type='heartbeat', data={"status": "up"}, id=id)
            retry_delay = 1  # Reset delay on success
            time.sleep(60)
        except Exception as e:
            logging.error(f"Heartbeat failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            # Exponential backoff with cap
            retry_delay = min(retry_delay * 2, max_retry_delay)


def main():
    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    parser = argparse.ArgumentParser(description="Smart bird feeder program")
    parser.add_argument('input', type=str, nargs='?',
                        help='Input source, camera/video file')
    parser.add_argument('--fake-motion', type=str, choices=['true', 'false'],
                        help='Use fake motion detector with motion or not')
    args = parser.parse_args()

    # Log startup configuration
    logging.info("=" * 80)
    logging.info("BirdLense Processor Starting")
    logging.info("=" * 80)
    
    # Instantiate all helper classes
    api = API()
    if args.fake_motion:
        motion = args.fake_motion.lower() == 'true'
        motion_detector = FakeMotionDetector(motion=motion, wait=10)
        logging.info(f"Motion Detector: FakeMotionDetector (motion={motion})")
    else:
        motion_detector = PIRMotionDetector()
        logging.info("Motion Detector: PIRMotionDetector")
        
    decision_maker = DecisionMaker(max_record_seconds=app_config.get(
        'processor.max_record_seconds'), max_inactive_seconds=app_config.get('processor.max_inactive_seconds'))
    main_size = (app_config.get('camera.video_width'),
                 app_config.get('camera.video_height'))
    camera_config = app_config.get('camera')
    media_source = MediaSource(main_size=main_size, camera_config=camera_config) if not args.input else VideoFileSource(
        args.input, main_size=main_size)
    
    # Log camera configuration
    logging.info(f"Camera Configuration:")
    logging.info(f"  Resolution: {main_size[0]}x{main_size[1]}")
    logging.info(f"  HDR Mode: {camera_config.get('hdr_mode', False)}")
    logging.info(f"  Focus Mode: {camera_config.get('focus_mode', 'auto')}")
    if camera_config.get('focus_mode') == 'manual':
        logging.info(f"  Lens Position: {camera_config.get('lens_position', 'N/A')}")
    
    # Initialize audio processor if enabled
    audio_enabled = app_config.get('processor.enable_audio_processing')
    audio_processor = None
    regional_species = []  # Empty list signals "use all species from included_bird_families config"
    
    if audio_enabled:
        audio_processor = AudioProcessor(lat=app_config.get(
            'secrets.latitude'), lon=app_config.get('secrets.longitude'), spectrogram_px_per_sec=app_config.get('processor.spectrogram_px_per_sec'))
        regional_species = audio_processor.get_regional_species() + ["Squirrel"]
        logging.info("Audio Processing: ENABLED")
        logging.info(f"  Location: Configured")
        logging.info(f"  Spectrogram Resolution: {app_config.get('processor.spectrogram_px_per_sec')} px/sec")
    else:
        logging.info("Audio Processing: DISABLED")
        
    regional_species = api.set_active_species(regional_species)

    # Initialize LLM verifier if API key is configured
    gemini_api_key = app_config.get('ai.gemini_api_key')
    llm_verifier = None
    if gemini_api_key:
        llm_verifier = LLMVerifier(
            api_key=gemini_api_key,
            model=app_config.get('ai.model'),
            min_confidence=app_config.get('ai.llm_verification.min_confidence'),
            max_calls_per_hour=app_config.get('ai.llm_verification.max_calls_per_hour'),
            max_calls_per_day=app_config.get('ai.llm_verification.max_calls_per_day'),
            latitude=app_config.get('secrets.latitude'),
            longitude=app_config.get('secrets.longitude'),
            log_dir=os.path.join('data', 'llm_verification_logs'),
        )
        logging.info("LLM Verification: ENABLED")
        logging.info(f"  Model: {app_config.get('ai.model')}")
        logging.info(f"  Min Confidence Threshold: {app_config.get('ai.llm_verification.min_confidence')}")
        logging.info(f"  Rate Limits: {app_config.get('ai.llm_verification.max_calls_per_hour')}/hour, {app_config.get('ai.llm_verification.max_calls_per_day')}/day")
    else:
        logging.info("LLM Verification: DISABLED")

    # Configure Detection Strategy
    strategy_type = app_config.get('processor.detection_strategy', 'single_stage')
    if strategy_type == 'global_two_stage':
        # Global strategy: YOLO binary detector + iNaturalist classifier (10,000+ species)
        # Supports Australian birds (cockatoos, king parrots, etc.) and other non-NABirds species
        detection_strategy = GlobalTwoStageStrategy(
            binary_model_path=app_config.get('processor.models.binary'),
            inat_model_name=app_config.get('processor.models.inat_classifier', 'rope_vit_reg4_b14_capi-inat21-224px'),
            regional_species=regional_species
        )
        logging.info("Detection Strategy: GLOBAL_TWO_STAGE (iNaturalist - 10,000+ species)")
        logging.info(f"  Binary Model: {app_config.get('processor.models.binary')}")
        logging.info(f"  iNat Classifier: {app_config.get('processor.models.inat_classifier', 'rope_vit_reg4_b14_capi-inat21-224px')}")
        logging.info("  Supports: Australian birds, Asian birds, European birds, etc.")
    elif strategy_type == 'two_stage':
        detection_strategy = TwoStageStrategy(
            binary_model_path=app_config.get('processor.models.binary'),
            classifier_model_path=app_config.get('processor.models.classifier'),
            regional_species=regional_species
        )
        logging.info("Detection Strategy: TWO_STAGE (NABirds - North American species only)")
        logging.info(f"  Binary Model: {app_config.get('processor.models.binary')}")
        logging.info(f"  Classifier Model: {app_config.get('processor.models.classifier')}")
    else:
        detection_strategy = SingleStageStrategy(
            model_path=app_config.get('processor.models.single_stage'),
            regional_species=regional_species
        )
        logging.info("Detection Strategy: SINGLE_STAGE")
        logging.info(f"  Model: {app_config.get('processor.models.single_stage')}")
    
    logging.info(f"Tracker: {app_config.get('processor.tracker')}")
    logging.info(f"Max Record Duration: {app_config.get('processor.max_record_seconds')}s")
    logging.info(f"Max Inactive Duration: {app_config.get('processor.max_inactive_seconds')}s")
    logging.info(f"Regional Species Count: {len(regional_species)}")
    logging.info("=" * 80)

    frame_processor = FrameProcessor(
        detection_strategy=detection_strategy,
        tracker=app_config.get('processor.tracker'), 
        save_images=app_config.get('processor.save_images')
    )
    fps_tracker = FPSTracker()

    # Main motion detection loop
    logging.info("Entering main motion detection loop - waiting for motion...")
    motion_event_count = 0
    while True:
        try:
            if not motion_detector.detect():
                logging.debug("Motion detector returned False, continuing to wait...")
                continue
            
            motion_event_count += 1
            api.notify_motion()

            # Configure video sources
            output_path = get_output_path()
            video_output = f"{output_path}/video.mp4"

            media_source.start_recording(video_output)

            logging.info(
                f'Motion event #{motion_event_count}: Starting video/audio recording to "{video_output}"')
            start_time = datetime.now(timezone.utc)

            # Video processing loop
            try:
                frame_processor.reset()
                decision_maker.reset()
                fps_tracker.reset()
                while True:
                    frame = media_source.capture()
                    if frame is None:
                        break
                    with fps_tracker:
                        has_detections = frame_processor.run(frame)

                    # Decision making
                    decision_maker.update_has_detections(has_detections)
                    species = decision_maker.decide_species(frame_processor.tracks)
                    if species is not None:
                        api.notify_species(species)
                    if decision_maker.decide_stop_recording():
                        break
                fps_tracker.log_summary()
            finally:
                media_source.stop_recording()
                end_time = datetime.now(timezone.utc)

            try:
                video_detections = decision_maker.get_results(
                    frame_processor.tracks)
                audio_detections, spectrogram_path = [], None
                if video_detections and audio_enabled and audio_processor:
                    audio_detections, spectrogram_path = audio_processor.run(
                        video_output)
                    
                    # LLM validation (if enabled)
                    if llm_verifier:
                        video_detections = llm_verifier.validate_detections(video_detections, start_time)
                        
                # Log summary without best_frame arrays
                video_summary = [{k: v for k, v in d.items() if k != 'best_frame'} for d in video_detections]
                logging.info(
                    f'Processing stopped. Video Result: {video_summary}; Audio Result: {audio_detections}')
                if len(video_detections) > 0:
                    api.create_video(video_detections, audio_detections, start_time,
                                     end_time, video_output, spectrogram_path)
                else:
                    # no detections, delete folder
                    try:
                        shutil.rmtree(output_path)
                    except Exception as e:
                        # Catch broad exception as shutil.rmtree can raise various errors
                        # (OSError, PermissionError, FileNotFoundError, etc.)
                        logging.warning(f"Failed to delete empty recording folder {output_path}: {e}")
            except Exception as e:
                logging.error(f"Error processing video results: {e}", exc_info=True)
        except KeyboardInterrupt:
            logging.info("Received shutdown signal, cleaning up...")
            break
        except Exception as e:
            logging.error(f"Error in main processing loop: {e}", exc_info=True)
            # Brief sleep before retrying to avoid tight error loop
            time.sleep(5)

    # Cleanup
    logging.info("Shutting down media source...")
    try:
        media_source.close()
    except Exception as e:
        logging.error(f"Error closing media source: {e}")


if __name__ == "__main__":
    main()
