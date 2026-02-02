import logging
from datetime import datetime
from gpiozero import MotionSensor

logger = logging.getLogger(__name__)


class PIRMotionDetector():
    def __init__(self, pin=4):
        self.pir = MotionSensor(pin)
        self.detection_count = 0
        logger.info(f"PIRMotionDetector initialized on GPIO pin {pin}")

    def detect(self):
        logger.debug("Waiting for motion on PIR sensor...")
        wait_start = datetime.now()
        self.pir.wait_for_motion()
        wait_duration = (datetime.now() - wait_start).total_seconds()
        self.detection_count += 1
        logger.info(
            f"🔔 MOTION DETECTED by PIR sensor! "
            f"(detection #{self.detection_count}, waited {wait_duration:.1f}s)"
        )
        return True
