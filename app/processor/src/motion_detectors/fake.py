import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class FakeMotionDetector():
    def __init__(self, wait=60, motion=False):
        self.wait = wait
        self.motion = motion
        self.detection_count = 0
        logger.info(
            f"FakeMotionDetector initialized (wait={wait}s, motion={motion})"
        )

    def detect(self):
        logger.debug(f"FakeMotionDetector: Simulating wait for {self.wait}s...")
        wait_start = datetime.now()
        time.sleep(self.wait)
        self.detection_count += 1
        if self.motion:
            logger.info(
                f"🔔 MOTION DETECTED (simulated)! "
                f"(detection #{self.detection_count}, waited {self.wait}s)"
            )
        else:
            logger.debug(
                f"FakeMotionDetector: No motion (simulated) after {self.wait}s wait "
                f"(check #{self.detection_count})"
            )
        return self.motion
