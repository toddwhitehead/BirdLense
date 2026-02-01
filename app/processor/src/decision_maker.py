import logging
import time
from collections import Counter

logger = logging.getLogger(__name__)

# Minimum combined confidence to process a detection further (LLM validation, saving).
# Below this threshold, detections are discarded as likely false positives.
MIN_CONFIDENCE_TO_PROCESS = 0.10


class DecisionMaker():
    def __init__(self,  max_record_seconds=60, max_inactive_seconds=10, min_track_duration=2):
        # Validate input parameters
        if max_record_seconds <= 0:
            raise ValueError("max_record_seconds must be positive")
        if max_inactive_seconds <= 0:
            raise ValueError("max_inactive_seconds must be positive")
        if min_track_duration < 0:
            raise ValueError("min_track_duration must be non-negative")
            
        self.max_record_seconds = max_record_seconds
        self.max_inactive_seconds = max_inactive_seconds
        self.min_track_duration = min_track_duration
        self.reset()

    def reset(self):
        self.stop_recording_decided = False
        self.species_decided = False
        self.start_time = time.time()
        self.inactive_start_time = None

    def update_has_detections(self, has_detections):
        if not has_detections:
            if self.inactive_start_time is None:
                self.inactive_start_time = time.time()
        else:
            self.inactive_start_time = None

    def decide_stop_recording(self):
        if self.stop_recording_decided:
            # already decided once
            return False
        reached_max_record_seconds = (
            time.time() - self.start_time) >= self.max_record_seconds
        reached_max_inactive_seconds = self.inactive_start_time and (
            time.time() - self.inactive_start_time) >= self.max_inactive_seconds
        decision = reached_max_inactive_seconds or reached_max_record_seconds
        self.stop_recording_decided = decision
        return decision

    def decide_species(self, tracks):
        if self.species_decided:
            # already decided once
            return None
        results = self.get_results(tracks)
        if len(results) > 0:
            self.species_decided = True
            return results[0]['species_name']
        return None

    def get_results(self, tracks):
        if not isinstance(tracks, dict):
            logger.error(f"Invalid tracks type: {type(tracks)}")
            return []
            
        result = []
        for track_id, track in tracks.items():
            try:
                # Validate track structure
                if not isinstance(track, dict):
                    logger.warning(f"Invalid track structure for track_id {track_id}")
                    continue
                    
                # Skip tracks with no predictions yet
                if not track.get('preds'):
                    continue
                    
                # Validate required fields
                if 'start_time' not in track or 'end_time' not in track:
                    logger.warning(f"Missing time fields for track_id {track_id}")
                    continue
                    
                # Find most common prediction for each track
                # preds is a list of (species_name, confidence)
                species_only = [p[0] for p in track['preds']]
                if not species_only:
                    continue
                    
                pred_counts = Counter(species_only)
                species_name, count = pred_counts.most_common(1)[0]
                
                voting_confidence = count / len(track['preds'])
                
                # Calculate average classifier confidence for the winning species
                relevant_confs = [p[1] for p in track['preds'] if p[0] == species_name]
                if not relevant_confs:
                    continue
                    
                avg_classifier_conf = sum(relevant_confs) / len(relevant_confs)
                
                # Combine confidences
                confidence = voting_confidence * avg_classifier_conf
                
                # Skip tracks with very low confidence - likely false positives
                if confidence < MIN_CONFIDENCE_TO_PROCESS:
                    logger.debug(f"Skipping track {track_id} with {confidence:.0%} confidence - below threshold")
                    continue
                
                # Only consider species with at least min_track_duration
                duration = track['end_time'] - track['start_time']
                if duration >= self.min_track_duration:
                    logger.info(
                        f'Track {track_id} ACCEPTED: {species_name} | '
                        f'confidence: {confidence:.1%} (voting: {voting_confidence:.1%}, avg_cls: {avg_classifier_conf:.1%}) | '
                        f'duration: {duration:.1f}s | predictions: {len(track["preds"])}'
                    )
                    result.append({
                        'track_id': track_id,
                        'species_name': species_name,
                        'start_time': track['start_time'],
                        'end_time': track['end_time'],
                        'confidence': confidence,
                        'best_frame': track.get('best_frame'),
                        'source': 'video',
                        'frames': track.get('frames', [])  # Per-frame bounding box data
                    })
                else:
                    logger.debug(
                        f'Track {track_id} REJECTED (duration): {species_name} | '
                        f'duration: {duration:.1f}s < min: {self.min_track_duration}s'
                    )
            except Exception as e:
                logger.error(f"Error processing track {track_id}: {e}", exc_info=True)
                continue
        
        if result:
            logger.info(f'Final results: {len(result)} tracks accepted')
        else:
            logger.debug(f'No tracks met acceptance criteria (processed {len(tracks)} tracks)')

        return result
