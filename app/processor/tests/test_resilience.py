"""
Tests for resilience improvements
"""

import unittest
import time
import sys
import os

# Ensure project root is in path to import app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.abspath(os.path.join(current_dir, '../src'))
sys.path.append(src_path)

from decision_maker import DecisionMaker


class TestDecisionMakerResilience(unittest.TestCase):
    """Test resilience improvements in DecisionMaker"""
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise ValueError"""
        with self.assertRaises(ValueError):
            DecisionMaker(max_record_seconds=-1)
        
        with self.assertRaises(ValueError):
            DecisionMaker(max_inactive_seconds=0)
        
        with self.assertRaises(ValueError):
            DecisionMaker(min_track_duration=-1)
    
    def test_valid_parameters(self):
        """Test that valid parameters work correctly"""
        dm = DecisionMaker(max_record_seconds=30, max_inactive_seconds=5, min_track_duration=1)
        self.assertEqual(dm.max_record_seconds, 30)
        self.assertEqual(dm.max_inactive_seconds, 5)
        self.assertEqual(dm.min_track_duration, 1)
    
    def test_invalid_tracks_type(self):
        """Test that invalid tracks type returns empty list"""
        dm = DecisionMaker()
        result = dm.get_results(None)
        self.assertEqual(result, [])
        
        result = dm.get_results([])
        self.assertEqual(result, [])
        
        result = dm.get_results("invalid")
        self.assertEqual(result, [])
    
    def test_malformed_track_data(self):
        """Test that malformed track data is handled gracefully"""
        dm = DecisionMaker()
        
        # Track with missing required fields
        tracks = {
            1: {
                'preds': [('Cardinal', 0.9)]
                # Missing start_time and end_time
            }
        }
        result = dm.get_results(tracks)
        self.assertEqual(result, [])
        
        # Track with empty preds
        tracks = {
            1: {
                'start_time': time.time(),
                'end_time': time.time() + 1,
                'preds': []
            }
        }
        result = dm.get_results(tracks)
        self.assertEqual(result, [])
    
    def test_track_with_valid_data(self):
        """Test that valid track data is processed correctly"""
        dm = DecisionMaker(min_track_duration=0)
        
        tracks = {
            1: {
                'start_time': time.time(),
                'end_time': time.time() + 1,
                'preds': [('Cardinal', 0.9)] * 5,
                'best_frame': None
            }
        }
        result = dm.get_results(tracks)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['species_name'], 'Cardinal')


if __name__ == '__main__':
    unittest.main()
