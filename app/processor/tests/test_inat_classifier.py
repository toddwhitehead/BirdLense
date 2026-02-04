"""
Tests for the iNaturalist classifier.

Run with: pytest tests/test_inat_classifier.py -v
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestINatClassifier:
    """Tests for INatClassifier class."""
    
    def test_common_name_mapping(self):
        """Test that Australian bird scientific names map to common names."""
        # Import the mapping directly
        from inat_classifier import INatClassifier
        
        # Test known mappings
        assert INatClassifier.COMMON_NAMES.get("Cacatua galerita") == "Sulphur-crested Cockatoo"
        assert INatClassifier.COMMON_NAMES.get("Alisterus scapularis") == "Australian King-Parrot"
        assert INatClassifier.COMMON_NAMES.get("Trichoglossus moluccanus") == "Rainbow Lorikeet"
        assert INatClassifier.COMMON_NAMES.get("Eolophus roseicapilla") == "Galah"
        assert INatClassifier.COMMON_NAMES.get("Dacelo novaeguineae") == "Laughing Kookaburra"
    
    def test_birder_not_available(self):
        """Test graceful handling when birder is not installed."""
        with patch.dict('sys.modules', {'birder': None}):
            # Force reimport to pick up the mock
            import importlib
            from inat_classifier import create_inat_classifier
            
            # Should return None when birder is not available
            # (actual behavior depends on BIRDER_AVAILABLE flag)
    
    @patch('inat_classifier.BIRDER_AVAILABLE', True)
    @patch('inat_classifier.birder')
    def test_classifier_initialization(self, mock_birder):
        """Test classifier initialization with mocked birder."""
        mock_net = Mock()
        mock_model_info = Mock()
        mock_model_info.signature = "test_signature"
        mock_model_info.rgb_stats = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
        mock_model_info.class_to_idx = {"Cacatua galerita": 0, "Alisterus scapularis": 1}
        
        mock_birder.load_pretrained_model.return_value = (mock_net, mock_model_info)
        mock_birder.get_size_from_signature.return_value = 224
        mock_birder.classification_transform.return_value = Mock()
        
        from inat_classifier import INatClassifier
        classifier = INatClassifier(model_name="test_model")
        
        assert classifier.net is mock_net
        assert classifier.size == 224
    
    def test_get_common_name_with_mapping(self):
        """Test _get_common_name with known scientific names."""
        with patch('inat_classifier.BIRDER_AVAILABLE', True), \
             patch('inat_classifier.birder') as mock_birder:
            
            mock_net = Mock()
            mock_model_info = Mock()
            mock_model_info.signature = "test"
            mock_model_info.rgb_stats = {}
            mock_model_info.class_to_idx = {}
            
            mock_birder.load_pretrained_model.return_value = (mock_net, mock_model_info)
            mock_birder.get_size_from_signature.return_value = 224
            mock_birder.classification_transform.return_value = Mock()
            
            from inat_classifier import INatClassifier
            classifier = INatClassifier()
            
            # Test known mapping
            assert classifier._get_common_name("Cacatua galerita") == "Sulphur-crested Cockatoo"
            
            # Test unknown scientific name (should return cleaned version)
            assert classifier._get_common_name("Aves_Unknown_species") == "Unknown species"
    
    def test_is_bird_class(self):
        """Test bird class detection."""
        with patch('inat_classifier.BIRDER_AVAILABLE', True), \
             patch('inat_classifier.birder') as mock_birder:
            
            mock_net = Mock()
            mock_model_info = Mock()
            mock_model_info.signature = "test"
            mock_model_info.rgb_stats = {}
            mock_model_info.class_to_idx = {}
            
            mock_birder.load_pretrained_model.return_value = (mock_net, mock_model_info)
            mock_birder.get_size_from_signature.return_value = 224
            mock_birder.classification_transform.return_value = Mock()
            
            from inat_classifier import INatClassifier
            classifier = INatClassifier()
            
            # Bird classes
            assert classifier._is_bird_class("Aves_Cacatua_galerita") == True
            assert classifier._is_bird_class("Aves/Psittaciformes") == True
            
            # Non-bird classes
            assert classifier._is_bird_class("Mammalia_Sciurus") == False
            assert classifier._is_bird_class("Insecta_Apis") == False


class TestCreateINatClassifier:
    """Tests for the factory function."""
    
    def test_factory_returns_none_when_birder_unavailable(self):
        """Test that factory returns None when birder is not installed."""
        with patch('inat_classifier.BIRDER_AVAILABLE', False):
            from inat_classifier import create_inat_classifier
            result = create_inat_classifier()
            assert result is None
