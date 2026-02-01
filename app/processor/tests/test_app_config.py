
import unittest
import os
import sys
import tempfile
import yaml

# Ensure project root is in path to import app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
# app/processor/tests -> app
app_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(app_path)

from app_config.app_config import AppConfig


class TestAppConfigEnvOverrides(unittest.TestCase):
    def tearDown(self):
        """Clean up environment variables."""
        if 'ENABLE_AUDIO_PROCESSING' in os.environ:
            del os.environ['ENABLE_AUDIO_PROCESSING']

    def test_env_override_true_string(self):
        """Test environment variable override with 'true' string."""
        os.environ['ENABLE_AUDIO_PROCESSING'] = 'true'
        config = {'processor': {'enable_audio_processing': False}}
        app_config = AppConfig.__new__(AppConfig)
        result = app_config.apply_env_overrides(config)
        self.assertTrue(result['processor']['enable_audio_processing'])

    def test_env_override_false_string(self):
        """Test environment variable override with 'false' string."""
        os.environ['ENABLE_AUDIO_PROCESSING'] = 'false'
        config = {'processor': {'enable_audio_processing': True}}
        app_config = AppConfig.__new__(AppConfig)
        result = app_config.apply_env_overrides(config)
        self.assertFalse(result['processor']['enable_audio_processing'])

    def test_env_override_1(self):
        """Test environment variable override with '1'."""
        os.environ['ENABLE_AUDIO_PROCESSING'] = '1'
        config = {'processor': {'enable_audio_processing': False}}
        app_config = AppConfig.__new__(AppConfig)
        result = app_config.apply_env_overrides(config)
        self.assertTrue(result['processor']['enable_audio_processing'])

    def test_env_override_0(self):
        """Test environment variable override with '0'."""
        os.environ['ENABLE_AUDIO_PROCESSING'] = '0'
        config = {'processor': {'enable_audio_processing': True}}
        app_config = AppConfig.__new__(AppConfig)
        result = app_config.apply_env_overrides(config)
        self.assertFalse(result['processor']['enable_audio_processing'])

    def test_env_override_yes(self):
        """Test environment variable override with 'yes'."""
        os.environ['ENABLE_AUDIO_PROCESSING'] = 'yes'
        config = {'processor': {'enable_audio_processing': False}}
        app_config = AppConfig.__new__(AppConfig)
        result = app_config.apply_env_overrides(config)
        self.assertTrue(result['processor']['enable_audio_processing'])

    def test_env_override_no(self):
        """Test environment variable override with 'no'."""
        os.environ['ENABLE_AUDIO_PROCESSING'] = 'no'
        config = {'processor': {'enable_audio_processing': True}}
        app_config = AppConfig.__new__(AppConfig)
        result = app_config.apply_env_overrides(config)
        self.assertFalse(result['processor']['enable_audio_processing'])

    def test_env_override_case_insensitive(self):
        """Test that environment variable values are case insensitive."""
        os.environ['ENABLE_AUDIO_PROCESSING'] = 'FALSE'
        config = {'processor': {'enable_audio_processing': True}}
        app_config = AppConfig.__new__(AppConfig)
        result = app_config.apply_env_overrides(config)
        self.assertFalse(result['processor']['enable_audio_processing'])

    def test_no_env_override(self):
        """Test that config is unchanged when no env var is set."""
        config = {'processor': {'enable_audio_processing': True}}
        app_config = AppConfig.__new__(AppConfig)
        result = app_config.apply_env_overrides(config)
        self.assertTrue(result['processor']['enable_audio_processing'])


if __name__ == '__main__':
    unittest.main()
