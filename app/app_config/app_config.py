import yaml
import os
import logging


class AppConfig:
    def __init__(self, user_config='user_config.yaml', default_config='default_config.yaml'):
        self.user_config_file = f"{os.path.dirname(__file__)}/{user_config}"
        self.default_config_file = f"{os.path.dirname(__file__)}/{default_config}"
        self.config = self.load_and_merge_configs()

    def load_and_merge_configs(self):
        # Load default config
        if not os.path.exists(self.default_config_file):
            raise FileNotFoundError(
                f"Default configuration file {self.default_config_file} not found."
            )

        with open(self.default_config_file, 'r') as file:
            default_config = yaml.safe_load(file) or {}

        # Load user config if it exists
        user_config = {}
        if os.path.exists(self.user_config_file):
            with open(self.user_config_file, 'r') as file:
                user_config = yaml.safe_load(file) or {}

        # Merge configs (user_config overrides default_config)
        merged = self.merge_dicts(default_config, user_config)
        
        # Apply environment variable overrides
        merged = self.apply_env_overrides(merged)
        
        return merged

    @staticmethod
    def merge_dicts(base, overrides):
        """Recursively merges two dictionaries."""
        for key, value in overrides.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = AppConfig.merge_dicts(base[key], value)
            else:
                base[key] = value
        return base

    def apply_env_overrides(self, config):
        """Apply environment variable overrides to config."""
        logger = logging.getLogger(__name__)
        
        # Map environment variables to config keys
        env_mappings = {
            'ENABLE_AUDIO_PROCESSING': 'processor.enable_audio_processing',
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Convert string to boolean for boolean settings
                lower_value = env_value.lower()
                if lower_value in ('true', '1', 'yes'):
                    value = True
                elif lower_value in ('false', '0', 'no'):
                    value = False
                else:
                    logger.warning(
                        f"Invalid boolean value '{env_value}' for {env_var}. "
                        f"Expected: true/false/1/0/yes/no. Ignoring override."
                    )
                    continue
                
                # Set the value in config
                keys = config_key.split('.')
                config_section = config
                for k in keys[:-1]:
                    config_section = config_section.setdefault(k, {})
                config_section[keys[-1]] = value
        
        return config

    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is None:
                return default
        return value

    def set(self, key, value):
        keys = key.split('.')
        config_section = self.config
        for k in keys[:-1]:
            config_section = config_section.setdefault(k, {})
        config_section[keys[-1]] = value

    def save(self, filename=None):
        save_file = filename or self.user_config_file
        with open(save_file, 'w') as file:
            yaml.safe_dump(self.config, file)


app_config = AppConfig()
