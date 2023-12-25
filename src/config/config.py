import os
import logging
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Config:
    """Configuration management class to read settings from a YAML file."""

    def __init__(self, config_file):
        """Initializes the Config object by loading the YAML configuration file."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Loading configuration from {config_file}")

        if not os.path.exists(config_file):
            self.logger.error(f"Configuration file {config_file} not found")
            raise FileNotFoundError(f"{config_file} does not exist")

        with open(config_file, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, key, default=None):
        """Retrieves a value for a given key from the configuration with a default fallback."""
        return self.config.get(key, default)
