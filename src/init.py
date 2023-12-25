# __init__.py

import logging
from config.config import Config
from index.builder import IndexBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main function to initialize and run the index building process."""
    try:
        # Load configuration
        config = Config('./config/config.yaml')

        # Initialize IndexBuilder with the loaded configuration
        index_builder = IndexBuilder(config)

        # Create the index using the specified index path from the configuration
        index_path = config.get('index_path')
        index_builder.create_index(index_path)
        
        logging.info("Index building process completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
