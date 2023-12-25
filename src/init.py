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
        # Initialize IndexBuilder with the loaded configuration
        index_builder = IndexBuilder()
        # Create the index using the specified index path from the configuration
        index_builder.create_index('/app/output/index')
        
        logging.info("Index building process completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
