import logging
import sys

def setup_logging():
    # prevent duplicate handlers
    if logging.getLogger().handlers:
        return
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log"),
        ],
    )
    