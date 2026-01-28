import logging
import os
from datetime import datetime

def get_logger(log_dir="logs", name="experiment"):
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(name)
    logger.info(f"Logger initialized. Writing to {log_file}")
    return logger
