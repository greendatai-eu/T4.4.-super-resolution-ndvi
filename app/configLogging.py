import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("app.service")
logger.setLevel(
    logging.INFO
)  # Set the logging level as desired (DEBUG, INFO, WARNING, ERROR, CRITICAL)

log_filename = "record.log"
max_bytes = 1024 * 1024 * 5  # 5MB
backup_count = 1  # Number of backup files to keep

handler = RotatingFileHandler(
    log_filename, maxBytes=max_bytes, backupCount=backup_count
)
handler.setLevel(logging.INFO)  # Set the desired level for the handler
formatter = logging.Formatter(
    "%(asctime)s,%(msecs)d %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)
