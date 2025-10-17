import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def ReadDirectory(directory: Optional[str] = None) -> Optional[List[Dict[str, str]]]:
    """
    Scan a directory for files matching the pattern XXX_LOT_EVENT_INTERVAL.xlsx or .stdf,
    extract LOT, EVENT, INTERVAL, and prompt the user for confirmation.
    Returns a list of dicts with filename and extracted metadata if confirmed, else None.
    """
    if directory is None:
        directory = os.getcwd()
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.error("Directory does not exist: %s", dir_path)
        return None
    if not dir_path.is_dir():
        logger.error("Path is not a directory: %s", dir_path)
        return None
    regex_str = r".+_(?P<lot>[^_]+)_(?P<event>[^_]+)_(?P<interval>[^_.]+)\.(xlsx|stdf)$"
    logger.debug("Using regex: %s", regex_str)
    pattern = re.compile(regex_str, re.IGNORECASE)
    matched_files = []
    for file in dir_path.iterdir():
        if file.is_file() and file.suffix.lower() in {'.xlsx', '.stdf'}:
            m = pattern.match(file.name)
            if m:
                logger.debug("Matched file: %s", file.name)
                matched_files.append({
                    "filename": str(file.resolve()),
                    "lot": m.group("lot"),
                    "event": m.group("event"),
                    "interval": m.group("interval"),
                })
            else:
                logger.debug("File did not match pattern: %s", file.name)
        else:
            logger.debug("Skipped file (not target extension): %s", file.name)
    if not matched_files:
        print("No files matching the required pattern were found.")
        return None
    print("Detected files:")
    print(f"{'Filename':60} {'LOT':10} {'EVENT':10} {'INTERVAL':10}")
    for f in matched_files:
        print(f"{os.path.basename(f['filename']):60} {f['lot']:10} {f['event']:10} {f['interval']:10}")
    # CLI confirmation (placeholder for GUI)
    try:
        confirm = input("Are these correct? (y/n): ").strip().lower()
    except Exception as e:
        logger.error("Input failed: %s", e)
        return None
    if confirm != 'y':
        print("Aborted by user.")
        return None
    return matched_files

# Placeholder for future GUI integration
# def ReadDirectoryGUI(...):
#     pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("read_directory.py script started")
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    ReadDirectory(directory)
