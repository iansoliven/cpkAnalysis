import os
import re
from pathlib import Path
from typing import List, Dict, Optional

def ReadDirectory(directory: Optional[str] = None) -> Optional[List[Dict[str, str]]]:
    """
    Scan a directory for files matching the pattern XXX_LOT_EVENT_INTERVAL.xlsx or .stdf,
    extract LOT, EVENT, INTERVAL, and prompt the user for confirmation.
    Returns a list of dicts with filename and extracted metadata if confirmed, else None.
    """
    print(f"[DEBUG] Using directory: {directory if directory else os.getcwd()}")
    if directory is None:
        directory = os.getcwd()
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"[ERROR] Directory does not exist: {dir_path}")
        return None
    if not dir_path.is_dir():
        print(f"[ERROR] Path is not a directory: {dir_path}")
        return None
    print(f"[DEBUG] Directory contents: {[f.name for f in dir_path.iterdir()]}")
    regex_str = r".+_(?P<lot>[^_]+)_(?P<event>[^_]+)_(?P<interval>[^_.]+)\.(xlsx|stdf)$"
    print(f"[DEBUG] Using regex: {regex_str}")
    pattern = re.compile(regex_str, re.IGNORECASE)
    matched_files = []
    for file in dir_path.iterdir():
        print(f"[DEBUG] Checking file: {file.name}")
        if file.is_file() and file.suffix.lower() in {'.xlsx', '.stdf'}:
            m = pattern.match(file.name)
            print(f"[DEBUG] Regex match result for '{file.name}': {m}")
            if m:
                print(f"[DEBUG] Matched pattern: {file.name}")
                matched_files.append({
                    "filename": str(file.resolve()),
                    "lot": m.group("lot"),
                    "event": m.group("event"),
                    "interval": m.group("interval"),
                })
            else:
                print(f"[DEBUG] Did not match pattern: {file.name}")
        else:
            print(f"[DEBUG] Skipped (not file or wrong extension): {file.name}")
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
        print(f"[ERROR] Input failed: {e}")
        return None
    if confirm != 'y':
        print("Aborted by user.")
        return None
    return matched_files

# Placeholder for future GUI integration
# def ReadDirectoryGUI(...):
#     pass

if __name__ == "__main__":
    print("[DEBUG] read_directory.py script started")
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else None
    ReadDirectory(directory)
