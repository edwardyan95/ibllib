"""
Copy selected mouse sessions from Cerebellum_Imaging while:
  - keeping directory structure,
  - including only session folders whose names EXACTLY match:
        1) ^\d+$                     (e.g. 20250914)
        2) ^\d+_valvesilent$
        3) ^\d+_valvesilent_punishsilent$
  - copying only files <= max_size_bytes (default 5 * 1024**3),
  - skipping files that already exist at the destination with same size and mtime,
  - logging actions to CSV (optional),
  - supporting a dry-run mode and disk-space pre-check.

Usage:
  
  python copy_imaging_sessions.py "Z:\TM_Lab\Edward\Cerebellum_Imaging\Pcp2-jgcamp8m" "I:\Cerebellum_imaging" --mice AF_R3,AF_L2
"""

import argparse
import csv
import os
import re
import sys
import time
import shutil
from pathlib import Path

# Optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

SESSION_PATTERNS = [
    re.compile(r'^\d+$'),                              # e.g. 20250914
    re.compile(r'^\d+_valvesilent$'),                  # e.g. 20250914_valvesilent
    re.compile(r'^\d+_valvesilent_punishsilent$'),     # e.g. 20250914_valvesilent_punishsilent
]

def is_valid_session(name: str) -> bool:
    return any(p.fullmatch(name) for p in SESSION_PATTERNS)

def approx_same_mtime(src: Path, dst: Path, tol_sec: float = 2.0) -> bool:
    """Return True if mtimes are within tol_sec."""
    try:
        sm = src.stat().st_mtime
        dm = dst.stat().st_mtime
        return abs(sm - dm) <= tol_sec
    except FileNotFoundError:
        return False

def should_skip_copy(src: Path, dst: Path) -> bool:
    """Skip if destination exists with same size and (approx) same mtime."""
    if not dst.exists():
        return False
    try:
        s = src.stat()
        d = dst.stat()
        if s.st_size == d.st_size and approx_same_mtime(src, dst):
            return True
    except FileNotFoundError:
        return False
    return False

def human_bytes(n: int) -> str:
    for unit in ['B','KB','MB','GB','TB','PB']:
        if n < 1024 or unit == 'PB':
            return f"{n:.1f} {unit}"
        n /= 1024

def scan_files_to_copy(src_root: Path, mice: list[str], max_size_bytes: int):
    """
    Returns a list of (src_file_path, rel_path) for all eligible files
    and a set of directories (relative) that will need to be created.
    Only files <= max_size_bytes from valid session folders are included.
    """
    files = []
    needed_dirs = set()

    for mouse in mice:
        mouse_dir = src_root / mouse
        if not mouse_dir.is_dir():
            print(f"[WARN] Mouse folder not found: {mouse_dir}", file=sys.stderr)
            continue

        # session folders under mouse directory
        for entry in mouse_dir.iterdir():
            if not entry.is_dir():
                continue
            session_name = entry.name
            if not is_valid_session(session_name):
                continue

            # Walk the session tree
            for dirpath, dirnames, filenames in os.walk(entry):
                dpath = Path(dirpath)
                rel_dir = dpath.relative_to(src_root)
                needed_dirs.add(rel_dir.as_posix())

                for fname in filenames:
                    fpath = dpath / fname
                    try:
                        st = fpath.stat()
                    except FileNotFoundError:
                        continue
                    if not fpath.is_file():
                        continue
                    if st.st_size <= max_size_bytes:
                        rel_file = fpath.relative_to(src_root).as_posix()
                        files.append((fpath, rel_file))
    return files, needed_dirs

def ensure_dirs(dest_root: Path, rel_dirs: set[str], dry_run: bool):
    for rel in sorted(rel_dirs):
        target_dir = dest_root / rel
        if dry_run:
            print(f"[DRY] mkdir -p {target_dir}")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)

def precheck_space(dest_root: Path, files: list[tuple[Path, str]]) -> tuple[int,int]:
    """
    Estimate total bytes to copy (sum of source sizes for files that either don't exist
    or differ), and return (needed_bytes, free_bytes_at_dest).
    """
    try:
        free = shutil.disk_usage(dest_root).free
    except FileNotFoundError:
        # If dest root does not exist yet, check its parent
        parent = dest_root if dest_root.exists() else dest_root.parent
        free = shutil.disk_usage(parent).free

    needed = 0
    for fpath, rel in files:
        dst = dest_root / rel
        if dst.exists():
            # If identical (size+mtime), skip its size
            try:
                if should_skip_copy(fpath, dst):
                    continue
            except Exception:
                pass
        try:
            needed += fpath.stat().st_size
        except FileNotFoundError:
            pass
    return needed, free

def write_log_header(log_writer: csv.writer):
    log_writer.writerow(["timestamp", "action", "status", "source", "destination", "bytes", "note"])

def main():
    parser = argparse.ArgumentParser(description="Copy selected mouse sessions with file-size limit.")
    parser.add_argument("src_root", type=str, help="Path to Cerebellum_Imaging source root")
    parser.add_argument("dest_root", type=str, help="Destination root")
    parser.add_argument("--mice", type=str, required=True,
                        help="Comma-separated list of mouse folder names to include, e.g. M123,M456")
    parser.add_argument("--max-gb", type=float, default=5.0,
                        help="Max file size in GB (default 5.0)")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not copy")
    parser.add_argument("--log", type=str, default=None, help="CSV log file path (optional)")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even if estimated needed space exceeds free space")
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    dest_root = Path(args.dest_root).resolve()
    mice = [m.strip() for m in args.mice.split(",") if m.strip()]
    max_size_bytes = int(args.max_gb * (1024**3))

    if not src_root.is_dir():
        print(f"[ERROR] Source root does not exist or is not a directory: {src_root}", file=sys.stderr)
        sys.exit(1)

    # Gather eligible files and needed dirs
    print("[INFO] Scanning eligible files...")
    files, rel_dirs = scan_files_to_copy(src_root, mice, max_size_bytes)
    print(f"[INFO] Eligible files (<= {args.max_gb} GB): {len(files)}")
    print(f"[INFO] Directories to ensure: {len(rel_dirs)}")

    # Precheck disk space
    needed, free = precheck_space(dest_root, files)
    print(f"[INFO] Estimated bytes to copy: {human_bytes(needed)}")
    print(f"[INFO] Free space at destination: {human_bytes(free)}")
    if needed > free and not args.force:
        print("[WARN] Estimated needed space exceeds free space at destination.")
        print("       Use --force to proceed anyway, or free up space / change destination.")
        if args.dry_run:
            print("[INFO] Continuing because this is a dry-run.")
        else:
            sys.exit(2)

    # Logging
    log_fp = None
    log_writer = None
    if args.log:
        log_fp = open(args.log, "w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_fp)
        write_log_header(log_writer)

    # Create directories
    ensure_dirs(dest_root, rel_dirs, args.dry_run)

    # Copy files
    iterator = files
    if tqdm is not None:
        iterator = tqdm(files, desc="Copying", unit="file")

    copied_bytes = 0
    copied_count = 0
    skipped_count = 0
    error_count = 0

    for src_path, rel in iterator:
        dst_path = dest_root / rel

        # Make sure parent dir exists (just in case)
        if not args.dry_run:
            dst_path.parent.mkdir(parents=True, exist_ok=True)

        action = "COPY"
        note = ""
        status = "OK"
        size = 0

        try:
            if args.dry_run:
                print(f"[DRY] copy2 {src_path} -> {dst_path}")
                status = "SKIPPED"
                note = "dry-run"
            else:
                # Skip if identical (size + ~mtime)
                if should_skip_copy(src_path, dst_path):
                    action = "SKIP"
                    status = "OK"
                    note = "exists_same_size_mtime"
                    skipped_count += 1
                else:
                    size = src_path.stat().st_size
                    shutil.copy2(src_path, dst_path)
                    # Align mtime (copy2 should do it, but we keep consistent)
                    try:
                        os.utime(dst_path, (src_path.stat().st_atime, src_path.stat().st_mtime))
                    except Exception:
                        pass
                    copied_bytes += size
                    copied_count += 1
        except Exception as e:
            status = "ERROR"
            note = str(e)
            error_count += 1

        # Log
        if log_writer:
            log_writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                action, status,
                str(src_path),
                str(dst_path),
                size,
                note
            ])

    if log_fp:
        log_fp.close()

    print("\n[SUMMARY]")
    print(f"  Files considered: {len(files)}")
    print(f"  Copied:          {copied_count} ({human_bytes(copied_bytes)})")
    print(f"  Skipped:         {skipped_count}")
    print(f"  Errors:          {error_count}")
    if args.log:
        print(f"  Log written to:  {args.log}")

if __name__ == "__main__":
    main()