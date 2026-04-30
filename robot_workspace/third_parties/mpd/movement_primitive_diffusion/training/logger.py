import sys
from pathlib import Path
from typing import Optional


class TeeWriter:
    """Custom writer that writes to both console and file."""
    
    def __init__(self, file_path: Path, original_stream):
        self.file = open(file_path, 'w', buffering=1)  # Line buffered
        self.original = original_stream
        
    def write(self, message):
        self.original.write(message)
        self.file.write(message)
        
    def flush(self):
        self.original.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()


def setup_training_logger(logging_path: Path, log_filename: str = "train.log") -> None:
    """Set up stdout/stderr redirection to both console and log file.
    
    Args:
        logging_path: Directory where log file will be created
        log_filename: Name of the log file
    """
    logging_path.mkdir(parents=True, exist_ok=True)
    log_file = logging_path / log_filename
    
    # Redirect stdout and stderr
    sys.stdout = TeeWriter(log_file, sys.stdout)
    sys.stderr = TeeWriter(log_file, sys.stderr)
    
    # Print output directory for user reference
    print(f"\n{'='*70}")
    print(f"📁 Output directory: {logging_path}")
    print(f"📝 Training log: {log_file}")
    print(f"{'='*70}\n")


def restore_original_streams(stdout_writer: Optional[TeeWriter] = None, 
                            stderr_writer: Optional[TeeWriter] = None) -> None:
    """Restore original stdout/stderr streams.
    
    Args:
        stdout_writer: TeeWriter for stdout (if any)
        stderr_writer: TeeWriter for stderr (if any)
    """
    if stdout_writer:
        stdout_writer.close()
    if stderr_writer:
        stderr_writer.close()
