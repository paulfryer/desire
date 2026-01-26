"""
Checkpoint management utilities for training.
"""

import re
from pathlib import Path


def find_checkpoints(checkpoint_dir: Path | str, pattern: str = "*.zip") -> list[Path]:
    """
    Find all checkpoint files in a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints
        pattern: Glob pattern for checkpoint files

    Returns:
        List of checkpoint paths sorted by timestep (oldest first)
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []

    checkpoints = list(checkpoint_dir.glob(pattern))

    # Sort by timestep extracted from filename
    def extract_timestep(path: Path) -> int:
        match = re.search(r"_(\d+)_steps", path.name)
        return int(match.group(1)) if match else 0

    return sorted(checkpoints, key=extract_timestep)


def find_latest_checkpoint(checkpoint_dir: Path | str, pattern: str = "*.zip") -> Path | None:
    """
    Find the most recent checkpoint in a directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints
        pattern: Glob pattern for checkpoint files

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    checkpoints = find_checkpoints(checkpoint_dir, pattern)
    return checkpoints[-1] if checkpoints else None


def cleanup_old_checkpoints(
    checkpoint_dir: Path | str,
    keep_last: int = 5,
    pattern: str = "*.zip",
) -> list[Path]:
    """
    Remove old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of checkpoints to keep
        pattern: Glob pattern for checkpoint files

    Returns:
        List of deleted checkpoint paths
    """
    if keep_last <= 0:
        return []

    checkpoints = find_checkpoints(checkpoint_dir, pattern)

    if len(checkpoints) <= keep_last:
        return []

    # Delete oldest checkpoints
    to_delete = checkpoints[:-keep_last]
    deleted = []

    for checkpoint in to_delete:
        try:
            checkpoint.unlink()
            deleted.append(checkpoint)
        except OSError:
            pass  # Ignore deletion errors

    return deleted


def get_resume_timestep(checkpoint_path: Path | str) -> int:
    """
    Extract the timestep from a checkpoint filename.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Timestep number extracted from filename, or 0 if not found
    """
    checkpoint_path = Path(checkpoint_path)
    match = re.search(r"_(\d+)_steps", checkpoint_path.name)
    return int(match.group(1)) if match else 0
