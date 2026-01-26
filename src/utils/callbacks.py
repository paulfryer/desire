"""
Custom callbacks for training.
"""

from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback

from src.utils.checkpoint import cleanup_old_checkpoints


class CheckpointCallbackWithCleanup(CheckpointCallback):
    """
    Checkpoint callback that automatically cleans up old checkpoints.

    Extends stable-baselines3 CheckpointCallback to remove old checkpoints
    after each save, keeping only the most recent K checkpoints.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        keep_last: int = 5,
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        """
        Initialize the callback.

        Args:
            save_freq: Save checkpoints every `save_freq` steps
            save_path: Path to the folder where checkpoints will be saved
            name_prefix: Prefix for the checkpoint filenames
            keep_last: Number of checkpoints to keep (older ones are deleted)
            save_replay_buffer: Save the replay buffer (for off-policy algorithms)
            save_vecnormalize: Save the VecNormalize statistics
            verbose: Verbosity level
        """
        super().__init__(
            save_freq=save_freq,
            save_path=save_path,
            name_prefix=name_prefix,
            save_replay_buffer=save_replay_buffer,
            save_vecnormalize=save_vecnormalize,
            verbose=verbose,
        )
        self.keep_last = keep_last

    def _on_step(self) -> bool:
        """
        Called at each step. Save checkpoint if needed and cleanup old ones.

        Returns:
            True to continue training, False to stop
        """
        result = super()._on_step()

        # Check if a checkpoint was saved this step
        if self.n_calls % self.save_freq == 0:
            # Cleanup old checkpoints
            deleted = cleanup_old_checkpoints(
                checkpoint_dir=self.save_path,
                keep_last=self.keep_last,
                pattern=f"{self.name_prefix}_*_steps.zip",
            )
            if deleted and self.verbose > 0:
                print(f"Cleaned up {len(deleted)} old checkpoint(s)")

        return result
