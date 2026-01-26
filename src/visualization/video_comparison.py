"""
Video generation for agent behavior comparison.

Creates side-by-side video comparisons of different agents
performing the same tasks.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional video dependencies
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from src.agents.base_agent import BaseAgent
from src.environments.foraging_grid import ForagingGridEnv
from src.utils.logging import EpisodeLog


def generate_episode_video(
    episode: EpisodeLog,
    grid_size: int,
    output_path: Path,
    fps: int = 10,
    cell_size: int = 20,
    resource_positions: Optional[List[Tuple[int, int]]] = None,
) -> None:
    """
    Generate video of a single episode.

    Args:
        episode: Episode log to visualize
        grid_size: Environment grid size
        output_path: Path to save video
        fps: Frames per second
        cell_size: Pixel size of each grid cell
        resource_positions: Positions of resources
    """
    if not HAS_CV2:
        print("Warning: cv2 not available, skipping video generation")
        return

    # Video dimensions
    width = grid_size * cell_size
    height = grid_size * cell_size

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Colors (BGR format for OpenCV)
    BACKGROUND = (220, 220, 220)
    AGENT = (0, 0, 255)
    RESOURCE = (0, 200, 0)
    TRAIL = (200, 200, 255)

    trajectory = []

    for step in episode.steps:
        # Create frame
        frame = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)

        # Draw trail
        trajectory.append(step.position)
        for pos in trajectory[:-1]:
            x, y = int(pos[0]), int(pos[1])
            cv2.rectangle(
                frame,
                (y * cell_size, x * cell_size),
                ((y + 1) * cell_size, (x + 1) * cell_size),
                TRAIL,
                -1,
            )

        # Draw resources
        if resource_positions:
            for rx, ry in resource_positions:
                cv2.rectangle(
                    frame,
                    (ry * cell_size + 2, rx * cell_size + 2),
                    ((ry + 1) * cell_size - 2, (rx + 1) * cell_size - 2),
                    RESOURCE,
                    -1,
                )

        # Draw agent
        ax, ay = int(step.position[0]), int(step.position[1])
        cv2.circle(
            frame,
            (ay * cell_size + cell_size // 2, ax * cell_size + cell_size // 2),
            cell_size // 3,
            AGENT,
            -1,
        )

        # Add info text
        cv2.putText(
            frame,
            f"Step: {step.step} Energy: {step.energy:.0f}",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )

        out.write(frame)

    out.release()


def generate_comparison_video(
    episodes_dict: Dict[str, EpisodeLog],
    grid_size: int,
    output_path: Path,
    fps: int = 10,
    cell_size: int = 15,
) -> None:
    """
    Generate side-by-side comparison video of multiple agents.

    Args:
        episodes_dict: Dictionary of agent_name -> episode
        grid_size: Environment grid size
        output_path: Path to save video
        fps: Frames per second
        cell_size: Pixel size of each cell
    """
    if not HAS_CV2:
        print("Warning: cv2 not available, skipping video generation")
        return

    n_agents = len(episodes_dict)
    agent_names = list(episodes_dict.keys())

    # Panel dimensions
    panel_width = grid_size * cell_size
    panel_height = grid_size * cell_size + 30  # Extra space for label

    # Total dimensions
    total_width = panel_width * n_agents
    total_height = panel_height

    # Find max steps
    max_steps = max(e.total_steps for e in episodes_dict.values())

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (total_width, total_height))

    # Colors
    BACKGROUND = (220, 220, 220)
    AGENT = (0, 0, 255)
    TRAIL = (200, 200, 255)

    trajectories = {name: [] for name in agent_names}

    for step_idx in range(max_steps):
        # Create combined frame
        frame = np.full((total_height, total_width, 3), 255, dtype=np.uint8)

        for agent_idx, (name, episode) in enumerate(episodes_dict.items()):
            # Panel offset
            x_offset = agent_idx * panel_width

            # Draw panel background
            cv2.rectangle(
                frame,
                (x_offset, 0),
                (x_offset + panel_width, panel_height),
                BACKGROUND,
                -1,
            )

            # Add agent name
            cv2.putText(
                frame,
                name,
                (x_offset + 5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

            # Get step data if available
            if step_idx < len(episode.steps):
                step = episode.steps[step_idx]
                trajectories[name].append(step.position)

                # Draw trail
                for pos in trajectories[name][:-1]:
                    px, py = int(pos[0]), int(pos[1])
                    cv2.rectangle(
                        frame,
                        (x_offset + py * cell_size, 30 + px * cell_size),
                        (x_offset + (py + 1) * cell_size, 30 + (px + 1) * cell_size),
                        TRAIL,
                        -1,
                    )

                # Draw agent
                ax, ay = int(step.position[0]), int(step.position[1])
                cv2.circle(
                    frame,
                    (x_offset + ay * cell_size + cell_size // 2,
                     30 + ax * cell_size + cell_size // 2),
                    cell_size // 3,
                    AGENT,
                    -1,
                )

                # Add step info
                cv2.putText(
                    frame,
                    f"E:{step.energy:.0f}",
                    (x_offset + 5, panel_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 0),
                    1,
                )

        out.write(frame)

    out.release()


def create_matplotlib_animation(
    episode: EpisodeLog,
    grid_size: int,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 10),
    interval: int = 100,
) -> Optional[FuncAnimation]:
    """
    Create matplotlib animation of episode.

    Args:
        episode: Episode to animate
        grid_size: Environment grid size
        output_path: Path to save (optional)
        figsize: Figure size
        interval: Milliseconds between frames

    Returns:
        Animation object (or None if matplotlib unavailable)
    """
    if not HAS_MATPLOTLIB:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Initialize elements
    agent_dot, = ax.plot([], [], 'ro', markersize=15)
    trail_line, = ax.plot([], [], 'b-', alpha=0.5, linewidth=2)
    title = ax.set_title('')

    trajectory_x = []
    trajectory_y = []

    def init():
        agent_dot.set_data([], [])
        trail_line.set_data([], [])
        return agent_dot, trail_line

    def animate(frame_idx):
        if frame_idx < len(episode.steps):
            step = episode.steps[frame_idx]
            x, y = step.position

            trajectory_x.append(y)  # Swap for visualization
            trajectory_y.append(x)

            agent_dot.set_data([y], [x])
            trail_line.set_data(trajectory_x, trajectory_y)
            title.set_text(f'Step {step.step} | Energy: {step.energy:.1f}')

        return agent_dot, trail_line

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(episode.steps),
        interval=interval,
        blit=True,
    )

    if output_path:
        try:
            writer = FFMpegWriter(fps=10, metadata=dict(artist='DESIRE'))
            anim.save(str(output_path), writer=writer)
        except Exception as e:
            print(f"Warning: Could not save video: {e}")

    return anim


def record_agent_episode(
    agent: BaseAgent,
    env: ForagingGridEnv,
    output_path: Path,
    fps: int = 10,
    cell_size: int = 20,
    seed: Optional[int] = None,
) -> EpisodeLog:
    """
    Record an episode and generate video.

    Args:
        agent: Agent to record
        env: Environment
        output_path: Path for video output
        fps: Frames per second
        cell_size: Pixel size of cells
        seed: Random seed

    Returns:
        Episode log of recorded episode
    """
    from src.utils.logging import StepLog

    obs, info = env.reset(seed=seed)
    agent.reset()

    steps = []
    total_reward = 0.0

    for step_idx in range(env.max_steps):
        action = agent.act(obs)
        internal_state = agent.get_internal_state()
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        steps.append(StepLog(
            step=step_idx,
            action=action,
            position=tuple(env.agent_position),
            energy=env.agent_energy,
            reward=reward,
            internal_state=internal_state,
            info=info.copy(),
        ))

        agent.set_energy(env.agent_energy)

        if terminated or truncated:
            break

        obs = next_obs

    episode = EpisodeLog(
        episode_id=0,
        agent_type=agent.name,
        environment_mode=str(env.mode),
        total_steps=len(steps),
        total_reward=total_reward,
        final_energy=env.agent_energy,
        survived=env.agent_energy > 0,
        steps=steps,
    )

    # Generate video
    resource_positions = [(r.x, r.y) for r in env.resources if r.is_active]
    generate_episode_video(
        episode=episode,
        grid_size=env.grid_size,
        output_path=output_path,
        fps=fps,
        cell_size=cell_size,
        resource_positions=resource_positions,
    )

    return episode
