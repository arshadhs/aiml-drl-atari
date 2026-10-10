"""
Evaluate a trained DQN agent on Atari Breakout.

Loads:
    results/dqn_breakout.pth

Runs the agent without training and reports evaluation rewards.
Optionally saves gameplay frames as a GIF.

Based on:
Mnih et al. (2013)
"Playing Atari with Deep Reinforcement Learning"
"""

import os
from collections import deque

import ale_py
import gymnasium as gym
import imageio
import numpy as np
import torch

from gymnasium.wrappers import AtariPreprocessing

from dqn import DQN


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ENV_NAME = "ALE/Breakout-v5"

MODEL_PATH = "results/dqn_breakout.pth"

NUM_EVAL_EPISODES = 1000

FRAME_STACK = 4

# Set to 0 for fully greedy evaluation.
# A small value such as 0.05 can also be used.
EPSILON = 0.5

SAVE_GIF = True

GIF_PATH = "results/gameplay.gif"

SEED = 100


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

def create_environment(render_mode=None):
    """
    Create Breakout environment.

    render_mode="rgb_array"
    is required if gameplay frames are to be saved.
    """

    gym.register_envs(ale_py)

    env = gym.make(
        ENV_NAME,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode=render_mode
    )

    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False
    )

    return env


# ---------------------------------------------------------
# Frame stacking
# ---------------------------------------------------------

def initialise_frame_stack(frame):
    """
    Repeat the initial frame four times.

    Output:
        deque containing 4 frames.
    """

    return deque(
        [frame.copy() for _ in range(FRAME_STACK)],
        maxlen=FRAME_STACK
    )


def stack_frames(frames):
    """
    Stack four frames into:

        (4, 84, 84)
    """

    return np.stack(
        frames,
        axis=0
    )


# ---------------------------------------------------------
# Action selection
# ---------------------------------------------------------

def select_action(
    model,
    state,
    env,
    epsilon=0.0
):
    """
    Choose an action using epsilon-greedy evaluation.

    Usually epsilon=0 during evaluation.
    """

    if np.random.random() < epsilon:
        return env.action_space.sample()

    state_tensor = torch.from_numpy(
        state
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(
            state_tensor
        )

    return q_values.argmax(
        dim=1
    ).item()


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate():

    render_mode = (
        "rgb_array"
        if SAVE_GIF
        else None
    )

    env = create_environment(
        render_mode=render_mode
    )

    num_actions = env.action_space.n

    print("Environment :", ENV_NAME)
    print("Actions     :", num_actions)
    print("Model       :", MODEL_PATH)
    print("Episodes    :", NUM_EVAL_EPISODES)
    print("Epsilon     :", EPSILON)

    # -----------------------------------------------------
    # Load DQN
    # -----------------------------------------------------

    model = DQN(
        num_actions=num_actions
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    print("Model loaded successfully.")

    rewards = []

    recorded_frames = []

    # -----------------------------------------------------
    # Episodes
    # -----------------------------------------------------

    for episode in range(
        NUM_EVAL_EPISODES
    ):

        frame, info = env.reset(
            seed=SEED + episode
        )
        
        # Breakout requires FIRE to start the ball.
        # Action 1 corresponds to FIRE in Breakout.
        frame, reward, terminated, truncated, info = env.step(1)

        frames = initialise_frame_stack(
            frame
        )

        state = stack_frames(
            frames
        )

        episode_reward = 0.0

        terminated = False
        truncated = False

        while not (
            terminated or truncated
        ):

            # ---------------------------------------------
            # Record visual frame
            # ---------------------------------------------

            if (
                SAVE_GIF
                and episode == 0
            ):

                rgb_frame = env.render()

                if rgb_frame is not None:
                    recorded_frames.append(
                        rgb_frame
                    )

            # ---------------------------------------------
            # Choose greedy action
            # ---------------------------------------------

            action = select_action(
                model,
                state,
                env,
                EPSILON
            )

            # ---------------------------------------------
            # Step environment
            # ---------------------------------------------

            next_frame, reward, terminated, truncated, info = \
                env.step(action)

            frames.append(
                next_frame
            )

            state = stack_frames(
                frames
            )

            episode_reward += reward

        rewards.append(
            episode_reward
        )

        print(
            f"Episode {episode + 1:3d} | "
            f"Reward {episode_reward:7.2f}"
        )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    rewards_array = np.array(
        rewards,
        dtype=np.float32
    )

    print()
    print("Evaluation complete.")
    print(
        f"Mean reward : "
        f"{rewards_array.mean():.2f}"
    )
    print(
        f"Min reward  : "
        f"{rewards_array.min():.2f}"
    )
    print(
        f"Max reward  : "
        f"{rewards_array.max():.2f}"
    )

    # -----------------------------------------------------
    # Save gameplay GIF
    # -----------------------------------------------------

    if (
        SAVE_GIF
        and len(recorded_frames) > 0
    ):

        os.makedirs(
            "results",
            exist_ok=True
        )

        imageio.mimsave(
            GIF_PATH,
            recorded_frames,
            fps=30
        )

        print(
            "Gameplay saved to:",
            GIF_PATH
        )

    env.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    evaluate()