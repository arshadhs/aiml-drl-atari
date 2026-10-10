# environment setup, preprocessing, epsilon schedule, training loop, logging

"""
Train a Deep Q-Network on Atari Breakout.

Based on:
Mnih et al. (2013)
"Playing Atari with Deep Reinforcement Learning"

This is a small-scale educational reproduction intended for the
DRL assignment, not a full reproduction of the paper's training budget.
"""

import os
import random
from collections import deque

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from gymnasium.wrappers import AtariPreprocessing

from dqn import DQN
from replay_buffer import ReplayBuffer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ENV_NAME = "ALE/Breakout-v5"

GAMMA = 0.99
LEARNING_RATE = 0.00025

BATCH_SIZE = 32
REPLAY_CAPACITY = 50_000
LEARNING_STARTS = 5_000 # 500 

NUM_EPISODES = 100

EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 100_000

FRAME_STACK = 4

SEED = 42


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

def create_environment():
    """
    Create Atari Breakout environment.

    The base environment uses frameskip=1 because
    AtariPreprocessing will handle frame skipping.
    """

    gym.register_envs(ale_py)

    env = gym.make(
        ENV_NAME,
        frameskip=1,
        repeat_action_probability=0.0
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
    At the beginning of an episode we do not yet have
    four different frames.

    Therefore the first frame is copied four times.

    Output shape:
        (4, 84, 84)
    """

    frames = deque(
        [frame.copy() for _ in range(FRAME_STACK)],
        maxlen=FRAME_STACK
    )

    return frames


def stack_frames(frames):
    """
    Convert the deque of four 84x84 frames into
    a NumPy array of shape:

        (4, 84, 84)
    """

    return np.stack(frames, axis=0)


# ---------------------------------------------------------
# Epsilon schedule
# ---------------------------------------------------------

def get_epsilon(step):
    """
    Linearly reduce epsilon from 1.0 to 0.1.
    """

    fraction = min(
        step / EPSILON_DECAY_STEPS,
        1.0
    )

    epsilon = (
        EPSILON_START
        + fraction
        * (EPSILON_END - EPSILON_START)
    )

    return epsilon


# ---------------------------------------------------------
# Action selection
# ---------------------------------------------------------

def select_action(model, state, epsilon, env):
    """
    epsilon-greedy action selection.

    With probability epsilon:
        choose random action

    Otherwise:
        choose action with highest Q-value
    """

    if random.random() < epsilon:
        return env.action_space.sample()

    state_tensor = torch.from_numpy(
        state
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(state_tensor)

    return q_values.argmax(dim=1).item()


# ---------------------------------------------------------
# DQN learning step
# ---------------------------------------------------------

def optimise_model(
    model,
    optimiser,
    replay_buffer
):
    """
    Sample one mini-batch from replay memory
    and perform one DQN gradient update.
    """

    if len(replay_buffer) < BATCH_SIZE:
        return None

    states, actions, rewards, next_states, dones = \
        replay_buffer.sample(
            BATCH_SIZE,
            device
        )

    # -----------------------------------------------------
    # Current prediction Q(s, a)
    # -----------------------------------------------------

    all_q_values = model(states)

    current_q = all_q_values.gather(
        1,
        actions.unsqueeze(1)
    ).squeeze(1)

    # -----------------------------------------------------
    # DQN target:
    #
    # y = r + gamma * max_a Q(s', a)
    #
    # For terminal states:
    # y = r
    #
    # This follows the original 2013 DQN formulation.
    # -----------------------------------------------------

    with torch.no_grad():

        next_q_values = model(next_states)

        max_next_q = next_q_values.max(
            dim=1
        ).values

        target_q = (
            rewards
            + GAMMA
            * max_next_q
            * (1.0 - dones)
        )

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    loss = F.mse_loss(
        current_q,
        target_q
    )

    # -----------------------------------------------------
    # Backpropagation
    # -----------------------------------------------------

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    return loss.item()


# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------

def save_results(
    rewards_history,
    loss_history
):
    """
    Save training graphs.
    """

    os.makedirs(
        "results",
        exist_ok=True
    )

    # Rewards
    plt.figure()

    plt.plot(rewards_history)

    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("DQN Training Rewards")

    plt.savefig(
        "results/training_rewards.png"
    )

    plt.close()

    # Loss
    plt.figure()

    plt.plot(loss_history)

    plt.xlabel("Training Update")
    plt.ylabel("Loss")
    plt.title("DQN Training Loss")

    plt.savefig(
        "results/loss_curve.png"
    )

    plt.close()


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = create_environment()

    num_actions = env.action_space.n

    print("Environment :", ENV_NAME)
    print("Actions     :", num_actions)

    model = DQN(
        num_actions=num_actions
    ).to(device)

    optimiser = torch.optim.RMSprop(
        model.parameters(),
        lr=LEARNING_RATE
    )

    replay_buffer = ReplayBuffer(
        capacity=REPLAY_CAPACITY
    )

    rewards_history = []
    loss_history = []

    global_step = 0

    # -----------------------------------------------------
    # Episodes
    # -----------------------------------------------------

    for episode in range(NUM_EPISODES):

        frame, info = env.reset(
            seed=SEED + episode
        )

        frames = initialise_frame_stack(
            frame
        )

        state = stack_frames(
            frames
        )

        episode_reward = 0.0

        terminated = False
        truncated = False

        # Breakout requires FIRE to launch the ball.
        #
        # We don't force it here yet because the agent
        # should eventually learn from the action space.
        # During very short test runs you may see long
        # periods before FIRE is selected.
        while not (
            terminated or truncated
        ):

            epsilon = get_epsilon(
                global_step
            )

            # ---------------------------------------------
            # Choose action
            # ---------------------------------------------

            action = select_action(
                model,
                state,
                epsilon,
                env
            )

            # ---------------------------------------------
            # Environment step
            # ---------------------------------------------

            next_frame, reward, terminated, truncated, info = \
                env.step(action)

            done = (
                terminated
                or truncated
            )

            # ---------------------------------------------
            # Reward clipping
            # ---------------------------------------------

            clipped_reward = float(
                np.sign(reward)
            )

            # ---------------------------------------------
            # Build next state
            # ---------------------------------------------

            frames.append(
                next_frame
            )

            next_state = stack_frames(
                frames
            )

            # ---------------------------------------------
            # Save transition
            # ---------------------------------------------

            replay_buffer.push(
                state,
                action,
                clipped_reward,
                next_state,
                done
            )

            # ---------------------------------------------
            # Training
            # ---------------------------------------------

            if (
                len(replay_buffer)
                >= LEARNING_STARTS
            ):

                loss = optimise_model(
                    model,
                    optimiser,
                    replay_buffer
                )

                if loss is not None:
                    loss_history.append(
                        loss
                    )

            state = next_state

            episode_reward += reward

            global_step += 1

        rewards_history.append(
            episode_reward
        )

        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        recent_rewards = rewards_history[
            -10:
        ]

        mean_reward = np.mean(
            recent_rewards
        )

        print(
            f"Episode {episode + 1:4d} | "
            f"Steps {global_step:7d} | "
            f"Reward {episode_reward:7.2f} | "
            f"Avg10 {mean_reward:7.2f} | "
            f"Epsilon {epsilon:.3f} | "
            f"Replay {len(replay_buffer):6d}"
        )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        "results/dqn_breakout.pth"
    )

    save_results(
        rewards_history,
        loss_history
    )

    env.close()

    print()
    print("Training finished.")
    print(
        "Model saved to "
        "results/dqn_breakout.pth"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    train()