# Playing Atari with Deep Reinforcement Learning

**Paper:** Mnih et al. (2013)  
**arXiv:** https://arxiv.org/abs/1312.5602

---

## 1. Objective of the Paper

The paper investigates whether a reinforcement-learning agent can learn successful control policies directly from high-dimensional visual input.

The key idea is to combine:

- Q-learning
- convolutional neural networks
- experience replay

so that the agent can learn action-values directly from Atari game frames without hand-crafted features.

---

## 2. Main Motivation

Traditional reinforcement-learning methods worked well when states could be represented compactly.

Atari presents a harder problem because the state is represented by raw image pixels.

The paper therefore asks whether a single deep-learning architecture can learn useful representations and control policies directly from visual observations.

---

## 3. Core DQN Idea

Instead of storing a Q-table:

$$
Q(s,a)
$$

the paper approximates the action-value function using a neural network:

$$
Q(s,a;\theta)
$$

where $\theta$ represents the neural-network parameters.

For a given state, the network outputs one Q-value for each available action.

The greedy action is:

$$
a^* = \arg\max_a Q(s,a;\theta)
$$

---

## 4. State Representation

The agent does not use a single game frame as the full state.

Several consecutive frames are used together so that the network can infer motion.

For example, in Breakout a single frame shows the position of the ball, but multiple frames allow the agent to infer whether the ball is moving upward or downward.

Our implementation uses:

```text
4 stacked grayscale frames
84 × 84 pixels each
```

giving an input shape of:

```text
4 × 84 × 84
```

---

## 5. Neural Network Architecture

The implementation follows the small DQN architecture associated with the 2013 paper:

```text
Input: 4 × 84 × 84
        ↓
Conv2D: 16 filters, 8×8 kernel, stride 4
        ↓
Conv2D: 32 filters, 4×4 kernel, stride 2
        ↓
Fully Connected: 256 units
        ↓
Output: one Q-value per action
```

For Breakout, the output corresponds to:

```text
NOOP
FIRE
RIGHT
LEFT
```

---

## 6. Q-Learning Update

For a sampled transition:

$$
(s_t,a_t,r_t,s_{t+1})
$$

the DQN target is:

$$
y_t = r_t + \gamma \max_{a'}Q(s_{t+1},a';\theta)
$$

for non-terminal states.

For terminal states:

$$
y_t = r_t
$$

The network learns by reducing the difference between:

$$
Q(s_t,a_t;\theta)
$$

and the target $y_t$.

---

## 7. Experience Replay

The paper uses experience replay to store transitions:

$$
(s,a,r,s',done)
$$

Instead of training only on consecutive observations, random mini-batches are sampled from replay memory.

This helps:

- reduce temporal correlation;
- reuse previous experience;
- improve data efficiency;
- stabilise learning.

Our implementation stores these transitions in:

```text
src/replay_buffer.py
```

and samples random mini-batches during training.

---

## 8. Exploration

The agent uses an ε-greedy policy.

With probability:

$$
\epsilon
$$

the agent selects a random action.

Otherwise it selects:

$$
\arg\max_a Q(s,a)
$$

This allows the agent to balance:

- exploration of new actions;
- exploitation of learned Q-values.

---

## 9. Reward Handling

Rewards are clipped during training:

```text
positive reward → +1
zero reward     → 0
negative reward → -1
```

This reduces differences in reward scale and helps make the learning process more stable.

The original game reward is still useful for reporting episode performance.

---

## 10. Our Implementation Mapping

| Paper Concept | Our Implementation |
|---|---|
| Atari visual input | `ALE/Breakout-v5` |
| Frame preprocessing | `src/train.py` |
| 4-frame state | `src/train.py` |
| CNN Q-network | `src/dqn.py` |
| Experience replay | `src/replay_buffer.py` |
| ε-greedy exploration | `src/train.py` |
| TD target | `src/train.py` |
| Backpropagation | PyTorch in `src/train.py` |
| Saved model | `results/dqn_breakout.pth` |
| Evaluation | `src/evaluate.py` |

---

## 11. What We Implemented

Our repository demonstrates the full learning pipeline:

```text
Atari frame
   ↓
preprocessing
   ↓
4-frame state
   ↓
DQN
   ↓
Q-values
   ↓
ε-greedy action
   ↓
reward + next state
   ↓
replay buffer
   ↓
random mini-batch
   ↓
TD target
   ↓
loss
   ↓
backpropagation
```

---

## 12. What We Did Not Reproduce

This is not a full reproduction of the paper.

Differences include:

- only Breakout is used;
- training is much shorter;
- modern Gymnasium/ALE is used;
- hardware differs from the original work;
- evaluation is assignment-scale;
- our aim is to demonstrate the algorithm rather than reproduce the published benchmark table.

---

## 13. Experimental Observation

In one particular evaluation run, the trained model was tested over 1,000 Breakout episodes.

Observed results:

| Metric | Value |
|---|---:|
| Mean reward | 1.63 |
| Minimum reward | 0 |
| Maximum reward | 12 |

These results show that the agent learned behaviour capable of scoring, but performance remained variable.

This should be interpreted as evidence that the implementation works, not as a reproduction of the original paper's benchmark results.

---

## 14. Limitations

The small-scale experiment has several limitations:

- limited training budget;
- one Atari game only;
- high variance between episodes;
- no guarantee of convergence;
- modern environment implementation differs from the original setup.

---

## 15. Later DQN Improvements Not Used Here

The assigned paper predates several later improvements.

We do not present the following as part of the original 2013 method:

- Double DQN
- Dueling DQN
- Prioritized Experience Replay

These are later developments in the DQN family.

---

## 16. Conclusion

The paper demonstrated that deep neural networks can approximate Q-values directly from raw visual input and learn Atari control policies.

Our implementation reproduces the core learning pipeline at assignment scale:

- visual state input;
- CNN Q-function;
- ε-greedy exploration;
- experience replay;
- TD learning;
- gradient-based optimisation;
- independent evaluation.

The experiment validates the implementation while remaining separate from the original paper's benchmark claims.

---

## Reference

Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013).

**Playing Atari with Deep Reinforcement Learning**

arXiv:1312.5602

https://arxiv.org/abs/1312.5602
