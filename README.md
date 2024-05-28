# RNN-RL
RNN+Transformer based reinforcement learning. Currently implemented for solving procedurally generated knapsack problems (https://en.wikipedia.org/wiki/Knapsack_problem)

Features:
- Transformer-based input encoder.
- GRU combines encodings with action history, using hidden layer for RL state representation.
- Pytorch only DQL implementation. With memory buffer and double-Q implementation.
- Bayesian Q-value output. Use MDN representation of Q-value for a richer understanding of expected reward. Could be used for custom exploration or inference strategies.

Documentation and code-cleaning work in progress.
