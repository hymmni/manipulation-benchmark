import numpy as np

from data.replay_buffer import ReplayBuffer


class SelfCollectedBuffer(ReplayBuffer):
    """High-quality self-collected episode buffer.

    Identical to ReplayBuffer but semantically reserved for shortcut episodes
    that passed the efficiency filter. Efficiency meta keys (is_success,
    path_len, n_steps, had_collision) are stored alongside each episode.
    """

    def add_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray | None = None,
        meta: dict | None = None,
    ) -> None:
        super().add_episode(states, actions, meta)
