import os

import numpy as np
import torch

from data.zarr_io import save_episode, load_episodes


class ReplayBuffer:
    """Sliding-window sampler built on top of the Zarr episode store.

    Guarantees windows never cross episode boundaries (avoids non-physical
    trajectories from mixing distinct demos).
    """

    def __init__(self, store_path: str, horizon: int, action_horizon: int) -> None:
        self.store_path = store_path
        self.horizon = horizon
        self.action_horizon = action_horizon
        self._data: dict = {}
        self._valid_starts: list[int] = []
        self._reload_index()

    @classmethod
    def from_zarr(cls, store_path: str, horizon: int, action_horizon: int) -> "ReplayBuffer":
        return cls(store_path, horizon, action_horizon)

    # ── index management ──────────────────────────────────────────────────────

    def _reload_index(self) -> None:
        if not os.path.exists(self.store_path):
            self._data = {}
            self._valid_starts = []
            return

        self._data = load_episodes(self.store_path)
        episode_ends = self._data["episode_ends"]

        win = max(self.horizon, self.action_horizon)
        starts: list[int] = []
        prev = 0
        for end in episode_ends:
            # Window [i, i+win) must lie fully within [prev, end)
            for i in range(prev, end - win + 1):
                starts.append(i)
            prev = int(end)
        self._valid_starts = starts

    # ── public API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._valid_starts)

    def add_episode(
        self,
        states: np.ndarray,
        actions: np.ndarray | None = None,
        meta: dict | None = None,
    ) -> None:
        save_episode(self.store_path, states, actions, meta)
        self._reload_index()

    def sample(self, batch_size: int) -> dict:
        if not self._valid_starts:
            raise RuntimeError(
                f"Buffer at '{self.store_path}' has no sample-able windows "
                f"(horizon={self.horizon}, action_horizon={self.action_horizon}). "
                "Add longer episodes or reduce horizon."
            )
        idxs = np.random.choice(self._valid_starts, size=batch_size, replace=True)
        states_arr = self._data["states"]
        actions_arr = self._data["actions"]

        state_windows = np.stack([states_arr[i : i + self.horizon] for i in idxs])
        action_windows = np.stack([actions_arr[i : i + self.action_horizon] for i in idxs])

        return {
            "states": torch.tensor(state_windows, dtype=torch.float32),
            "actions": torch.tensor(action_windows, dtype=torch.float32),
        }

    def episodes(self) -> dict:
        """Return the raw episode dict (states, actions, episode_ends, meta)."""
        return self._data
