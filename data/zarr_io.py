import os
import numpy as np
import zarr


def save_episode(
    store_path: str,
    states: np.ndarray,
    actions: np.ndarray | None = None,
    meta: dict | None = None,
) -> None:
    """Append one episode to a Zarr store (creates store if absent).

    Schema (diffusion-policy convention):
      /states        (total_T, 3)  float32
      /actions       (total_T, 3)  float32
      /episode_ends  (n_ep,)       int64   — cumulative end indices
      /meta/...      scalar arrays per key
    """
    os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
    states = np.asarray(states, dtype=np.float32)
    T = len(states)

    if actions is None:
        actions = np.zeros((T, 3), dtype=np.float32)
    else:
        actions = np.asarray(actions, dtype=np.float32)

    root = zarr.open(store_path, mode="a")

    if "states" not in root:
        root.create_dataset(
            "states", data=states, chunks=(min(T, 1000), 3), dtype="float32"
        )
        root.create_dataset(
            "actions", data=actions, chunks=(min(T, 1000), 3), dtype="float32"
        )
        root.create_dataset(
            "episode_ends", data=np.array([T], dtype=np.int64), chunks=(1000,), dtype="int64"
        )
    else:
        prev_total = int(root["states"].shape[0])
        root["states"].append(states)
        root["actions"].append(actions)
        new_end = prev_total + T
        root["episode_ends"].append(np.array([new_end], dtype=np.int64))

    if meta:
        if "meta" not in root:
            root.create_group("meta")
        meta_grp = root["meta"]
        for k, v in meta.items():
            arr = np.array([v]) if np.isscalar(v) else np.asarray(v)
            if k in meta_grp:
                meta_grp[k].append(arr)
            else:
                meta_grp.create_dataset(k, data=arr, chunks=(1000,))


def load_episodes(store_path: str) -> dict:
    """Load all episodes from a Zarr store.

    Returns dict with keys: states, actions, episode_ends, meta (dict).
    """
    root = zarr.open(store_path, mode="r")
    states = root["states"][:]
    actions = root["actions"][:]
    episode_ends = root["episode_ends"][:]

    meta = {}
    if "meta" in root:
        for k in root["meta"]:
            meta[k] = root["meta"][k][:]

    return {
        "states": states,
        "actions": actions,
        "episode_ends": episode_ends,
        "meta": meta,
    }


def num_episodes(store_path: str) -> int:
    """Return the number of episodes stored."""
    root = zarr.open(store_path, mode="r")
    return int(root["episode_ends"].shape[0])
