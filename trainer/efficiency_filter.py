import numpy as np


def baseline_cost(demo_episodes: dict) -> float:
    """Compute the mean path_len across all demo episodes as the inefficiency baseline.

    Uses path_len (pixel distance) as the cost metric because it is independent of
    time-step size and directly reflects trajectory efficiency.
    Falls back to step count if path_len is unavailable.
    """
    meta = demo_episodes.get("meta", {})
    if "path_len" in meta and len(meta["path_len"]) > 0:
        return float(np.mean(meta["path_len"]))
    if "n_steps" in meta and len(meta["n_steps"]) > 0:
        return float(np.mean(meta["n_steps"]))
    # No meta available — return a large sentinel so all episodes pass (permissive fallback)
    return float("inf")


def is_efficient(
    episode_meta: dict,
    baseline: float,
    margin: float,
    discriminator=None,
) -> bool:
    """Return True if the episode qualifies as a shortcut.

    Conditions (all must hold):
      1. is_success == True            — episode reached the goal
      2. had_collision == False        — physically clean trajectory
      3. cost < baseline * (1 - margin) — strictly faster than the demo average

    Args:
        episode_meta: dict with keys is_success, path_len, had_collision (optional).
        baseline:     reference cost from baseline_cost().
        margin:       fraction by which cost must beat baseline (BufferConfig.efficiency_margin).
        discriminator: optional callable(episode_meta) -> bool for plausibility checks
                       (AdaptDiffuser-style OOD filter — reserved, not used in MVP because
                       physical feasibility is already guaranteed by real env rollout).
    """
    if not episode_meta.get("is_success", False):
        return False
    if episode_meta.get("had_collision", False):
        return False

    cost = episode_meta.get("path_len", episode_meta.get("n_steps", float("inf")))
    threshold = baseline * (1.0 - margin)
    if cost >= threshold:
        return False

    if discriminator is not None and not discriminator(episode_meta):
        return False

    return True
