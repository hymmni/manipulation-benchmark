# Step 3 — Models: Interface + Minimal Stubs

**Date**: 2026-06-04
**Branch**: feat/0-mvp

## Overview

Implemented three neural network stubs for the manipulation pipeline:
`CVAE`, `LatentTrajectoryDiffusion`, `InverseDynamicsDiffusion`.
Goal: correct output shapes on CPU, trainable loss, and a wired-up interface
for Step 4's training loop — not production-quality performance.

## Modules Created

| File | Class | Key Signature |
|---|---|---|
| `models/nets.py` | `SinusoidalTimeEmbedding`, `MLPResNet`, `ConditionalDenoiser` | Shared backbone |
| `models/base.py` | `BaseGenerativeModel(ABC)` | `_build_scheduler(backbone)`, `loss(batch)`, `sample(cond)` |
| `models/cvae.py` | `CVAE(nn.Module)` | `encode/reparameterize/decode/forward/loss` |
| `models/latent_traj_diffusion.py` | `LatentTrajectoryDiffusion(BaseGenerativeModel)` | `sample(cond, num_samples, guidance_scale) → (ns, H, D)` |
| `models/inverse_dynamics.py` | `InverseDynamicsDiffusion(BaseGenerativeModel)` | `sample(cond, num_samples, exploration_noise) → (ns, AH, AD)` |

## Backbone: diffusers DDPMScheduler

- `DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")`
- Training: `add_noise(x0, eps, t)` → denoiser → MSE loss
- Sampling: `set_timesteps` → reverse loop over `scheduler.timesteps` → `step().prev_sample`
- Backbone abstraction in `BaseGenerativeModel._build_scheduler()` — `"flow_matching"` / `"shortcut"` reserved with `NotImplementedError`

## Classifier-Free Guidance (CFG) in LatentTrajectoryDiffusion

- During training: condition is zeroed out with probability `cfg.cond_dropout_prob` (default 0.1)
- During sampling: `eps = eps_uncond + scale * (eps_cond - eps_uncond)` when `guidance_scale > 0`
- `guidance_scale` defaults to `cfg.guidance_scale` but is overridable per-call

## exploration_noise in InverseDynamicsDiffusion (Self-Growth Core)

- `exploration_noise > 0` adds `N(0, exploration_noise²)` to `x` at **each reverse-diffusion step**
  (not to the final action) — on-manifold exploration aligned with DPPO's analysis
- `exploration_noise = 0`: near-deterministic under fixed seed (verified by test)
- `exploration_noise = 0.5`: significantly higher sample std than `= 0.0` (verified by test)

## Test Results

```
47 passed in 1.41s
```

New tests in `tests/test_models_shapes.py` (11 tests):
- CVAE: forward shapes, no-cond variant, scalar loss dict
- LatentTrajectoryDiffusion: sample shape (2, H, D), loss scalar, cpu-only
- InverseDynamicsDiffusion: sample shape (4, AH, AD), loss scalar, diversity test, determinism test, cpu-only
