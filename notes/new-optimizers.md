# New optimizers

HAT-f supports two modern optimizers beyond BasicSR's built-in Adam/AdamW:
`Prodigy` and `ProdigyPlusScheduleFree`. Both eliminate manual LR tuning
via Prodigy's adaptive step-size mechanism.

## Prodigy (simple, needs scheduler)

Plain Prodigy adapts the learning rate; you still need a LR scheduler for decay.

```bash
uv pip install prodigyplus
```

```yaml
train:
  optim_g:
    type: Prodigy
    lr: 1.0
    betas: [0.9, 0.99]
    weight_decay: 0
  scheduler:
    type: MultiStepLR
    milestones: [125000, 200000, 225000, 240000]
    gamma: 0.5
```

Common `d0` / `d_coef` adjustments — pass them directly in the YAML:

```yaml
  optim_g:
    type: Prodigy
    lr: 1.0
    d0: 1e-6
    d_coef: 1.0
```

## ProdigyPlusScheduleFree (no scheduler needed)

Combines Prodigy's adaptive LR with Schedule-Free, which approximates a
decaying schedule internally. No `scheduler` section in the config.

```bash
uv pip install prodigy-plus-schedule-free
```

```yaml
train:
  optim_g:
    type: ProdigyPlusScheduleFree
    lr: 1.0
    betas: [0.9, 0.99]
    weight_decay: 0
  scheduler: ~
```

### Notable parameters

All optimizer parameters are passed directly from YAML to the constructor
via `**kwargs` — nothing special needed.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_schedulefree` | `True` | Enable Schedule-Free logic (no external scheduler needed) |
| `use_stableadamw` | `True` | Internal gradient scaling, replaces external gradient clipping |
| `use_speed` | `False` | Alternative adaptation: directional-progress-based (lower memory, experimental) |
| `factored` | `True` | Low-rank second moment (Adafactor-style). Saves memory. |
| `d_limiter` | `True` | Prevent d from growing too fast during early training |
| `d0` | `1e-6` | Initial guess for Prodigy's adaptive LR |
| `d_coef` | `1.0` | Scalar multiplier on the adapted LR |
| `prodigy_steps` | `0` | Freeze d and free Prodigy state after N steps (0 = never freeze) |
| `split_groups` | `True` | Per-parameter-group LR adaptation |

Example with `use_speed`:

```yaml
train:
  optim_g:
    type: ProdigyPlusScheduleFree
    lr: 1.0
    betas: [0.95, 0.99]
    use_speed: true
    use_stableadamw: true
    factored: true
    d_limiter: true
  scheduler: ~
```

### Required: optimizer.train() / eval()

Schedule-Free needs explicit mode switching. `HATModel` already handles this
automatically — `optimizer.train()` is called before each training step,
`optimizer.eval()` before validation. No user action needed.
