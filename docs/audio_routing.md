# Audio routing matrix

Sonaflow maps realtime audio features onto img2img feedback motion through a
**routing matrix**: a list of routes, each binding a source (and optional
amplitude source) to a motion target.

The matrix is split into two layers:

- **Structure** (what a route does) lives in an external JSON file you edit by
  hand: `audio_routes.json`.
- **Tuning** (`gain` and `enable`) is exposed in the settings UI as one clean,
  custom-labeled slider + toggle per route, so you can dial values at runtime.

## The routes file

Default location: `audio_routes.json` next to `visualizer_settings.json` (or
set `react_routes_path` to point elsewhere). On first run it is written
automatically from the built-in defaults.

```json
{
  "routes": [
    {
      "source": "bass",
      "amp": "one",
      "target": "zoom",
      "curve": 1.0,
      "mode": "add",
      "gain": 0.008,
      "enable": true,
      "label": "Bass -> Zoom",
      "tooltip": "Bass energy adds a steady forward zoom.",
      "group": "Audio Routes",
      "gain_min": 0.0,
      "gain_max": 0.05,
      "gain_step": 0.001
    }
  ]
}
```

### Fields

| Field | Edited where | Description |
|-------|--------------|-------------|
| `source` | file | Primary audio feature (see below) |
| `amp` | file | Second feature multiplied in (`one` = no scaling) |
| `target` | file | Motion parameter to drive |
| `curve` | file | Exponent applied to the source (`1` = linear) |
| `mode` | file | `add`, `mul`, or `max` |
| `gain` | **UI slider** | Signed scale factor |
| `enable` | **UI toggle** | Turn the route on/off |
| `label` | file | Slider label shown in the UI |
| `tooltip` | file | Hover help in the UI |
| `group` | file | UI section the controls appear under |
| `gain_min` / `gain_max` / `gain_step` | file | Slider range/precision |

Add, remove, or reorder routes by editing the `routes` array. The UI rebuilds
its controls from the file on the next launch. There is no fixed slot count.

**Contribution formula:**

```
contrib = gain * source^curve * amp
```

- **add** — summed into the target (before clamp)
- **mul** — multiplied into the target (e.g. tempo noise scaling)
- **max** — only the largest contribution among `max` routes is added

Final values are clamped: zoom to `[base, react_zoom_max]`, pan to
`+/-react_pan_max`, rotate to `+/-react_rotate_max`, noise to `[0, 1]`.

## Sources

### Continuous (EMA-smoothed)

`rms`, `bass`, `mid`, `treble`, `onset`, `centroid`, `rolloff`, `flux`,
`spread`, `zcr`, `pitch`, `pitch_conf`

### Tempo

- `bpm_mapped` — BPM linearly mapped `[react_bpm_map_min, react_bpm_map_max]` to `[0, 1]`
- `bpm_noise_mult` — legacy tempo to noise multiplier curve

### Events

- `beat` — decaying impulse on each detected beat
- `beat_bass` — bass level captured at the last beat hit
- `beat_loud` — signed loud-beat rotate punch (direction follows spin trend)

### Oscillators

- `orbit_sin`, `orbit_cos` — pan orbit at `react_pan_orbit_rate` rad/s

### Constant

- `one` — always `1.0` (use as `amp` when no scaling is needed)
- `none` — disabled / zero

## Targets

`zoom`, `pan_x`, `pan_y`, `rotate`, `noise_drift`

These map directly to `StreamPipeline.set_transform()` and update every
generation frame.

## Default routes

The generated `audio_routes.json` reproduces the legacy reactive behavior:

1. `Bass -> Zoom`
2. `Beat Zoom Punch`
3. `Beat Zoom Punch (bass-scaled)`
4. `Onset -> Noise`
5. `Tempo Noise Scale` (`mul`)
6. `Treble -> Rotate`
7. `Loud Beat -> Rotate`
8. `Mid Pan Orbit X`
9. `Mid Pan Orbit Y`
10. `Tempo -> Zoom (constant)` — disabled by default; enable in the UI for a
    constant zoom push that scales with detected tempo

## Global reactive controls

Under **Reactive** in the UI: caps (`react_zoom_max`, `react_rotate_max`,
`react_pan_max`), BPM mapping range, BPM noise curve, beat loud thresholds,
smoothing/decay taus, orbit rate, and analyzer norm decay.

## Persistence

Routes are stored only in the routes file. Clicking **Save Defaults** in the UI
writes both `visualizer_settings.json` (without routes) and the current routes
file (with your tuned `gain`/`enable` values). The main settings file keeps a
`react_routes_path` pointer but no route data.
