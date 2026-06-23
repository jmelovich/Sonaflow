# CLAP vocabulary lists

CLAP tag matching uses **parallel vocabulary lists** stored as JSON files under
`clap_vocabs/` (next to `audio_routes.json` in the data directory).

Each audio scoring cycle embeds the window once, then compares it against every
**enabled** list independently. Top matches from each list are stabilized
(EMA + hysteresis) and appended to the prompt as compel-weighted tags.

## Default lists

On first run, three lists are created:

| File | Purpose |
|------|---------|
| `mood.json` | Specific emotional textures (not broad labels like "happy") |
| `style.json` | Hyper-specific art styles and rendering aesthetics |
| `imagery.json` | Concrete scene and object descriptors |

## File format

Each `*.json` file is one list:

```json
{
  "_comment": "CLAP vocabulary list. Edit phrases here; tune runtime fields in the UI.",
  "id": "mood",
  "name": "Mood",
  "order": 0,
  "group": "CLAP: Mood",
  "phrases": ["wistful nostalgia", "..."],
  "enable": true,
  "interval_sec": 1.0,
  "top_k": 5,
  "max_tags": 2,
  "enter_threshold": 0.12,
  "exit_threshold": 0.06,
  "score_tau": 1.5,
  "min_hold_sec": 1.5,
  "weight_min": 0.7,
  "weight_max": 1.4,
  "weight_step": 0.1,
  "softmax_temp": 0.02,
  "calibrate": true,
  "calib_alpha": 0.001,
  "calib_warmup": 8,
  "calib_spike_guard": 0.05
}
```

### Structural fields (edit in JSON, restart required)

- `id` — stable identifier (used internally; filename should match)
- `name` — display name
- `order` — prompt merge order (lower first)
- `group` — UI section label
- `phrases` — non-empty array of strings to match against

### Runtime fields (tune in UI, saved back to JSON on Save Defaults)

- `enable` — include this list in scoring
- `interval_sec` — seconds between CLAP searches for this list
- `top_k` — phrases retrieved each cycle
- `max_tags` — max injected tags after stabilization
- Threshold, weight, and calibration fields (same semantics as the old global CLAP settings)

## Adding a list

1. Create `clap_vocabs/my_list.json` with a unique `id` and a `phrases` array.
2. Restart the visualizer (new files and phrase edits require a restart).
3. A new UI section appears for the list's runtime controls.

## Path override

Set `clap_vocabs_dir` in `visualizer_settings.json` (Integration group) or leave
blank for `<data_dir>/clap_vocabs/`.

## Prompt merge order

Tags are appended in `order` (then `id`) across lists: base prompt, then mood
tags, then style tags, then imagery tags, etc.
