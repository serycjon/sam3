# Streaming-tracker memory visualization tool — plan & status

Goal: a FastAPI server + vanilla-JS page that visualizes, per tracked frame, what the
SAM3 streaming tracker had in memory, what it attended to, and what it predicted —
driven entirely by artifacts saved during tracking (no model needed at viz time).

## Background (already implemented in the repo)

- `SAM3StreamingTracker(debug=True)` records one JSON-serializable dict per
  `init()`/`track()`/`correct()` call into `tracker.debug_log`
  (schema: see `SAM3StreamingTracker._record_debug` docstring). Key contents:
  - `attention.cond_selected/cond_unselected` — conditioning frames attended (slot 0);
  - `attention.spatial_mem` — `{frame_idx, t_pos, is_cond}` per attended spatial
    memory (`t_pos` 1..6 = non-cond slots, 6 = most recent);
  - `attention.obj_ptrs` — `{frame_idx, pos, is_cond}` per attended object pointer
    (`pos` = true distance for cond, recency rank for non-cond);
  - `scores` — current frame `eff_iou_score` / `iou_score` / `object_score_logits`;
  - `mem_state` — full bank after the call: cond ids + `{frame_idx: eff_iou_score}`;
  - `trimmed` / `cleared` / `evicted_corrections` — deletions made by the call.
  All frame indices absolute (init = 0, k-th `track()` = k).
  `tracker.debug_config` holds `mf_threshold`, `num_maskmem`, etc.
- `track(frame, return_all_masks=True)` returns `(mask, candidates)` where
  `candidates` has the 3 multimask logits/masks/ious **and** the single-mask token
  output (`token0_logits/mask/iou/stability`) — the mask the decoder would use with
  multimask off; computed every frame, normally discarded (`MaskDecoder.forward`
  stashes it when `expose_token0_output` is set; the streaming tracker sets it).
- Deliberately out of scope: attention *weights* (only the roster is recorded).

## Architecture

```
tracking script                              viz time
---------------                              --------
SAM3StreamingTracker(debug=True)             python scripts/memory_viz/server.py DUMP_DIR
  + DebugDumpWriter(out_dir)    --> DUMP --> FastAPI reads dump, serves JSON + images
    .add_frame(...) per frame        DIR    static/index.html + app.js render the UI
    .finalize(tracker) at end
```

### Dump directory format (produced by `sam3/debug_dump.py::DebugDumpWriter`)

```
dump_dir/
  debug.json               # {"config": tracker.debug_config, "log": tracker.debug_log}
  thumbs/000123.jpg        # per-frame thumbnail (default width 384, aspect kept)
  masks/000123.png         # final output mask, thumb resolution, 0/255 grayscale
  multimask/000123_0.png   # candidate mask k (binary 0/255), one PNG per candidate
  multimask/000123_token0.png  # single-mask-token output (binary), when present
  multimask.json           # {frame: {ious, n_candidates, token0_iou,
                           #          token0_stability}} — scalar sidecar metadata
```

Candidate/token0 masks are stored as binary PNGs (the viz only ever thresholds
them at logit 0, and the candidates dict already carries the thresholded booleans),
not float logits — this keeps the dump small. Only the scalar IoU/stability values
go in the sidecar `multimask.json`.

Writer usage in a tracking loop:

```python
tracker = SAM3StreamingTracker(debug=True)
dump = DebugDumpWriter("runs/seq01_dump")
mask = tracker.init(frame0, init_mask)
dump.add_frame(0, frame0, mask)
for i, frame in enumerate(frames, start=1):
    mask, cands = tracker.track(frame, return_all_masks=True)
    dump.add_frame(i, frame, mask, candidates=cands)
dump.finalize(tracker)   # writes debug.json
```

### Server (`scripts/memory_viz/server.py`)

Deps: none — Python standard library only (`http.server`). This is deliberate:
the HPC module env shadows the venv with an old `typing_extensions`, which breaks
fastapi/pydantic import; a stdlib server sidesteps all of it.
Run: `python scripts/memory_viz/server.py DUMP_DIR [--host 127.0.0.1] [--port 8123]`.

Endpoints:
- `GET /` → `static/index.html`; `GET /static/*` → js/css.
- `GET /api/meta` → `{config, num_records}`.
- `GET /api/log` → the full record list (fine up to ~100k records; paginate later
  if ever needed).
- `GET /thumbs/{frame:06d}.jpg`, `GET /masks/{frame:06d}.png` → files from the dump.
- `GET /api/multimask/{frame}` → `{available, ious, n_candidates, token0_iou,
  token0_stability}` (served from the preloaded `multimask.json`).
- `GET /multimask/{frame}/{k}.png` → candidate k (0..2) or `token0`, served
  directly as the stored binary PNG.

### Frontend (`scripts/memory_viz/static/`: `index.html`, `app.js`, `style.css`)

Vanilla JS, no build step, fetch + canvas. Master position = index into the record
log (so `correct` events are their own steps); frame number shown alongside.

Layout:
- **Top bar**: record slider, frame number, prev/next (also ← / →), play/pause
  (space), playback FPS.
- **Current frame panel** (left): thumbnail with mask overlay on a canvas.
  Overlay selector: final mask / candidate 0-2 / token0. Score readout:
  `eff_iou_score` (vs `mf_threshold`), `object_score_logits`, `iou_score`,
  per-candidate predicted IoUs, token0 IoU + stability. Event badge (init/track/
  correct; for correct also what was cleared/evicted).
- **Attention pools panel** (right), from `record.attention` — cells show thumbnail
  (with that frame's mask overlay), frame number, and that frame's bank score:
  1. *Conditioning* (slot 0): up to 4 cells, pinned first frame marked.
  2. *Non-cond spatial*: fixed slots 1..6 (empty slots rendered empty).
  3. *Object pointers*: up to 19 cells ordered as attended; cond-sourced marked;
     pointer-only frames (no spatial slot) visually distinct.
- **Memory bank strip** (bottom): every frame in `mem_state` in index order, colored
  by role this step: cond / spatial+ptr / ptr-only / retained-but-unattended;
  frames in `record.trimmed` shown as just-deleted.
- **Score timeline** (bottom): canvas plot of `eff_iou_score` per track record with
  the `mf_threshold` line and correct-event markers; click to seek. Doubles as the
  occlusion-regime view (score ≈ 0 stretches vs. above-threshold blips).

## Status / task checklist

- [x] Debug records (`debug=True`, `debug_log`, `save_debug_log`) — done earlier.
- [x] Token0 (single-mask token) exposure via `return_all_masks=True`.
- [x] `sam3/debug_dump.py` — `DebugDumpWriter`.
- [x] `scripts/memory_viz/server.py` — FastAPI app.
- [x] `scripts/memory_viz/static/` — index.html / app.js / style.css.
- [ ] Smoke test on the cluster: run a real sequence with `debug=True` +
      `DebugDumpWriter`, then `server.py` on the dump; check pools match
      expectations (≤4 cond + 6 spatial slots; ptr-only frames appear).
- [ ] Nice-to-have: image sizes >384px option; export current view as PNG; log-scale
      toggle for the score timeline; jump-to-frame-by-number input.

## Open questions / future

- Attention weights deliberately skipped (would need chunked softmax recompute
  around the 4 cross-attn layers; roster-only is the current decision).
- Very long runs: `debug.json` served whole; add range endpoints if it hurts.
- Correction workflows: `correct()` records have `attention: null`; the UI shows
  the event + bank mutations only.
