# Context: OmniParser Conditional UI Filtering Integration

## 1. The Research Goal
`frame_change_threshold_with_min_max.py` relies on pixel differencing to detect target and valid UI changes. Results obtained are overly broad and missing dialog windows and menus.
- a target and valid UI change is a resting state of the UI then the appearance of a dialog window or drop down menu, or new set of options now appearing on the screen.

**Hypothesis**: invoking `OmniParser` model for inference on selected frames for filtering, will identify an increased cluster of interactive icons and text. Signifying a suitable change. If there is no increase, no valid change must be present.

## 2. Implementation
Do not over engineer a solution.
Run a rapid experiment to validate if `OmniParser` can detect an increase and cluster of interactive icons and text in a frame pair, with one being the resting state and the other being the increased detection. The experiment should be small and have an ablation toggle allowing for reviewing of with and without `OmniParser` detection.

**Logic Flow**
1. Maintain current `predictive_grounding_generator.py` pipeline in `vlm_candidate_engine`, with initial sweep of 1 FPS.
2. The conditional trigger is if a frame-to-frame pixel difference exceeds the threshold. Do not draw a bounding box.
3. Instead, invoke OmniParser on that specific pre/mid/post frame.
4. If OmniParser detects `> X` new UI elements, mark frame as valid and continue with pipeline processing.
5. If OmniParser does not detect new UI elements, filter out the action as invalid.

## 3. Strict Constraints and Boundaries
* Ablation toggle: build this as a separate module. It must not break or permanently alter the existing `chrome_matcher.py` or the baseline `smoke_test_121.sh` pipeline.
* **v1 is a post-hoc dir scorer, not an inline `module_c` hook** — it reads existing sample dirs and emits a sidecar verdict; `vlm_candidate_engine` is untouched. The feature-flag-inside-module_c phrasing above is reserved for *after* the experiment proves out. See ADR-0001.

## Resolved design — see [ADR-0001](docs/adr/0001-omniparser-posthoc-recovery-gate-experiment.md)
Seven branches pinned 2026-06-21 (home, mechanism, metric, locality, calib set, integration, go/no-go).

## Glossary

### New Element
An element detected in the POST frame with **no IoU match (≥ 0.5)** to any element in
the PRE frame. Counts **both** OmniParser output types — `interactivity:true` icons and
`interactivity:false` OCR text — because a dropdown/dialog opening introduces both. The raw
new-count is **not** the accept signal on its own (and never `len(POST) − len(PRE)`): see
Clustered Change.

### Clustered Change (the accept signal)
The accept decision is **not** "≥X new elements" but "the new elements form a tight spatial
**cluster** of ≥5". Resolved 2026-06-22 from qualitative review: a high *scattered* new-count
is detector jitter/relayout (a false positive — e.g. an accepted node-click scored 38 new
spread across the whole frame), whereas a genuine new UI (dropdown / menu / panel) shows as
one localized group. Operationalized as proximity clustering (DBSCAN-like on new-element box
centres, `eps` ≈ a small fraction of frame width); **accept iff the largest cluster ≥5**,
scattered news reject regardless of total. `eps` + min-cluster-size are calibrated on the
hand set. Needs no resting-state profile — it is a property of the new-element set itself.

### PRE / MID / POST
The three frames a `module_c` sample dir already holds. The pixel-diff gate proposes them
(the trigger); OmniParser only *confirms* on them (the hybrid mechanism). In v1 the PRE
frame **is** the baseline — there is no separate resting-state profile (that is v2).

### Resting-State Profile (v2 only)
A per-template baseline of mean element count + per-dock-region distribution over settled
editor frames. **Not built in v1.** Added only if per-anchor PRE-vs-POST counting proves
too noisy. Distinct from "New Element", which needs no profile.

### Recovery Positive vs Guardrail Negative
- **Recovery positive** — a frame pair the plateau gate *false-rejected* (a real
  menu/dialog change). Has no saved pngs; must be re-seeked from the h264 via module_c's
  B-stage scan. These test the hypothesis.
- **Guardrail** — an existing accepted dir (known-good). OmniParser must **not** flag it.
  These test that the new gate doesn't lose precision on the current 9,333.

### Detector Config
YOLO detector + OCR **on**, Florence captioner **off** (`use_local_semantics=False`).
We count and localize elements; we do not caption them.

