# OmniParser-as-filter is a post-hoc recovery-gate experiment, not an inline module_c gate (yet)

Status: accepted (2026-06-21)

## Context & decision

The `vlm_candidate_engine` plateau/NCC gate accepts 24.9% of candidate anchors;
`no_endpoint_qualifies` drives 67% of rejects via one hand-tuned threshold
(`MAX_ENDPOINT_PLATEAU = 0.035`). Hypothesis: a UI-element-aware signal (OmniParser
detector + OCR) distinguishes a real UI change (a localized cluster of new elements —
menu/dialog/panel) from pixel-plateau settling more robustly.

We are testing **validity before adoption**. Seven decisions were pinned in a
`grill-with-docs` session:

1. **Home.** Code lives on a branch of the existing fork `0bi0n3/OmniParser`
   (weights + batch harness already present), **not** a new repo and **not** inside
   `vlm_candidate_engine`. Keeps the main pipeline repo untouched.
2. **Mechanism = hybrid now, profile later.** Pixel-diff stays the trigger; OmniParser
   only *confirms* on the proposed pre/mid/post frames. The per-anchor PRE frame is the
   baseline — **no** resting-state profile, regional dock-grid, or forward-scan in v1.
   Those are v2, added only if per-anchor PRE-vs-POST proves too noisy.
3. **Delta metric = both types, IoU-matched new count.** Count elements in POST with no
   IoU match (start 0.5) to any PRE element; count both `interactivity:true` icons and
   `interactivity:false` OCR text (a dropdown adds both). Raw global `len` delta rejected
   — OCR flicker swings count ±5–10 (observed 130 vs 158 on near-identical frames).
4. **Locality = RESOLVED: require a proximity cluster of ≥5 new elements** (2026-06-22,
   from qualitative review of 6 Godot pairs). A raw new-count does NOT separate change from
   noise: scattered new detections are detector jitter/relayout (e.g. an accepted node-click
   gave new-count 38 spread across the frame — a false positive), while genuine new UI
   (dropdown/menu/panel) appears as a tight spatial group. Accept rule = cluster the
   `new_set` by spatial proximity (DBSCAN-like on box centres, `eps` ≈ a small fraction of
   frame width) and **accept iff the largest cluster has ≥5 members**; scattered news reject
   regardless of total count. This is a property of the new-element set (needs no v2 profile),
   and confirms the original "localized cluster" spec empirically. `eps`/min-cluster-size are
   the tunables to calibrate on the hand set.
   **Validation (6-pair check + human read, 2026-06-22):** single-linkage proximity clustering
   is the right metric — the human confirmed `click_node` (cluster≈25, new 38) is a GENUINE
   dense cluster of new UI, NOT a chaining artifact, and `drag` (cluster 6, new 7) is noise.
   So the chaining worry is dropped; gate on **largest-cluster size**. But the cut is **higher
   than 5**: real changes scored cluster 15–25, noise scored ≤6. The two middle cases
   (`click_button` cluster 5, `type` cluster 7) are unadjudicated and sit exactly on any
   boundary — so the threshold is NOT set from these 6; it is calibrated on the ~30–50 hand
   set (#8) via the go/no-go (#9). `eps` (tried 4/6/8% of width) also pins there.
5. **Calibration set = ~30–50 hand-picked pairs**, single reviewer, drawn from existing
   sample dirs (negatives/guardrail) **and** re-seeked false-rejects (recovery positives).
6. **Integration = post-hoc dir scorer.** A standalone script reads existing module_c
   sample dirs (`pre/mid/post.png`), runs OmniParser, writes accept/reject + new-count to
   a sidecar JSON. **Zero** changes to `vlm_candidate_engine`. Supersedes the earlier
   CONTEXT.md note about a feature-flagged hook inside `module_c` / `chrome_matcher.py` —
   that is reserved for *after* this experiment proves out.
7. **Detector config.** YOLO detector + OCR on, Florence captioner **off**
   (`use_local_semantics=False`, no 1.1G load — counts not captions). `BOX_TRESHOLD`
   starts 0.05. New conda env `omniparser`; weights from existing HF cache.

## Pre-registered go/no-go

Promote to inline integration **iff**, on the ~30–50 hand set: (a) the **largest-cluster
size** of new elements (per decision 4, not the raw new-count) cleanly separates menu-open
positives from no-op/Run negatives (ROC-AUC ≥ 0.8 or a visible threshold gap around the ≥5
cut), **AND** (b) zero of the known-good accepted dirs are wrongly flagged. Both hold →
scale to the ~3.1k top-7 reject sweep; else stop. (The raw-count formulation is superseded:
review showed scattered high counts are false positives — see decision 4.)

## Intended integration target (North Star, 2026-06-25)

The post-hoc dir scorer (decision 6) is the **validation harness**, not the destination.
Once the cluster-gate proves out, OmniParser is intended to **reinforce or replace the NCC
template-matching editor gate** (`chrome_matcher` / ADR-0016 in vce) — i.e. it becomes the
model that decides "a real editor UI change happened," not the hand-tuned pixel/template
heuristics. Production operating mode = the **forward-scan from the anchor** (the v2 mode
deferred in decision 2): given frames scanned forward from the anchor, OmniParser gates and
**fires on the first frame that presents the increased UI-element cluster** (largest cluster
≥ threshold). This widens the eventual scope beyond "swap the plateau quality gate" to
"supersede the template-matching gate, in forward-scan mode." It does **not** change v1: the
post-hoc scorer on existing pairs, with the NCC gate held fixed, remains the clean A/B that
must pass the go/no-go first. The cluster-gate function (`largest_cluster_size`) is identical
in both deployments — only the frame source (existing pair vs anchor-forward-scan) differs.

## Consequences / watch-items

- **Recovery needs frames rejects don't have.** Reject entries in `_results.json` carry
  `ts` + `reason` but no saved pngs; the re-seek utility must re-run module_c's B-stage
  scan from the h264 to obtain candidate pre/mid/post.
- **`actions[]` is unreliable for selecting recovery positives.** The 2026-06-21 re-extract
  did not attach the Gemini closed-vocab labels to new tuts (most reject `actions` are `[]`),
  so the "top-7 closed-vocab" filter must join to the external relabel or be hand-picked
  visually — do not trust `_results.json`'s `actions` field.
- **Domain gap is the live risk.** OmniParser is web/OS-UI trained; Godot is a dark-theme
  IDE. Smoke-test detector recall on editor icons before investing (mitigations: lower conf,
  fine-tune, or conclude the signal can't beat the plateau gate).
- **Confounder discipline.** Only the accept/reject *gate* moves; the triplet/pairing
  scaffold stays fixed, so any yield change is attributable.
