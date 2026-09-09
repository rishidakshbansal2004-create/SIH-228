# Trusted CV — Final Phases 6–9 (Model-Only)

No calibration image folder is required.

## Run image
```bash
python run_pipeline.py --image test.png --weights best.pt
```

## Run webcam
```bash
python run_pipeline.py --camera 0 --weights best.pt
```
Press `Q` or `Esc` to stop.

## Pipeline
- Phase 6: model-derived OOD/distribution-shift gate using confidence, margin, perturbation consistency and image-quality sanity checks.
- Phase 7: full YOLO inference using the supplied `best.pt`.
- Phase 8: inference integrity using calibration proxy, robustness, confidence and OOD/distribution-shift consistency. Actual accuracy requires labelled ground truth.
- Phase 9: append-only SHA-256 audit ledger containing the inference event.

This is a calibration-free deployment prototype. It should not be presented as a universal semantic OOD detector; thresholds are conservative engineering heuristics derived from the model's own outputs.
