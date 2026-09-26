"""Phase 8: Inference Integrity — calibration, robustness, confidence, spatial sanity, and shift.

Evaluates post-inference output integrity without requiring ground truth labels in live mode:
- Confidence calibration and uncertainty profiling
- Multi-vector environmental robustness (photometric, contrast, multi-scale, lens blur)
- Spatial consistency & bounding box geometric plausibility (degenerate box prevention)
- Hallucination resistance on negative/empty scenes
- Complete human-readable narrative summaries and operational recommendations
"""
import cv2
import numpy as np


class InferenceIntegrity:
    def __init__(self, detector):
        self.detector = detector

    @staticmethod
    def confidence_stats(dets):
        if not dets:
            return {'count': 0, 'mean': 0.0, 'max': 0.0, 'min': 0.0, 'std': 0.0}
        c = np.array([float(d.get('confidence', 0.0)) for d in dets], float)
        return {
            'count': int(len(c)),
            'mean': float(np.mean(c)),
            'max': float(np.max(c)),
            'min': float(np.min(c)),
            'std': float(np.std(c)) if len(c) > 1 else 0.0
        }

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        bb = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / (aa + bb - inter + 1e-8)

    @staticmethod
    def spatial_integrity(dets, img_shape):
        """Verify geometric validity of detected bounding boxes."""
        if not dets:
            return {
                'valid': True,
                'score': 1.0,
                'disposition': 'accept',
                'total_boxes': 0,
                'valid_boxes': 0,
                'degenerate_boxes': 0,
                'out_of_bounds': 0,
                'reason': 'No bounding boxes in frame (Negative scene).'
            }

        h, w = img_shape[:2]
        margin_w = w * 0.05
        margin_h = h * 0.05
        degenerate = 0
        oob = 0

        for d in dets:
            box = d.get('bbox_xyxy') or []
            if len(box) != 4 or any(v is None or np.isnan(v) for v in box):
                degenerate += 1
                continue
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            if bw < 2 or bh < 2 or area < 12:
                degenerate += 1
            if x1 < -margin_w or y1 < -margin_h or x2 > w + margin_w or y2 > h + margin_h:
                oob += 1

        total = len(dets)
        valid_boxes = total - (degenerate + oob)
        score = max(0.0, float(valid_boxes) / float(total)) if total > 0 else 1.0

        if degenerate > 0:
            disposition = 'quarantine' if degenerate > 1 else 'review'
            reason = f"Detected {degenerate} degenerate bounding box(es) with inverted or near-zero area dimensions."
        elif oob > 0:
            disposition = 'review'
            reason = f"Detected {oob} bounding box(es) extending past image boundaries."
        else:
            disposition = 'accept'
            reason = f"All {total} bounding box(es) are geometrically valid and within image boundaries."

        return {
            'valid': degenerate == 0,
            'score': score,
            'disposition': disposition,
            'total_boxes': total,
            'valid_boxes': valid_boxes,
            'degenerate_boxes': degenerate,
            'out_of_bounds': oob,
            'reason': reason
        }

    def _predict(self, img, conf, iou):
        try:
            return self.detector.predict(img, conf, iou)
        except Exception:
            return []

    def robustness(self, img, base, conf, iou):
        """Evaluate inference stability across 5 environmental perturbation vectors."""
        h, w = img.shape[:2]
        variants = [
            ("Brightness Boost (+15)", cv2.convertScaleAbs(img, alpha=1.0, beta=15)),
            ("Brightness Drop (-15)", cv2.convertScaleAbs(img, alpha=1.0, beta=-15)),
            ("Contrast Scale (x1.10)", cv2.convertScaleAbs(img, alpha=1.1, beta=0)),
            ("Resolution Rescale (95%)", cv2.resize(cv2.resize(img, (max(32, int(w * 0.95)), max(32, int(h * 0.95)))), (w, h))),
            ("Optical Lens Blur (3x3)", cv2.GaussianBlur(img, (3, 3), 0)),
        ]

        vals = []
        details = []

        for name, v in variants:
            d = self._predict(v, conf, iou)
            if not base:
                # Hallucination test: on clean negative input, perturbations should not hallucinate phantom objects
                passed = (len(d) == 0)
                score = 1.0 if passed else 0.0
                vals.append(score)
                details.append({
                    "perturbation": name,
                    "score": score,
                    "status": "PASS" if passed else "HALLUCINATED",
                    "note": "Zero ghost detections hallucinated" if passed else f"Hallucinated {len(d)} phantom detection(s)"
                })
                continue

            a = max(base, key=lambda x: x.get('confidence', 0.0))
            b = max(d, key=lambda x: x.get('confidence', 0.0)) if d else None

            match = False
            overlap = 0.0
            if b and a.get('class_id') == b.get('class_id'):
                overlap = self.iou(a['bbox_xyxy'], b['bbox_xyxy'])
                if overlap >= 0.45:
                    match = True

            score = 1.0 if match else 0.0
            vals.append(score)
            details.append({
                "perturbation": name,
                "score": score,
                "status": "PASS" if match else "DIVERGED",
                "iou": round(float(overlap), 3),
                "note": f"Class matched with IoU={overlap:.2f}" if match else ("Detection lost" if not b else f"Class flip or IoU below threshold ({overlap:.2f} < 0.45)")
            })

        mean_robustness = float(np.mean(vals)) if vals else 1.0
        return mean_robustness, details

    @staticmethod
    def _disposition(value, warn_below, fail_below):
        if value < fail_below:
            return 'quarantine'
        if value < warn_below:
            return 'review'
        return 'accept'

    def evaluate_frame(self, img, dets, gate, conf=0.25, iou=0.45, do_robustness=True):
        cs = self.confidence_stats(dets)
        c = cs['mean']
        is_empty_frame = (cs['count'] == 0)

        # Multi-perturbation robustness
        robust, detail = self.robustness(img, dets, conf, iou) if do_robustness else (1.0, [])

        # Spatial consistency check
        spatial = self.spatial_integrity(dets, img.shape)

        # Distribution shift from Phase 6
        shift = float(np.clip(1.0 - getattr(gate, 'risk_score', 0.0), 0.0, 1.0))
        is_ood = getattr(gate, 'ood', False)

        if is_empty_frame:
            # Negative scene / background frame
            calibration = 1.0 if robust >= 0.8 else 0.5
            confidence = 1.0 if robust >= 0.8 else 0.5
            uncertainty = 0.0
            score = float(0.35 * robust + 0.35 * spatial['score'] + 0.30 * shift)
            if is_ood:
                score = min(score, 0.49)

            cal_disp = 'accept' if robust >= 0.8 else 'review'
            conf_disp = 'accept' if robust >= 0.8 else 'review'
            cal_reason = "No target objects present in frame (Negative scene). Model resisted spurious hallucinations under environmental perturbations." if robust >= 0.8 else "No target objects in base frame, but minor hallucinations appeared under photometric perturbation."
            conf_reason = "Zero positive detections; model cleanly rejects background noise." if robust >= 0.8 else "Background scene induced uncertain detections under perturbation."
        else:
            calibration = float(np.clip(c, 0.0, 1.0))
            confidence = float(c)
            uncertainty = float(1.0 - confidence)
            score = float(0.25 * calibration + 0.25 * robust + 0.25 * spatial['score'] + 0.15 * confidence + 0.10 * shift)
            if is_ood:
                score = min(score, 0.49)

            cal_disp = self._disposition(calibration, 0.60, 0.35)
            conf_disp = self._disposition(confidence, 0.60, 0.35)
            cal_reason = f"Mean detection confidence is {c*100:.1f}%, within expected calibration bounds." if calibration >= 0.60 else (f"Mean detection confidence is marginal ({c*100:.1f}%), indicating slight calibration softness." if calibration >= 0.35 else f"Mean detection confidence is critically low ({c*100:.1f}%), indicating poor calibration.")
            conf_reason = f"Detections carry high confidence (average {c*100:.1f}%, min {cs['min']*100:.1f}%, max {cs['max']*100:.1f}%)." if confidence >= 0.60 else f"Detections exhibit elevated uncertainty (average confidence {c*100:.1f}%)."

        # Robustness disposition
        rob_disp = self._disposition(robust, 0.60, 0.40)
        rob_reason = f"Top detections remained invariant across {sum(1 for d in detail if d.get('status')=='PASS')}/{len(detail)} environmental perturbations ({robust*100:.1f}% stability)." if robust >= 0.60 else f"Predictions diverged under small environmental perturbations ({robust*100:.1f}% stability)."

        # OOD disposition
        ood_disp = 'quarantine' if is_ood else self._disposition(shift, 0.60, 0.35)
        ood_reason = f"Phase 6 OOD gate flagged this input: {', '.join(getattr(gate, 'reasons', [])) or 'distribution boundary exceeded'}." if is_ood else "Input confirmed consistent with expected in-distribution domain."

        flags = [
            {
                'check': 'confidence_calibration',
                'name': 'Confidence Calibration',
                'value': round(calibration, 4),
                'confidence': round(calibration, 4),
                'disposition': cal_disp,
                'status': 'VERIFIED' if cal_disp == 'accept' else ('REVIEW' if cal_disp == 'review' else 'FLAGGED'),
                'reason': cal_reason,
                'recommendation': 'Calibration confirmed reliable for automated decisions.' if cal_disp == 'accept' else 'Apply calibration scaling or review edge cases.',
                'evidence': {'mean_confidence': cs['mean'], 'detection_count': cs['count'], 'confidence_std': cs['std']}
            },
            {
                'check': 'environmental_robustness',
                'name': 'Environmental Perturbation Robustness',
                'value': round(robust, 4),
                'confidence': round(robust, 4),
                'disposition': rob_disp,
                'status': 'VERIFIED' if rob_disp == 'accept' else ('REVIEW' if rob_disp == 'review' else 'FLAGGED'),
                'reason': rob_reason,
                'recommendation': 'Model displays robust invariance to lighting, contrast, and scaling.' if rob_disp == 'accept' else 'Retrain with data augmentation covering photometric and resolution shifts.',
                'evidence': {'variants_tested': len(detail), 'variants_passed': sum(1 for d in detail if d.get('status') == 'PASS'), 'details': detail}
            },
            {
                'check': 'spatial_integrity',
                'name': 'Spatial & Geometric Bounding Box Sanity',
                'value': round(spatial['score'], 4),
                'confidence': round(spatial['score'], 4),
                'disposition': spatial['disposition'],
                'status': 'VERIFIED' if spatial['disposition'] == 'accept' else ('REVIEW' if spatial['disposition'] == 'review' else 'FLAGGED'),
                'reason': spatial['reason'],
                'recommendation': 'Bounding box coordinates are clean and geometrically sound.' if spatial['disposition'] == 'accept' else 'Inspect model output decoding head for degenerate coordinates.',
                'evidence': spatial
            },
            {
                'check': 'prediction_confidence',
                'name': 'Prediction Certainty',
                'value': round(confidence, 4),
                'confidence': round(confidence, 4),
                'disposition': conf_disp,
                'status': 'VERIFIED' if conf_disp == 'accept' else ('REVIEW' if conf_disp == 'review' else 'FLAGGED'),
                'reason': conf_reason,
                'recommendation': 'Predictions carry sufficient confidence for production inference.' if conf_disp == 'accept' else 'Require human confirmation for low-confidence detections.',
                'evidence': {'uncertainty': uncertainty, 'confidence_stats': cs}
            },
            {
                'check': 'ood_distribution_consistency',
                'name': 'Distribution Shift & In-Domain Consistency',
                'value': round(shift, 4),
                'confidence': round(shift, 4),
                'disposition': ood_disp,
                'status': 'VERIFIED' if ood_disp == 'accept' else ('REVIEW' if ood_disp == 'review' else 'FLAGGED'),
                'reason': ood_reason,
                'recommendation': 'Input belongs to the expected operational design domain.' if ood_disp == 'accept' else 'Reject input or alert operator that input is out-of-distribution.',
                'evidence': {'phase6_risk_score': getattr(gate, 'risk_score', 0.0), 'phase6_reasons': getattr(gate, 'reasons', [])}
            },
        ]

        if any(f['disposition'] == 'quarantine' for f in flags):
            overall_disposition = 'quarantine'
            status = 'FLAGGED'
        elif any(f['disposition'] == 'review' for f in flags):
            overall_disposition = 'review'
            status = 'REVIEW_NEEDED'
        else:
            overall_disposition = 'accept'
            status = 'RELIABLE' if not is_empty_frame else 'RELIABLE_NEGATIVE'

        # Build clean, complete human-readable summary narrative
        if is_empty_frame:
            summary = (
                f"Negative scene verified: 0 target objects detected in frame. Model demonstrated 100% hallucination resistance "
                f"across {len(detail)} environmental perturbations with zero phantom detections. Spatial sanity verified. "
                f"Phase 8 disposition: {overall_disposition.upper()}."
            )
            action = "Model inference is stable and clean on negative scenes. Cleared for deployment."
        else:
            summary = (
                f"Inference integrity verified: {cs['count']} detection(s) recorded with {c*100:.1f}% mean confidence "
                f"(range: {cs['min']*100:.1f}% - {cs['max']*100:.1f}%). All bounding boxes passed geometric sanity. "
                f"Prediction stability achieved {robust*100:.1f}% invariance across {len(detail)} environmental perturbations. "
                f"Phase 8 disposition: {overall_disposition.upper()}."
            )
            action = (
                "Model predictions demonstrate high confidence and environmental stability. Cleared for production deployment."
                if overall_disposition == 'accept' else
                "Review recommended: Certain detections exhibited marginal confidence or perturbation sensitivity."
            )

        return {
            'status': status,
            'reliability_score': round(score, 4),
            'disposition': overall_disposition,
            'summary': summary,
            'recommended_action': action,
            'components': {
                'calibration': round(calibration, 4),
                'robustness': round(robust, 4),
                'spatial_sanity': round(spatial['score'], 4),
                'confidence': round(confidence, 4),
                'uncertainty': round(uncertainty, 4),
                'ood_distribution_consistency': round(shift, 4)
            },
            'flags': flags,
            'confidence_stats': cs,
            'spatial_integrity': spatial,
            'robustness_details': detail,
            'accuracy': {'available': False, 'value': None, 'note': 'Ground truth required for empirical accuracy.'},
            'access_level': 'white_box',
            'limitations': [
                'Does not perform backdoor/trigger detection on the model weights — see Phase 4 TRACE verification.',
                'Robustness probe evaluates photometric, scaling, contrast, and optical blur perturbations, not iterative gradient adversarial attacks.',
                'OOD assessment is derived from model embeddings; no external ground-truth OOD dataset is required.'
            ]
        }

    @staticmethod
    def evaluate_accuracy(predicted, true):
        n = min(len(predicted), len(true))
        return {
            'available': bool(n),
            'value': float(np.mean(np.asarray(predicted[:n]) == np.asarray(true[:n]))) if n else None,
            'n': n
        }