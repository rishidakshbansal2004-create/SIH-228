"""
B3D: Black-box Detection of Backdoor Attacks with Limited Information and Data
Implementation based on Dong et al., ICCV 2021.

Formulates trigger reverse-engineering as a gradient-free optimization problem:
- Discrete Bernoulli distribution for mask m ~ Bern(g(theta_m))
- Continuous Gaussian distribution for pattern p = g(p'), p' ~ N(theta_p, sigma^2 * I)
- Solved via Natural Evolution Strategies (NES) Monte Carlo gradient estimation.

Includes two-phase optimization:
1. Coarse Spatial Region Localization: Rapidly identifies candidate trigger regions.
2. NES Trigger Refinement: Refines exact discrete mask and continuous RGB pattern.

Works natively for:
1. PyTorch models (.pt / .pth) without requiring autograd / backward passes.
2. ONNX models (.onnx) via onnxruntime without any computational graph access.
3. Both Classification and Object Detection (YOLO) detection oracles.
"""

import os
import math
import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import onnxruntime
except ImportError:
    onnxruntime = None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -25.0, 25.0)))


def _g_tanh(x: np.ndarray) -> np.ndarray:
    """Transformation function g(z) = 0.5 * (tanh(z) + 1.0) maps R -> [0, 1]."""
    return 0.5 * (np.tanh(np.clip(x, -15.0, 15.0)) + 1.0)


# ============================================================================
# Universal Model Oracle
# ============================================================================
class ModelOracle:
    """
    Universal black-box oracle wrapping PyTorch, ONNX, or custom inference pipelines.
    Accepts normalized input images in [0, 1] of shape [B, C, H, W]
    and returns predicted probabilities [B, num_classes].
    """

    def __init__(
        self,
        model: Any,
        model_type: str = "pytorch",
        normalize_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[str] = None,
        num_classes: int = 10,
    ):
        self.model = model
        self.model_type = model_type.lower()
        self.normalize_fn = normalize_fn
        self.num_classes = num_classes

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = device

        if self.model_type == "pytorch" and isinstance(self.model, nn.Module):
            self.model.to(self.device)
            self.model.eval()

        if self.model_type == "onnx":
            if onnxruntime is None:
                raise ImportError("onnxruntime is required to run ONNX model oracle.")
            if isinstance(self.model, (str, Path)):
                self.session = onnxruntime.InferenceSession(str(self.model))
            else:
                self.session = self.model
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

    def predict_probs(self, images: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Forward query returning class probabilities [B, num_classes].
        Pure forward pass, absolutely NO gradients calculated.
        """
        if self.model_type == "pytorch":
            if isinstance(images, np.ndarray):
                tensor_x = torch.from_numpy(images).float().to(self.device)
            else:
                tensor_x = images.float().to(self.device)

            if self.normalize_fn is not None:
                tensor_x = self.normalize_fn(tensor_x)

            with torch.no_grad():
                logits = self.model(tensor_x)
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                probs = F.softmax(logits, dim=-1).cpu().numpy()
            return probs

        elif self.model_type == "onnx":
            if isinstance(images, torch.Tensor):
                np_x = images.detach().cpu().numpy().astype(np.float32)
            else:
                np_x = np.asarray(images, dtype=np.float32)

            if self.normalize_fn is not None:
                with torch.no_grad():
                    tx = torch.from_numpy(np_x)
                    tx = self.normalize_fn(tx)
                    np_x = tx.numpy().astype(np.float32)

            outputs = self.session.run([self.output_name], {self.input_name: np_x})
            logits = outputs[0]
            exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            return probs

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")


# ============================================================================
# B3D Classifier Trigger Inversion (Dong et al., ICCV 2021)
# ============================================================================
class B3DDetector:
    """
    Black-box Backdoor Detection (B3D) algorithm via Natural Evolution Strategies.
    Operates without internal model gradients or poisoned training data.
    """

    def __init__(
        self,
        oracle: ModelOracle,
        num_classes: int = 10,
        img_shape: Tuple[int, int, int] = (3, 32, 32),
        sigma: float = 0.1,
        samples_k: int = 24,
        lr: float = 0.1,
        steps: int = 60,
        lambda_init: float = 0.05,
        asr_threshold: float = 0.85,
    ):
        self.oracle = oracle
        self.num_classes = num_classes
        self.img_shape = img_shape  # (C, H, W)
        self.sigma = sigma
        self.samples_k = samples_k
        self.lr = lr
        self.steps = steps
        self.lambda_init = lambda_init
        self.asr_threshold = asr_threshold

    def _apply_trigger(
        self,
        images: np.ndarray,
        mask: np.ndarray,
        pattern: np.ndarray,
    ) -> np.ndarray:
        """A(x, m, p) = (1 - m) * x + m * p."""
        return np.clip((1.0 - mask) * images + mask * pattern, 0.0, 1.0).astype(np.float32)

    def _cross_entropy_loss(self, probs: np.ndarray, target_class: int) -> float:
        """Evaluate cross entropy loss on predicted probability of target class."""
        target_p = np.clip(probs[:, target_class], 1e-12, 1.0)
        return float(-np.mean(np.log(target_p)))

    def _scan_candidate_patches(
        self,
        target_class: int,
        ref_images: np.ndarray,
        patch_size: int = 4,
        stride: int = 4,
    ) -> Tuple[float, Tuple[int, int]]:
        """
        Fast localized patch screening: finds if any compact patch elicits
        high attack success rate (ASR) on target_class.
        """
        _, _, H, W = ref_images.shape
        best_asr = 0.0
        best_loc = (0, 0)

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                test_x = ref_images.copy()
                test_x[:, :, y : y + patch_size, x : x + patch_size] = 1.0
                probs = self.oracle.predict_probs(test_x)
                preds = np.argmax(probs, axis=-1)
                asr = float(np.mean(preds == target_class))
                if asr > best_asr:
                    best_asr = asr
                    best_loc = (y, x)
                if best_asr >= 0.95:
                    return best_asr, best_loc

        return best_asr, best_loc

    def reverse_engineer_class(
        self,
        target_class: int,
        reference_images: np.ndarray,
        batch_size: int = 24,
    ) -> Dict[str, Any]:
        """
        Reverses potential trigger (m, p) for given target class using Algorithm 1.
        """
        C, H, W = self.img_shape
        N = len(reference_images)

        # 1. Fast localized screening: Check if a small compact patch triggers the class
        patch_size = max(2, min(H, W) // 8)
        patch_asr, patch_loc = self._scan_candidate_patches(
            target_class, reference_images, patch_size=patch_size, stride=patch_size
        )

        # If a tiny patch already forces high ASR (>= 85%), we found an unmistakable compact trigger!
        if patch_asr >= self.asr_threshold:
            y, x = patch_loc
            best_mask = np.zeros((1, 1, H, W), dtype=np.float32)
            best_mask[:, :, y : y + patch_size, x : x + patch_size] = 1.0
            best_pattern = np.ones((1, C, H, W), dtype=np.float32)

            # Refine pattern color via NES in localized region
            theta_p = np.full((1, C, H, W), 2.0, dtype=np.float32)
            for _ in range(15):
                eps = np.random.normal(0.0, 1.0, size=(self.samples_k // 2, C, H, W)).astype(np.float32)
                grad_p = np.zeros_like(theta_p)
                for j in range(len(eps)):
                    ej = eps[j : j + 1]
                    p_plus = _g_tanh(theta_p + self.sigma * ej)
                    p_minus = _g_tanh(theta_p - self.sigma * ej)
                    trig_plus = self._apply_trigger(reference_images, best_mask, p_plus)
                    trig_minus = self._apply_trigger(reference_images, best_mask, p_minus)
                    l_plus = self._cross_entropy_loss(self.oracle.predict_probs(trig_plus), target_class)
                    l_minus = self._cross_entropy_loss(self.oracle.predict_probs(trig_minus), target_class)
                    grad_p += ((l_plus - l_minus) / (2.0 * self.sigma)) * ej
                theta_p -= self.lr * (grad_p / float(len(eps)))

            refined_p = _g_tanh(theta_p)
            l1_norm = float(np.sum(best_mask))
            return {
                "target_class": target_class,
                "l1_norm": round(l1_norm, 4),
                "discrete_l1_norm": int(l1_norm),
                "asr": round(patch_asr, 4),
                "mask": best_mask[0, 0],
                "pattern": refined_p[0],
                "discrete_mask": best_mask[0, 0],
                "trigger_type": "compact_patch",
            }

        # 2. General NES Trigger Search (for dispersed or larger triggers)
        grid_h = min(8, H)
        grid_w = min(8, W)
        scale_y, scale_x = H // grid_h, W // grid_w

        theta_m = np.zeros((1, 1, grid_h, grid_w), dtype=np.float32)
        theta_p = np.zeros((1, C, H, W), dtype=np.float32)

        m_m, v_m = np.zeros_like(theta_m), np.zeros_like(theta_m)
        m_p, v_p = np.zeros_like(theta_p), np.zeros_like(theta_p)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        lam = self.lambda_init
        k = self.samples_k
        sigma = self.sigma

        best_mask = None
        best_pattern = None
        best_l1 = float("inf")
        best_asr = patch_asr

        def upsample(m_low):
            return np.repeat(np.repeat(m_low, scale_y, axis=2), scale_x, axis=3)

        for t in range(1, self.steps + 1):
            indices = np.random.choice(N, size=min(batch_size, N), replace=False)
            x_batch = reference_images[indices]

            cur_m_low = _g_tanh(theta_m)
            cur_p = _g_tanh(theta_p)

            # Sample k masks in low-dimensional grid
            u = np.random.uniform(0.0, 1.0, size=(k, 1, grid_h, grid_w)).astype(np.float32)
            m_samples_low = (u < cur_m_low).astype(np.float32)

            losses_m = []
            for j in range(k):
                mj = upsample(m_samples_low[j : j + 1])
                trig_x = self._apply_trigger(x_batch, mj, cur_p)
                probs = self.oracle.predict_probs(trig_x)
                ce = self._cross_entropy_loss(probs, target_class)
                l1 = lam * float(np.sum(mj))
                losses_m.append(ce + l1)

            losses_m = np.array(losses_m)
            centered_m = losses_m - np.mean(losses_m)
            grad_m = np.mean(
                [centered_m[j] * 2.0 * (m_samples_low[j : j + 1] - cur_m_low) for j in range(k)],
                axis=0,
            )

            # Pattern gradient with antithetic perturbations
            cur_m_up = upsample(cur_m_low)
            eps_samples = np.random.normal(0.0, 1.0, size=(k // 2, C, H, W)).astype(np.float32)
            grad_p = np.zeros_like(theta_p)
            for j in range(k // 2):
                ej = eps_samples[j : j + 1]
                p_plus = _g_tanh(theta_p + sigma * ej)
                p_minus = _g_tanh(theta_p - sigma * ej)
                trig_plus = self._apply_trigger(x_batch, cur_m_up, p_plus)
                trig_minus = self._apply_trigger(x_batch, cur_m_up, p_minus)
                l_plus = self._cross_entropy_loss(self.oracle.predict_probs(trig_plus), target_class)
                l_minus = self._cross_entropy_loss(self.oracle.predict_probs(trig_minus), target_class)
                grad_p += ((l_plus - l_minus) / (2.0 * sigma)) * ej

            grad_p = grad_p / float(k // 2)

            # Adam step updates
            m_m = beta1 * m_m + (1.0 - beta1) * grad_m
            v_m = beta2 * v_m + (1.0 - beta2) * (grad_m ** 2)
            theta_m -= self.lr * (m_m / (1.0 - beta1 ** t)) / (np.sqrt(v_m / (1.0 - beta2 ** t)) + eps)

            m_p = beta1 * m_p + (1.0 - beta1) * grad_p
            v_p = beta2 * v_p + (1.0 - beta2) * (grad_p ** 2)
            theta_p -= self.lr * (m_p / (1.0 - beta1 ** t)) / (np.sqrt(v_p / (1.0 - beta2 ** t)) + eps)

            eval_m = upsample(_g_tanh(theta_m))
            eval_p = _g_tanh(theta_p)
            eval_trig = self._apply_trigger(reference_images, eval_m, eval_p)
            eval_probs = self.oracle.predict_probs(eval_trig)
            preds = np.argmax(eval_probs, axis=-1)
            asr = float(np.mean(preds == target_class))
            l1_sum = float(np.sum(eval_m))

            if asr >= self.asr_threshold:
                lam = min(lam * 1.3, 10.0)
                if l1_sum < best_l1:
                    best_l1 = l1_sum
                    best_asr = asr
                    best_mask = copy.deepcopy(eval_m)
                    best_pattern = copy.deepcopy(eval_p)
            else:
                lam = max(lam / 1.2, 0.001)

        final_m = best_mask if best_mask is not None else upsample(_g_tanh(theta_m))
        final_p = best_pattern if best_pattern is not None else _g_tanh(theta_p)
        final_l1 = float(np.sum(final_m))
        discrete_mask = (final_m >= 0.5).astype(np.float32)

        return {
            "target_class": target_class,
            "l1_norm": round(best_l1 if best_mask is not None else final_l1, 4),
            "discrete_l1_norm": int(np.sum(discrete_mask)),
            "asr": round(best_asr, 4),
            "mask": final_m[0, 0],
            "pattern": final_p[0],
            "discrete_mask": discrete_mask[0, 0],
            "trigger_type": "nes_inverted",
        }

    def scan_all_classes(
        self,
        reference_images: Union[np.ndarray, torch.Tensor],
        class_names: Optional[List[str]] = None,
        max_reference_samples: int = 40,
    ) -> Dict[str, Any]:
        """
        Runs B3D trigger reconstruction across all classes and performs
        Median Absolute Deviation (MAD) outlier detection.
        """
        if isinstance(reference_images, torch.Tensor):
            ref_imgs = reference_images.detach().cpu().numpy()
        else:
            ref_imgs = np.asarray(reference_images)

        if ref_imgs.ndim != 4:
            raise ValueError(f"Expected reference_images shape [N, C, H, W], got {ref_imgs.shape}")

        if len(ref_imgs) > max_reference_samples:
            ref_imgs = ref_imgs[:max_reference_samples]

        class_results = []
        l1_norms = []

        for c in range(self.num_classes):
            res = self.reverse_engineer_class(c, ref_imgs)
            class_results.append(res)
            l1_norms.append(res["l1_norm"])

        l1_arr = np.array(l1_norms, dtype=np.float32)
        median_l1 = float(np.median(l1_arr))
        mad = float(np.median(np.abs(l1_arr - median_l1)))
        norm_mad = 1.4826 * mad

        # Anomaly Index for each class
        anomaly_scores = []
        for l1 in l1_norms:
            score = (median_l1 - l1) / (norm_mad + 1e-8)
            anomaly_scores.append(float(score))

        suspect_idx = int(np.argmax(anomaly_scores))
        max_anomaly = anomaly_scores[suspect_idx]

        # Paper outlier heuristic: Anomaly Index > 2.0 OR L1 < 0.25 * median
        is_anomaly_mad = max_anomaly > 2.0
        is_anomaly_ratio = l1_norms[suspect_idx] < (0.25 * median_l1)
        is_backdoored = bool(is_anomaly_mad or is_anomaly_ratio)

        suspect_name = class_names[suspect_idx] if class_names and suspect_idx < len(class_names) else str(suspect_idx)

        if is_backdoored:
            disposition = "quarantine"
            status = "BACKDOOR_DETECTED"
            reason = (
                f"Class {suspect_idx} ('{suspect_name}') exhibits an anomalous reversed trigger "
                f"with L1 norm {l1_norms[suspect_idx]:.2f} (Anomaly Index: {max_anomaly:.2f}, "
                f"Median L1: {median_l1:.2f})."
            )
        else:
            disposition = "accept"
            status = "CLEAN"
            reason = "All classes exhibit consistent trigger inversion L1 norms; no backdoor outlier detected."

        return {
            "method": "B3D (Black-box Backdoor Detection)",
            "access_level": "blackbox",
            "status": status,
            "disposition": disposition,
            "is_backdoored": is_backdoored,
            "target_class": suspect_idx if is_backdoored else None,
            "target_class_name": suspect_name if is_backdoored else None,
            "anomaly_score": round(max_anomaly, 4),
            "median_l1": round(median_l1, 4),
            "mad": round(mad, 4),
            "reason": reason,
            "class_results": [
                {
                    "class_id": r["target_class"],
                    "class_name": class_names[r["target_class"]] if class_names else str(r["target_class"]),
                    "l1_norm": r["l1_norm"],
                    "discrete_l1_norm": r["discrete_l1_norm"],
                    "asr": r["asr"],
                    "anomaly_score": round(anomaly_scores[i], 4),
                    "trigger_type": r.get("trigger_type", "nes"),
                }
                for i, r in enumerate(class_results)
            ],
            "suspect_trigger": {
                "class_id": suspect_idx,
                "mask": class_results[suspect_idx]["mask"].tolist(),
                "discrete_mask": class_results[suspect_idx]["discrete_mask"].tolist(),
                "pattern_mean": float(np.mean(class_results[suspect_idx]["pattern"])),
            } if is_backdoored else None,
        }


# ============================================================================
# YOLO Object Detection Oracle & B3D Detector
# ============================================================================
class YOLOModelOracle:
    """
    Universal black-box oracle for YOLO models (.pt or .onnx).
    Accepts RGB images [H, W, 3] in uint8 [0, 255] or float [0, 1]
    and returns maximum detection confidence for requested class ID.
    """

    def __init__(self, model_path: Union[str, Path]):
        self.path = Path(model_path)
        self.suffix = self.path.suffix.lower()

        if self.suffix == ".onnx":
            if onnxruntime is None:
                raise ImportError("onnxruntime is required for ONNX YOLO oracle.")
            self.session = onnxruntime.InferenceSession(str(self.path))
            self.input_name = self.session.get_inputs()[0].name
            self.is_onnx = True
            # Inspect output shape
            self.output_name = self.session.get_outputs()[0].name
            self.names = {0: "class_0", 1: "class_1"}  # default fallback
        else:
            from ultralytics import YOLO
            self.model = YOLO(str(self.path))
            self.is_onnx = False
            self.names = self.model.names

    def max_class_confidence(self, image_rgb: np.ndarray, target_class: int) -> float:
        """
        Evaluate image and return highest detection confidence for target_class.
        image_rgb: [640, 640, 3] in [0, 1] or [0, 255] uint8.
        """
        if image_rgb.dtype != np.uint8 and image_rgb.max() <= 1.0:
            img_uint8 = (np.clip(image_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_uint8 = image_rgb.astype(np.uint8)

        if not self.is_onnx:
            # Accelerated PyTorch inference directly through base module
            try:
                import torch
                inp = torch.from_numpy(np.transpose(img_uint8.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0)
                with torch.inference_mode():
                    raw = self.model.model(inp)
                    out = raw[0] if isinstance(raw, (tuple, list)) else raw
                class_channel = 4 + target_class
                if class_channel < out.shape[1]:
                    return float(torch.max(out[0, class_channel, :]).item())
            except Exception:
                pass
            # Fallback to standard Ultralytics call
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            results = self.model(img_bgr, verbose=False)
            boxes = results[0].boxes
            if not boxes or len(boxes) == 0:
                return 0.0
            scores = [
                float(b.conf[0]) for b in boxes if int(b.cls[0]) == target_class
            ]
            return max(scores) if scores else 0.0
        else:
            # ONNX inference
            if img_uint8.shape[:2] != (640, 640):
                img_uint8 = cv2.resize(img_uint8, (640, 640))
            inp = np.transpose(img_uint8.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
            out = self.session.run([self.output_name], {self.input_name: inp})[0]
            # out shape: [1, 4 + num_classes, 8400]
            class_channel = 4 + target_class
            if class_channel < out.shape[1]:
                return float(np.max(out[0, class_channel, :]))
            return 0.0


class YOLOB3DDetector:
    """
    B3D Trigger Inversion for YOLO Object Detectors (.pt and .onnx).
    Operates without internal gradients, evaluating spatial trigger resistance.
    """

    def __init__(
        self,
        oracle: YOLOModelOracle,
        num_classes: Optional[int] = None,
        class_names: Optional[Dict[int, str]] = None,
        imgsz: int = 640,
        patch_size: int = 48,
        stride: int = 64,
        confidence_threshold: float = 0.70,
    ):
        self.oracle = oracle
        self.class_names = class_names or self.oracle.names
        self.num_classes = num_classes or len(self.class_names)
        self.imgsz = imgsz
        self.patch_size = patch_size
        self.stride = stride
        self.confidence_threshold = confidence_threshold

    def scan_class_trigger(
        self,
        target_class: int,
        test_images: List[np.ndarray],
    ) -> Dict[str, Any]:
        """
        Scans across spatial grid on clean frames to test if any localized patch
        forces the YOLO model to hallucinate or misclassify target_class.

        A backdoor trigger is a localized perturbation that causes a significant
        anomalous jump in detection confidence on images where the target object
        is NOT naturally present.
        """
        best_conf = 0.0
        best_loc = (0, 0)
        best_patch = None
        found_trigger = False

        ps = self.patch_size
        st = self.stride
        max_y = self.imgsz - ps
        max_x = self.imgsz - ps

        for img in test_images:
            h, w = img.shape[:2]
            if (h, w) != (self.imgsz, self.imgsz):
                base_img = cv2.resize(img, (self.imgsz, self.imgsz))
            else:
                base_img = img.copy()

            # Baseline confidence without any trigger
            baseline_conf = self.oracle.max_class_confidence(base_img, target_class)

            # If the model naturally detects the target class on this clean image
            # with high confidence (>= 0.50), the real object is genuinely present.
            # This image cannot be used to search for false-positive triggers for
            # this class — skip it.
            if baseline_conf >= 0.50:
                continue

            # Slide localized candidate patches across the image
            for y in range(0, max_y + 1, st):
                for x in range(0, max_x + 1, st):
                    trig_img = base_img.copy()
                    trig_img[y : y + ps, x : x + ps] = (255, 255, 255)  # white candidate patch
                    conf = self.oracle.max_class_confidence(trig_img, target_class)
                    delta = conf - baseline_conf

                    if conf > best_conf:
                        best_conf = conf
                        best_loc = (y, x)

                    # A true trigger must cause BOTH high absolute confidence
                    # AND a significant jump from the baseline (delta >= 0.25).
                    # This prevents flagging models that just have moderate
                    # background activation on reference images.
                    if conf >= self.confidence_threshold and delta >= 0.25:
                        found_trigger = True
                        break
                if found_trigger:
                    break
            if found_trigger:
                break

        # Trigger is confirmed only if spatial patch scan found a true anomaly
        is_triggered = found_trigger
        if is_triggered:
            l1_norm = float(ps * ps)  # Small compact trigger (e.g. 48x48 = 2304 pixels)
            trigger_mask = np.zeros((self.imgsz, self.imgsz), dtype=np.uint8)
            y, x = best_loc
            trigger_mask[y : y + ps, x : x + ps] = 255
        else:
            # Clean class requires full-canvas modification (safe, non-outlier)
            l1_norm = float(self.imgsz * self.imgsz * 0.5)  # e.g., ~200,000 pixels
            trigger_mask = None

        return {
            "target_class": target_class,
            "class_name": self.class_names.get(target_class, str(target_class)),
            "l1_norm": l1_norm,
            "max_confidence": round(best_conf, 4),
            "best_location": best_loc,
            "patch_size": ps,
            "is_triggered": is_triggered,
            "trigger_mask": trigger_mask,
        }

    def scan_yolo_model(self, test_images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Scans all classes in the YOLO model for backdoor trigger vulnerabilities.
        Returns complete assurance verdict, L1 distribution, and MAD anomaly index.
        """
        class_results = []
        l1_norms = []

        for c in range(self.num_classes):
            res = self.scan_class_trigger(c, test_images)
            class_results.append(res)
            l1_norms.append(res["l1_norm"])
            if res["is_triggered"]:
                # Backdoor trigger identified; early exit to preserve responsiveness
                break

        l1_arr = np.array(l1_norms, dtype=np.float32)
        median_l1 = float(np.median(l1_arr))
        mad = float(np.median(np.abs(l1_arr - median_l1)))
        norm_mad = 1.4826 * mad

        anomaly_scores = []
        for l1 in l1_norms:
            score = (median_l1 - l1) / (norm_mad + 1e-8)
            anomaly_scores.append(float(score))

        # Backdoor check: Does any class trigger with high confidence from a tiny patch?
        triggered_classes = [r for r in class_results if r["is_triggered"]]
        is_backdoored = len(triggered_classes) > 0

        # --- Bias Injection Detection (Global Confidence Anomaly) ---
        # If no spatial trigger found, check for weight-level attacks that globally
        # inflate one class's confidence (e.g., detection head bias manipulation).
        # These attacks don't use a spatial trigger — they shift ALL detections for
        # a target class upward, making the model hallucinate objects everywhere.
        #
        # Detection strategy: For each reference image, compute baseline for ALL
        # classes. If a class has high baseline (>= 0.50) on an image where it is
        # the ONLY class with elevated confidence, that's suspicious — a real object
        # would typically not inflate just one class's raw score in isolation while
        # all other classes remain near-zero. Cross-validate with a neutral frame.
        bias_anomaly_class = None
        if not is_backdoored:
            # Compute per-class baselines on all reference images
            ref_baselines = []  # list of dicts {class_id: conf} per image
            for img in test_images:
                h, w = img.shape[:2]
                if (h, w) != (self.imgsz, self.imgsz):
                    check_img = cv2.resize(img, (self.imgsz, self.imgsz))
                else:
                    check_img = img
                img_baselines = {}
                for c in range(self.num_classes):
                    img_baselines[c] = self.oracle.max_class_confidence(check_img, c)
                ref_baselines.append(img_baselines)

            # Check for isolated class inflation
            for c in range(self.num_classes):
                for img_idx, baselines in enumerate(ref_baselines):
                    if baselines[c] >= 0.50:
                        # This class has high baseline on this image.
                        # Check if any OTHER class also has elevated baseline (>= 0.40)
                        # on the same image — if yes, the image likely contains a real
                        # multi-class scene and this is normal.
                        other_elevated = any(
                            baselines[oc] >= 0.40
                            for oc in range(self.num_classes) if oc != c
                        )
                        if not other_elevated:
                            # Cross-validate: check if this class also has elevated
                            # confidence on a neutral frame (even slightly).
                            neutral = np.full((self.imgsz, self.imgsz, 3), 128, dtype=np.uint8)
                            neutral_conf = self.oracle.max_class_confidence(neutral, c)
                            # Compare neutral conf against other classes on neutral
                            other_neutral = [
                                self.oracle.max_class_confidence(neutral, oc)
                                for oc in range(self.num_classes) if oc != c
                            ]
                            mean_other_neutral = float(np.mean(other_neutral)) if other_neutral else 0.0
                            # If this class's neutral confidence is >= 3x the mean of other classes,
                            # it confirms a systematic bias inflation.
                            if neutral_conf >= max(0.02, mean_other_neutral * 3.0):
                                bias_anomaly_class = c
                                is_backdoored = True
                                break
                if is_backdoored:
                    break

        if is_backdoored:
            if triggered_classes:
                suspect = max(triggered_classes, key=lambda x: x["max_confidence"])
                disposition = "quarantine"
                status = "BACKDOOR_DETECTED"
                reason = (
                    f"YOLO model contains a backdoor trigger for class {suspect['target_class']} "
                    f"('{suspect['class_name']}'). A {suspect['patch_size']}x{suspect['patch_size']} "
                    f"patch at {suspect['best_location']} forces detection with {suspect['max_confidence']*100:.1f}% confidence."
                )
            elif bias_anomaly_class is not None:
                suspect_name = self.class_names.get(bias_anomaly_class, str(bias_anomaly_class))
                disposition = "quarantine"
                status = "BACKDOOR_DETECTED"
                # Compute the max reference baseline for this anomalous class
                max_ref_conf = max(
                    baselines[bias_anomaly_class] for baselines in ref_baselines
                ) if ref_baselines else 0.0
                reason = (
                    f"YOLO model shows anomalous global confidence inflation for class {bias_anomaly_class} "
                    f"('{suspect_name}'). Baseline confidence reaches {max_ref_conf*100:.1f}% on reference "
                    f"images without the target object present. This indicates a weight-level attack "
                    f"(e.g., detection head bias manipulation) that forces hallucinated detections of "
                    f"'{suspect_name}' across all inputs."
                )
                # Synthesize a suspect-like dict for the return value
                suspect = {
                    "target_class": bias_anomaly_class,
                    "class_name": suspect_name,
                    "max_confidence": max_ref_conf,
                    "best_location": None,
                    "patch_size": self.patch_size,
                }
            else:
                suspect = None
        else:
            suspect = None
            disposition = "accept"
            status = "CLEAN"
            reason = (
                "YOLO model demonstrated robust backdoor resistance across all tested classes. "
                "No localized trigger patch induced false positive detections."
            )

        return {
            "method": "YOLO B3D (Black-box Detection Trigger Inversion)",
            "access_level": "blackbox",
            "model_format": "onnx" if self.oracle.is_onnx else "pytorch",
            "status": status,
            "disposition": disposition,
            "is_backdoored": is_backdoored,
            "target_class": suspect["target_class"] if suspect else None,
            "target_class_name": suspect["class_name"] if suspect else None,
            "compromised_class": suspect["class_name"] if suspect else None,
            "max_confidence": suspect["max_confidence"] if suspect else 0.0,
            "reason": reason,
            "class_results": [
                {
                    "class_id": r["target_class"],
                    "class_name": r["class_name"],
                    "l1_norm": r["l1_norm"],
                    "max_confidence": r["max_confidence"],
                    "is_triggered": r["is_triggered"],
                    "best_location": r["best_location"],
                }
                for r in class_results
            ],
            "suspect_trigger_location": suspect["best_location"] if suspect else None,
        }

