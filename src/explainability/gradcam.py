"""Grad-CAM (Selvaraju et al., 2017) for the EfficientNet classifier.

Returns the *raw* class-activation map (float32 in [0, 1]) so every downstream
explainability module works on real attention values, not on the colours of a
rendered overlay.
"""

import cv2
import numpy as np
import torch


class GradCAM:

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._activations = None
        self._gradients = None
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        self._activations = output
        if output.requires_grad:
            # tensor hook: gradient of the score w.r.t. this layer's output
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self._gradients = grad

    def generate(self, image_tensor: torch.Tensor, class_idx: int = None):
        """Run one forward + backward pass.

        Returns (probabilities [num_classes], class_idx, cam HxW float32 in [0,1]).
        """
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            image_tensor = image_tensor.requires_grad_(False)
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

            if class_idx is None:
                class_idx = int(np.argmax(probs))

            logits[0, class_idx].backward()

        activations = self._activations.detach()[0]          # C x h x w
        gradients = self._gradients.detach()[0]              # C x h x w
        weights = gradients.mean(dim=(1, 2))                 # C

        cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))
        cam = cam.cpu().numpy().astype(np.float32)

        peak = float(cam.max())
        if peak > 1e-8:
            cam /= peak
        else:  # no positive evidence for this class -> empty map instead of NaNs
            cam = np.zeros_like(cam)

        self._activations = self._gradients = None
        self.model.zero_grad(set_to_none=True)

        return probs, class_idx, cam


def upsample_cam(cam: np.ndarray, size: int, fov_mask: np.ndarray = None) -> np.ndarray:
    """Resize a low-resolution CAM to the display size and zero it outside the retina."""
    big = cv2.resize(cam, (size, size), interpolation=cv2.INTER_CUBIC)
    big = np.clip(big, 0, None)
    if fov_mask is not None:
        big = big * fov_mask
    peak = big.max()
    return (big / peak).astype(np.float32) if peak > 1e-8 else np.zeros_like(big, dtype=np.float32)


def render_heatmap_overlay(image_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Colour the CAM with the JET palette (red = strongest) and blend it onto the image."""
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, heat, alpha, 0)
