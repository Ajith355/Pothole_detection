import cv2
import numpy as np
import math
import sys

# ENABLED: MiDaS Neural Network
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch not found. Installing: pip install torch torchvision timm")

# Global model cache
midas_model = None
midas_transform = None
device = None

def undistort_image(img, K=None, dist=None):
    # ...existing code...
    if K is None:
        return img
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 1)
    return cv2.undistort(img, K, dist, None, newK)

def detect_wet_muddy(img):
    """Return labels and masks: {'wet':bool, 'muddy':bool, 'spec_mask':binary mask}"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2].astype(float)/255.0
    s = hsv[:,:,1].astype(float)/255.0
    h = hsv[:,:,0].astype(float)
    # specular detection: very bright and low saturation
    spec_mask = (v > 0.85) & (s < 0.25)
    # muddy detection: brownish hue and moderate saturation
    muddy_mask = ((h > 5) & (h < 35)) & (s > 0.15) & (v < 0.6)
    wet = spec_mask.mean() > 0.02 or v.mean() < 0.45
    muddy = muddy_mask.mean() > 0.02
    return {'wet': bool(wet), 'muddy': bool(muddy), 'spec_mask': (spec_mask.astype(np.uint8)*255)}

def simple_pothole_segmentation(img):
    """Fallback segmentation: edge + morphology. Prefer a trained model in production."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    mask = np.zeros((h,w), dtype=np.uint8)
    # choose largest plausible contour
    max_area = 0
    best = None
    for c in contours:
        a = cv2.contourArea(c)
        if a > max_area and a > 500:  # area threshold (tune)
            max_area = a
            best = c
    if best is not None:
        cv2.drawContours(mask, [best], -1, 255, -1)
    # smooth mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    return mask

def load_midas():
    """Loads the MiDaS model once."""
    global midas_model, midas_transform, device
    if midas_model is not None or not HAS_TORCH:
        return

    print("🧠 Loading MiDaS Neural Network (MAX ACCURACY MODE - DPT_Large)...")
    try:
        # ACCURACY UPGRADE: Using DPT_Large.
        # "By any means" implies trading speed for the best possible depth accuracy.
        model_type = "DPT_Large"

        # Load from torch hub
        midas_model = torch.hub.load("intel-isl/MiDaS", model_type)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas_model.to(device)
        midas_model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "MiDaS_small":
            midas_transform = midas_transforms.small_transform
        elif model_type == "DPT_Large":
             midas_transform = midas_transforms.dpt_transform
        else:
            midas_transform = midas_transforms.dpt_transform

        print(f"✅ MiDaS {model_type} loaded on {device}")
    except Exception as e:
        print(f"❌ Failed to load MiDaS: {e}")
        midas_model = None

def run_midas_depth(img):
    """
    Runs MiDaS Neural Network to get high-accuracy relative depth map.
    Returns: float32 depth map (Inverse Depth: High=Close, Low=Far).
    """
    global midas_model

    # Try loading model if not loaded
    if midas_model is None and HAS_TORCH:
        load_midas()

    if midas_model is not None:
        try:
            # Resize image to a small fixed size for faster inference
            # MiDaS small works well with 256x256 or 384x384
            # We preserve aspect ratio during transform usually,
            # but let's ensure input isn't massive.

            # Transform input
            input_batch = midas_transform(img).to(device)

            # Inference
            with torch.no_grad():
                prediction = midas_model(input_batch)

                # Resize to original resolution
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic", # Reverting to Bicubic for higher quality interpolation
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()

            # Normalize to 0-1 for consistent scaling with our heuristic
            d_min = depth_map.min()
            d_max = depth_map.max()
            depth_map_norm = (depth_map - d_min) / (d_max - d_min + 1e-8)

            # ENHANCEMENT: Apply local contrast enhancement (CLAHE) to depth map
            # This helps bring out the pothole structure from the road surface
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            depth_map_norm_u8 = (depth_map_norm * 255).astype(np.uint8)
            depth_enhanced = clahe.apply(depth_map_norm_u8).astype(np.float32) / 255.0

            # Blend enhanced with original for stability
            return (0.7 * depth_map_norm + 0.3 * depth_enhanced).astype(np.float32)

        except Exception as e:
            print(f"⚠️ MiDaS Inference Error: {e}, falling back to Sobel.")

    # Fallback to simple gradient-based pseudo-depth (very rough).
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    grady = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    depth_rel = cv2.magnitude(gradx, grady)
    depth_rel = cv2.GaussianBlur(depth_rel, (31,31), 0)
    depth_rel = (depth_rel - depth_rel.min()) / (depth_rel.max() - depth_rel.min() + 1e-8)
    # invert so pothole interiors (dark valleys) tend to be larger depth in relative map
    depth_rel = 1.0 - depth_rel
    return depth_rel.astype(np.float32)

def estimate_metric_scale(depth_rel, mask_rim_pixels, cam_params):
    """Estimate scale factor s to convert relative depth to metric using camera geometry.
    cam_params: dict with 'f', 'cx', 'cy', 'H', optional 'pitch' (radians)
    Returns scale s and estimated real-rim-distance in meters.
    """
    f = cam_params.get('f', None)
    cx = cam_params.get('cx', 0)
    cy = cam_params.get('cy', 0)
    H = cam_params.get('H', None)
    pitch = cam_params.get('pitch', 0.0)
    if f is None or H is None:
        return None, None
    # compute median image row v of rim points
    vs = mask_rim_pixels[:,1]
    v_med = np.median(vs)
    # correct for pitch: for small pitch approximate v_corr = v_med - cy + f * tan(pitch)
    v_corr = (v_med - cy) - f * math.tan(pitch)
    # avoid division by zero
    if abs(v_corr) < 1e-3:
        return None, None
    # approximate ground distance Z (meters) for rim
    Z_rim = (H * f) / (v_corr + 1e-8)
    # median relative depth at rim
    rel_vals = depth_rel[mask_rim_pixels[:,1].astype(int), mask_rim_pixels[:,0].astype(int)]
    r_rel = np.median(rel_vals) if rel_vals.size>0 else np.median(depth_rel)
    if r_rel <= 0:
        return None, None
    s = Z_rim / r_rel
    return s, Z_rim

def rim_pixels_from_mask(mask):
    # find contour, sample rim points
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0,2), dtype=int)
    # pick largest
    c = max(contours, key=cv2.contourArea)
    c = c.reshape(-1,2)
    # take a sample (every Nth point)
    if c.shape[0] > 200:
        idx = np.linspace(0, c.shape[0]-1, 200).astype(int)
        c = c[idx]
    return c

def refine_depth(depth_metric, mask, spec_mask):
    """Simple refinement: inpaint specular regions, smooth inside mask and enforce rim boundary = 0."""
    h, w = depth_metric.shape
    depth = depth_metric.copy()
    # inpaint specular areas in depth
    inpaint_mask = (spec_mask>0).astype(np.uint8)
    if inpaint_mask.sum()>0:
        depth = cv2.inpaint((depth/np.max(depth)*255).astype(np.uint8), inpaint_mask, 3, cv2.INPAINT_TELEA).astype(np.float32)
        depth = depth / 255.0 * depth_metric.max()
    # smooth inside mask
    kernel = np.ones((5,5), np.float32)/25
    depth_sm = cv2.filter2D(depth, -1, kernel)
    # enforce rim pixels to be near zero depth (road plane)
    rim = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((7,7),np.uint8))
    depth_sm[rim>0] = np.minimum(depth_sm[rim>0], np.percentile(depth_sm[mask>0], 20))
    return depth_sm

def estimate_pothole_depth(image_path, cam_params=None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    # undistort if available
    img_u = undistort_image(img, K=cam_params.get('K') if cam_params else None, dist=cam_params.get('dist') if cam_params else None)
    # detect scene condition
    cond = detect_wet_muddy(img_u)
    # segment pothole (replace with model in practice)
    mask = simple_pothole_segmentation(img_u)
    if mask.sum() == 0:
        return {'error': 'no_pothole_detected'}
    # get rim pixels
    rim_pixels = rim_pixels_from_mask(mask)
    # get relative depth map
    depth_rel = run_midas_depth(img_u)
    # estimate metric scale
    s, Z_rim = estimate_metric_scale(depth_rel, rim_pixels, cam_params or {})
    metric_available = s is not None
    if not metric_available:
        # warn: only relative depth
        s = 1.0
    depth_metric = depth_rel * s
    # refine depth (specular-aware)
    depth_refined = refine_depth(depth_metric, mask, cond['spec_mask'])
    # compute pothole depth relative to rim (assume rim->road plane)
    # compute median rim height (near zero) and interior minimum
    rim_vals = depth_refined[rim_pixels[:,1].astype(int), rim_pixels[:,0].astype(int)] if rim_pixels.size>0 else depth_refined[mask>0]
    rim_median = np.median(rim_vals) if rim_vals.size>0 else np.median(depth_refined[mask>0])
    interior_vals = depth_refined[mask>0]
    bottom = np.min(interior_vals) if interior_vals.size>0 else rim_median
    pothole_depth = max(0.0, rim_median - bottom)
    # estimate confidence: variance between geometric Z_rim and median scaled depth at rim
    confidence = 1.0
    if metric_available:
        scaled_rim = np.median(depth_rel[rim_pixels[:,1].astype(int), rim_pixels[:,0].astype(int)]) * s
        rel_err = abs(Z_rim - scaled_rim) / (Z_rim + 1e-6)
        confidence = max(0.0, 1.0 - rel_err)
    else:
        confidence = 0.5  # low for unscaled monocular
    return {
        'pothole_depth_m': float(pothole_depth) if metric_available else None,
        'pothole_depth_relative': float(rim_median - bottom),
        'metric_available': bool(metric_available),
        'scale_s': float(s) if metric_available else None,
        'Z_rim_estimated_m': float(Z_rim) if metric_available else None,
        'scene_condition': cond,
        'confidence': float(confidence),
        'mask': mask,
        'depth_map': depth_refined
    }

if __name__ == '__main__':
    # example usage:
    # python depth_estimation.py image.jpg
    if len(sys.argv) < 2:
        print("Usage: python depth_estimation.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    # Example camera params: fill with real calibration for metric results
    cam_params = {
        'f': 1200.0,   # focal length in pixels (example)
        'cx': 640.0,
        'cy': 360.0,
        'H': 1.25,     # camera height in meters
        'pitch': 0.0
    }
    out = estimate_pothole_depth(image_path, cam_params)
    print("Result:", {k: out[k] for k in ['pothole_depth_m','pothole_depth_relative','metric_available','confidence','scene_condition']})
    # Save diagnostic images
    cv2.imwrite('pothole_mask.png', out['mask'])
    depth_vis = (out['depth_map'] - out['depth_map'].min()) / (out['depth_map'].ptp()+1e-8) * 255
    cv2.imwrite('pothole_depth.png', depth_vis.astype(np.uint8))

'''
# Pothole Depth Estimation — Algorithm and Cautions

Summary (inputs)
- Images: single image preferred, stereo or short forward video helpful.
- Camera intrinsics: focal length f (pixels), principal point (cx, cy).
- Camera height H above road and pitch θ (optional but highly recommended).
- Optional: reference object of known size (wheel or a placed marker).

Core pipeline
1. Preprocess
   - Undistort, denoise, convert to RGB/HSV/Lab.
2. Appearance classification (dry / mud / wet)
   - Compute HSV mean and specular mask S:
     - wet if (low V, high highlights ratio) OR strong mirror-like reflections.
     - muddy if (brownish hue, high texture).
3. Pothole segmentation
   - Prefer semantic segmentation model (e.g., U-Net/DeepLab trained on potholes).
   - Fallback: Canny -> morphological closing -> contour filter by area/shape -> convexity check.
4. Road-plane & rim extraction
   - Extract rim contour C_rim (largest contour boundary).
   - Assume rim lies on road plane. Use rim pixels to estimate the road-plane anchor.
5. Depth estimation methods (combine)
   - Learned monocular depth (MiDaS/DPT): produces relative depth D_rel(x,y).
   - Geometry anchoring:
     - If camera intrinsics & height H known, for an image row v (pixels), approximate ground distance along optical axis:
       D_ground ≈ H * f / (v_corr)
       where v_corr = (v - cy) corrected for pitch θ; derive exact mapping from extrinsics.
     - Compute median D_ground over rim pixels → D_rim_real.
   - Scale factor s = median(D_rim_real) / median(D_rel at rim).
   - Metric depth map: D_metric = s * D_rel.
6. Pothole depth numeric
   - Fit road-plane height z_plane using rim (assume z_plane = 0).
   - Pothole depth at pixel p = z_plane - z(p) where z(p) is the reconstructed metric height (ensure non-negative).
   - Report depth = max_p (pothole depth) and mean depth in the deepest 5% area.
7. Refinement
   - Inpaint specular regions in imagery and depth map, re-run local SfS refinement:
     - Solve Poisson equation ∇^2 z = divergence of gradient field from shading cues constrained by boundary (rim).
     - Regularize by curvature and smoothness, keep depth >= 0.
   - Combine learned and geometric depth via weighted average where weight ∝ confidence (texture, specular fraction, presence of ref object).
8. Uncertainty & flags
   - If no intrinsics or reference: only relative depth — report "relative depth" and do not claim metric value.
   - If specular fraction > 15% or muddy texture detected, increase uncertainty and warn user.

Key formulas (simplified)
- Pinhole projection: u = f X / Z + cx, v = f Y / Z + cy.
- For a forward-facing camera at height H and small pitch, ground point at image row v corresponds approximately to distance along Z:
  Z ≈ (H * f) / (v - cy)   (approx; correct for camera pitch θ)
- Scale metricization:
  s = D_rim_real / median(D_rel[rim_pixels])
  D_metric = s * D_rel

Practical cautions
- Metric scale is ambiguous for monocular methods unless you provide H/pitch or a reference object.
- Wet surfaces produce specular highlights and reflections that mimic depth variation — detect and inpaint.
- Mud can fill potholes partially: visual bottom may be occluded — use multiple images / video to see rim vs bottom.
- Use stereo or structure-from-motion (SfM) across frames if accuracy required.
- Validate with a physical measurement for calibration and reporting.

Outputs
- depth_m (single value with 95% CI), depth_map (metric where possible), scene_condition (dry/muddy/wet), confidence score, diagnostic images (segmentation, rim, specular mask).

References & implementation notes
- Use MiDaS/DPT for strong monocular priors.
- Use RANSAC to reject outlier rim points when estimating plane.
- For final accuracy targets (<2 cm) use stereo or multi-view photogrammetry and known calibration.
'''
