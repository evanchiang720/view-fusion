"""
causal_patching.py
------------------
Causal Mediation Analysis and Activation Patching Framework for ViewFusion.
Evaluated with Thresholded Symmetric Epipolar Distance (TSED) 3D Consistency Metric.

Features:
  1. Safe, robust multi-object loader supporting index-based selection.
  2. Window-based activation patching (e.g., window_size=100 restricts patching to [t-100, t]).
  3. Geometric TSED evaluation based on multi-view camera epipolar constraints (GT verified).
  4. Simultaneous export of:
     - Quantitative Causal Attribution Heatmap (real_causal_heatmap_{tag}.png)
     - Qualitative Visual Comparison Grid (causal_visual_grid_{tag}.png)

Authors: ECE 285 Course Project Team
"""

import os
import tarfile
import io
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import yaml

from model.unet import UNet
from model.view_fusion import ViewFusion


# ============================================================================
# Part 1: Forward Hook Manager (Window-based Activation Patching)
# ============================================================================

class HookManager:
    def __init__(self):
        # Format: {mod_name: {timestep: tensor}}
        self.cached_activations: Dict[str, Dict[int, torch.Tensor]] = {}
        self.mode: str = "IDLE"
        self.patch_target_name: Optional[str] = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.timestep_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.current_timestep: Optional[int] = None
        self.target_timestep: Optional[int] = None
        self.window_size: Optional[int] = None
        self.step_counter: int = 1000

    def register_timestep_tracker(self, denoise_module: nn.Module, total_steps: int = 1000):
        self.remove_timestep_tracker()
        self.step_counter = total_steps

        def pre_hook(module, args, kwargs):
            self.step_counter -= 1
            t_val = kwargs.get("time", None)
            if t_val is None and len(args) > 2:
                t_val = args[2]

            if t_val is not None and torch.is_tensor(t_val) and t_val.numel() == 1:
                val = t_val.item()
                self.current_timestep = int(val) if val > 1.0 else self.step_counter
            else:
                self.current_timestep = self.step_counter

        self.timestep_handle = denoise_module.register_forward_pre_hook(
            pre_hook, with_kwargs=True
        )

    def remove_timestep_tracker(self):
        if self.timestep_handle is not None:
            self.timestep_handle.remove()
            self.timestep_handle = None

    def register_record_hooks(self, candidate_modules: Dict[str, nn.Module], target_timestep: int, window_size: int = 100):
        self.remove_hooks()
        self.mode = "RECORD_ALL"
        self.target_timestep = target_timestep
        self.window_size = window_size
        lower_bound = max(0, target_timestep - window_size)

        for name, module in candidate_modules.items():
            if name not in self.cached_activations:
                self.cached_activations[name] = {}

            def make_hook(mod_name):
                def hook_fn(sub_module, input_tensor, output_tensor):
                    if self.current_timestep is not None and (lower_bound <= self.current_timestep <= self.target_timestep):
                        out = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
                        self.cached_activations[mod_name][self.current_timestep] = out.detach().clone()
                    return output_tensor
                return hook_fn

            handle = module.register_forward_hook(make_hook(name))
            self.hook_handles.append(handle)

    def register_patch_hook(self, target_module: nn.Module, target_name: str, target_timestep: int, window_size: int = 100):
        self.remove_hooks()
        self.mode = "PATCH"
        self.patch_target_name = target_name
        self.target_timestep = target_timestep
        self.window_size = window_size
        lower_bound = max(0, target_timestep - window_size)

        if target_name not in self.cached_activations:
            raise KeyError(f"No clean activation cached for target: {target_name}")

        def patch_fn(sub_module, input_tensor, output_tensor):
            # Only patch within the interval [target_timestep - window_size, target_timestep]
            if self.current_timestep is not None and (lower_bound <= self.current_timestep <= self.target_timestep):
                clean_step_dict = self.cached_activations[target_name]
                if self.current_timestep in clean_step_dict:
                    clean_tensor = clean_step_dict[self.current_timestep]
                    if isinstance(output_tensor, tuple):
                        return (clean_tensor.to(output_tensor[0].device, output_tensor[0].dtype),) + output_tensor[1:]
                    return clean_tensor.to(output_tensor.device, output_tensor.dtype)
            return output_tensor

        handle = target_module.register_forward_hook(patch_fn)
        self.hook_handles.append(handle)

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def remove_all(self):
        self.remove_hooks()
        self.remove_timestep_tracker()
        self.cached_activations.clear()


# ============================================================================
# Part 2: Module Discovery
# ============================================================================

def find_candidate_components(model: nn.Module) -> Dict[str, nn.Module]:
    candidates = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ in ("ResnetBlock", "SelfAttention"):
            candidates[name] = module
    return candidates


# ============================================================================
# Part 3: Robust Multi-Object Dataset Loader
# ============================================================================

def find_shard_path(shard_name: str = "NMR-test-00.tar") -> str:
    candidates = [
        os.path.join("data/nmr/NMR_sharded_100_4", shard_name),
        os.path.join("/home/chc302/public/users/chc302/NMR_sharded_100_4", shard_name),
        shard_name
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def load_real_nmr_sample(
    device: torch.device,
    object_index: int = 0, 
    shard_name: str = "NMR-test-00.tar",
    target_offset_deg: float = 15.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Extracts multi-view images for a specific 3D CAD object using object_index.
    Clean target is set relative to context reference with a baseline offset angle.
    """
    shard_path = find_shard_path(shard_name=shard_name)
    print(f"[*] Loading object index [{object_index}] from shard: {shard_path}")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    extracted_images = []
    skip_images = object_index * 24
    image_counter = 0

    with tarfile.open(shard_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".png") or member.name.endswith(".jpg"):
                image_counter += 1
                if image_counter <= skip_images:
                    continue

                f = tar.extractfile(member)
                if f is not None:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                    extracted_images.append(transform(img))
                    if len(extracted_images) >= 3:
                        break

    if len(extracted_images) < 3:
        raise RuntimeError(f"Could not extract sufficient views for object index {object_index}.")

    # Context views: (1, 3, 3, 64, 64)
    y_cond = torch.stack(extracted_images[:3], dim=0).unsqueeze(0).to(device)
    view_count = torch.tensor([3], device=device)

    context_angle_rad = 0.0
    clean_angle_rad = context_angle_rad + np.radians(target_offset_deg)
    clean_angle = torch.tensor([[clean_angle_rad]], dtype=torch.float32, device=device)

    return y_cond, view_count, clean_angle, context_angle_rad


# ============================================================================
# Part 4: TSED 3D Epipolar Consistency Metric (GT Verified)
# ============================================================================

def get_nmr_camera_matrices(
    azimuth_rad: float,
    elevation_deg: float = 30.0,
    dist: float = 2.732,
    fov_deg: float = 30.0,
    img_size: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard OpenCV projection matrices for NMR ShapeNet rendering setup:
      1. Negate theta (-azimuth_rad) to match NMR index rotation convention.
      2. Camera coordinate system: X-right, Y-down, Z-forward.
      3. FOV fixed at 30 degrees (verified with ground-truth camera extrinsics).
    """
    phi = np.radians(elevation_deg)
    theta = -float(azimuth_rad)

    cam_pos = np.array([
        dist * np.cos(phi) * np.sin(theta),
        dist * np.sin(phi),
        dist * np.cos(phi) * np.cos(theta)
    ], dtype=np.float32)

    forward = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-8)
    down = np.cross(forward, right)
    down = down / (np.linalg.norm(down) + 1e-8)

    rot = np.vstack([right, down, forward])
    trans = -rot @ cam_pos

    focal = (img_size / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    intrinsics = np.array([
        [focal, 0.0, img_size / 2.0],
        [0.0, focal, img_size / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return intrinsics, rot, trans


def compute_fundamental_matrix(
    k1: np.ndarray, r1: np.ndarray, t1: np.ndarray,
    k2: np.ndarray, r2: np.ndarray, t2: np.ndarray
) -> np.ndarray:
    r_rel = r2 @ r1.T
    t_rel = t2 - r_rel @ t1

    tx = np.array([
        [0.0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0.0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0.0]
    ], dtype=np.float32)

    essential = tx @ r_rel
    fundamental = np.linalg.inv(k2).T @ essential @ np.linalg.inv(k1)
    return fundamental


def evaluate_3d_consistency_metric(
    synthesized_view: torch.Tensor,
    context_ref_view: torch.Tensor,
    target_angle_rad: float,
    context_angle_rad: float = 0.0,
    threshold: float = 4.0,
    eval_img_size: int = 256,
    ratio_test: float = 0.85
) -> Tuple[float, int, float]:
    """
    Evaluates Thresholded Symmetric Epipolar Distance (TSED) consistency metric.
    Includes fallback matching heuristics optimized for low-texture CAD objects.
    """
    def to_cv2_image(t: torch.Tensor) -> np.ndarray:
        arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        arr = np.clip(arr * 0.5 + 0.5, 0.0, 1.0) * 255.0
        uint_img = arr.astype(np.uint8)
        return cv2.resize(uint_img, (eval_img_size, eval_img_size), interpolation=cv2.INTER_CUBIC)

    img_synth = to_cv2_image(synthesized_view)
    img_context = to_cv2_image(context_ref_view)

    # High sensitivity SIFT configuration for smooth geometries
    sift = cv2.SIFT_create(
        nfeatures=1000,
        contrastThreshold=0.005,
        edgeThreshold=30,
        sigma=1.2
    )
    
    kp1, des1 = sift.detectAndCompute(img_synth, None)
    kp2, des2 = sift.detectAndCompute(img_context, None)

    # Fallback to ORB detector when keypoints are sparse
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        orb = cv2.ORB_create(nfeatures=500, fastThreshold=5)
        kp1, des1 = orb.detectAndCompute(img_synth, None)
        kp2, des2 = orb.detectAndCompute(img_context, None)
        is_binary = True
    else:
        is_binary = False

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0.0, 0, 999.0

    norm_type = cv2.NORM_HAMMING if is_binary else cv2.NORM_L2
    bf = cv2.BFMatcher(norm_type)
    
    matches_raw = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for pair in matches_raw:
        if len(pair) == 2 and pair[0].distance < ratio_test * pair[1].distance:
            good_matches.append(pair[0])

    # Cross-check fallback for small match count
    if len(good_matches) < 4:
        bf_cross = cv2.BFMatcher(norm_type, crossCheck=True)
        good_matches = bf_cross.match(des1, des2)
        if len(good_matches) < 4:
            return 0.0, len(good_matches), 999.0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Calculate fundamental matrix and symmetric epipolar distances
    k1, r1, t1 = get_nmr_camera_matrices(target_angle_rad, img_size=eval_img_size)
    k2, r2, t2 = get_nmr_camera_matrices(context_angle_rad, img_size=eval_img_size)
    f_matrix = compute_fundamental_matrix(k1, r1, t1, k2, r2, t2)

    p1_h = np.hstack([pts1, np.ones((len(pts1), 1), dtype=np.float32)])
    p2_h = np.hstack([pts2, np.ones((len(pts2), 1), dtype=np.float32)])

    l2 = (f_matrix @ p1_h.T).T
    l1 = (f_matrix.T @ p2_h.T).T

    num = np.abs(np.sum(p2_h * l2, axis=1))
    d2 = num / (np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2) + 1e-8)
    d1 = num / (np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2) + 1e-8)
    sed = (d1 + d2) / 2.0

    valid_ratio = float(np.mean(sed < threshold))
    median_sed = float(np.median(sed))
    return valid_ratio, len(good_matches), median_sed


def corrupt_target_pose(clean_angle: torch.Tensor, delta_rad: float = np.pi / 2) -> torch.Tensor:
    """Applies a deterministic +90 degree perturbation to induce geometric violation."""
    return (clean_angle + delta_rad) % (2 * np.pi)


def compute_causal_effects(
    m_clean: float,
    m_corrupted: float,
    m_patched: float,
    te_threshold: float = 1e-4
) -> Dict[str, float]:
    total_effect = m_clean - m_corrupted
    eps = 1e-6

    if abs(total_effect) < te_threshold:
        return {"Total_Effect": total_effect, "Indirect_Effect": 0.0, "unstable": True}

    indirect_effect = (m_patched - m_corrupted) / (total_effect + eps)
    return {"Total_Effect": total_effect, "Indirect_Effect": indirect_effect, "unstable": False}


# ============================================================================
# Part 5: Visualizations (Heatmap & Generated Image Grid)
# ============================================================================

def plot_causal_heatmap(
    heatmap_matrix: np.ndarray,
    component_names: List[str],
    timesteps: List[int],
    title: str = "ViewFusion Causal Attribution Heatmap (TSED Consistency)",
    save_path: str = "results/real_causal_heatmap_sofa_0.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig_height = max(8, len(component_names) * 0.35)
    plt.figure(figsize=(12, fig_height))

    clean_labels = [
        ".".join(name.split(".")[-3:]) if len(name.split(".")) >= 3 else name
        for name in component_names
    ]

    vmax = max(0.05, float(np.max(heatmap_matrix)))
    im = plt.imshow(heatmap_matrix, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=vmax)
    cbar = plt.colorbar(im)
    cbar.set_label("Indirect Effect on TSED Consistency ($IE_c$)", rotation=270, labelpad=18, fontsize=11)

    plt.xticks(ticks=range(len(timesteps)), labels=[str(t) for t in timesteps], fontsize=10)
    plt.yticks(ticks=range(len(clean_labels)), labels=clean_labels, fontsize=8)

    plt.xlabel("Diffusion Timestep $t$ (Denoising trajectory: High Noise -> Low Noise)", fontsize=11, labelpad=10)
    plt.ylabel("U-Net Candidate Component ($c \\in \\mathcal{C}$)", fontsize=11, labelpad=10)
    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Causal heatmap exported to: {save_path}")


def tensor_to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    """Converts (-1 to 1) PyTorch image tensor to (0 to 1) NumPy RGB image."""
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip(arr * 0.5 + 0.5, 0.0, 1.0)


def plot_causal_image_grid(
    saved_images: Dict[Tuple[str, int], torch.Tensor],
    clean_img: torch.Tensor,
    corrupted_img: torch.Tensor,
    selected_components: List[str],
    timesteps: List[int],
    clean_deg: float,
    corrupted_deg: float,
    save_path: str = "results/causal_visual_grid_{tag}.png"
):
    """
    Renders a qualitative comparison table:
      - Row 0: Clean Baseline vs Corrupted Baseline
      - Subsequent Rows: Synthesized images patched at each component across timesteps.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_rows = len(selected_components) + 1
    n_cols = len(timesteps)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2), squeeze=False)

    # Row 0: Baselines
    for col_idx, t_step in enumerate(timesteps):
        ax = axes[0, col_idx]
        if col_idx == 0:
            ax.imshow(tensor_to_numpy_img(clean_img))
            ax.set_title(f"Clean Target\n({clean_deg:.1f}°)", fontsize=9, fontweight="bold", color="darkgreen")
        elif col_idx == 1:
            ax.imshow(tensor_to_numpy_img(corrupted_img))
            ax.set_title(f"Corrupted\n({corrupted_deg:.1f}°)", fontsize=9, fontweight="bold", color="darkred")
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    # Subsequent Rows: Patched generations
    for row_idx, c_name in enumerate(selected_components, start=1):
        short_name = ".".join(c_name.split(".")[-3:])
        for col_idx, t_step in enumerate(timesteps):
            ax = axes[row_idx, col_idx]
            img_tensor = saved_images.get((c_name, t_step), None)
            if img_tensor is not None:
                ax.imshow(tensor_to_numpy_img(img_tensor))
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 1:
                ax.set_title(f"t = {t_step}", fontsize=10, pad=5)
            if col_idx == 0:
                ax.set_ylabel(short_name, fontsize=10, labelpad=8, fontweight="bold")

    plt.suptitle("Qualitative Causal Intervention Grid: Synthesized Novel Views", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Visual qualitative comparison table exported to: {save_path}")


# ============================================================================
# Part 6: Full Sweep Execution Pipeline
# ============================================================================

def load_viewfusion_model(device: torch.device) -> ViewFusion:
    config_path = "logs/pretrained/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    denoise_fn = UNet(**config["model"]["denoise_net_params"])
    model = ViewFusion(
        denoise_fn,
        config["model"]["view_fusion_params"]["beta_schedule"],
        config["model"]["view_fusion_params"].get("weighting_train", True),
        config["model"]["view_fusion_params"].get("weighting_inference", True),
    ).to(device)

    ckpt_path = "logs/pretrained/best_model_all.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
    clean_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict, strict=False)
    model.set_new_noise_schedule(device=device, phase="test")
    model.eval()
    return model


def run_full_causal_sweep(
    object_index: int = 0,
    shard_name: str = "NMR-test-00.tar",
    tag: str = "sofa_tsed",
    window_size: int = 100,
    target_offset_deg: float = 15.0,
    tsed_threshold: float = 4.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Robust Causal Sweep (TSED Metric) on: {device}")

    model = load_viewfusion_model(device)
    components = find_candidate_components(model)
    comp_names = list(components.keys())
    print(f"[✓] Registered {len(comp_names)} candidate components.")

    timesteps = [900, 700, 500, 300, 100]
    heatmap_matrix = np.zeros((len(comp_names), len(timesteps)))
    saved_visuals: Dict[Tuple[str, int], torch.Tensor] = {}

    hook_manager = HookManager()
    total_diffusion_steps = 1000

    # 1. Load multi-view dataset sample
    y_cond, view_count, clean_angle, context_angle_rad = load_real_nmr_sample(
        device, object_index=object_index, shard_name=shard_name, target_offset_deg=target_offset_deg
    )
    corrupted_angle = corrupt_target_pose(clean_angle, delta_rad=np.pi / 2)

    clean_angle_rad = clean_angle.item()
    context_ref = y_cond[0, 0].detach().clone()

    clean_deg = np.degrees(clean_angle_rad)
    corrupted_deg = np.degrees(corrupted_angle.item())
    context_deg = np.degrees(context_angle_rad)
    print(f"[*] Setup: Clean Target = {clean_deg:.1f}° | Corrupted Target = {corrupted_deg:.1f}° | Context Ref = {context_deg:.1f}°")
    print(f"[*] Patching Window Size = {window_size} steps")

    # Step 1: Clean Baseline Reference
    print("\n[1/3] Generating Clean Baseline Reference...")
    hook_manager.remove_hooks()
    torch.manual_seed(42)
    with torch.no_grad():
        clean_out = model(y_cond=y_cond, view_count=view_count, angle=clean_angle, generate=True)
    m_clean, n_clean, med_clean = evaluate_3d_consistency_metric(
        clean_out[0], context_ref, target_angle_rad=clean_angle_rad, context_angle_rad=context_angle_rad, threshold=tsed_threshold
    )

    # Step 2: Corrupted Baseline Reference
    print("[2/3] Generating Corrupted Baseline Reference...")
    torch.manual_seed(42)
    with torch.no_grad():
        corrupted_out = model(y_cond=y_cond, view_count=view_count, angle=corrupted_angle, generate=True)
    m_corrupted, n_corr, med_corr = evaluate_3d_consistency_metric(
        corrupted_out[0], context_ref, target_angle_rad=clean_angle_rad, context_angle_rad=context_angle_rad, threshold=tsed_threshold
    )

    total_effect = m_clean - m_corrupted
    print(f"[*] TSED Clean     : {m_clean:.4f} (Matches: {n_clean}, Median SED: {med_clean:.2f} px)")
    print(f"[*] TSED Corrupted : {m_corrupted:.4f} (Matches: {n_corr}, Median SED: {med_corr:.2f} px)")
    print(f"[*] Total Effect   : {total_effect:.4f}")

    if abs(total_effect) < 1e-3:
        print("\n" + "!" * 75)
        print("[WARNING] Total Effect is near zero! Activation patching will result in all 0s.")
        print("          Please verify that the generated view contains sufficient keypoints.")
        print("!" * 75 + "\n")

    # Step 3: Run Activation Patching Grid (Window-based)
    print("\n[3/3] Executing Activation Patching Grid...")
    total_runs = len(comp_names) * len(timesteps)
    completed = 0

    for j, t_step in enumerate(timesteps):
        t_end = max(0, t_step - window_size)
        print(f"\n---> [Timestep Window: t={t_step} ~ {t_end}] Caching clean activations across all components...")

        hook_manager.cached_activations.clear()
        hook_manager.register_timestep_tracker(model.denoise_fn, total_steps=total_diffusion_steps)
        hook_manager.register_record_hooks(components, target_timestep=t_step, window_size=window_size)
        torch.manual_seed(42)
        with torch.no_grad():
            _ = model(y_cond=y_cond, view_count=view_count, angle=clean_angle, generate=True)

        for i, c_name in enumerate(comp_names):
            c_module = components[c_name]

            hook_manager.register_timestep_tracker(model.denoise_fn, total_steps=total_diffusion_steps)
            hook_manager.register_patch_hook(c_module, target_name=c_name, target_timestep=t_step, window_size=window_size)

            torch.manual_seed(42)
            with torch.no_grad():
                patched_out = model(y_cond=y_cond, view_count=view_count, angle=corrupted_angle, generate=True)

            m_patched, _, _ = evaluate_3d_consistency_metric(
                patched_out[0], context_ref, target_angle_rad=clean_angle_rad, context_angle_rad=context_angle_rad, threshold=tsed_threshold
            )
            effects = compute_causal_effects(m_clean, m_corrupted, m_patched)
            ie_score = float(np.clip(effects["Indirect_Effect"], 0.0, 1.0))
            heatmap_matrix[i, j] = ie_score

            # Save generated images for visual qualitative comparison table
            saved_visuals[(c_name, t_step)] = patched_out[0].detach().cpu()

            completed += 1
            short_name = ".".join(c_name.split(".")[-3:])
            status = " [UNSTABLE]" if effects["unstable"] else ""
            print(f"  [{completed:03d}/{total_runs}] Comp: {short_name:<28} | t={t_step:<4}~{t_end:<4} | IE_c={ie_score:.4f}{status}")

    hook_manager.remove_all()

    # Step 4: Export Heatmap
    heatmap_path = f"results/real_causal_heatmap_{tag}.png"
    plot_causal_heatmap(
        heatmap_matrix,
        comp_names,
        timesteps,
        title=f"ViewFusion Causal Attribution Heatmap (TSED @ {tsed_threshold:.1f}px, Win={window_size})",
        save_path=heatmap_path
    )

    # Step 5: Dynamically select Top-3 components based on mean causal effect
    mean_effects = np.mean(heatmap_matrix, axis=1)
    ranked_indices = np.argsort(mean_effects)[::-1]

    print("\n" + "=" * 50)
    print(" [Debug] All Components Ranked by Mean IE_c:")
    for rank, idx in enumerate(ranked_indices, start=1):
        print(f"   Rank {rank:02d}: {comp_names[idx]:<30} | Mean={mean_effects[idx]:.4f} | Peak={np.max(heatmap_matrix[idx]):.4f}")
    print("=" * 50 + "\n")

    top_k = min(3, len(comp_names))
    visual_components = [comp_names[idx] for idx in ranked_indices[:top_k]]

    # Append Negative Control
    control_name = "downs.1.res_block"
    if control_name in comp_names:
        bottom_idx = comp_names.index(control_name)
    else:
        bottom_idx = ranked_indices[-1]
    if comp_names[bottom_idx] not in visual_components:
        visual_components.append(comp_names[bottom_idx])

    grid_path = f"results/causal_visual_grid_{tag}.png"
    plot_causal_image_grid(
        saved_images=saved_visuals,
        clean_img=clean_out[0].detach().cpu(),
        corrupted_img=corrupted_out[0].detach().cpu(),
        selected_components=visual_components,
        timesteps=timesteps,
        clean_deg=clean_deg,
        corrupted_deg=corrupted_deg,
        save_path=grid_path
    )

    print("\n" + "=" * 60)
    print("Full Causal Sweep & Visual Table Successfully Exported!")
    print(f"1. Quantitative Heatmap: {heatmap_path}")
    print(f"2. Qualitative Image Grid: {grid_path}")
    print("=" * 60)


if __name__ == "__main__":
    """
    NMR-test-00.tar = {0~633: sofa, 634~1442: airplane, 1443~1905: lamp, 1906~2115: telephone, 2116~2186: vessel/boat}
    NMR-test-01.tar = {0~315: vessel/boat, 316~638: speaker, 639~1993: chair, 1994~2186: cabinet}
    NMR-test-02.tar = {0~120: cabinet, 121~1821: table, 1822~2040: display, 2041~2186: car}
    NMR-test-03.tar = {0~1352: car, 1353~1715: bench, 1716~2186: rifle}
    NMR-test-04.tar = {0~2: rifle}
    """
    run_full_causal_sweep(
        object_index=0,
        shard_name="NMR-test-03.tar",
        tag="car_tsed_win50_90_other",
        window_size=50,
        target_offset_deg=90.0,
        tsed_threshold=4.0
    )