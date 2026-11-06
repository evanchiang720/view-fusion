"""
causal_patching.py
------------------
Causal Mediation Analysis and Activation Patching Framework for ViewFusion.

Features:
  1. Safe, robust multi-object loader supporting index-based selection.
  2. Persistent activation patching across candidate U-Net components.
  3. Simultaneous export of:
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
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import yaml

from model.unet import UNet
from model.view_fusion import ViewFusion


# ============================================================================
# Part 1: Forward Hook Manager
# ============================================================================

class HookManager:
    def __init__(self):
        self.cached_activations: Dict[str, torch.Tensor] = {}
        self.mode: str = "IDLE"
        self.patch_target_name: Optional[str] = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.timestep_handle: Optional[torch.utils.hooks.RemovableHandle] = None
        self.current_timestep: Optional[int] = None
        self.target_timestep: Optional[int] = None
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

    def register_record_hooks(self, candidate_modules: Dict[str, nn.Module], target_timestep: int):
        self.remove_hooks()
        self.mode = "RECORD_ALL"
        self.target_timestep = target_timestep

        for name, module in candidate_modules.items():
            def make_hook(mod_name):
                def hook_fn(sub_module, input_tensor, output_tensor):
                    if self.current_timestep == self.target_timestep:
                        self.cached_activations[mod_name] = output_tensor.detach().clone()
                    return output_tensor
                return hook_fn

            handle = module.register_forward_hook(make_hook(name))
            self.hook_handles.append(handle)

    def register_patch_hook(self, target_module: nn.Module, target_name: str, target_timestep: int):
        self.remove_hooks()
        self.mode = "PATCH"
        self.patch_target_name = target_name
        self.target_timestep = target_timestep

        if target_name not in self.cached_activations:
            raise KeyError(f"No clean activation cached for target: {target_name}")

        clean_tensor = self.cached_activations[target_name]

        def patch_fn(sub_module, input_tensor, output_tensor):
            # Intervene persistently from target_timestep downwards
            if self.current_timestep is not None and self.current_timestep <= self.target_timestep:
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
        if "attn" in name and len(list(module.children())) == 0:
            candidates[name] = module
        elif "mid" in name and module.__class__.__name__ == "ResnetBlocWithAttn":
            candidates[name] = module
        elif name.endswith("final_conv"):
            candidates[name] = module
    return candidates


# ============================================================================
# Part 3: Robust Multi-Object Dataset Loader
# ============================================================================

def find_shard_path(shard_name: str = "NMR-test-00.tar") -> str:
    path = os.path.join("data/nmr/NMR_sharded_100_4", shard_name)
    if os.path.exists(path):
        return path
    return os.path.join("/home/chc302/public/users/chc302/NMR_sharded_100_4", shard_name)


def load_real_nmr_sample(
    device: torch.device,
    object_index: int = 0, 
    shard_name: str = "NMR-test-00.tar"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts multi-view images for a specific 3D CAD object using object_index.
    Each object in NMR ShapeNet has 24 views. Skipping (object_index * 24)
    allows cleanly selecting different objects (e.g. object 0, object 1, object 2).
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

    # Context views: 3 distinct input angles
    y_cond = torch.stack(extracted_images[:3], dim=0).unsqueeze(0).to(device)  # (1, 3, 3, 64, 64)
    view_count = torch.tensor([3], device=device)
    clean_angle = torch.tensor([[np.pi / 2]], dtype=torch.float32, device=device)  # Target: 90 degrees

    return y_cond, view_count, clean_angle


# ============================================================================
# Part 4: Consistency Metric & Causal Effects
# ============================================================================

def evaluate_3d_consistency_metric(
    synthesized_view: torch.Tensor,
    reference_view: torch.Tensor
) -> float:
    """
    Measures novel view alignment (PSNR in dB) relative to reference view.
    """
    mse = torch.mean((synthesized_view - reference_view) ** 2).item()
    if mse < 1e-8:
        return 86.0
    return float(10.0 * np.log10(4.0 / mse))


def corrupt_target_pose(clean_angle: torch.Tensor, min_delta_rad: float = np.pi / 2) -> torch.Tensor:
    perturbation = torch.empty_like(clean_angle).uniform_(min_delta_rad, 2 * np.pi - min_delta_rad)
    return (clean_angle + perturbation) % (2 * np.pi)


def compute_causal_effects(
    m_clean: float,
    m_corrupted: float,
    m_patched: float,
    te_threshold: float = 1e-3
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
    title: str = "ViewFusion Causal Attribution Heatmap (Target-View Consistency)",
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
    cbar.set_label("Indirect Effect on Consistency ($IE_c$)", rotation=270, labelpad=18, fontsize=11)

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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))

    # Row 0: Baselines
    for col_idx, t_step in enumerate(timesteps):
        ax = axes[0, col_idx]
        if col_idx == 0:
            ax.imshow(tensor_to_numpy_img(clean_img))
            ax.set_title("Clean Reference\n(Target: 90°)", fontsize=9, fontweight="bold", color="darkgreen")
        elif col_idx == 1:
            ax.imshow(tensor_to_numpy_img(corrupted_img))
            ax.set_title("Corrupted\n", fontsize=9, fontweight="bold", color="darkred")
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
    print(f" Visual qualitative comparison table exported to: {save_path}")


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


def run_full_causal_sweep(object_index: int = 0, shard_name="NMR-test-00.tar", tag="sofa0"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Real Causal Sweep on: {device}")

    model = load_viewfusion_model(device)
    components = find_candidate_components(model)
    comp_names = list(components.keys())
    print(f"[✓] Registered {len(comp_names)} candidate components.")

    timesteps = [900, 700, 500, 300, 100]
    heatmap_matrix = np.zeros((len(comp_names), len(timesteps)))
    saved_visuals: Dict[Tuple[str, int], torch.Tensor] = {}

    hook_manager = HookManager()
    total_diffusion_steps = 1000

    # 1. Load multi-view sample (change object_index to test different objects)
    y_cond, view_count, clean_angle = load_real_nmr_sample(device, object_index=object_index, shard_name=shard_name)
    corrupted_angle = corrupt_target_pose(clean_angle)

    print(f"[*] Real Sample Loaded | Clean target: {clean_angle.item():.3f} rad | Corrupted target: {corrupted_angle.item():.3f} rad")

    # Step 1: Clean Baseline Reference
    print("\n[1/3] Generating Clean Baseline Reference...")
    hook_manager.remove_hooks()
    torch.manual_seed(42)
    with torch.no_grad():
        clean_out = model(y_cond=y_cond, view_count=view_count, angle=clean_angle, generate=True)
    clean_ref = clean_out[0].detach().clone()
    m_clean = evaluate_3d_consistency_metric(clean_out[0], clean_ref)

    # Step 2: Corrupted Baseline Reference
    print("[2/3] Generating Corrupted Baseline Reference...")
    torch.manual_seed(42)
    with torch.no_grad():
        corrupted_out = model(y_cond=y_cond, view_count=view_count, angle=corrupted_angle, generate=True)
    m_corrupted = evaluate_3d_consistency_metric(corrupted_out[0], clean_ref)

    total_effect = m_clean - m_corrupted
    print(f"[*] Consistency Baselines: M(Clean) = {m_clean:.2f} dB | M(Corrupted) = {m_corrupted:.2f} dB | Total Effect (TE) = {total_effect:.2f} dB\n")

    # Step 3: Run Activation Patching Grid
    print("[3/3] Executing Activation Patching Grid...")
    total_runs = len(comp_names) * len(timesteps)
    completed = 0

    for j, t_step in enumerate(timesteps):
        print(f"\n---> [Timestep Group: t = {t_step}] Caching clean activations across all components...")

        hook_manager.register_timestep_tracker(model.denoise_fn, total_steps=total_diffusion_steps)
        hook_manager.register_record_hooks(components, target_timestep=t_step)
        torch.manual_seed(42)
        with torch.no_grad():
            _ = model(y_cond=y_cond, view_count=view_count, angle=clean_angle, generate=True)

        for i, c_name in enumerate(comp_names):
            c_module = components[c_name]

            hook_manager.register_timestep_tracker(model.denoise_fn, total_steps=total_diffusion_steps)
            hook_manager.register_patch_hook(c_module, target_name=c_name, target_timestep=t_step)

            torch.manual_seed(42)
            with torch.no_grad():
                patched_out = model(y_cond=y_cond, view_count=view_count, angle=corrupted_angle, generate=True)

            m_patched = evaluate_3d_consistency_metric(patched_out[0], clean_ref)
            effects = compute_causal_effects(m_clean, m_corrupted, m_patched)
            ie_score = float(np.clip(effects["Indirect_Effect"], 0.0, 1.0))
            heatmap_matrix[i, j] = ie_score

            # Save generated images for visual qualitative comparison table
            saved_visuals[(c_name, t_step)] = patched_out[0].detach().cpu()

            completed += 1
            short_name = ".".join(c_name.split(".")[-3:])
            print(f"  [{completed:03d}/{total_runs}] Comp: {short_name:<28} | t={t_step:<4} | IE_c={ie_score:.4f}")

    hook_manager.remove_all()

    # Step 4: Export Heatmap
    heatmap_path = f"results/real_causal_heatmap_{tag}.png"
    plot_causal_heatmap(
        heatmap_matrix,
        comp_names,
        timesteps,
        title="ViewFusion Causal Attribution Heatmap (Target-View Consistency)",
        save_path=heatmap_path
    )

    # Step 5: Dynamically select Top-3 components based on peak causal effect
    # Compute the maximum Indirect Effect (IE_c) across timesteps for each component
    peak_effects = np.max(heatmap_matrix, axis=1)
    ranked_indices = np.argsort(peak_effects)[::-1]  # Sort in descending order

    top_k = 3
    visual_components = [comp_names[idx] for idx in ranked_indices[:top_k]]

    # Also append the lowest-ranked component (Bottom-1) as a Negative Control
    bottom_idx = ranked_indices[-1]
    visual_components.append(comp_names[bottom_idx])

    print("\n[*] Dynamically selected components for Visual Qualitative Grid:")
    for rank, idx in enumerate(ranked_indices[:top_k], start=1):
        print(f"    Rank {rank} (Top-{rank}): {comp_names[idx]} | Peak IE_c = {peak_effects[idx]:.4f}")
    print(f"    Negative Control (Bottom-1): {comp_names[bottom_idx]} | Peak IE_c = {peak_effects[bottom_idx]:.4f}\n")

    grid_path = f"results/causal_visual_grid_{tag}.png"
    plot_causal_image_grid(
        saved_images=saved_visuals,
        clean_img=clean_out[0].detach().cpu(),
        corrupted_img=corrupted_out[0].detach().cpu(),
        selected_components=visual_components,
        timesteps=timesteps,
        save_path=grid_path
    )

    print("\n" + "=" * 60)
    print(f"Full Causal Sweep & Visual Table Successfully Exported!")
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
    # Change object_index to test different 3D objects (e.g. 0, 1, 2, 3...)
    run_full_causal_sweep(object_index=0, shard_name="NMR-test-00.tar", tag="sofa0")