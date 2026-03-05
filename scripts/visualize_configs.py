#!/usr/bin/env python3
"""
Render the robot under various Phase F configurations to visualize
what the training data diversity looks like.

Outputs a grid of images showing:
  Row 1: Baseline (no randomization) — what training looked like before
  Row 2: Wide position randomization (±10cm single-arm, ±8cm bimanual)
  Row 3: Random starting postures
  Row 4: Visual domain randomization (colors, lighting, distractors)
  Row 5: Everything combined

Each row has 4 random samples.

Usage:
  cd ~/projects/humanoid_vla
  MUJOCO_GL=egl python3 scripts/visualize_configs.py
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
from physics_sim import PhysicsSim
from domain_randomization import DomainRandomizer

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "visualizations")


def render_ego(sim):
    """Render ego camera view (480x640x3)."""
    return sim.render_camera()


def render_scene(sim):
    """Render scene camera view (480x640x3)."""
    sim.renderer.update_scene(sim.data, camera="scene_camera")
    return sim.renderer.render().copy()


def make_grid(images, ncols=4, pad=4):
    """Arrange images into a grid with padding."""
    nrows = (len(images) + ncols - 1) // ncols
    h, w, c = images[0].shape
    grid = np.ones((nrows * (h + pad) - pad, ncols * (w + pad) - pad, c),
                   dtype=np.uint8) * 40  # dark grey background
    for idx, img in enumerate(images):
        r, col = divmod(idx, ncols)
        y = r * (h + pad)
        x = col * (w + pad)
        grid[y:y+h, x:x+w] = img
    return grid


def add_label(img, text, position=(10, 30)):
    """Add simple text label by drawing white pixels (no PIL needed).
    Just returns the image as-is if we can't import PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        # Draw text with shadow for readability
        x, y = position
        draw.text((x+1, y+1), text, fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255))
        return np.array(pil_img)
    except ImportError:
        return img


def main():
    sim = PhysicsSim()
    randomizer = DomainRandomizer(sim.model, sim.data)
    rng = np.random.default_rng(42)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_images = []
    labels = []

    # ── Row 1: Baseline (no randomization) ──
    print("Row 1: Baseline...")
    for i in range(4):
        sim.reset_with_noise(rng, noise_x=0.02, noise_y=0.02)
        img = render_ego(sim)
        img = add_label(img, f"Baseline #{i+1}")
        all_images.append(img)

    # ── Row 2: Wide position randomization ──
    print("Row 2: Wide position...")
    for i in range(4):
        sim.reset_with_noise(rng, noise_x=0.08, noise_y=0.06)
        img = render_ego(sim)
        img = add_label(img, f"Wide position #{i+1}")
        all_images.append(img)

    # ── Row 3: Random starting postures ──
    print("Row 3: Random posture...")
    for i in range(4):
        sim.reset_with_noise(rng, noise_x=0.02, noise_y=0.02)
        sim.random_arm_start(rng, arm='both', spread=0.3)
        img = render_ego(sim)
        img = add_label(img, f"Random posture #{i+1}")
        all_images.append(img)

    # ── Row 4: Visual domain randomization ──
    print("Row 4: Visual randomization...")
    for i in range(4):
        sim.reset_with_noise(rng, noise_x=0.02, noise_y=0.02)
        randomizer.randomize(rng)
        mujoco.mj_forward(sim.model, sim.data)
        img = render_ego(sim)
        img = add_label(img, f"Visual rand #{i+1}")
        randomizer.restore()
        all_images.append(img)

    # ── Row 5: Everything combined ──
    print("Row 5: All combined...")
    for i in range(4):
        sim.reset_with_noise(rng, noise_x=0.08, noise_y=0.06)
        sim.random_arm_start(rng, arm='both', spread=0.3)
        randomizer.randomize(rng)
        mujoco.mj_forward(sim.model, sim.data)
        img = render_ego(sim)
        img = add_label(img, f"Combined #{i+1}")
        randomizer.restore()
        all_images.append(img)

    # Build grid
    grid = make_grid(all_images, ncols=4, pad=4)

    # Save
    try:
        from PIL import Image
        out_path = os.path.join(OUT_DIR, "phase_f_configs.png")
        Image.fromarray(grid).save(out_path)
        print(f"\nSaved grid: {out_path}")
        print(f"  Size: {grid.shape[1]}x{grid.shape[0]} pixels")
        print(f"  Layout: 5 rows x 4 cols = {len(all_images)} images")
    except ImportError:
        # Fallback: save as raw numpy
        out_path = os.path.join(OUT_DIR, "phase_f_configs.npy")
        np.save(out_path, grid)
        print(f"\nPIL not available. Saved raw array: {out_path}")

    # Also save individual scene camera views for a second perspective
    print("\nRendering scene camera views...")
    scene_images = []
    for i in range(4):
        sim.reset_with_noise(rng, noise_x=0.08, noise_y=0.06)
        sim.random_arm_start(rng, arm='both', spread=0.3)
        randomizer.randomize(rng)
        mujoco.mj_forward(sim.model, sim.data)
        img = render_scene(sim)
        img = add_label(img, f"Scene view #{i+1}")
        scene_images.append(img)
        randomizer.restore()

    scene_grid = make_grid(scene_images, ncols=4, pad=4)
    try:
        from PIL import Image
        scene_path = os.path.join(OUT_DIR, "phase_f_scene_views.png")
        Image.fromarray(scene_grid).save(scene_path)
        print(f"Saved scene views: {scene_path}")
    except ImportError:
        pass

    print("\nDone!")


if __name__ == "__main__":
    main()
