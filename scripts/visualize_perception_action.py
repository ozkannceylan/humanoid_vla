#!/usr/bin/env python3
"""
Visualize the robot's perception-action loop step by step.

Shows what the ego camera sees at each trajectory phase, alongside
the scene camera view, for both baseline and randomized configurations.

Also renders a side-by-side comparison: same task with different
box positions to show how the visual input changes.

Output:
  1. Trajectory strip: ego + scene views at key phases of bimanual pick
  2. Position diversity: 8 different box positions from ego camera
  3. Full diversity: combined randomization from ego camera
"""

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(__file__))
from physics_sim import (
    PhysicsSim, LEFT_ARM_CTRL, RIGHT_ARM_CTRL, BOTH_ARMS_CTRL,
)
from domain_randomization import DomainRandomizer
from generate_bimanual_demos import (
    plan_bimanual_trajectory, BOX_POS_NOMINAL,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "visualizations")


def render_ego(sim):
    return sim.render_camera()

def render_scene(sim):
    sim.renderer.update_scene(sim.data, camera="scene_camera")
    return sim.renderer.render().copy()


def label_img(img, text):
    """Add text label to image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        # Background rectangle for readability
        bbox = draw.textbbox((0, 0), text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([8, 6, 14 + tw, 12 + th], fill=(0, 0, 0, 180))
        draw.text((10, 8), text, fill=(255, 255, 255))
        return np.array(pil)
    except ImportError:
        return img


def hstack(images, pad=2):
    """Horizontally stack images with padding."""
    h = max(img.shape[0] for img in images)
    total_w = sum(img.shape[1] for img in images) + pad * (len(images) - 1)
    out = np.ones((h, total_w, 3), dtype=np.uint8) * 40
    x = 0
    for img in images:
        out[:img.shape[0], x:x+img.shape[1]] = img
        x += img.shape[1] + pad
    return out


def vstack(images, pad=2):
    """Vertically stack images with padding."""
    w = max(img.shape[1] for img in images)
    total_h = sum(img.shape[0] for img in images) + pad * (len(images) - 1)
    out = np.ones((total_h, w, 3), dtype=np.uint8) * 40
    y = 0
    for img in images:
        out[y:y+img.shape[0], :img.shape[1]] = img
        y += img.shape[0] + pad
    return out


def resize_img(img, scale=0.5):
    """Resize image by scale factor."""
    try:
        from PIL import Image
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        return np.array(Image.fromarray(img).resize((new_w, new_h)))
    except ImportError:
        return img


def run_trajectory_and_capture(sim, rng, noise_x=0.02, noise_y=0.02,
                                random_start=0.0, domain_rand=False,
                                randomizer=None):
    """Run a bimanual trajectory and capture ego+scene at key phases.

    Returns list of (phase_name, ego_img, scene_img) tuples.
    """
    sim.reset_with_noise(rng, noise_x=noise_x, noise_y=noise_y)

    if random_start > 0:
        q_L, q_R = sim.random_arm_start(rng, arm='both', spread=random_start)
    else:
        q_L, q_R = None, None

    if domain_rand and randomizer:
        randomizer.randomize(rng)

    box_pos = sim.box_pos.copy()

    # Plan trajectory
    result = plan_bimanual_trajectory(
        sim, box_pos, rng, q_home_L=q_L, q_home_R=q_R)
    traj_L = result['left_traj']
    traj_R = result['right_traj']
    total_frames = result['n_frames']

    # Key phases with frame indices (clamped to actual length)
    phases = [
        ("1. Start (home)", 0),
        ("2. Pre-approach", min(30, total_frames - 1)),
        ("3. Approach", min(50, total_frames - 1)),
        ("4. Squeeze", min(80, total_frames - 1)),
        ("5. Squeeze hold", min(90, total_frames - 1)),
        ("6. Lifting", min(120, total_frames - 1)),
        ("7. Lift complete", min(150, total_frames - 1)),
        ("8. Hold at top", min(total_frames - 1, 168)),
    ]

    captures = []
    last_frame = 0

    for phase_name, target_frame in phases:
        # Execute from last_frame to target_frame
        for f in range(last_frame, min(target_frame + 1, total_frames)):
            sim.target_pos[LEFT_ARM_CTRL] = traj_L[f]
            sim.target_pos[RIGHT_ARM_CTRL] = traj_R[f]
            sim.step_frame()
        last_frame = target_frame + 1

        ego = render_ego(sim)
        scene = render_scene(sim)
        captures.append((phase_name, ego, scene))

    if domain_rand and randomizer:
        randomizer.restore()

    return captures


def main():
    sim = PhysicsSim()
    randomizer = DomainRandomizer(sim.model, sim.data)
    rng = np.random.default_rng(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ════════════════════════════════════════════════════════
    # 1. Trajectory strip: baseline config
    # ════════════════════════════════════════════════════════
    print("=== 1. Baseline trajectory strip ===")
    captures = run_trajectory_and_capture(sim, rng)

    ego_strip = []
    scene_strip = []
    for name, ego, scene in captures:
        ego_strip.append(label_img(resize_img(ego, 0.5), name))
        scene_strip.append(label_img(resize_img(scene, 0.5), name))

    baseline_grid = vstack([
        hstack(ego_strip[:4]),
        hstack(scene_strip[:4]),
        hstack(ego_strip[4:]),
        hstack(scene_strip[4:]),
    ], pad=4)

    try:
        from PIL import Image
        path = os.path.join(OUT_DIR, "trajectory_baseline.png")
        Image.fromarray(baseline_grid).save(path)
        print(f"  Saved: {path}")
    except ImportError:
        pass

    # ════════════════════════════════════════════════════════
    # 2. Same trajectory with full randomization
    # ════════════════════════════════════════════════════════
    print("=== 2. Randomized trajectory strip ===")
    rng2 = np.random.default_rng(99)
    captures_rand = run_trajectory_and_capture(
        sim, rng2, noise_x=0.08, noise_y=0.06,
        random_start=0.25, domain_rand=True, randomizer=randomizer)

    ego_strip_r = []
    scene_strip_r = []
    for name, ego, scene in captures_rand:
        ego_strip_r.append(label_img(resize_img(ego, 0.5), name))
        scene_strip_r.append(label_img(resize_img(scene, 0.5), name))

    rand_grid = vstack([
        hstack(ego_strip_r[:4]),
        hstack(scene_strip_r[:4]),
        hstack(ego_strip_r[4:]),
        hstack(scene_strip_r[4:]),
    ], pad=4)

    try:
        from PIL import Image
        path = os.path.join(OUT_DIR, "trajectory_randomized.png")
        Image.fromarray(rand_grid).save(path)
        print(f"  Saved: {path}")
    except ImportError:
        pass

    # ════════════════════════════════════════════════════════
    # 3. Position diversity: ego view at START for 8 different positions
    # ════════════════════════════════════════════════════════
    print("=== 3. Position diversity (ego at start) ===")
    pos_imgs = []
    rng3 = np.random.default_rng(123)
    for i in range(8):
        sim.reset_with_noise(rng3, noise_x=0.08, noise_y=0.06)
        box = sim.box_pos.copy()
        offset = box - BOX_POS_NOMINAL
        ego = render_ego(sim)
        ego = label_img(resize_img(ego, 0.5),
                       f"box dx={offset[0]:+.2f} dy={offset[1]:+.2f}")
        pos_imgs.append(ego)

    pos_grid = vstack([hstack(pos_imgs[:4]), hstack(pos_imgs[4:])], pad=4)
    try:
        from PIL import Image
        path = os.path.join(OUT_DIR, "position_diversity_ego.png")
        Image.fromarray(pos_grid).save(path)
        print(f"  Saved: {path}")
    except ImportError:
        pass

    # Also scene camera for the same positions
    scene_imgs = []
    rng3b = np.random.default_rng(123)  # same seed to match
    for i in range(8):
        sim.reset_with_noise(rng3b, noise_x=0.08, noise_y=0.06)
        box = sim.box_pos.copy()
        offset = box - BOX_POS_NOMINAL
        scene = render_scene(sim)
        scene = label_img(resize_img(scene, 0.5),
                         f"box dx={offset[0]:+.2f} dy={offset[1]:+.2f}")
        scene_imgs.append(scene)

    scene_grid = vstack([hstack(scene_imgs[:4]), hstack(scene_imgs[4:])], pad=4)
    try:
        from PIL import Image
        path = os.path.join(OUT_DIR, "position_diversity_scene.png")
        Image.fromarray(scene_grid).save(path)
        print(f"  Saved: {path}")
    except ImportError:
        pass

    # ════════════════════════════════════════════════════════
    # 4. Full diversity comparison: 2x4 grid, everything varies
    # ════════════════════════════════════════════════════════
    print("=== 4. Full diversity (all combined) ===")
    full_imgs_ego = []
    full_imgs_scene = []
    rng4 = np.random.default_rng(777)
    for i in range(8):
        sim.reset_with_noise(rng4, noise_x=0.08, noise_y=0.06)
        sim.random_arm_start(rng4, arm='both', spread=0.3)
        randomizer.randomize(rng4)
        mujoco.mj_forward(sim.model, sim.data)

        ego = label_img(resize_img(render_ego(sim), 0.5), f"Config #{i+1}")
        scene = label_img(resize_img(render_scene(sim), 0.5), f"Config #{i+1}")
        full_imgs_ego.append(ego)
        full_imgs_scene.append(scene)
        randomizer.restore()

    full_grid = vstack([
        hstack(full_imgs_ego[:4]),
        hstack(full_imgs_scene[:4]),
        hstack(full_imgs_ego[4:]),
        hstack(full_imgs_scene[4:]),
    ], pad=4)
    try:
        from PIL import Image
        path = os.path.join(OUT_DIR, "full_diversity.png")
        Image.fromarray(full_grid).save(path)
        print(f"  Saved: {path}")
    except ImportError:
        pass

    print("\nDone! All visualizations saved to data/visualizations/")


if __name__ == "__main__":
    main()
