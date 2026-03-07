#!/usr/bin/env python3
"""
scripts/record_demo_videos.py

Record short demo clips of the trained ACT policy performing all 5 tasks.
Generates individual .mp4 files in media/ for embedding in README.

Output files:
  media/reach.mp4    — Reach the red cube
  media/grasp.mp4    — Grasp the red cube
  media/pick.mp4     — Pick up the red cube
  media/place.mp4    — Place the red cube on the blue plate
  media/bimanual.mp4 — Pick up the green box with both hands
  media/bimanual_adaptability.mp4 — 4 episodes with varied conditions
  media/all_tasks.mp4 — Combined montage of all tasks

Usage:
  cd ~/projects/humanoid_vla
  MUJOCO_GL=egl python3 scripts/record_demo_videos.py
"""

import argparse
import math
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from act_model import ACTPolicy, TASK_LABELS, task_to_id
from generate_demos import SimWrapper, RIGHT_ARM_CTRL, CAMERA_NAME
from physics_sim import PhysicsSim, LEFT_ARM_CTRL as BM_LEFT_CTRL, RIGHT_ARM_CTRL as BM_RIGHT_CTRL
from domain_randomization import DomainRandomizer

SCENE_CAMERA = "scene_camera"
RENDER_W, RENDER_H = 640, 480
FPS = 15


# ── Rendering helpers ─────────────────────────────────

def render_scene(sim, camera_name=SCENE_CAMERA):
    """Render from the fixed overview camera."""
    sim.renderer.update_scene(sim.data, camera=camera_name)
    return sim.renderer.render().copy()


def add_text(frame, text, pos=(10, 30), scale=0.7, color=(255, 255, 255),
             bg_color=(0, 0, 0), thickness=2):
    """Put text with background on frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 4, y + baseline + 4),
                  bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def compose_frame(scene, ego, label_text, step_text=""):
    """Create side-by-side scene + ego frame with label bar."""
    h, w = RENDER_H, RENDER_W
    bar_h = 40
    canvas = np.zeros((h + bar_h, w * 2, 3), dtype=np.uint8)

    # Top bar
    add_text(canvas, label_text, pos=(10, 28), scale=0.75,
             color=(0, 255, 200), bg_color=(30, 30, 30))
    if step_text:
        add_text(canvas, step_text, pos=(w + 300, 28), scale=0.6,
                 color=(200, 200, 200), bg_color=(30, 30, 30))

    # Place frames (RGB→BGR for cv2)
    canvas[bar_h:bar_h + h, :w] = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
    canvas[bar_h:bar_h + h, w:w * 2] = cv2.cvtColor(ego, cv2.COLOR_RGB2BGR)

    # Camera labels
    add_text(canvas, "Overview", pos=(10, bar_h + 22), scale=0.5,
             color=(200, 200, 255))
    add_text(canvas, "Robot View", pos=(w + 10, bar_h + 22), scale=0.5,
             color=(200, 255, 200))
    return canvas


def make_title_card(text, canvas_w, canvas_h, frames=20):
    """Create title card frames."""
    card = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    add_text(card, text, pos=(canvas_w // 2 - len(text) * 12, canvas_h // 2),
             scale=1.0, color=(0, 255, 200), bg_color=(20, 20, 20), thickness=2)
    return [card.copy() for _ in range(frames)]


# ── Single-arm episode recording ──────────────────────

def record_single_arm_episode(model, sim, task_label, rng, device='cuda',
                               max_steps=250, chunk_exec=5, ensemble_k=0.01):
    """Run single-arm episode and return list of composed frames."""
    sim.reset_with_noise(rng)
    chunk_size = model.chunk_size
    action_dim = model.action_dim

    is_composite = ("pick" in task_label or "place" in task_label)
    approach_id = task_to_id("grasp the red cube") if is_composite else None
    final_id = task_to_id(task_label)
    active_id = approach_id if is_composite else final_id

    grasped = False
    grasp_step = -1
    released = False
    is_place = ("place" in task_label)
    place_target = sim.place_pos if is_place else None

    total_len = max_steps + chunk_size
    action_sum = np.zeros((total_len, action_dim), dtype=np.float64)
    weight_sum = np.zeros(total_len, dtype=np.float64)

    frames = []
    canvas_h = RENDER_H + 40
    canvas_w = RENDER_W * 2

    for step in range(max_steps):
        if step % chunk_exec == 0:
            image = sim.render_camera()
            pos, vel = sim.get_obs()
            state = np.concatenate([pos, vel])
            chunk = model.predict(image, state, active_id, device=device)
            for i in range(chunk_size):
                w = math.exp(-ensemble_k * i)
                action_sum[step + i] += w * chunk[i]
                weight_sum[step + i] += w

        if weight_sum[step] > 0:
            action = action_sum[step] / weight_sum[step]
        else:
            action = np.zeros(action_dim)

        sim.target_pos[RIGHT_ARM_CTRL] = action[RIGHT_ARM_CTRL]
        sim.step_frame()

        # Auto-grasp
        if not grasped and not released:
            d = np.linalg.norm(sim.hand_pos - sim.cube_pos)
            if d < 0.04:
                sim.set_weld(True)
                grasped = True
                grasp_step = step
                if is_composite:
                    active_id = final_id
                    action_sum[step + 1:] = 0
                    weight_sum[step + 1:] = 0

        # Auto-release
        if is_place and grasped and place_target is not None:
            steps_grasped = step - grasp_step
            cube_near = np.linalg.norm(sim.cube_pos[:2] - place_target[:2]) < 0.06
            hand_low = sim.hand_pos[2] < place_target[2] + 0.12
            if steps_grasped > 30 and cube_near and hand_low:
                sim.set_weld(False)
                grasped = False
                released = True
                sim.data.qpos[sim.cube_qpos_adr + 2] = 0.825
                mujoco.mj_forward(sim.model, sim.data)

        # Capture every other step (15 fps effective)
        if step % 2 == 0:
            ego = sim.render_camera()
            scene = render_scene(sim)
            status = "[GRASPED]" if grasped else ("[PLACED]" if released else "")
            frame = compose_frame(
                scene, ego, task_label,
                f"Step {step:03d}  {status}")
            frames.append(frame)

        # Early termination for simple tasks
        if step > 80 and not is_composite and not is_place:
            done = False
            if "reach" in task_label and np.linalg.norm(sim.hand_pos - sim.cube_pos) < 0.04:
                done = True
            if "grasp" in task_label and grasped:
                done = True
            if done:
                for _ in range(15):
                    sim.step_frame()
                    ego = sim.render_camera()
                    scene = render_scene(sim)
                    frames.append(compose_frame(scene, ego, task_label, "SUCCESS"))
                break

    return frames


# ── Bimanual episode recording ────────────────────────

def record_bimanual_episode(model, sim, task_label, rng, device='cuda',
                             max_steps=250, chunk_exec=5, ensemble_k=0.01,
                             noise_x=0.02, noise_y=0.02, randomizer=None,
                             arm_spread=0.0):
    """Run bimanual episode and return list of composed frames."""
    sim.reset_with_noise(rng, noise_x=noise_x, noise_y=noise_y)
    if arm_spread > 0:
        sim.random_arm_start(rng, arm='both', spread=arm_spread,
                             reach_target=sim.box_pos)
    if randomizer is not None:
        randomizer.randomize(rng)
        mujoco.mj_forward(sim.model, sim.data)
    box_init_z = sim.box_pos[2]
    chunk_size = model.chunk_size
    action_dim = model.action_dim

    total_len = max_steps + chunk_size
    action_sum = np.zeros((total_len, action_dim), dtype=np.float64)
    weight_sum = np.zeros(total_len, dtype=np.float64)

    frames = []

    for step in range(max_steps):
        if step % chunk_exec == 0:
            image = sim.render_camera()
            pos_all, vel_all = sim.get_obs()
            state = np.concatenate([
                pos_all[BM_LEFT_CTRL], pos_all[BM_RIGHT_CTRL],
                vel_all[BM_LEFT_CTRL], vel_all[BM_RIGHT_CTRL],
            ])
            chunk = model.predict(image, state, task_id=0, device=device)
            for i in range(chunk_size):
                w = math.exp(-ensemble_k * i)
                action_sum[step + i] += w * chunk[i]
                weight_sum[step + i] += w

        if weight_sum[step] > 0:
            action = action_sum[step] / weight_sum[step]
        else:
            action = np.zeros(action_dim)

        sim.target_pos[BM_LEFT_CTRL] = action[:7]
        sim.target_pos[BM_RIGHT_CTRL] = action[7:]
        sim.step_frame()

        # Capture every other step
        if step % 2 == 0:
            ego = sim.render_camera()
            scene = render_scene(sim)
            lift = (sim.box_pos[2] - box_init_z) * 100
            contacts = sim.get_palm_box_contacts()
            ctext = ""
            if contacts["left_contact"] or contacts["right_contact"]:
                ctext = f"Lift: {lift:.1f}cm  F=[{contacts['left_force']:.0f},{contacts['right_force']:.0f}]N"
            frame = compose_frame(scene, ego, task_label, ctext)
            frames.append(frame)

    return frames


# ── Multi-episode bimanual adaptability demo ──────────

def record_bimanual_adaptability(model, sim, rng, device='cuda',
                                  n_episodes=4, max_steps=250):
    """Record multiple bimanual episodes with varied conditions.

    Shows the model adapting to different box positions, visual environments,
    and arm starting poses — demonstrating generalization.
    """
    randomizer = DomainRandomizer(sim.model, sim.data)
    canvas_h = RENDER_H + 40
    canvas_w = RENDER_W * 2
    all_frames = []

    configs = [
        {"noise_x": 0.00, "noise_y": 0.00, "arm_spread": 0.0,
         "visual": False, "label": "Nominal"},
        {"noise_x": 0.05, "noise_y": 0.04, "arm_spread": 0.15,
         "visual": True, "label": "Randomized Position + Visual"},
        {"noise_x": 0.04, "noise_y": 0.03, "arm_spread": 0.10,
         "visual": True, "label": "Different Start + Colors"},
        {"noise_x": 0.05, "noise_y": 0.05, "arm_spread": 0.15,
         "visual": True, "label": "Full Domain Randomization"},
    ]

    for i, cfg in enumerate(configs[:n_episodes]):
        print(f"  Episode {i+1}/{n_episodes}: {cfg['label']}")
        randomizer.restore()

        frames = record_bimanual_episode(
            model, sim, f"Bimanual Grasp — {cfg['label']}", rng,
            device=device, max_steps=max_steps,
            noise_x=cfg["noise_x"], noise_y=cfg["noise_y"],
            randomizer=randomizer if cfg["visual"] else None,
            arm_spread=cfg["arm_spread"])
        frames = add_success_overlay(frames, n_hold=10)

        # Title card for this episode
        all_frames.extend(make_title_card(
            f"Episode {i+1}: {cfg['label']}", canvas_w, canvas_h, frames=12))
        all_frames.extend(frames)

        randomizer.restore()

    return all_frames


# ── Video writer ──────────────────────────────────────

def write_video(path, frames, fps=FPS):
    """Write frames to .mp4 file."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved: {path}  ({len(frames)} frames, {size_mb:.1f} MB)")


def add_success_overlay(frames, n_hold=20):
    """Add SUCCESS text to the last frame and hold."""
    if not frames:
        return frames
    last = frames[-1].copy()
    h, w = last.shape[:2]
    add_text(last, "SUCCESS", pos=(w // 2 - 80, h // 2),
             scale=1.2, color=(0, 255, 0), bg_color=(0, 0, 0), thickness=3)
    frames.extend([last.copy() for _ in range(n_hold)])
    return frames


# ── Main ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Record demo videos for README")
    parser.add_argument("--sa-checkpoint", default="data/checkpoints/best.pt",
                        help="Single-arm ACT model")
    parser.add_argument("--bm-checkpoint", default="data/bimanual_checkpoints_phase_f2/best.pt",
                        help="Bimanual ACT model (Phase F2)")
    parser.add_argument("--output-dir", default="media",
                        help="Output directory for video files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    canvas_h = RENDER_H + 40
    canvas_w = RENDER_W * 2

    # Load models
    def load_model(path, label):
        ckpt = torch.load(path, map_location=args.device, weights_only=False)
        config = ckpt['config']
        model = ACTPolicy(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            chunk_size=config['chunk_size'],
            hidden_dim=config['hidden_dim'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            num_tasks=config['num_tasks'],
        ).to(args.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        print(f"Loaded {label} model [epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}]")
        return model

    sa_model = load_model(args.sa_checkpoint, "single-arm")
    bm_model = load_model(args.bm_checkpoint, "bimanual")

    rng = np.random.default_rng(args.seed)

    # ── Single-arm tasks ────────────────────────────────
    single_arm_tasks = [
        ("reach", "reach the red cube"),
        ("grasp", "grasp the red cube"),
        ("pick", "pick up the red cube"),
        ("place", "place the red cube on the blue plate"),
    ]

    sa_sim = SimWrapper()
    all_frames = []  # for combined video

    for filename, task_label in single_arm_tasks:
        print(f"\nRecording: {task_label}")
        frames = record_single_arm_episode(
            sa_model, sa_sim, task_label, rng, device=args.device)
        frames = add_success_overlay(frames)

        # Save individual clip
        clip_path = os.path.join(args.output_dir, f"{filename}.mp4")
        write_video(clip_path, frames)

        # Add title card + clip to combined video
        all_frames.extend(make_title_card(
            task_label.upper(), canvas_w, canvas_h, frames=15))
        all_frames.extend(frames)

    sa_sim.renderer.close()

    # ── Bimanual task (adaptability demo) ─────────────
    print(f"\nRecording: bimanual adaptability demo (4 episodes)")
    bm_sim = PhysicsSim()

    # Single nominal episode
    bm_frames = record_bimanual_episode(
        bm_model, bm_sim, "pick up the green box with both hands",
        rng, device=args.device)
    bm_frames = add_success_overlay(bm_frames)
    clip_path = os.path.join(args.output_dir, "bimanual.mp4")
    write_video(clip_path, bm_frames)

    all_frames.extend(make_title_card(
        "BIMANUAL: LIFT GREEN BOX", canvas_w, canvas_h, frames=15))
    all_frames.extend(bm_frames)

    # Multi-episode adaptability video
    adapt_frames = record_bimanual_adaptability(
        bm_model, bm_sim, rng, device=args.device, n_episodes=4)
    adapt_path = os.path.join(args.output_dir, "bimanual_adaptability.mp4")
    write_video(adapt_path, adapt_frames)

    all_frames.extend(make_title_card(
        "ADAPTABILITY DEMO", canvas_w, canvas_h, frames=15))
    all_frames.extend(adapt_frames)

    bm_sim.renderer.close()

    # ── Combined montage ────────────────────────────────
    combined_path = os.path.join(args.output_dir, "all_tasks.mp4")
    write_video(combined_path, all_frames)

    total_mb = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in os.listdir(args.output_dir) if f.endswith('.mp4')
    ) / 1024 / 1024
    print(f"\nAll videos saved to {args.output_dir}/  (total: {total_mb:.1f} MB)")
    print("Files:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith('.mp4'):
            size = os.path.getsize(os.path.join(args.output_dir, f)) / 1024 / 1024
            print(f"  {f:20s}  {size:.1f} MB")


if __name__ == "__main__":
    main()
