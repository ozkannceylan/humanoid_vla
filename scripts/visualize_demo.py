#!/usr/bin/env python3
"""
scripts/visualize_demo.py

Render a video of the trained ACT policy performing all 4 tasks in MuJoCo.
Shows side-by-side: scene camera (overview) + ego camera (robot's view).

Outputs: videos/demo_all_tasks.mp4

Usage:
  cd ~/projects/humanoid_vla
  MUJOCO_GL=egl python3 scripts/visualize_demo.py --checkpoint data/checkpoints/best.pt
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

SCENE_CAMERA = "scene_camera"
RENDER_W, RENDER_H = 640, 480
FPS = 30


def render_scene(sim, camera_name=SCENE_CAMERA):
    """Render from the fixed scene (overview) camera."""
    sim.renderer.update_scene(sim.data, camera=camera_name)
    return sim.renderer.render().copy()


def add_text(frame, text, pos=(10, 30), scale=0.8, color=(255, 255, 255),
             bg_color=(0, 0, 0), thickness=2):
    """Put text on frame with background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 4, y + baseline + 4),
                  bg_color, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return frame


def run_visual_episode(model, sim, task_label, rng, device='cuda',
                       max_steps=250, chunk_exec=5, ensemble_k=0.01,
                       auto_grasp_dist=0.04, auto_release_delay=30):
    """Run one episode and return list of (scene_frame, ego_frame) pairs."""
    sim.reset_with_noise(rng)
    chunk_size = model.chunk_size
    action_dim = model.action_dim

    is_composite = ("pick" in task_label or "place" in task_label)
    approach_task_id = task_to_id("grasp the red cube") if is_composite else None
    final_task_id = task_to_id(task_label)
    active_task_id = approach_task_id if is_composite else final_task_id

    grasped = False
    grasp_step = -1
    released = False
    is_place = ("place" in task_label)
    place_target = sim.place_pos if is_place else None

    total_len = max_steps + chunk_size
    action_sum = np.zeros((total_len, action_dim), dtype=np.float64)
    weight_sum = np.zeros(total_len, dtype=np.float64)

    frames = []

    for step in range(max_steps):
        # Re-plan
        if step % chunk_exec == 0:
            image = sim.render_camera()
            pos, vel = sim.get_obs()
            state = np.concatenate([pos, vel])
            chunk = model.predict(image, state, active_task_id, device=device)
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
            if d < auto_grasp_dist:
                sim.set_weld(True)
                grasped = True
                grasp_step = step
                if is_composite:
                    active_task_id = final_task_id
                    action_sum[step+1:] = 0
                    weight_sum[step+1:] = 0

        # Auto-release
        if is_place and grasped and place_target is not None:
            steps_grasped = step - grasp_step
            cube_near = np.linalg.norm(sim.cube_pos[:2] - place_target[:2]) < 0.06
            hand_low = sim.hand_pos[2] < place_target[2] + 0.12
            if steps_grasped > auto_release_delay and cube_near and hand_low:
                sim.set_weld(False)
                grasped = False
                released = True
                sim.data.qpos[sim.cube_qpos_adr + 2] = 0.825
                mujoco.mj_forward(sim.model, sim.data)
                sim.set_weld(False)
                grasped = False

        # Capture frames every other step (15 fps effective to keep video shorter)
        if step % 2 == 0:
            ego = sim.render_camera()                      # 480×640×3 RGB
            scene = render_scene(sim)                      # 480×640×3 RGB
            frames.append((scene, ego, grasped, step))

        # Early termination for reach/grasp (once successful, hold a few more frames)
        if step > 80 and not is_composite and not is_place:
            if task_label == "reach the red cube" and np.linalg.norm(sim.hand_pos - sim.cube_pos) < 0.04:
                # Hold final pose for 15 more frames
                for _ in range(15):
                    sim.step_frame()
                    ego = sim.render_camera()
                    scene = render_scene(sim)
                    frames.append((scene, ego, grasped, step))
                break
            if task_label == "grasp the red cube" and grasped:
                for _ in range(15):
                    sim.step_frame()
                    ego = sim.render_camera()
                    scene = render_scene(sim)
                    frames.append((scene, ego, grasped, step))
                break

    return frames


def compose_frame(scene, ego, task_label, step, grasped, frame_h=480, frame_w=640):
    """Create a side-by-side composition with labels."""
    # Convert RGB→BGR for cv2
    scene_bgr = cv2.cvtColor(scene, cv2.COLOR_RGB2BGR)
    ego_bgr = cv2.cvtColor(ego, cv2.COLOR_RGB2BGR)

    # Side by side
    canvas = np.zeros((frame_h + 50, frame_w * 2, 3), dtype=np.uint8)

    # Top bar with task label
    add_text(canvas, f"Task: {task_label}", pos=(10, 35), scale=0.9,
             color=(0, 255, 200), bg_color=(30, 30, 30))
    step_text = f"Step: {step:03d}"
    if grasped:
        step_text += "  [GRASPED]"
    add_text(canvas, step_text, pos=(frame_w + 200, 35), scale=0.7,
             color=(200, 200, 200), bg_color=(30, 30, 30))

    # Place frames
    canvas[50:50+frame_h, :frame_w] = scene_bgr
    canvas[50:50+frame_h, frame_w:frame_w*2] = ego_bgr

    # Camera labels
    add_text(canvas, "Scene Camera", pos=(10, 75), scale=0.6, color=(200, 200, 255))
    add_text(canvas, "Ego Camera (Robot View)", pos=(frame_w + 10, 75), scale=0.6,
             color=(200, 255, 200))

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize ACT policy in MuJoCo")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="videos/demo_all_tasks.mp4")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
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
    print(f"Loaded model [epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}]")

    sim = SimWrapper()
    rng = np.random.default_rng(args.seed)

    tasks = [
        "reach the red cube",
        "grasp the red cube",
        "pick up the red cube",
        "place the red cube on the blue plate",
    ]

    # Video writer: side-by-side = 1280×530, at 15 fps
    canvas_h = RENDER_H + 50
    canvas_w = RENDER_W * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, 15, (canvas_w, canvas_h))

    for task_label in tasks:
        print(f"\n  Recording: {task_label}")
        episode_frames = run_visual_episode(
            model, sim, task_label, rng, device=args.device
        )

        # Add 15 frames of title card before each task
        title_card = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        add_text(title_card, task_label.upper(), pos=(canvas_w // 2 - 250, canvas_h // 2),
                 scale=1.2, color=(0, 255, 200), bg_color=(20, 20, 20), thickness=2)
        for _ in range(20):
            writer.write(title_card)

        # Write episode frames
        for scene, ego, grasped, step in episode_frames:
            canvas = compose_frame(scene, ego, task_label, step, grasped)
            writer.write(canvas)

        # Hold last frame briefly
        if episode_frames:
            scene, ego, grasped, step = episode_frames[-1]
            final = compose_frame(scene, ego, task_label, step, grasped)
            # Add success overlay
            add_text(final, "DONE", pos=(canvas_w // 2 - 40, canvas_h // 2),
                     scale=1.5, color=(0, 255, 0), bg_color=(0, 0, 0), thickness=3)
            for _ in range(30):
                writer.write(final)

        print(f"    → {len(episode_frames)} frames captured")

    writer.release()
    print(f"\n✅ Video saved: {args.output}")
    print(f"   Resolution: {canvas_w}×{canvas_h} @ 15fps")
    print(f"   Duration: ~{sum(1 for _ in range(4)) * 10}s per task")


if __name__ == "__main__":
    main()
