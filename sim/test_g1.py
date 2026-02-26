"""
sim/test_g1.py — Phase A Milestone 2
Validates: MuJoCo loads G1 model, camera renders, passive viewer runs.
No ROS2 dependency.

Run: MUJOCO_GL=egl python3 sim/test_g1.py
     (or just: python3 sim/test_g1.py — MUJOCO_GL is set below)
"""

import os
import sys
import time
import numpy as np

# Must be set before any mujoco import
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "g1_with_camera.xml")
CAMERA_NAME = "ego_camera"
RENDER_WIDTH = 640
RENDER_HEIGHT = 480
PHYSICS_HZ = 500          # simulation steps per second
RENDER_HZ = 30            # camera rendering target
DISPLAY_PREVIEW = True    # show cv2 window (set False if no display)


def main():
    # -----------------------------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------------------------
    print(f"\nLoading model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    print(f"  Model loaded OK")
    print(f"  DOF (nv):       {model.nv}")
    print(f"  Actuators (nu): {model.nu}")
    print(f"  Bodies (nbody): {model.nbody}")

    # -----------------------------------------------------------------------
    # 2. Validate camera
    # -----------------------------------------------------------------------
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        print(f"\nERROR: Camera '{CAMERA_NAME}' not found in model.")
        print("Check sim/g1_with_camera.xml — the body name must match exactly.")
        sys.exit(1)
    print(f"\nCamera '{CAMERA_NAME}' found (id={cam_id})")

    # -----------------------------------------------------------------------
    # 3. Print actuator (joint) names and indices
    # -----------------------------------------------------------------------
    print(f"\nActuators ({model.nu} total) — these are the ctrl[] indices:")
    print(f"  {'idx':>3}  {'name':<35}  {'ctrlrange'}")
    print(f"  {'-'*3}  {'-'*35}  {'-'*20}")
    for i in range(model.nu):
        act = model.actuator(i)
        lo = model.actuator_ctrlrange[i][0]
        hi = model.actuator_ctrlrange[i][1]
        print(f"  {i:>3}  {act.name:<35}  [{lo:.0f}, {hi:.0f}] Nm")

    # -----------------------------------------------------------------------
    # 4. Create offscreen renderer (EGL — must stay on this thread)
    # -----------------------------------------------------------------------
    print(f"\nCreating renderer ({RENDER_WIDTH}x{RENDER_HEIGHT}) ...")
    renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
    print("  Renderer OK")

    # -----------------------------------------------------------------------
    # 5. Launch passive viewer (GLFW — its own context, non-blocking)
    # -----------------------------------------------------------------------
    print("Launching passive viewer (close window to exit) ...")
    viewer = mujoco.viewer.launch_passive(model, data)

    # -----------------------------------------------------------------------
    # 6. Main loop: physics + camera render + cv2 preview
    # -----------------------------------------------------------------------
    physics_dt = 1.0 / PHYSICS_HZ
    render_interval = 1.0 / RENDER_HZ
    last_render = 0.0
    frame_count = 0
    step_count = 0
    t_start = time.perf_counter()

    print(f"\nRunning at {PHYSICS_HZ} Hz physics, {RENDER_HZ} Hz camera render.")
    if DISPLAY_PREVIEW:
        print("Camera preview window open. Press 'q' in cv2 window to quit.")
    print("Press Ctrl+C or close the viewer window to exit.\n")

    try:
        while viewer.is_running():
            step_start = time.perf_counter()

            # Physics step (lock so viewer sync doesn't race)
            with viewer.lock():
                mujoco.mj_step(model, data)
            step_count += 1

            # Camera render at target Hz
            now = time.perf_counter()
            if now - last_render >= render_interval:
                renderer.update_scene(data, camera=CAMERA_NAME)
                rgb = renderer.render()   # H x W x 3, uint8, RGB

                if DISPLAY_PREVIEW:
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Ego Camera — G1", bgr)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break

                last_render = now
                frame_count += 1

                # Print status every 3 seconds
                elapsed = now - t_start
                if frame_count % (RENDER_HZ * 3) == 0:
                    actual_fps = frame_count / elapsed
                    print(f"  t={data.time:.1f}s | "
                          f"frames={frame_count} | "
                          f"actual_fps={actual_fps:.1f}")

            # Sync viewer
            viewer.sync()

            # Maintain physics timestep
            elapsed_step = time.perf_counter() - step_start
            sleep_time = physics_dt - elapsed_step
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        renderer.close()
        if DISPLAY_PREVIEW:
            cv2.destroyAllWindows()
        viewer.close()

    elapsed_total = time.perf_counter() - t_start
    print(f"\nDone.")
    print(f"  Total sim time:    {data.time:.2f}s")
    print(f"  Wall time:         {elapsed_total:.2f}s")
    print(f"  Steps:             {step_count}")
    print(f"  Camera frames:     {frame_count}")
    print(f"  Avg camera fps:    {frame_count / elapsed_total:.1f}")


if __name__ == "__main__":
    main()
