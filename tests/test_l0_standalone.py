"""
tests/test_l0_standalone.py — Level 0 Verification

Automated test for standalone MuJoCo simulation (no ROS2).
Tests: Model loading, camera rendering, physics performance, VRAM usage.

Usage:
    python3 tests/test_l0_standalone.py [--duration SECONDS] [--headless]

Requirements:
    - nvidia-smi must be available for VRAM monitoring
    - Run from project root directory
"""

import os
import sys
import time
import subprocess
import argparse
import numpy as np

# Set before import
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer
import cv2


class PerformanceMonitor:
    """Track performance metrics during test run."""

    def __init__(self):
        self.frame_times = []
        self.physics_times = []
        self.vram_samples = []
        self.start_time = None
        self.frame_count = 0
        self.step_count = 0

    def start(self):
        self.start_time = time.perf_counter()

    def record_frame(self):
        self.frame_count += 1
        self.frame_times.append(time.perf_counter() - self.start_time)

    def record_step(self):
        self.step_count += 1

    def sample_vram(self):
        """Query current VRAM usage via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                vram_mb = float(result.stdout.strip())
                self.vram_samples.append(vram_mb)
        except Exception as e:
            print(f"Warning: Could not query VRAM: {e}")

    def get_stats(self):
        """Return performance statistics."""
        elapsed = time.perf_counter() - self.start_time
        return {
            "elapsed_sec": elapsed,
            "frame_count": self.frame_count,
            "step_count": self.step_count,
            "avg_camera_fps": self.frame_count / elapsed if elapsed > 0 else 0,
            "physics_realtime_factor": (self.step_count / 500.0) / elapsed if elapsed > 0 else 0,
            "vram_peak_mb": max(self.vram_samples) if self.vram_samples else None,
            "vram_avg_mb": np.mean(self.vram_samples) if self.vram_samples else None,
        }


def run_test(duration_sec: float, headless: bool) -> dict:
    """
    Run standalone simulation test.

    Args:
        duration_sec: How long to run simulation
        headless: If True, don't launch viewer window

    Returns:
        Dictionary with test results and metrics
    """
    MODEL_PATH = "sim/g1_with_camera.xml"
    CAMERA_NAME = "ego_camera"
    RENDER_WIDTH = 640
    RENDER_HEIGHT = 480
    PHYSICS_HZ = 500
    RENDER_HZ = 30

    results = {
        "success": False,
        "errors": [],
        "metrics": {},
    }

    monitor = PerformanceMonitor()

    # -----------------------------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------------------------
    print(f"\n[L0 Test] Loading model: {MODEL_PATH}")
    try:
        t0 = time.perf_counter()
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
        load_time_ms = (time.perf_counter() - t0) * 1000
        print(f"  ✓ Model loaded in {load_time_ms:.1f} ms")
        print(f"    DOF (nv): {model.nv}, Actuators (nu): {model.nu}")
        results["metrics"]["model_load_ms"] = load_time_ms
    except Exception as e:
        results["errors"].append(f"Model loading failed: {e}")
        return results

    # -----------------------------------------------------------------------
    # 2. Validate camera
    # -----------------------------------------------------------------------
    print(f"\n[L0 Test] Validating camera: {CAMERA_NAME}")
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        results["errors"].append(f"Camera '{CAMERA_NAME}' not found in model")
        return results
    print(f"  ✓ Camera found (id={cam_id})")

    # -----------------------------------------------------------------------
    # 3. Create renderer
    # -----------------------------------------------------------------------
    print(f"\n[L0 Test] Creating renderer ({RENDER_WIDTH}x{RENDER_HEIGHT})")
    try:
        renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
        print(f"  ✓ Renderer created (EGL)")
    except Exception as e:
        results["errors"].append(f"Renderer creation failed: {e}")
        return results

    # -----------------------------------------------------------------------
    # 4. Launch viewer (optional)
    # -----------------------------------------------------------------------
    viewer = None
    if not headless:
        print(f"\n[L0 Test] Launching viewer window")
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
            print(f"  ✓ Viewer launched")
        except Exception as e:
            print(f"  Warning: Viewer launch failed: {e}")
            print(f"  Continuing in headless mode...")

    # -----------------------------------------------------------------------
    # 5. Main simulation loop
    # -----------------------------------------------------------------------
    physics_dt = 1.0 / PHYSICS_HZ
    render_interval = 1.0 / RENDER_HZ
    last_render = 0.0
    last_vram_sample = 0.0
    vram_sample_interval = 1.0  # Sample VRAM every 1 second

    print(f"\n[L0 Test] Running simulation for {duration_sec}s")
    print(f"  Physics: {PHYSICS_HZ} Hz, Camera: {RENDER_HZ} Hz")
    print(f"  Mode: {'with viewer' if viewer else 'headless'}")

    monitor.start()

    # Sample initial VRAM
    monitor.sample_vram()

    try:
        while True:
            # Check duration
            if time.perf_counter() - monitor.start_time >= duration_sec:
                break

            # Check viewer
            if viewer and not viewer.is_running():
                print("\n  Viewer closed by user")
                break

            step_start = time.perf_counter()

            # Physics step
            if viewer:
                with viewer.lock():
                    mujoco.mj_step(model, data)
            else:
                mujoco.mj_step(model, data)
            monitor.record_step()

            # Camera render at target Hz
            now = time.perf_counter()
            if now - last_render >= render_interval:
                renderer.update_scene(data, camera=CAMERA_NAME)
                rgb = renderer.render()
                monitor.record_frame()
                last_render = now

                # Progress indicator every 3 seconds
                elapsed = now - monitor.start_time
                if monitor.frame_count % (RENDER_HZ * 3) == 0:
                    stats = monitor.get_stats()
                    print(f"  t={elapsed:.1f}s | frames={monitor.frame_count} | "
                          f"fps={stats['avg_camera_fps']:.1f} | "
                          f"rt_factor={stats['physics_realtime_factor']:.2f}")

            # Sample VRAM periodically
            if now - last_vram_sample >= vram_sample_interval:
                monitor.sample_vram()
                last_vram_sample = now

            # Sync viewer
            if viewer:
                viewer.sync()

            # Maintain physics timestep
            elapsed_step = time.perf_counter() - step_start
            sleep_time = physics_dt - elapsed_step
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    finally:
        renderer.close()
        if viewer:
            viewer.close()

    # -----------------------------------------------------------------------
    # 6. Collect results
    # -----------------------------------------------------------------------
    monitor.sample_vram()  # Final sample
    stats = monitor.get_stats()
    results["metrics"].update(stats)

    # Success criteria
    camera_fps_ok = stats["avg_camera_fps"] >= 28.0  # >93% of 30Hz
    physics_ok = 0.95 <= stats["physics_realtime_factor"] <= 1.05

    if camera_fps_ok and physics_ok:
        results["success"] = True
    else:
        if not camera_fps_ok:
            results["errors"].append(
                f"Camera FPS too low: {stats['avg_camera_fps']:.1f} < 28 Hz"
            )
        if not physics_ok:
            results["errors"].append(
                f"Physics real-time factor out of range: {stats['physics_realtime_factor']:.2f}"
            )

    return results


def print_results(results: dict):
    """Pretty-print test results."""
    print("\n" + "="*70)
    print("L0 VERIFICATION RESULTS")
    print("="*70)

    if results["success"]:
        print("\n✅ TEST PASSED")
    else:
        print("\n❌ TEST FAILED")
        print("\nErrors:")
        for err in results["errors"]:
            print(f"  - {err}")

    print("\nPerformance Metrics:")
    print("-" * 70)
    m = results["metrics"]

    if "model_load_ms" in m:
        print(f"  Model load time:        {m['model_load_ms']:.1f} ms")
    if "elapsed_sec" in m:
        print(f"  Test duration:          {m['elapsed_sec']:.1f} s")
        print(f"  Physics steps:          {m['step_count']}")
        print(f"  Camera frames:          {m['frame_count']}")
        print(f"  Avg camera FPS:         {m['avg_camera_fps']:.1f} Hz (target: 30 Hz)")
        print(f"  Physics realtime factor: {m['physics_realtime_factor']:.3f} (target: 1.0)")

    if m.get("vram_peak_mb"):
        print(f"  VRAM peak:              {m['vram_peak_mb']:.0f} MB")
        print(f"  VRAM average:           {m['vram_avg_mb']:.0f} MB")

        # Calculate headroom for Phase B
        total_vram_mb = 6144  # RTX 4050
        headroom_mb = total_vram_mb - m['vram_peak_mb']
        headroom_gb = headroom_mb / 1024
        print(f"\n  VRAM headroom (6GB - peak): {headroom_mb:.0f} MB ({headroom_gb:.1f} GB)")

        # Model fit assessment
        act_size_mb = 500
        groot_size_mb = 4000
        print(f"\n  ACT model (~500MB):     {'✓ Fits' if headroom_mb > act_size_mb else '✗ Too tight'}")
        print(f"  GR00T N1 INT8 (~4GB):   {'✓ Fits' if headroom_mb > groot_size_mb else '✗ Too tight'}")
    else:
        print(f"  VRAM monitoring:        Not available (nvidia-smi not found)")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Phase A Level 0 Verification Test")
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Test duration in seconds (default: 60)")
    parser.add_argument("--headless", action="store_true",
                       help="Run without viewer window")
    args = parser.parse_args()

    # Check if running from project root
    if not os.path.exists("sim/g1_with_camera.xml"):
        print("ERROR: Must run from project root directory")
        print("Current dir:", os.getcwd())
        sys.exit(1)

    print("="*70)
    print("PHASE A - LEVEL 0 VERIFICATION TEST")
    print("="*70)
    print(f"Test duration: {args.duration}s")
    print(f"Mode: {'headless' if args.headless else 'with viewer'}")

    results = run_test(args.duration, args.headless)
    print_results(results)

    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
