# Developer entry points. `make setup` on a fresh clone gets you to a runnable sim.

.PHONY: setup assets install lint test test-all smoke train-single train-bimanual

setup: assets install  ## fresh-clone bootstrap

assets:
	./scripts/fetch_assets.sh

install:
	pip3 install -e .[dev,track]

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest -m "not slow" -q

test-all:
	pytest -q

# Loads the MJCF (needs assets) and steps physics briefly — no GL required.
smoke:
	python3 -c "import mujoco; m = mujoco.MjModel.from_xml_path('sim/g1_with_camera.xml'); d = mujoco.MjData(m); [mujoco.mj_step(m, d) for _ in range(100)]; print('smoke OK:', m.nu, 'actuators,', m.nq, 'qpos')"

train-single:
	python3 -m humanoid_vla.train --demos data/demos --output data/checkpoints_v2

train-bimanual:
	python3 -m humanoid_vla.train --demos data/bimanual_demos_phase_f2 \
		--output data/bimanual_checkpoints_v2 --filter-success
