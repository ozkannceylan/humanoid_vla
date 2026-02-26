import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/media/orka/storage/robotics/ros2_ws/install/vla_mujoco_bridge'
