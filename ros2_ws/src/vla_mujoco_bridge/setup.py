from setuptools import find_packages, setup

package_name = 'vla_mujoco_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ozkan Ceylan',
    maintainer_email='ozkannceylan@gmail.com',
    description='MuJoCo-ROS2 bridge for Unitree G1 simulation (Phase A+B)',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'bridge_node    = vla_mujoco_bridge.bridge_node:main',
            'teleop_node    = vla_mujoco_bridge.teleop_node:main',
            'arm_teleop_node = vla_mujoco_bridge.arm_teleop_node:main',
            'demo_recorder  = vla_mujoco_bridge.demo_recorder:main',
        ],
    },
)
