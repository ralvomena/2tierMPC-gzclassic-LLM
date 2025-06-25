import os
from glob import glob
from setuptools import setup

package_name = 'spawn_pkg'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # Path to the mobile robot sdf file
        (os.path.join('share', package_name,'models/agv/'), glob('./models/agv/*')),
        (os.path.join('share', package_name,'models/agv_big/'), glob('./models/agv_big/*')),
        (os.path.join('share', package_name,'models/cylinder/'), glob('./models/cylinder/*')),

        # Path to the world file
        (os.path.join('share', package_name,'worlds/'), glob('./worlds/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rômulo Omena',
    maintainer_email='romulo.omena@ee.ufcg.edu.br',
    description='Package to launch the scenarios in Gazebo and spawn AGVs.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "spawn_node = spawn_pkg.spawn_node:main",
            "scn_1 = spawn_pkg.scn_1:main",
            "scn_2 = spawn_pkg.scn_2:main",
            "scn_3 = spawn_pkg.scn_3:main",
            "scn_4 = spawn_pkg.scn_4:main",
        ],
    },
)