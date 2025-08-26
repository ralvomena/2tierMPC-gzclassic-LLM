from setuptools import setup

package_name = 'edge_tier_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rômulo Omena',
    maintainer_email='romulo.omena@ee.ufcg.edu.br',
    description='Package for the edge nodes tier.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "mpc_node = edge_tier_pkg.mpc_node:main",
            "supervisor = edge_tier_pkg.supervisor:main",
            "netem_edge = edge_tier_pkg.netem_node_edge:main",
        ],
    },
)
