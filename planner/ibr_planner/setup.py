from setuptools import setup
import os
from glob import glob

package_name = 'ibr_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ychuang',
    maintainer_email='ychuang@ethz.ch',
    description='Iterative Best Response (IBR) planner, ported from the AirSim NeurIPS 2019 drone racing baseline',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ibr_planner = ibr_planner.ibr_node:main',
        ],
    },
)
