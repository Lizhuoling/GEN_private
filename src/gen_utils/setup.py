from glob import glob
import os

from setuptools import setup


package_name = 'gen_utils'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zhuoling',
    maintainer_email='lizhuoling98@gmail.com',
    description='An importable library for ROS2 navigation in python3',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'isaac_warehouse_gt_pose = gen_utils.isaac_warehouse_gt_pose:main',
        ],
    },
)
