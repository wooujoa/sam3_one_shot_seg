from setuptools import find_packages, setup

package_name = 'sam3_one_shot_seg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jwg',
    maintainer_email='wjddnrud4487@kw.ac.kr',
    description='One-shot RGBD capture, external SAM3 inference, masked backprojection point cloud generation',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sam3_one_shot_node = sam3_one_shot_seg.sam3_one_shot_node:main',
            'sam3_robot_r_node = sam3_one_shot_seg.sam3_robot_r_node:main',
            'sam3_robot_node = sam3_one_shot_seg.sam3_robot_node:main',
        ],
    },
)