#!/usr/bin/env python3
# SAM3_DOOR.py
# Door-handle one-shot segmentation node for shelf_1 HANDLE_OPEN stage.
#
# MASTER flow:
#   HANDLE_OPEN -> /sam3_door_start true
#               -> /cali_zed_door_start true
#               -> /arm_door_start true
#   MASTER does NOT wait /sam3_door_finish. It waits /arm_door_finish.
#
# Main output for CALI_ZED_DOOR:
#   /sam3_door/handle_center_camera   geometry_msgs/PointStamped
#
# Sensor input policy:
#   compressed RGB + compressedDepth only. Raw image/depth are NOT subscribed.

import os
import time
import yaml
import shutil
import struct
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from cv_bridge import CvBridge

from std_msgs.msg import Bool, String
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import (
    CameraInfo,
    CompressedImage,
    Image as RosImage,
    PointCloud2,
    PointField,
)
from sensor_msgs_py import point_cloud2


# ============================================================
# Topic / parameter defaults
# ============================================================
NODE_NAME = 'sam3_door_node'

DEFAULT_START_TOPIC = '/sam3_door_start'
DEFAULT_PROMPT_TOPIC = '/sam3_door_text_prompt'
DEFAULT_FINISH_TOPIC = '/sam3_door_finish'
DEFAULT_PROMPT = 'door handle'

DEFAULT_COLOR_COMPRESSED_TOPIC = '/zedm/zed_node/left/image_rect_color/compressed'
DEFAULT_DEPTH_TOPIC = '/zedm/zed_node/depth/depth_registered'
DEFAULT_DEPTH_COMPRESSED_TOPIC = '/zedm/zed_node/depth/depth_registered/compressedDepth'
DEFAULT_CAMERA_INFO_TOPIC = '/zedm/zed_node/left/camera_info'
DEFAULT_FALLBACK_CAMERA_INFO_TOPIC = '/sam3_door/camera_info_fallback'
DEFAULT_LOCAL_CAMERA_INFO_CACHE_YAML = '~/colcon_ws/src/master_capstone/config/zed_left_camera_info.yaml'
DEFAULT_CAMERA_FRAME_FALLBACK = 'zedm_left_camera_optical_frame'

DEFAULT_HANDLE_CENTER_TOPIC = '/sam3_door/handle_center_camera'
DEFAULT_TARGET_MASK_TOPIC = '/sam3_door/target_mask'
DEFAULT_TARGET_PC_TOPIC = '/sam3_door/target_pc'
DEFAULT_DEBUG_IMAGE_TOPIC = '/sam3_door/debug_image'


@dataclass
class CachedColor:
    bgr: np.ndarray
    header: Any
    source: str


@dataclass
class CachedDepth:
    depth_m: np.ndarray
    header: Any
    source: str
    raw_encoding: str


class Sam3DoorNode(Node):
    def __init__(self) -> None:
        super().__init__(NODE_NAME)

        # ============================================================
        # Master control / prompt
        # ============================================================
        self.declare_parameter('start_topic', DEFAULT_START_TOPIC)
        self.declare_parameter('prompt_topic', DEFAULT_PROMPT_TOPIC)
        self.declare_parameter('finish_topic', DEFAULT_FINISH_TOPIC)
        self.declare_parameter('prompt', DEFAULT_PROMPT)

        # ============================================================
        # ZED RGB-D input topics
        # ============================================================
        self.declare_parameter('color_compressed_topic', DEFAULT_COLOR_COMPRESSED_TOPIC)
        self.declare_parameter('depth_topic', DEFAULT_DEPTH_TOPIC)
        self.declare_parameter('depth_compressed_topic', DEFAULT_DEPTH_COMPRESSED_TOPIC)
        self.declare_parameter('camera_info_topic', DEFAULT_CAMERA_INFO_TOPIC)
        self.declare_parameter('fallback_camera_info_topic', DEFAULT_FALLBACK_CAMERA_INFO_TOPIC)
        self.declare_parameter('local_camera_info_cache_yaml', DEFAULT_LOCAL_CAMERA_INFO_CACHE_YAML)
        self.declare_parameter('camera_frame_fallback', DEFAULT_CAMERA_FRAME_FALLBACK)

        # IMPORTANT: compressed-only input, same as SAM3_L/R.
        # Raw RGB/depth subscriptions are intentionally not used because they can
        # overload DDS and delay/drop compressedDepth delivery on the robot split setup.
        # depth_topic remains only as a reference/backward-compatible parameter.

        # ============================================================
        # Outputs
        # ============================================================
        self.declare_parameter('handle_center_topic', DEFAULT_HANDLE_CENTER_TOPIC)
        self.declare_parameter('target_mask_topic', DEFAULT_TARGET_MASK_TOPIC)
        self.declare_parameter('target_pc_topic', DEFAULT_TARGET_PC_TOPIC)
        self.declare_parameter('debug_image_topic', DEFAULT_DEBUG_IMAGE_TOPIC)

        # ============================================================
        # External SAM3 inference
        # ============================================================
        self.declare_parameter('save_dir', '/home/jwg/sam3_door_output')
        self.declare_parameter('conda_env_name', 'sam3')
        self.declare_parameter('conda_exe', '')
        self.declare_parameter('sam3_infer_script', '/home/jwg/sam3/test_sam3_external_infer.py')
        self.declare_parameter('sam3_timeout_sec', 180.0)
        self.declare_parameter('save_debug_files', True)

        # Expected external script CLI:
        #   python SCRIPT --image_path input_rgb.png --prompt "door handle" \
        #                 --mask_path mask.png --overlay_path result_overlay.png \
        #                 --meta_npz_path boxes_scores.npz
        # This matches the existing SAM3_L/R external-call structure.

        # ============================================================
        # Depth / RGB-D sync / point reconstruction
        # ============================================================
        self.declare_parameter('depth_scale', 0.001)            # 16UC1 mm -> m
        self.declare_parameter('float_depth_is_meters', True)   # ZED raw 32FC1 depth is usually meters
        self.declare_parameter('depth_min_m', 0.10)
        self.declare_parameter('depth_max_m', 2.50)
        self.declare_parameter('rgbd_sync_tolerance_sec', 0.30)
        self.declare_parameter('resize_depth_to_color', False)  # prefer registered depth; enable only if needed
        self.declare_parameter('min_valid_points', 50)
        self.declare_parameter('pointcloud_pixel_stride', 1)
        self.declare_parameter('max_pointcloud_points', 120000)

        # Robust center calculation
        # center_method is used only as a fallback. The normal output uses
        # front depth-band + RANSAC plane fitting on the protruded handle surface.
        self.declare_parameter('center_method', 'median')       # 'median' or 'mean'
        self.declare_parameter('z_outlier_threshold_m', 0.08)   # keep |z - median(z)| <= threshold
        self.declare_parameter('xy_outlier_mad_scale', 4.0)     # robust xy filtering after z filtering

        # Protruded handle surface extraction.
        # The output point is NOT the 2D bbox center and NOT the whole-handle
        # geometric center. It is a stable point on the camera-facing protruded
        # surface of the handle.
        self.declare_parameter('handle_front_percentile', 20.0)
        self.declare_parameter('handle_front_band_m', 0.04)
        self.declare_parameter('min_front_points', 40)
        self.declare_parameter('enable_front_plane_ransac', True)
        self.declare_parameter('plane_ransac_iterations', 250)
        self.declare_parameter('plane_ransac_threshold_m', 0.012)
        self.declare_parameter('plane_min_inliers', 35)
        self.declare_parameter('plane_min_inlier_ratio', 0.25)
        self.declare_parameter('plane_refit_inliers', True)
        self.declare_parameter('plane_random_seed', 7)

        # 2D mask cleanup
        self.declare_parameter('min_mask_pixels', 80)
        self.declare_parameter('select_largest_component', True)
        self.declare_parameter('morph_open_kernel', 3)
        self.declare_parameter('morph_close_kernel', 5)
        self.declare_parameter('mask_dilate_iterations', 0)

        # Repeated debug publication. Center is still calculated only once.
        self.declare_parameter('publish_repeat_count', 5)
        self.declare_parameter('publish_repeat_period_sec', 0.20)

        # ============================================================
        # Fetch parameters
        # ============================================================
        self.start_topic = str(self.get_parameter('start_topic').value)
        self.prompt_topic = str(self.get_parameter('prompt_topic').value)
        self.finish_topic = str(self.get_parameter('finish_topic').value)
        self.prompt = self._normalize_prompt(str(self.get_parameter('prompt').value))

        self.color_compressed_topic = str(self.get_parameter('color_compressed_topic').value)
        self.depth_topic = str(self.get_parameter('depth_topic').value)
        self.depth_compressed_topic = str(self.get_parameter('depth_compressed_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.fallback_camera_info_topic = str(self.get_parameter('fallback_camera_info_topic').value)
        self.local_camera_info_cache_yaml = os.path.expanduser(
            str(self.get_parameter('local_camera_info_cache_yaml').value)
        )
        self.camera_frame_fallback = str(self.get_parameter('camera_frame_fallback').value).strip()

        self.handle_center_topic = str(self.get_parameter('handle_center_topic').value)
        self.target_mask_topic = str(self.get_parameter('target_mask_topic').value)
        self.target_pc_topic = str(self.get_parameter('target_pc_topic').value)
        self.debug_image_topic = str(self.get_parameter('debug_image_topic').value)

        self.save_dir = os.path.expanduser(str(self.get_parameter('save_dir').value))
        self.conda_env_name = str(self.get_parameter('conda_env_name').value)
        self.conda_exe_param = str(self.get_parameter('conda_exe').value).strip()
        self.sam3_infer_script = os.path.expanduser(str(self.get_parameter('sam3_infer_script').value))
        self.sam3_timeout_sec = float(self.get_parameter('sam3_timeout_sec').value)
        self.save_debug_files = bool(self.get_parameter('save_debug_files').value)

        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.float_depth_is_meters = bool(self.get_parameter('float_depth_is_meters').value)
        self.depth_min_m = float(self.get_parameter('depth_min_m').value)
        self.depth_max_m = float(self.get_parameter('depth_max_m').value)
        self.rgbd_sync_tolerance_sec = float(self.get_parameter('rgbd_sync_tolerance_sec').value)
        self.resize_depth_to_color = bool(self.get_parameter('resize_depth_to_color').value)
        self.min_valid_points = int(self.get_parameter('min_valid_points').value)
        self.pointcloud_pixel_stride = max(1, int(self.get_parameter('pointcloud_pixel_stride').value))
        self.max_pointcloud_points = max(1, int(self.get_parameter('max_pointcloud_points').value))

        self.center_method = str(self.get_parameter('center_method').value).strip().lower()
        if self.center_method not in ('median', 'mean'):
            self.get_logger().warn(f'Unknown center_method={self.center_method}. Falling back to median.')
            self.center_method = 'median'
        self.z_outlier_threshold_m = float(self.get_parameter('z_outlier_threshold_m').value)
        self.xy_outlier_mad_scale = float(self.get_parameter('xy_outlier_mad_scale').value)

        self.handle_front_percentile = float(self.get_parameter('handle_front_percentile').value)
        self.handle_front_percentile = float(np.clip(self.handle_front_percentile, 1.0, 60.0))
        self.handle_front_band_m = float(self.get_parameter('handle_front_band_m').value)
        self.min_front_points = int(self.get_parameter('min_front_points').value)
        self.enable_front_plane_ransac = bool(self.get_parameter('enable_front_plane_ransac').value)
        self.plane_ransac_iterations = int(self.get_parameter('plane_ransac_iterations').value)
        self.plane_ransac_threshold_m = float(self.get_parameter('plane_ransac_threshold_m').value)
        self.plane_min_inliers = int(self.get_parameter('plane_min_inliers').value)
        self.plane_min_inlier_ratio = float(self.get_parameter('plane_min_inlier_ratio').value)
        self.plane_refit_inliers = bool(self.get_parameter('plane_refit_inliers').value)
        self.plane_random_seed = int(self.get_parameter('plane_random_seed').value)
        self._rng = np.random.default_rng(self.plane_random_seed)

        self.min_mask_pixels = int(self.get_parameter('min_mask_pixels').value)
        self.select_largest_component = bool(self.get_parameter('select_largest_component').value)
        self.morph_open_kernel = int(self.get_parameter('morph_open_kernel').value)
        self.morph_close_kernel = int(self.get_parameter('morph_close_kernel').value)
        self.mask_dilate_iterations = int(self.get_parameter('mask_dilate_iterations').value)

        self.publish_repeat_count = int(self.get_parameter('publish_repeat_count').value)
        self.publish_repeat_period_sec = float(self.get_parameter('publish_repeat_period_sec').value)

        os.makedirs(self.save_dir, exist_ok=True)
        self.bridge = CvBridge()

        # ============================================================
        # Runtime state
        # ============================================================
        self.active = False
        self.done = False
        self.processing = False
        self.camera_info: Optional[CameraInfo] = None
        self.fallback_camera_info: Optional[CameraInfo] = None
        self.local_camera_info: Optional[CameraInfo] = self.load_camera_info_yaml(self.local_camera_info_cache_yaml)
        self.latest_color: Optional[CachedColor] = None
        self.latest_depth: Optional[CachedDepth] = None
        self.cached_outputs: Dict[str, Any] = {}
        self.publish_timer = None
        self.publish_remaining = 0
        self.start_count = 0

        # ============================================================
        # QoS
        # ============================================================
        self.qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Compressed RGB/depth from image_transport is often reliable in this project.
        self.qos_compressed_sensor = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.qos_camera_info_fallback = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ============================================================
        # Publishers
        # ============================================================
        self.pub_handle_center = self.create_publisher(PointStamped, self.handle_center_topic, self.qos_cmd)
        self.pub_finish = self.create_publisher(Bool, self.finish_topic, self.qos_cmd)
        self.pub_mask = self.create_publisher(RosImage, self.target_mask_topic, 10)
        self.pub_pc = self.create_publisher(PointCloud2, self.target_pc_topic, 10)
        self.pub_debug = self.create_publisher(RosImage, self.debug_image_topic, 10)

        # ============================================================
        # Subscribers
        # ============================================================
        self.create_subscription(Bool, self.start_topic, self.start_callback, self.qos_cmd)
        self.create_subscription(String, self.prompt_topic, self.prompt_callback, self.qos_cmd)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(
            CameraInfo,
            self.fallback_camera_info_topic,
            self.fallback_camera_info_callback,
            self.qos_camera_info_fallback,
        )

        # Compressed-only sensor subscriptions.
        # Do NOT subscribe to raw color/depth here. This matches SAM3_L/R.
        self.create_subscription(
            CompressedImage,
            self.color_compressed_topic,
            self.color_compressed_callback,
            self.qos_compressed_sensor,
        )
        self.create_subscription(
            CompressedImage,
            self.depth_compressed_topic,
            self.compressed_depth_callback,
            self.qos_compressed_sensor,
        )

        self.get_logger().info('========================================')
        self.get_logger().info('SAM3_DOOR node ready')
        self.get_logger().info(f'node_name                 : {NODE_NAME}')
        self.get_logger().info(f'start_topic               : {self.start_topic}')
        self.get_logger().info(f'prompt_topic              : {self.prompt_topic}')
        self.get_logger().info(f'finish_topic              : {self.finish_topic}  # debug only; master does not wait this')
        self.get_logger().info(f'prompt                    : {self.prompt}')
        self.get_logger().info(f'color_compressed_topic    : {self.color_compressed_topic}  # subscribed')
        self.get_logger().info(f'depth_topic               : {self.depth_topic}  # parameter only; NOT subscribed')
        self.get_logger().info(f'depth_compressed_topic    : {self.depth_compressed_topic}  # subscribed')
        self.get_logger().info(f'camera_info_topic         : {self.camera_info_topic}')
        self.get_logger().info(f'fallback_camera_info_topic: {self.fallback_camera_info_topic}')
        self.get_logger().info(f'local_camera_info_cache_yaml: {self.local_camera_info_cache_yaml}')
        self.get_logger().info(f'camera_frame_fallback     : {self.camera_frame_fallback}')
        if self.local_camera_info is not None:
            self.get_logger().info(
                f'[LOCAL CAMERA_INFO CACHE] loaded frame_id={self.local_camera_info.header.frame_id}, '
                f'size={self.local_camera_info.width}x{self.local_camera_info.height}, '
                f'fx={self.local_camera_info.k[0]:.3f}, fy={self.local_camera_info.k[4]:.3f}'
            )
        else:
            self.get_logger().warn('[LOCAL CAMERA_INFO CACHE] not found or invalid. Will use topic/fallback CameraInfo only.')
        self.get_logger().info(f'handle_center_topic       : {self.handle_center_topic}')
        self.get_logger().info(f'target_mask_topic         : {self.target_mask_topic}')
        self.get_logger().info(f'target_pc_topic           : {self.target_pc_topic}')
        self.get_logger().info(f'debug_image_topic         : {self.debug_image_topic}')
        self.get_logger().info(f'sam3_infer_script         : {self.sam3_infer_script}')
        self.get_logger().info(f'conda_env_name            : {self.conda_env_name}')
        self.get_logger().info(f'front_surface             : percentile={self.handle_front_percentile:.1f}, band={self.handle_front_band_m:.3f} m')
        self.get_logger().info(f'front_plane_ransac        : enabled={self.enable_front_plane_ransac}, threshold={self.plane_ransac_threshold_m:.3f} m, min_inliers={self.plane_min_inliers}, min_ratio={self.plane_min_inlier_ratio:.2f}')
        self.get_logger().info('QoS control               : RELIABLE / TRANSIENT_LOCAL / KEEP_LAST / depth=1')
        self.get_logger().info('Main chain                : SAM3_DOOR -> /sam3_door/handle_center_camera -> CALI_ZED_DOOR')
        self.get_logger().info('========================================')

    # ============================================================
    # Master callbacks
    # ============================================================
    def start_callback(self, msg: Bool) -> None:
        if msg.data:
            self.active = True
            self.done = False
            self.processing = False
            self.start_count += 1
            self.clear_runtime_cache(clear_camera_info=False)
            self.get_logger().info('[START] /sam3_door_start true')
            self.try_run_with_latest_rgbd(trigger='start')
        else:
            self.active = False
            self.done = False
            self.processing = False
            self.clear_runtime_cache(clear_camera_info=False)
            self.get_logger().info('[STOP] /sam3_door_start false')

    def prompt_callback(self, msg: String) -> None:
        self.prompt = self._normalize_prompt(msg.data)
        self.get_logger().info(f'[PROMPT UPDATED] {self.prompt}')

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.camera_info = msg
        self.try_run_with_latest_rgbd(trigger='camera_info')

    def fallback_camera_info_callback(self, msg: CameraInfo) -> None:
        self.fallback_camera_info = msg
        self.get_logger().info(
            f'[RX fallback camera_info] topic={self.fallback_camera_info_topic}, '
            f'frame_id={msg.header.frame_id}, size={msg.width}x{msg.height}, '
            f'fx={msg.k[0]:.3f}, fy={msg.k[4]:.3f}'
        )
        self.try_run_with_latest_rgbd(trigger='fallback_camera_info')

    # ============================================================
    # Sensor callbacks
    # ============================================================
    def color_compressed_callback(self, msg: CompressedImage) -> None:
        if not self.active or self.done:
            return
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                self.get_logger().warn('[RX color compressed] cv2.imdecode returned None. Drop frame.')
                return
            self.latest_color = CachedColor(bgr=bgr, header=msg.header, source='compressed_color')
            self.try_run_with_latest_rgbd(trigger='color')
        except Exception as exc:
            self.get_logger().warn(f'[RX color compressed] decode failed: {repr(exc)}')

    def compressed_depth_callback(self, msg: CompressedImage) -> None:
        if not self.active or self.done:
            return
        try:
            depth_m, enc = self.decode_compressed_depth_to_meters(msg)
            if depth_m is None:
                self.get_logger().warn(
                    f'[RX depth compressed] failed to decode. topic={self.depth_compressed_topic}, format={msg.format}'
                )
                return
            self.latest_depth = CachedDepth(
                depth_m=depth_m,
                header=msg.header,
                source='compressed_depth',
                raw_encoding=enc,
            )
            self.try_run_with_latest_rgbd(trigger='compressed_depth')
        except Exception as exc:
            self.get_logger().warn(f'[RX depth compressed] decode failed: {repr(exc)}')

    # ============================================================
    # One-shot run guard
    # ============================================================
    def try_run_with_latest_rgbd(self, trigger: str = '') -> None:
        if not self.active:
            return
        if self.done or self.processing:
            return
        camera_info_to_use, camera_info_source = self.select_camera_info_for_current_rgbd(trigger)
        if camera_info_to_use is None:
            if trigger == 'start':
                self.get_logger().warn(
                    'Waiting fresh RGB-D: camera_info, fallback_camera_info, or local yaml camera_info is not available yet.'
                )
            return
        if self.latest_color is None:
            if trigger == 'start':
                self.get_logger().warn('Waiting fresh RGB-D: color frame is not available yet.')
            return
        if self.latest_depth is None:
            if trigger == 'start':
                self.get_logger().warn('Waiting fresh RGB-D: depth frame is not available yet.')
            return

        color_t = self.stamp_to_float(self.latest_color.header.stamp)
        depth_t = self.stamp_to_float(self.latest_depth.header.stamp)
        if color_t > 0.0 and depth_t > 0.0:
            dt = abs(color_t - depth_t)
            if dt > self.rgbd_sync_tolerance_sec:
                # Do not mix frames from different moments. Drop the older one.
                if color_t < depth_t:
                    self.latest_color = None
                    older = 'color'
                else:
                    self.latest_depth = None
                    older = 'depth'
                self.get_logger().warn(
                    f'RGB-D timestamp mismatch dt={dt:.3f}s > {self.rgbd_sync_tolerance_sec:.3f}s. '
                    f'Dropped older {older}; waiting fresh pair.'
                )
                return

        self.processing = True
        self.get_logger().info('[SAM3_DOOR RUN] start inference')
        try:
            self.get_logger().info(f'[CAMERA_INFO] using {camera_info_source}')
            self.process_once(self.latest_color, self.latest_depth, camera_info_to_use)
            self.done = True
        except subprocess.TimeoutExpired:
            self.get_logger().error(f'[SAM3_DOOR ERROR] external SAM3 timeout after {self.sam3_timeout_sec:.1f}s')
        except Exception as exc:
            self.get_logger().error(f'[SAM3_DOOR ERROR] {repr(exc)}')
        finally:
            self.processing = False

    # ============================================================
    # CameraInfo fallback helpers
    # ============================================================
    def select_camera_info_for_current_rgbd(self, trigger: str = '') -> Tuple[Optional[CameraInfo], str]:
        """Choose CameraInfo with the same priority as SAM3_L/R.

        Priority:
          1) original camera_info topic
          2) /sam3_door/camera_info_fallback
          3) local yaml cache
        """
        candidates = [
            (self.camera_info, 'camera_info_topic'),
            (self.fallback_camera_info, 'fallback_camera_info_topic'),
            (self.local_camera_info, 'local_yaml_cache'),
        ]

        for camera_info, source in candidates:
            if camera_info is None:
                continue
            if self.latest_color is not None and self.latest_depth is not None:
                if not self.validate_camera_info_for_rgbd(camera_info, self.latest_color, self.latest_depth, source):
                    if source == 'fallback_camera_info_topic':
                        self.fallback_camera_info = None
                    elif source == 'local_yaml_cache':
                        self.local_camera_info = None
                    continue
            if source != 'camera_info_topic' and trigger in ('start', 'color', 'compressed_depth', 'fallback_camera_info'):
                self.get_logger().warn(
                    f'[CAMERA_INFO] using {source}: frame_id={camera_info.header.frame_id}, '
                    f'size={camera_info.width}x{camera_info.height}'
                )
            return camera_info, source

        return None, 'none'

    def validate_camera_info_for_rgbd(
        self,
        camera_info: CameraInfo,
        color: CachedColor,
        depth: CachedDepth,
        source: str,
    ) -> bool:
        try:
            color_h, color_w = color.bgr.shape[:2]
            depth_h, depth_w = depth.depth_m.shape[:2]
        except Exception:
            return False

        if int(camera_info.width) != int(color_w) or int(camera_info.height) != int(color_h):
            self.get_logger().error(
                f'CameraInfo size mismatch from {source}. '
                f'camera_info={camera_info.width}x{camera_info.height}, color={color_w}x{color_h}. '
                'Waiting for a matching CameraInfo.'
            )
            return False

        if int(depth_w) != int(color_w) or int(depth_h) != int(color_h):
            self.get_logger().error(
                f'RGB-depth size mismatch. color={color_w}x{color_h}, depth={depth_w}x{depth_h}. '
                'Use registered depth or set resize_depth_to_color:=true.'
            )
            if self.resize_depth_to_color:
                return True
            return False

        if len(camera_info.k) < 9 or float(camera_info.k[0]) <= 0.0 or float(camera_info.k[4]) <= 0.0:
            self.get_logger().error(f'CameraInfo from {source} has invalid K matrix.')
            return False

        if not camera_info.header.frame_id:
            camera_info.header.frame_id = self.camera_frame_fallback or DEFAULT_CAMERA_FRAME_FALLBACK
            self.get_logger().warn(
                f'CameraInfo from {source} had empty frame_id. '
                f'Using camera_frame_fallback={camera_info.header.frame_id}'
            )

        return True

    # ============================================================
    # Processing
    # ============================================================
    def process_once(self, color: CachedColor, depth: CachedDepth, camera_info: CameraInfo) -> None:
        bgr = color.bgr.copy()
        depth_m = depth.depth_m.copy()

        if depth_m.ndim == 3:
            depth_m = depth_m[:, :, 0]

        h_color, w_color = bgr.shape[:2]
        h_depth, w_depth = depth_m.shape[:2]

        if (h_depth, w_depth) != (h_color, w_color):
            if self.resize_depth_to_color:
                depth_m = cv2.resize(depth_m, (w_color, h_color), interpolation=cv2.INTER_NEAREST)
                self.get_logger().warn(
                    f'Depth resized to color size: depth {w_depth}x{h_depth} -> color {w_color}x{h_color}'
                )
            else:
                raise RuntimeError(
                    f'Color/depth size mismatch: color={w_color}x{h_color}, depth={w_depth}x{h_depth}. '
                    'Use registered depth or set resize_depth_to_color:=true.'
                )

        # Save RGB input for external SAM3.
        input_rgb_path = os.path.join(self.save_dir, 'door_input_rgb.png')
        input_depth_npy = os.path.join(self.save_dir, 'door_input_depth_m.npy')
        input_cam_yaml = os.path.join(self.save_dir, 'door_camera_info.yaml')
        mask_path = os.path.join(self.save_dir, 'mask.png')
        overlay_path = os.path.join(self.save_dir, 'result_overlay.png')
        meta_npz_path = os.path.join(self.save_dir, 'boxes_scores.npz')

        for old_path in (mask_path, overlay_path, meta_npz_path):
            if os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except Exception:
                    pass

        cv2.imwrite(input_rgb_path, bgr)
        if self.save_debug_files:
            np.save(input_depth_npy, depth_m)
            self.save_camera_info_yaml(camera_info, input_cam_yaml)

        self.call_external_sam3(
            image_path=input_rgb_path,
            prompt=self.prompt,
            mask_path=mask_path,
            overlay_path=overlay_path,
            meta_npz_path=meta_npz_path,
        )

        if not os.path.exists(mask_path):
            raise RuntimeError(f'SAM3 did not produce mask file: {mask_path}')

        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise RuntimeError(f'Failed to read SAM3 mask: {mask_path}')
        if mask_gray.shape[:2] != (h_color, w_color):
            mask_gray = cv2.resize(mask_gray, (w_color, h_color), interpolation=cv2.INTER_NEAREST)

        mask = self.refine_mask(mask_gray > 0)
        mask_pixels = int(np.count_nonzero(mask))
        if mask_pixels < self.min_mask_pixels:
            raise RuntimeError(f'SAM3 mask too small: pixels={mask_pixels} < {self.min_mask_pixels}')

        valid_depth = np.isfinite(depth_m) & (depth_m >= self.depth_min_m) & (depth_m <= self.depth_max_m)
        points, colors, us, vs = self.mask_to_points(mask, depth_m, bgr, camera_info, valid_depth)
        if points.shape[0] < self.min_valid_points:
            self.get_logger().warn(
                f'Not enough valid depth points inside mask: {points.shape[0]} < {self.min_valid_points}. Skip publish.'
            )
            return

        points_filtered = self.filter_outliers(points)
        if points_filtered.shape[0] < self.min_valid_points:
            self.get_logger().warn(
                f'Not enough points after outlier filtering: {points_filtered.shape[0]} < {self.min_valid_points}. Skip publish.'
            )
            return

        center, surface_info = self.compute_handle_surface_point(points_filtered)
        self.get_logger().info(
            '[SAM3_DOOR SURFACE] '
            f'method={surface_info.get("method")}, '
            f'mask_points={points.shape[0]}, filtered_points={points_filtered.shape[0]}, '
            f'front_points={surface_info.get("front_points", 0)}, '
            f'plane_inliers={surface_info.get("plane_inliers", 0)}, '
            f'plane_ratio={surface_info.get("plane_ratio", 0.0):.3f}, '
            f'z_front={surface_info.get("z_front", float("nan")):.4f}'
        )
        frame_id = self.resolve_frame_id(camera_info, color, depth)
        stamp = self.resolve_output_stamp(color, depth)

        center_msg = PointStamped()
        center_msg.header.frame_id = frame_id
        center_msg.header.stamp = stamp
        center_msg.point.x = float(center[0])
        center_msg.point.y = float(center[1])
        center_msg.point.z = float(center[2])

        mask_msg = self.bridge.cv2_to_imgmsg((mask.astype(np.uint8) * 255), encoding='mono8')
        mask_msg.header.frame_id = frame_id
        mask_msg.header.stamp = stamp

        pc_msg = self.make_xyzrgb_cloud(frame_id, stamp, points, colors)
        debug_msg = self.make_debug_image(bgr, mask, us, vs, center, camera_info, frame_id, stamp)

        self.cached_outputs = {
            'center': center_msg,
            'mask': mask_msg,
            'pc': pc_msg,
            'debug': debug_msg,
        }

        self.publish_remaining = max(1, self.publish_repeat_count)
        self.publish_once_bundle()
        if self.publish_remaining > 0:
            self.publish_timer = self.create_timer(self.publish_repeat_period_sec, self.publish_repeat_callback)

        self.get_logger().info(
            f'[SAM3_DOOR RESULT] handle_center_camera=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}), '
            f'frame_id={frame_id}'
        )

    # ============================================================
    # External SAM3
    # ============================================================
    def call_external_sam3(self, image_path: str, prompt: str, mask_path: str, overlay_path: str, meta_npz_path: str) -> None:
        if not os.path.exists(self.sam3_infer_script):
            raise RuntimeError(f'SAM3 infer script not found: {self.sam3_infer_script}')

        conda_exe = self.conda_exe_param or shutil.which('conda') or '/home/jwg/miniconda3/bin/conda'
        if not os.path.exists(conda_exe):
            raise RuntimeError(f'conda executable not found: {conda_exe}')

        cmd = [
            conda_exe, 'run', '--no-capture-output', '-n', self.conda_env_name,
            'python', self.sam3_infer_script,
            '--image_path', image_path,
            '--prompt', prompt,
            '--mask_path', mask_path,
            '--overlay_path', overlay_path,
            '--meta_npz_path', meta_npz_path,
        ]

        self.get_logger().info('Calling external SAM3 for door handle...')
        self.get_logger().info(' '.join(cmd))
        start = time.time()
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=self.sam3_timeout_sec,
        )
        elapsed = time.time() - start
        self.get_logger().info(f'SAM3 external call finished in {elapsed:.2f}s')
        if result.stdout:
            self.get_logger().info(f'[SAM3 stdout]\n{result.stdout}')
        if result.stderr:
            self.get_logger().info(f'[SAM3 stderr]\n{result.stderr}')
        if result.returncode != 0:
            raise RuntimeError(f'External SAM3 inference failed with return code {result.returncode}')

    # ============================================================
    # Depth decoding / conversion
    # ============================================================
    def depth_image_to_meters(self, depth_cv: np.ndarray, encoding: str) -> np.ndarray:
        enc = (encoding or '').upper()
        arr = np.asarray(depth_cv)
        if enc in ('32FC1', 'TYPE_32FC1') or arr.dtype == np.float32 or arr.dtype == np.float64:
            out = arr.astype(np.float32)
            if not self.float_depth_is_meters:
                out = out * float(self.depth_scale)
            return out
        if enc in ('16UC1', 'MONO16', 'TYPE_16UC1') or arr.dtype == np.uint16:
            return arr.astype(np.float32) * float(self.depth_scale)
        # Last-resort conversion for unusual encodings.
        self.get_logger().warn(f'Unknown depth encoding={encoding}, dtype={arr.dtype}. Applying depth_scale.')
        return arr.astype(np.float32) * float(self.depth_scale)

    def decode_compressed_depth_to_meters(self, msg: CompressedImage) -> Tuple[Optional[np.ndarray], str]:
        data = np.frombuffer(msg.data, dtype=np.uint8)
        fmt = (msg.format or '').lower()

        # compressed_depth_transport usually prepends a 12-byte header:
        # int format + float depthQuantA + float depthQuantB.
        header = None
        if data.size > 12:
            try:
                header = struct.unpack('iff', bytes(data[:12]))
            except Exception:
                header = None

        candidates = []
        if 'compresseddepth' in fmt and data.size > 12:
            candidates.append(data[12:])
        candidates.append(data)

        decoded = None
        for candidate in candidates:
            if candidate.size == 0:
                continue
            img = cv2.imdecode(candidate, cv2.IMREAD_UNCHANGED)
            if img is not None:
                decoded = img
                break
        if decoded is None:
            return None, 'unknown'
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]

        # 32FC1 compressedDepth is commonly inverse-depth encoded as uint16.
        # Recover meters when quantization parameters are available.
        if '32fc1' in fmt:
            raw = decoded.astype(np.float32)
            if header is not None:
                _, depth_quant_a, depth_quant_b = header
                if np.isfinite(depth_quant_a) and abs(depth_quant_a) > 1e-6:
                    denom = raw - float(depth_quant_b)
                    out = np.full(raw.shape, np.nan, dtype=np.float32)
                    valid = (raw > 0.0) & (np.abs(denom) > 1e-6)
                    out[valid] = float(depth_quant_a) / denom[valid]
                    return out, '32FC1_compressedDepth_recovered'
            # Fallback: if decoded is already float-like, use it as meters.
            return decoded.astype(np.float32), '32FC1_compressedDepth_fallback'

        # 16UC1 compressedDepth is generally PNG depth in original integer units.
        if '16uc1' in fmt or decoded.dtype == np.uint16:
            return decoded.astype(np.float32) * float(self.depth_scale), '16UC1_compressedDepth'

        return decoded.astype(np.float32) * float(self.depth_scale), 'compressedDepth_unknown_scaled'

    # ============================================================
    # Mask / 3D helpers
    # ============================================================
    def refine_mask(self, mask: np.ndarray) -> np.ndarray:
        out = mask.astype(bool)

        k_open = self._odd_kernel(self.morph_open_kernel)
        if k_open > 1:
            kernel = np.ones((k_open, k_open), dtype=np.uint8)
            out = cv2.morphologyEx(out.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel) > 0

        k_close = self._odd_kernel(self.morph_close_kernel)
        if k_close > 1:
            kernel = np.ones((k_close, k_close), dtype=np.uint8)
            out = cv2.morphologyEx(out.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0

        if self.select_largest_component and np.any(out):
            out = self.select_largest_connected_component(out)

        if self.mask_dilate_iterations > 0:
            kernel = np.ones((3, 3), dtype=np.uint8)
            out = cv2.dilate(out.astype(np.uint8) * 255, kernel, iterations=self.mask_dilate_iterations) > 0

        return out

    def select_largest_connected_component(self, mask: np.ndarray) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if num_labels <= 1:
            return mask.copy()
        best_label = 1
        best_area = 0
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_label = label
        return labels == best_label

    def mask_to_points(
        self,
        mask: np.ndarray,
        depth_m: np.ndarray,
        bgr: np.ndarray,
        camera_info: CameraInfo,
        valid_depth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k = camera_info.k
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        if fx <= 0.0 or fy <= 0.0:
            raise RuntimeError(f'Invalid CameraInfo intrinsics: fx={fx}, fy={fy}')

        use_mask = mask & valid_depth
        if self.pointcloud_pixel_stride > 1:
            stride_mask = np.zeros_like(use_mask, dtype=bool)
            stride_mask[::self.pointcloud_pixel_stride, ::self.pointcloud_pixel_stride] = True
            use_mask &= stride_mask

        vs, us = np.where(use_mask)
        if us.size == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), us, vs

        if us.size > self.max_pointcloud_points:
            idx = np.linspace(0, us.size - 1, self.max_pointcloud_points).astype(np.int64)
            us = us[idx]
            vs = vs[idx]

        z = depth_m[vs, us].astype(np.float32)
        x = (us.astype(np.float32) - cx) * z / fx
        y = (vs.astype(np.float32) - cy) * z / fy
        xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[vs, us].astype(np.uint8)
        return xyz, rgb, us.astype(np.int32), vs.astype(np.int32)

    def filter_outliers(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] == 0:
            return points

        z = points[:, 2]
        z_med = float(np.median(z))
        keep = np.abs(z - z_med) <= max(1e-6, self.z_outlier_threshold_m)
        pts = points[keep]
        if pts.shape[0] == 0:
            return points

        # Robust xy filtering using median absolute deviation.
        xy = pts[:, :2]
        med_xy = np.median(xy, axis=0)
        abs_dev = np.abs(xy - med_xy)
        mad = np.median(abs_dev, axis=0)
        mad = np.maximum(mad, 1e-6)
        scaled = abs_dev / mad
        keep_xy = np.all(scaled <= max(1.0, self.xy_outlier_mad_scale), axis=1)
        pts2 = pts[keep_xy]
        return pts2 if pts2.shape[0] > 0 else pts

    def compute_center(self, points: np.ndarray) -> np.ndarray:
        if self.center_method == 'mean':
            return np.mean(points, axis=0).astype(np.float32)
        return np.median(points, axis=0).astype(np.float32)

    def compute_handle_surface_point(self, points: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Return a stable point on the camera-facing protruded handle surface.

        This intentionally does not compute the 2D bbox center or the whole
        handle's geometric center. The output is:
          1) pick the front depth band inside the SAM3 mask,
          2) fit a plane to that front band using RANSAC,
          3) take the median of plane inliers, projected back onto the plane.

        If plane fitting is unreliable, fallback to the robust median of the
        front band.
        """
        info: Dict[str, Any] = {
            'method': 'fallback_filtered_median',
            'front_points': 0,
            'plane_inliers': 0,
            'plane_ratio': 0.0,
            'z_front': float('nan'),
        }

        if points.shape[0] == 0:
            raise RuntimeError('compute_handle_surface_point received empty point cloud')

        front_points, z_front = self.select_front_depth_band(points)
        info['front_points'] = int(front_points.shape[0])
        info['z_front'] = float(z_front)

        if front_points.shape[0] < max(3, self.min_front_points):
            # Not enough front-band points. Use all filtered points as a safe fallback.
            info['method'] = 'fallback_filtered_median_not_enough_front'
            return self.compute_center(points), info

        if not self.enable_front_plane_ransac:
            info['method'] = 'front_band_median_no_ransac'
            return self.compute_center(front_points), info

        plane = self.fit_plane_ransac(front_points)
        if plane is None:
            info['method'] = 'fallback_front_median_ransac_failed'
            return self.compute_center(front_points), info

        normal, d, inliers = plane
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = float(inlier_count) / max(1, int(front_points.shape[0]))
        info['plane_inliers'] = inlier_count
        info['plane_ratio'] = inlier_ratio

        if inlier_count < self.plane_min_inliers or inlier_ratio < self.plane_min_inlier_ratio:
            info['method'] = 'fallback_front_median_low_plane_support'
            return self.compute_center(front_points), info

        plane_points = front_points[inliers]

        if self.plane_refit_inliers and plane_points.shape[0] >= 3:
            refit = self.fit_plane_svd(plane_points)
            if refit is not None:
                normal, d = refit

        p_med = np.median(plane_points, axis=0).astype(np.float64)
        signed_dist = float(np.dot(normal, p_med) + d)
        p_on_plane = p_med - signed_dist * normal

        info['method'] = 'front_band_ransac_plane_projected_median'
        return p_on_plane.astype(np.float32), info

    def select_front_depth_band(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        z = points[:, 2].astype(np.float64)
        z_front = float(np.percentile(z, self.handle_front_percentile))
        z_hi = z_front + max(1e-4, self.handle_front_band_m)
        keep = z <= z_hi
        front = points[keep]
        if front.shape[0] == 0:
            return points, z_front
        return front, z_front

    def fit_plane_ransac(self, points: np.ndarray) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        n_points = int(points.shape[0])
        if n_points < 3:
            return None

        threshold = max(1e-4, float(self.plane_ransac_threshold_m))
        iterations = max(1, int(self.plane_ransac_iterations))

        best_inliers = None
        best_normal = None
        best_d = 0.0
        best_count = -1
        best_median_dist = float('inf')

        pts64 = points.astype(np.float64, copy=False)
        for _ in range(iterations):
            try:
                ids = self._rng.choice(n_points, size=3, replace=False)
            except Exception:
                return None
            p1, p2, p3 = pts64[ids]
            normal = np.cross(p2 - p1, p3 - p1)
            norm = float(np.linalg.norm(normal))
            if norm < 1e-9:
                continue
            normal = normal / norm
            d = -float(np.dot(normal, p1))

            dist = np.abs(pts64 @ normal + d)
            inliers = dist <= threshold
            count = int(np.count_nonzero(inliers))
            if count <= 0:
                continue
            median_dist = float(np.median(dist[inliers]))

            if count > best_count or (count == best_count and median_dist < best_median_dist):
                best_count = count
                best_median_dist = median_dist
                best_inliers = inliers
                best_normal = normal
                best_d = d

        if best_inliers is None or best_normal is None:
            return None
        return best_normal.astype(np.float64), float(best_d), best_inliers

    @staticmethod
    def fit_plane_svd(points: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        if points.shape[0] < 3:
            return None
        pts64 = points.astype(np.float64, copy=False)
        centroid = np.mean(pts64, axis=0)
        x = pts64 - centroid
        try:
            _, _, vh = np.linalg.svd(x, full_matrices=False)
        except Exception:
            return None
        normal = vh[-1]
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            return None
        normal = normal / norm
        d = -float(np.dot(normal, centroid))
        return normal.astype(np.float64), d

    # ============================================================
    # Publish helpers
    # ============================================================
    def publish_repeat_callback(self) -> None:
        if self.publish_remaining <= 0:
            if self.publish_timer is not None:
                self.publish_timer.cancel()
                self.publish_timer = None
            self.publish_finish(True)
            # Remain done=True. Even if active remains true, this start cycle will not run again.
            return
        self.publish_once_bundle()

    def publish_once_bundle(self) -> None:
        if not self.cached_outputs:
            return
        center_msg: PointStamped = self.cached_outputs['center']
        self.pub_handle_center.publish(center_msg)
        self.get_logger().info('[PUB] /sam3_door/handle_center_camera')

        self.pub_mask.publish(self.cached_outputs['mask'])
        self.pub_pc.publish(self.cached_outputs['pc'])
        self.pub_debug.publish(self.cached_outputs['debug'])

        self.publish_remaining -= 1

    def publish_finish(self, value: bool = True) -> None:
        msg = Bool()
        msg.data = bool(value)
        self.pub_finish.publish(msg)
        if value:
            self.get_logger().info('[PUB] /sam3_door_finish true')
        else:
            self.get_logger().info('[PUB] /sam3_door_finish false')

    def make_xyzrgb_cloud(self, frame_id: str, stamp, xyz: np.ndarray, rgb: np.ndarray) -> PointCloud2:
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        header = self.make_header(frame_id, stamp)
        points = []
        for p, c in zip(xyz, rgb):
            packed_rgb = (int(c[0]) << 16) | (int(c[1]) << 8) | int(c[2])
            points.append((float(p[0]), float(p[1]), float(p[2]), int(packed_rgb)))
        return point_cloud2.create_cloud(header, fields, points)

    def make_debug_image(
        self,
        bgr: np.ndarray,
        mask: np.ndarray,
        us: np.ndarray,
        vs: np.ndarray,
        center: np.ndarray,
        camera_info: CameraInfo,
        frame_id: str,
        stamp,
    ) -> RosImage:
        overlay = bgr.copy()
        green = np.zeros_like(overlay, dtype=np.uint8)
        green[:, :, 1] = 255
        alpha = 0.35
        overlay[mask] = cv2.addWeighted(overlay[mask], 1.0 - alpha, green[mask], alpha, 0.0)

        # Draw 2D mask median only as a weak visual reference.
        ys, xs = np.where(mask)
        if xs.size > 0:
            cx2d = int(np.median(xs))
            cy2d = int(np.median(ys))
            cv2.drawMarker(overlay, (cx2d, cy2d), (180, 180, 180), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)

        # Draw the actual 3D output point projected back into the RGB image.
        u_center, v_center = self.project_camera_point_to_pixel(center, camera_info)
        if u_center is not None and v_center is not None:
            if 0 <= u_center < overlay.shape[1] and 0 <= v_center < overlay.shape[0]:
                cv2.circle(overlay, (u_center, v_center), 8, (0, 0, 255), -1)
                cv2.drawMarker(overlay, (u_center, v_center), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
                text_xy = (max(5, u_center - 210), max(22, v_center - 16))
            else:
                text_xy = (10, 25)
            cv2.putText(
                overlay,
                f'plane point ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m',
                text_xy,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        return msg

    @staticmethod
    def project_camera_point_to_pixel(point: np.ndarray, camera_info: CameraInfo) -> Tuple[Optional[int], Optional[int]]:
        try:
            z = float(point[2])
            if not np.isfinite(z) or z <= 1e-6:
                return None, None
            k = camera_info.k
            fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
            if fx <= 0.0 or fy <= 0.0:
                return None, None
            u = int(round(fx * float(point[0]) / z + cx))
            v = int(round(fy * float(point[1]) / z + cy))
            return u, v
        except Exception:
            return None, None

    # ============================================================
    # State / utility
    # ============================================================
    def clear_runtime_cache(self, clear_camera_info: bool = False) -> None:
        self.latest_color = None
        self.latest_depth = None
        self.cached_outputs.clear()
        self.publish_remaining = 0
        if self.publish_timer is not None:
            self.publish_timer.cancel()
            self.publish_timer = None
        if clear_camera_info:
            self.camera_info = None

    def resolve_frame_id(self, camera_info: CameraInfo, color: CachedColor, depth: CachedDepth) -> str:
        if camera_info is not None and camera_info.header.frame_id:
            return camera_info.header.frame_id
        if color is not None and color.header.frame_id:
            return color.header.frame_id
        if depth is not None and depth.header.frame_id:
            return depth.header.frame_id
        return self.camera_frame_fallback or DEFAULT_CAMERA_FRAME_FALLBACK

    @staticmethod
    def resolve_output_stamp(color: CachedColor, depth: CachedDepth):
        # SAM3 mask is produced from the color image; use the color stamp as the primary stamp.
        if color is not None:
            return color.header.stamp
        return depth.header.stamp

    @staticmethod
    def make_header(frame_id: str, stamp):
        from std_msgs.msg import Header
        h = Header()
        h.frame_id = frame_id
        h.stamp = stamp
        return h

    @staticmethod
    def stamp_to_float(stamp) -> float:
        try:
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9
        except Exception:
            return 0.0

    @staticmethod
    def _odd_kernel(k: int) -> int:
        k = max(1, int(k))
        return k if k % 2 == 1 else k + 1

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        prompt = (prompt or '').strip()
        return prompt if prompt else DEFAULT_PROMPT

    @staticmethod
    def load_camera_info_yaml(yaml_path: str) -> Optional[CameraInfo]:
        if not yaml_path:
            return None
        path = os.path.expanduser(str(yaml_path))
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}

            msg = CameraInfo()
            msg.width = int(data.get('width', 0))
            msg.height = int(data.get('height', 0))
            msg.distortion_model = str(data.get('distortion_model', 'plumb_bob'))
            msg.d = [float(v) for v in data.get('d', [])]
            msg.k = [float(v) for v in data.get('k', [])]
            msg.r = [float(v) for v in data.get('r', [])]
            msg.p = [float(v) for v in data.get('p', [])]

            frame_id = ''
            if isinstance(data.get('header'), dict):
                frame_id = str(data.get('header', {}).get('frame_id', ''))
            if not frame_id:
                frame_id = str(data.get('frame_id', ''))
            msg.header.frame_id = frame_id

            if msg.width <= 0 or msg.height <= 0:
                return None
            if len(msg.k) < 9 or float(msg.k[0]) <= 0.0 or float(msg.k[4]) <= 0.0:
                return None
            if len(msg.r) < 9:
                msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            if len(msg.p) < 12:
                fx, fy, cx, cy = float(msg.k[0]), float(msg.k[4]), float(msg.k[2]), float(msg.k[5])
                msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            return msg
        except Exception:
            return None

    @staticmethod
    def save_camera_info_yaml(camera_info: CameraInfo, path: str) -> None:
        data = {
            'header': {
                'frame_id': camera_info.header.frame_id,
                'stamp': {
                    'sec': int(camera_info.header.stamp.sec),
                    'nanosec': int(camera_info.header.stamp.nanosec),
                },
            },
            'height': int(camera_info.height),
            'width': int(camera_info.width),
            'distortion_model': str(camera_info.distortion_model),
            'd': [float(v) for v in camera_info.d],
            'k': [float(v) for v in camera_info.k],
            'r': [float(v) for v in camera_info.r],
            'p': [float(v) for v in camera_info.p],
        }
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, sort_keys=False)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = Sam3DoorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if node is not None:
            node.get_logger().error(f'Unhandled exception: {repr(exc)}')
        else:
            print(f'Unhandled exception before node init: {repr(exc)}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()