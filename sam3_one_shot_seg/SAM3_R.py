#!/usr/bin/env python3
# SAM3 one-shot segmentation node for master_2 (RIGHT ARM).
# - waits for /sam3_r_start true before running
# - prompt from /sam3_r_text_prompt
# - publishes only pipeline-required topics plus target_pc for RViz through CALI
# - stays alive for repeated INIT2 cycles

import os
import cv2
import yaml
import time
import shutil
import subprocess
from typing import Optional, Tuple, Dict

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge

from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PointStamped


def to_builtin(obj):
    import numpy as _np
    if isinstance(obj, dict):
        return {to_builtin(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    return obj


class Sam3Master2Node(Node):
    def __init__(self):
        super().__init__('sam3_r_master2_node')

        # ---------------- master control ----------------
        self.declare_parameter('start_topic', '/sam3_r_start')
        self.declare_parameter('prompt_topic', '/sam3_r_text_prompt')
        self.declare_parameter('finish_topic', '/sam3_r_finish')

        # ---------------- ROS topics ----------------
        self.declare_parameter('color_topic', '/camera_r/camera_r/color/image_rect_raw')
        self.declare_parameter('depth_topic', '/camera_r/camera_r/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera_r/camera_r/aligned_depth_to_color/camera_info')

        # ---------------- Prompt / outputs ----------------
        self.declare_parameter('prompt', 'target object')
        self.declare_parameter('save_dir', '/home/jwg/sam3_ros_output_r')

        # ---------------- External SAM3 ----------------
        self.declare_parameter('conda_env_name', 'sam3')
        self.declare_parameter('sam3_infer_script', '/home/jwg/sam3/test_sam3_external_infer.py')
        self.declare_parameter('sam3_timeout_sec', 180.0)

        # ---------------- Depth filtering ----------------
        self.declare_parameter('depth_scale', 0.001)
        self.declare_parameter('depth_min_m', 0.10)
        self.declare_parameter('depth_max_m', 1.20)

        # ---------------- 2D mask refinement ----------------
        self.declare_parameter('bbox_expand_ratio', 0.08)
        self.declare_parameter('bbox_expand_min_px', 8)
        self.declare_parameter('min_component_pixels', 120)
        self.declare_parameter('core_erode_kernel', 3)
        self.declare_parameter('core_erode_iterations', 1)
        self.declare_parameter('safe_dilate_kernel', 5)
        self.declare_parameter('safe_dilate_iterations', 1)
        self.declare_parameter('min_object_core_pixels', 100)
        self.declare_parameter('target_boundary_margin_px', 5.0)
        self.declare_parameter('target_erode_kernel', 5)
        self.declare_parameter('target_erode_iterations', 1)
        self.declare_parameter('min_target_pixels', 60)
        self.declare_parameter('pixel_stride', 2)
        self.declare_parameter('use_depth_band', True)
        self.declare_parameter('depth_band_margin_m', 0.10)
        self.declare_parameter('min_object_points', 80)

        # ---------------- publishers ----------------
        self.declare_parameter('target_pc_topic', '/sam3_r/target_pc')
        self.declare_parameter('object_pc_topic', '/sam3_r/object_pc')
        self.declare_parameter('background_pc_topic', '/sam3_r/background_pc')
        self.declare_parameter('full_scene_pc_topic', '/sam3_r/full_scene_pc')
        self.declare_parameter('mask_image_topic', '/sam3_r/target_mask')
        self.declare_parameter('object_center_topic', '/sam3_r/object_center_camera')
        self.declare_parameter('publish_repeat_count', 20)
        self.declare_parameter('publish_repeat_period_sec', 0.25)
        self.declare_parameter('save_npy', True)
        self.declare_parameter('save_xyzrgb_npy', True)

        # ---------------- parameter fetch ----------------
        self.start_topic = self.get_parameter('start_topic').value
        self.prompt_topic = self.get_parameter('prompt_topic').value
        self.finish_topic = self.get_parameter('finish_topic').value
        self.color_topic = self.get_parameter('color_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.prompt = self.get_parameter('prompt').value
        self.save_dir = self.get_parameter('save_dir').value
        self.conda_env_name = self.get_parameter('conda_env_name').value
        self.sam3_infer_script = self.get_parameter('sam3_infer_script').value
        self.sam3_timeout_sec = float(self.get_parameter('sam3_timeout_sec').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.depth_min_m = float(self.get_parameter('depth_min_m').value)
        self.depth_max_m = float(self.get_parameter('depth_max_m').value)
        self.bbox_expand_ratio = float(self.get_parameter('bbox_expand_ratio').value)
        self.bbox_expand_min_px = int(self.get_parameter('bbox_expand_min_px').value)
        self.min_component_pixels = int(self.get_parameter('min_component_pixels').value)
        self.core_erode_kernel = int(self.get_parameter('core_erode_kernel').value)
        self.core_erode_iterations = int(self.get_parameter('core_erode_iterations').value)
        self.safe_dilate_kernel = int(self.get_parameter('safe_dilate_kernel').value)
        self.safe_dilate_iterations = int(self.get_parameter('safe_dilate_iterations').value)
        self.min_object_core_pixels = int(self.get_parameter('min_object_core_pixels').value)
        self.target_boundary_margin_px = float(self.get_parameter('target_boundary_margin_px').value)
        self.target_erode_kernel = int(self.get_parameter('target_erode_kernel').value)
        self.target_erode_iterations = int(self.get_parameter('target_erode_iterations').value)
        self.min_target_pixels = int(self.get_parameter('min_target_pixels').value)
        self.pixel_stride = int(self.get_parameter('pixel_stride').value)
        self.use_depth_band = bool(self.get_parameter('use_depth_band').value)
        self.depth_band_margin_m = float(self.get_parameter('depth_band_margin_m').value)
        self.min_object_points = int(self.get_parameter('min_object_points').value)
        self.publish_repeat_count = int(self.get_parameter('publish_repeat_count').value)
        self.publish_repeat_period_sec = float(self.get_parameter('publish_repeat_period_sec').value)
        self.save_npy = bool(self.get_parameter('save_npy').value)
        self.save_xyzrgb_npy = bool(self.get_parameter('save_xyzrgb_npy').value)

        self.target_pc_topic = self.get_parameter('target_pc_topic').value
        self.object_pc_topic = self.get_parameter('object_pc_topic').value
        self.background_pc_topic = self.get_parameter('background_pc_topic').value
        self.full_scene_pc_topic = self.get_parameter('full_scene_pc_topic').value
        self.mask_image_topic = self.get_parameter('mask_image_topic').value
        self.object_center_topic = self.get_parameter('object_center_topic').value

        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()
        self.camera_info: Optional[CameraInfo] = None
        self.active = False
        self.done = False
        self.publish_timer = None
        self.publish_remaining = 0
        self.cached_msgs: Dict[str, object] = {}

        # Latest RGB-D cache.
        # Robot arm is assumed static during detection, so exact color/depth timestamp sync
        # is not required. We run with the latest available color + latest available depth.
        self.latest_color_msg: Optional[RosImage] = None
        self.latest_depth_msg: Optional[RosImage] = None
        self.processing = False

        self.qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---------------- publishers ----------------
        self.target_pub = self.create_publisher(PointCloud2, self.target_pc_topic, 10)
        self.object_pub = self.create_publisher(PointCloud2, self.object_pc_topic, 10)
        self.background_pub = self.create_publisher(PointCloud2, self.background_pc_topic, 10)
        self.full_scene_pub = self.create_publisher(PointCloud2, self.full_scene_pc_topic, 10)
        self.mask_pub = self.create_publisher(RosImage, self.mask_image_topic, 10)
        self.object_center_pub = self.create_publisher(PointStamped, self.object_center_topic, 10)
        self.finish_pub = self.create_publisher(Bool, self.finish_topic, self.qos_cmd)

        # ---------------- subscribers ----------------
        self.create_subscription(Bool, self.start_topic, self.start_callback, self.qos_cmd)
        self.create_subscription(String, self.prompt_topic, self.prompt_callback, self.qos_cmd)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        # Do NOT use ApproximateTimeSynchronizer here.
        # For this task the arm is static while detection runs, so a strict timestamp pair
        # can unnecessarily block SAM3 when color/depth stamps differ.
        # Keep normal image QoS/default subscription behavior, same as old code style.
        self.create_subscription(RosImage, self.color_topic, self.color_callback, 10)
        self.create_subscription(RosImage, self.depth_topic, self.depth_callback, 10)

        self.get_logger().info('========================================')
        self.get_logger().info('SAM3 MASTER2 Node Ready (RIGHT ARM)')
        self.get_logger().info(f'start_topic       : {self.start_topic}')
        self.get_logger().info(f'prompt_topic      : {self.prompt_topic}')
        self.get_logger().info(f'finish_topic      : {self.finish_topic}')
        self.get_logger().info(f'prompt            : "{self.prompt}"')
        self.get_logger().info(f'color_topic       : {self.color_topic}')
        self.get_logger().info(f'depth_topic       : {self.depth_topic}')
        self.get_logger().info(f'target_pc_topic   : {self.target_pc_topic}')
        self.get_logger().info(f'object_pc_topic   : {self.object_pc_topic}')
        self.get_logger().info('========================================')

    # ============================================================
    # Master control
    # ============================================================
    def start_callback(self, msg: Bool):
        if msg.data:
            self.active = True
            self.done = False
            self.processing = False
            self.cached_msgs.clear()
            self.publish_remaining = 0
            if self.publish_timer is not None:
                self.publish_timer.cancel()
                self.publish_timer = None

            self.get_logger().info(
                f'[START] {self.start_topic} true. ' 
                f'using latest RGB-D if available. prompt="{self.prompt}"'
            )
            self.try_run_with_latest_rgbd(trigger='start')
        else:
            self.active = False
            self.done = False
            self.processing = False
            self.cached_msgs.clear()
            self.publish_remaining = 0
            if self.publish_timer is not None:
                self.publish_timer.cancel()
                self.publish_timer = None
            self.get_logger().info(f'[STOP] {self.start_topic} false. paused.')

    def prompt_callback(self, msg: String):
        prompt = msg.data.strip()
        if prompt:
            self.prompt = prompt
            self.get_logger().info(f'[PROMPT UPDATED] "{self.prompt}"')

    def publish_finish(self, value: bool = True):
        msg = Bool()
        msg.data = bool(value)
        self.finish_pub.publish(msg)
        self.get_logger().info(f'[PUB] {self.finish_topic} data={str(value).lower()}')

    # ============================================================
    # ROS callbacks
    # ============================================================
    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg
        self.try_run_with_latest_rgbd(trigger='camera_info')

    def color_callback(self, msg: RosImage):
        self.latest_color_msg = msg
        self.try_run_with_latest_rgbd(trigger='color')

    def depth_callback(self, msg: RosImage):
        self.latest_depth_msg = msg
        self.try_run_with_latest_rgbd(trigger='depth')

    def try_run_with_latest_rgbd(self, trigger: str = ''):
        if not self.active:
            return
        if self.done or self.processing:
            return
        if self.camera_info is None:
            if trigger == 'start':
                self.get_logger().warn('Waiting for camera_info...')
            return
        if self.latest_color_msg is None:
            if trigger == 'start':
                self.get_logger().warn('Waiting for latest color image...')
            return
        if self.latest_depth_msg is None:
            if trigger == 'start':
                self.get_logger().warn('Waiting for latest depth image...')
            return

        self.done = True
        self.processing = True

        color_stamp = self.stamp_to_float(self.latest_color_msg.header.stamp)
        depth_stamp = self.stamp_to_float(self.latest_depth_msg.header.stamp)
        dt = abs(color_stamp - depth_stamp)

        self.get_logger().info(
            'Using latest RGB-D frame without timestamp synchronization. '
            f'trigger={trigger}, color_stamp={color_stamp:.9f}, '
            f'depth_stamp={depth_stamp:.9f}, dt={dt:.3f}s'
        )

        try:
            self.run_pipeline(self.latest_color_msg, self.latest_depth_msg, self.camera_info)
        except Exception as e:
            self.get_logger().error(f'Pipeline failed: {repr(e)}')
            self.active = False
            self.publish_finish(False)
        finally:
            self.processing = False

    @staticmethod
    def stamp_to_float(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    # ============================================================
    # Main pipeline
    # ============================================================
    def run_pipeline(self, color_msg: RosImage, depth_msg: RosImage, camera_info: CameraInfo):
        color_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        input_rgb_path = os.path.join(self.save_dir, 'input_rgb.png')
        input_depth_npy = os.path.join(self.save_dir, 'input_depth.npy')
        input_depth_png = os.path.join(self.save_dir, 'input_depth_vis.png')
        input_cam_yaml = os.path.join(self.save_dir, 'camera_info.yaml')

        cv2.imwrite(input_rgb_path, cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))
        np.save(input_depth_npy, depth_cv)
        cv2.imwrite(input_depth_png, self.depth_to_vis(depth_cv))
        self.save_camera_info_yaml(camera_info, input_cam_yaml)

        mask_path = os.path.join(self.save_dir, 'mask.png')
        overlay_path = os.path.join(self.save_dir, 'result_overlay.png')
        meta_npz_path = os.path.join(self.save_dir, 'boxes_scores.npz')

        self.call_external_sam3(input_rgb_path, self.prompt, mask_path, overlay_path, meta_npz_path)

        if not os.path.exists(mask_path):
            raise RuntimeError(f'SAM3 did not produce mask file: {mask_path}')

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise RuntimeError('Failed to read mask.png')

        raw_mask = mask_img > 0
        if not np.any(raw_mask):
            raise RuntimeError('SAM3 mask is empty.')

        h, w = raw_mask.shape
        best_box = self.load_best_box(meta_npz_path, raw_mask)
        roi_mask = self.expand_bbox_to_roi_mask(best_box, h, w)
        raw_mask = raw_mask & roi_mask
        dominant_mask = self.select_dominant_component(raw_mask, best_box)
        object_core_mask, object_safe_mask = self.refine_object_masks(dominant_mask, roi_mask)

        depth_m = self.depth_to_meters(depth_cv)
        valid_depth = np.isfinite(depth_m) & (depth_m >= self.depth_min_m) & (depth_m <= self.depth_max_m)

        if self.use_depth_band:
            ref_mask = object_core_mask if np.any(object_core_mask) else dominant_mask
            obj_depths = depth_m[ref_mask & valid_depth]
            if obj_depths.size > 0:
                z_med = float(np.median(obj_depths))
                z_lo = max(self.depth_min_m, z_med - self.depth_band_margin_m)
                z_hi = min(self.depth_max_m, z_med + self.depth_band_margin_m)
                band_mask = valid_depth & (depth_m >= z_lo) & (depth_m <= z_hi)
                dominant_mask &= band_mask
                object_core_mask &= band_mask
                object_safe_mask &= band_mask
                self.get_logger().info(f'depth band = [{z_lo:.3f}, {z_hi:.3f}] m')

        target_mask = self.make_strict_target_mask(object_core_mask)
        if int(np.count_nonzero(target_mask)) < self.min_target_pixels:
            self.get_logger().warn('strict target mask too small. Fallback to object_core_mask.')
            target_mask = object_core_mask.copy()

        object_mask = object_safe_mask & valid_depth
        target_mask = target_mask & valid_depth
        background_mask = valid_depth & (~object_mask)

        sampled = self.build_sampled_points(color_rgb, depth_cv, camera_info)
        all_xyz = sampled['xyz']
        all_rgb = sampled['rgb']
        all_u = sampled['u']
        all_v = sampled['v']

        if all_xyz.shape[0] == 0:
            raise RuntimeError('No valid 3D points from depth.')

        target_idx = target_mask[all_v, all_u]
        object_idx = object_mask[all_v, all_u]
        background_idx = background_mask[all_v, all_u]
        full_idx = valid_depth[all_v, all_u]

        target_xyz = all_xyz[target_idx]
        target_rgb = all_rgb[target_idx]
        object_xyz = all_xyz[object_idx]
        object_rgb = all_rgb[object_idx]
        background_xyz = all_xyz[background_idx]
        background_rgb = all_rgb[background_idx]
        full_xyz = all_xyz[full_idx]
        full_rgb = all_rgb[full_idx]

        if target_xyz.shape[0] < self.min_object_points:
            raise RuntimeError(f'target_pc too small: {target_xyz.shape[0]} < {self.min_object_points}')
        if object_xyz.shape[0] < self.min_object_points:
            raise RuntimeError(f'object_pc too small: {object_xyz.shape[0]} < {self.min_object_points}')

        object_center_xyz = np.median(object_xyz, axis=0).astype(np.float32)

        self.get_logger().info(
            f'object_center_camera = ({object_center_xyz[0]:.4f}, {object_center_xyz[1]:.4f}, {object_center_xyz[2]:.4f})'
        )
        self.get_logger().info(
            f'points | target={target_xyz.shape[0]} object={object_xyz.shape[0]} background={background_xyz.shape[0]} full={full_xyz.shape[0]}'
        )

        if self.save_npy:
            np.save(os.path.join(self.save_dir, 'target_points_xyz.npy'), target_xyz)
            np.save(os.path.join(self.save_dir, 'object_points_xyz.npy'), object_xyz)
            np.save(os.path.join(self.save_dir, 'background_points_xyz.npy'), background_xyz)
        if self.save_xyzrgb_npy:
            np.save(os.path.join(self.save_dir, 'target_points_xyzrgb.npy'), self.merge_xyz_rgb(target_xyz, target_rgb))
            np.save(os.path.join(self.save_dir, 'object_points_xyzrgb.npy'), self.merge_xyz_rgb(object_xyz, object_rgb))
            np.save(os.path.join(self.save_dir, 'background_points_xyzrgb.npy'), self.merge_xyz_rgb(background_xyz, background_rgb))
            np.save(os.path.join(self.save_dir, 'full_scene_xyzrgb.npy'), self.merge_xyz_rgb(full_xyz, full_rgb))

        cv2.imwrite(os.path.join(self.save_dir, 'mask_raw.png'), (raw_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(self.save_dir, 'mask_dominant.png'), (dominant_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(self.save_dir, 'mask_object_core.png'), (object_core_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(self.save_dir, 'mask_target_strict.png'), (target_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(self.save_dir, 'mask_object_safe.png'), (object_mask.astype(np.uint8) * 255))
        cv2.imwrite(os.path.join(self.save_dir, 'mask_background.png'), (background_mask.astype(np.uint8) * 255))

        frame_id = camera_info.header.frame_id
        stamp = color_msg.header.stamp

        center_msg = PointStamped()
        center_msg.header.frame_id = frame_id
        center_msg.header.stamp = stamp
        center_msg.point.x = float(object_center_xyz[0])
        center_msg.point.y = float(object_center_xyz[1])
        center_msg.point.z = float(object_center_xyz[2])

        self.cached_msgs['target'] = self.make_xyz_cloud(frame_id, stamp, target_xyz)
        self.cached_msgs['object'] = self.make_xyz_cloud(frame_id, stamp, object_xyz)
        self.cached_msgs['background'] = self.make_xyz_cloud(frame_id, stamp, background_xyz)
        self.cached_msgs['full_scene'] = self.make_xyzrgb_cloud(frame_id, stamp, self.make_xyzrgb_tuples(full_xyz, full_rgb))
        self.cached_msgs['target_mask_img'] = self.bridge.cv2_to_imgmsg((target_mask.astype(np.uint8) * 255), encoding='mono8')
        self.cached_msgs['target_mask_img'].header.frame_id = frame_id
        self.cached_msgs['target_mask_img'].header.stamp = stamp
        self.cached_msgs['object_center'] = center_msg

        self.publish_remaining = max(1, int(self.publish_repeat_count))
        self.publish_once_bundle()
        self.get_logger().info(f'Start repeated publish: {self.publish_remaining} times, period={self.publish_repeat_period_sec:.2f}s')
        self.publish_timer = self.create_timer(self.publish_repeat_period_sec, self.publish_repeat_callback)

    # ============================================================
    # Repeated publish
    # ============================================================
    def publish_repeat_callback(self):
        if self.publish_remaining <= 0:
            if self.publish_timer is not None:
                self.publish_timer.cancel()
                self.publish_timer = None
            self.get_logger().info('Repeated publish finished. SAM3 node stays alive.')
            self.active = False
            self.publish_finish(True)
            return
        self.publish_once_bundle()

    def publish_once_bundle(self):
        msg = self.cached_msgs
        if 'target' in msg:
            self.target_pub.publish(msg['target'])
        if 'object' in msg:
            self.object_pub.publish(msg['object'])
        if 'background' in msg:
            self.background_pub.publish(msg['background'])
        if 'full_scene' in msg:
            self.full_scene_pub.publish(msg['full_scene'])
        if 'target_mask_img' in msg:
            self.mask_pub.publish(msg['target_mask_img'])
        if 'object_center' in msg:
            self.object_center_pub.publish(msg['object_center'])
        self.publish_remaining -= 1
        self.get_logger().info(f'Published SAM3 outputs. remaining={self.publish_remaining}')

    # ============================================================
    # External SAM3
    # ============================================================
    def call_external_sam3(self, image_path: str, prompt: str, mask_path: str, overlay_path: str, meta_npz_path: str):
        if not os.path.exists(self.sam3_infer_script):
            raise RuntimeError(f'SAM3 infer script not found: {self.sam3_infer_script}')
        conda_exe = shutil.which('conda') or '/home/jwg/miniconda3/bin/conda'
        if not os.path.exists(conda_exe):
            raise RuntimeError('`conda` command not found.')
        cmd = [
            conda_exe, 'run', '--no-capture-output', '-n', self.conda_env_name,
            'python', self.sam3_infer_script,
            '--image_path', image_path,
            '--prompt', prompt,
            '--mask_path', mask_path,
            '--overlay_path', overlay_path,
            '--meta_npz_path', meta_npz_path,
        ]
        self.get_logger().info('Calling external SAM3...')
        self.get_logger().info(' '.join(cmd))
        start = time.time()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=self.sam3_timeout_sec)
        elapsed = time.time() - start
        self.get_logger().info(f'SAM3 external call finished in {elapsed:.2f}s')
        if result.stdout:
            self.get_logger().info(f'[SAM3 stdout]\n{result.stdout}')
        if result.stderr:
            self.get_logger().info(f'[SAM3 stderr]\n{result.stderr}')
        if result.returncode != 0:
            raise RuntimeError(f'External SAM3 inference failed with return code {result.returncode}')

    # ============================================================
    # Mask refinement
    # ============================================================
    def load_best_box(self, meta_npz_path: str, mask_bin: np.ndarray) -> np.ndarray:
        if os.path.exists(meta_npz_path):
            try:
                meta = np.load(meta_npz_path)
                if 'best_box' in meta:
                    return meta['best_box'].astype(np.float32)
            except Exception:
                pass
        ys, xs = np.where(mask_bin)
        if ys.size == 0:
            return np.array([0, 0, mask_bin.shape[1] - 1, mask_bin.shape[0] - 1], dtype=np.float32)
        return np.array([np.min(xs), np.min(ys), np.max(xs), np.max(ys)], dtype=np.float32)

    def expand_bbox_to_roi_mask(self, box: np.ndarray, h: int, w: int) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in box]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        mx = max(int(self.bbox_expand_min_px), int(self.bbox_expand_ratio * bw))
        my = max(int(self.bbox_expand_min_px), int(self.bbox_expand_ratio * bh))
        ex1 = max(0, int(np.floor(x1)) - mx)
        ey1 = max(0, int(np.floor(y1)) - my)
        ex2 = min(w, int(np.ceil(x2)) + mx)
        ey2 = min(h, int(np.ceil(y2)) + my)
        roi = np.zeros((h, w), dtype=bool)
        roi[ey1:ey2, ex1:ex2] = True
        return roi

    def select_dominant_component(self, mask: np.ndarray, box: np.ndarray) -> np.ndarray:
        mask_u8 = mask.astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if num_labels <= 1:
            return mask.copy()
        cx = int(round(0.5 * (float(box[0]) + float(box[2]))))
        cy = int(round(0.5 * (float(box[1]) + float(box[3]))))
        cx = int(np.clip(cx, 0, mask.shape[1] - 1))
        cy = int(np.clip(cy, 0, mask.shape[0] - 1))
        center_label = int(labels[cy, cx])
        if center_label > 0 and stats[center_label, cv2.CC_STAT_AREA] >= self.min_component_pixels:
            return labels == center_label
        best_label = 1
        best_area = -1
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= self.min_component_pixels and area > best_area:
                best_area = area
                best_label = label
        return labels == best_label

    def refine_object_masks(self, obj_mask_raw: np.ndarray, roi_mask_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask_u8 = obj_mask_raw.astype(np.uint8) * 255
        core_k = self._odd_kernel(self.core_erode_kernel)
        safe_k = self._odd_kernel(self.safe_dilate_kernel)
        core_kernel = np.ones((core_k, core_k), dtype=np.uint8)
        safe_kernel = np.ones((safe_k, safe_k), dtype=np.uint8)
        obj_mask_core = cv2.erode(mask_u8, core_kernel, iterations=max(1, self.core_erode_iterations)) > 0
        if int(np.count_nonzero(obj_mask_core)) < self.min_object_core_pixels:
            obj_mask_core = obj_mask_raw.copy()
        obj_mask_safe = cv2.dilate(obj_mask_core.astype(np.uint8) * 255, safe_kernel, iterations=max(1, self.safe_dilate_iterations)) > 0
        obj_mask_safe &= roi_mask_2d
        return obj_mask_core, obj_mask_safe

    def make_strict_target_mask(self, object_core_mask: np.ndarray) -> np.ndarray:
        if not np.any(object_core_mask):
            return object_core_mask.copy()
        dist = cv2.distanceTransform(object_core_mask.astype(np.uint8), cv2.DIST_L2, 5)
        strict_mask = dist >= float(self.target_boundary_margin_px)
        target_mask = strict_mask if np.any(strict_mask) else object_core_mask.copy()
        k = self._odd_kernel(self.target_erode_kernel)
        kernel = np.ones((k, k), dtype=np.uint8)
        eroded = cv2.erode(target_mask.astype(np.uint8) * 255, kernel, iterations=max(1, self.target_erode_iterations)) > 0
        if np.count_nonzero(eroded) >= self.min_target_pixels:
            target_mask = eroded
        return target_mask

    @staticmethod
    def _odd_kernel(k: int) -> int:
        k = max(1, int(k))
        return k if (k % 2 == 1) else (k + 1)

    # ============================================================
    # Point cloud helpers
    # ============================================================
    def depth_to_meters(self, depth_cv: np.ndarray) -> np.ndarray:
        return depth_cv.astype(np.float32) * float(self.depth_scale)

    def build_sampled_points(self, color_rgb: np.ndarray, depth_cv: np.ndarray, camera_info: CameraInfo):
        depth_m = self.depth_to_meters(depth_cv)
        h, w = depth_m.shape[:2]
        k = camera_info.k
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        stride = max(1, int(self.pixel_stride))
        us = np.arange(0, w, stride, dtype=np.int32)
        vs = np.arange(0, h, stride, dtype=np.int32)
        uu, vv = np.meshgrid(us, vs)
        z = depth_m[vv, uu]
        valid = np.isfinite(z) & (z >= self.depth_min_m) & (z <= self.depth_max_m)
        if not np.any(valid):
            return {'xyz': np.zeros((0, 3), dtype=np.float32), 'rgb': np.zeros((0, 3), dtype=np.uint8), 'u': np.zeros((0,), dtype=np.int32), 'v': np.zeros((0,), dtype=np.int32)}
        uu = uu[valid].astype(np.float32)
        vv = vv[valid].astype(np.float32)
        z = z[valid].astype(np.float32)
        x = (uu - cx) * z / fx
        y = (vv - cy) * z / fy
        xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        u_i = uu.astype(np.int32)
        v_i = vv.astype(np.int32)
        rgb = color_rgb[v_i, u_i].astype(np.uint8)
        return {'xyz': xyz, 'rgb': rgb, 'u': u_i, 'v': v_i}

    @staticmethod
    def merge_xyz_rgb(xyz: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        return np.concatenate([xyz.astype(np.float32), rgb.astype(np.float32)], axis=1)

    @staticmethod
    def make_xyzrgb_tuples(xyz: np.ndarray, rgb: np.ndarray):
        if xyz.shape[0] == 0:
            return []
        return [(float(p[0]), float(p[1]), float(p[2]), int(c[0]), int(c[1]), int(c[2])) for p, c in zip(xyz, rgb)]

    def make_xyz_cloud(self, frame_id: str, stamp, xyz: np.ndarray) -> PointCloud2:
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        header = self._make_header(frame_id, stamp)
        points = [(float(p[0]), float(p[1]), float(p[2])) for p in xyz]
        return point_cloud2.create_cloud(header, fields, points)

    def make_xyzrgb_cloud(self, frame_id: str, stamp, xyzrgb_tuples) -> PointCloud2:
        header = self._make_header(frame_id, stamp)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        packed = []
        for x, y, z, r, g, b in xyzrgb_tuples:
            rgb = (int(r) << 16) | (int(g) << 8) | int(b)
            packed.append((float(x), float(y), float(z), int(rgb)))
        return point_cloud2.create_cloud(header, fields, packed)

    @staticmethod
    def _make_header(frame_id: str, stamp):
        from std_msgs.msg import Header
        h = Header()
        h.frame_id = frame_id
        h.stamp = stamp
        return h

    @staticmethod
    def depth_to_vis(depth_cv: np.ndarray) -> np.ndarray:
        d = depth_cv.astype(np.float32)
        valid = np.isfinite(d) & (d > 0)
        if not np.any(valid):
            return np.zeros_like(depth_cv, dtype=np.uint8)
        vmin = float(np.min(d[valid]))
        vmax = float(np.max(d[valid]))
        if vmax <= vmin:
            vmax = vmin + 1.0
        out = np.zeros_like(d, dtype=np.uint8)
        out[valid] = np.clip(255.0 * (d[valid] - vmin) / (vmax - vmin), 0, 255).astype(np.uint8)
        return out

    @staticmethod
    def save_camera_info_yaml(camera_info: CameraInfo, yaml_path: str):
        data = {
            'width': int(camera_info.width),
            'height': int(camera_info.height),
            'k': list(camera_info.k),
            'd': list(camera_info.d),
            'r': list(camera_info.r),
            'p': list(camera_info.p),
            'distortion_model': camera_info.distortion_model,
            'frame_id': camera_info.header.frame_id,
        }
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(to_builtin(data), f, sort_keys=False, allow_unicode=True)


def main(args=None):
    rclpy.init(args=args)
    node = Sam3Master2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()