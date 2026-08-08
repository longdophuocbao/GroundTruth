import os
import sys
import time
import csv
import json
from datetime import datetime
import numpy as np
import cv2
import pyrealsense2 as rs
import pyqtgraph as pg

from PySide6.QtCore import Qt, QThread, Signal, Slot, QMutex, QMutexLocker
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                             QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, 
                             QGroupBox, QComboBox, QFileDialog, QMessageBox, QCheckBox,
                             QDoubleSpinBox, QSpinBox, QDialog)
from PySide6.QtGui import QImage, QPixmap, QFont, QColor, QIcon

class TagTracker:
    def __init__(self, tag_id, alpha=0.7, max_lost_frames=15):
        self.tag_id = tag_id
        self.alpha = alpha
        self.max_lost_frames = max_lost_frames
        
        self.pos_pnp_filtered = None
        self.pos_depth_filtered = None
        self.corners_filtered = None
        self.rvec_filtered = None
        self.tvec_filtered = None
        
        self.lost_frames = 0
        self.is_tracked = False

    def update(self, corners, pos_pnp, pos_depth, rvec, tvec, alpha=0.7, max_lost_frames=15):
        self.alpha = alpha
        self.max_lost_frames = max_lost_frames
        self.lost_frames = 0
        self.is_tracked = True
        
        if self.pos_pnp_filtered is None:
            self.pos_pnp_filtered = np.array(pos_pnp, dtype=np.float32)
            self.pos_depth_filtered = np.array(pos_depth, dtype=np.float32)
            self.corners_filtered = np.array(corners, dtype=np.float32)
            self.rvec_filtered = np.array(rvec, dtype=np.float32) if rvec is not None else None
            self.tvec_filtered = np.array(tvec, dtype=np.float32) if tvec is not None else None
        else:
            self.pos_pnp_filtered = self.alpha * np.array(pos_pnp, dtype=np.float32) + (1.0 - self.alpha) * self.pos_pnp_filtered
            self.pos_depth_filtered = self.alpha * np.array(pos_depth, dtype=np.float32) + (1.0 - self.alpha) * self.pos_depth_filtered
            self.corners_filtered = self.alpha * np.array(corners, dtype=np.float32) + (1.0 - self.alpha) * self.corners_filtered
            
            if rvec is not None and self.rvec_filtered is not None:
                self.rvec_filtered = self.alpha * np.array(rvec, dtype=np.float32) + (1.0 - self.alpha) * self.rvec_filtered
            elif rvec is not None:
                self.rvec_filtered = np.array(rvec, dtype=np.float32)
                
            if tvec is not None and self.tvec_filtered is not None:
                self.tvec_filtered = self.alpha * np.array(tvec, dtype=np.float32) + (1.0 - self.alpha) * self.tvec_filtered
            elif tvec is not None:
                self.tvec_filtered = np.array(tvec, dtype=np.float32)

    def predict(self, max_lost_frames=15):
        self.max_lost_frames = max_lost_frames
        self.lost_frames += 1
        if self.lost_frames >= self.max_lost_frames:
            self.is_tracked = False

class CameraWorker(QThread):
    frame_ready = Signal(np.ndarray, dict) # Emits the annotated BGR/RGB frame and results dictionary
    error_occurred = Signal(str)
    status_msg = Signal(str)
    calibration_complete = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self._mutex = QMutex()
        
        # Thread-safe config parameters
        self._tag_size = 0.150 # meters (150mm)
        self._source_id = 1
        self._target_ids_str = ""
        self._coord_mode = "depth" # "pnp" or "depth"
        
        # Tracking config
        self._filter_alpha = 0.7
        self._max_lost_frames = 15
        self._enable_smoothing = True
        self._enable_keep_alive = True
        
        # RealSense SDK post-processing filters config
        self._enable_decimation = False
        self._enable_hole_filling = False
        self._enable_spatial = False
        self._enable_temporal = False
        self._enable_threshold = False
        self._threshold_min = 1.0
        self._threshold_max = 4.0
        self._laser_power = 150
        
        # Reference Map state
        self._use_ref_map = False
        self._reference_map = None
        self._use_imu = True
        self._filtered_R_A = None
        self._filtered_P_A = None
        
        # Calibration state
        self._calibration_active = False
        self._calibration_frames_collected = 0
        self._calibration_data_p = {}
        self._calibration_data_r = {}
        self._calibration_anchor_id = None
        
        self._trackers = {}
        self._distance_history = []
        self._window_size = 10
        
        # Logger state
        self._log_file_path = ""
        self._is_logging = False
        self._log_file = None
        self._log_writer = None
        self._log_start_time = 0.0

    # Thread-safe getters/setters
    def update_config(self, tag_size, source_id, target_ids_str, coord_mode,
                      filter_alpha, max_lost_frames, enable_smoothing, enable_keep_alive, window_size,
                      enable_decimation, enable_hole_filling, enable_spatial, enable_temporal, enable_threshold,
                      threshold_min, threshold_max, laser_power, use_ref_map, reference_map, use_imu):
        with QMutexLocker(self._mutex):
            self._tag_size = tag_size
            self._source_id = source_id
            self._target_ids_str = target_ids_str
            self._coord_mode = coord_mode
            self._filter_alpha = filter_alpha
            self._max_lost_frames = max_lost_frames
            self._enable_smoothing = enable_smoothing
            self._enable_keep_alive = enable_keep_alive
            self._window_size = window_size
            self._enable_decimation = enable_decimation
            self._enable_hole_filling = enable_hole_filling
            self._enable_spatial = enable_spatial
            self._enable_temporal = enable_temporal
            self._enable_threshold = enable_threshold
            self._threshold_min = threshold_min
            self._threshold_max = threshold_max
            self._laser_power = laser_power
            self._use_ref_map = use_ref_map
            self._reference_map = reference_map
            self._use_imu = use_imu
            self._filtered_R_A = None
            self._filtered_P_A = None

    def start_calibration(self, reset=False):
        with QMutexLocker(self._mutex):
            self._calibration_active = True
            self._calibration_frames_collected = 0
            self._calibration_data_p = {}
            self._calibration_data_r = {}
            
            # Khởi tạo lại bản đồ nếu yêu cầu làm mới
            if reset:
                self._reference_map = None
                self._calibration_anchor_id = None
                
            self._filtered_R_A = None
            self._filtered_P_A = None

    def stop(self):
        self.running = False

    def start_logging(self, file_path):
        with QMutexLocker(self._mutex):
            try:
                self._log_file_path = file_path
                # Open CSV and write header
                self._log_file = open(file_path, mode='w', newline='', encoding='utf-8')
                self._log_writer = csv.writer(self._log_file)
                self._log_writer.writerow([
                    "Timestamp_Sec", "Time_Formatted", 
                    "Source_ID", "Source_X_mm", "Source_Y_mm", "Source_Z_mm",
                    "Num_Targets", "Polyline_Dist_mm",
                    "Target_IDs", "Selected_Coord_Mode"
                ])
                self._is_logging = True
                self._log_start_time = time.time()
                return True
            except Exception as e:
                self._is_logging = False
                if self._log_file:
                    self._log_file.close()
                    self._log_file = None
                raise e

    def stop_logging(self):
        with QMutexLocker(self._mutex):
            self._is_logging = False
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            self._log_file_path = ""

    def run(self):
        self.running = True
        self.status_msg.emit("Đang quét kết nối thiết bị RealSense...")
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                self.error_occurred.emit("Không tìm thấy camera Intel RealSense nào đang kết nối. Vui lòng kiểm tra lại cáp cắm.")
                self.running = False
                return
        except Exception as e:
            self.error_occurred.emit(f"Lỗi khi kiểm tra thiết bị: {str(e)}")
            self.running = False
            return
            
        self.status_msg.emit("Đang khởi tạo camera RealSense...")
        
        # Create pipeline and config
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure streams (D455 supports 640x480 for color and depth)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Try to enable IMU streams (accel and gyro)
        enable_imu = True
        try:
            config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
            config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
        except Exception as imu_err:
            print(f"Không thể kích hoạt luồng IMU: {imu_err}. Chạy không có IMU.")
            enable_imu = False
            
        align = rs.align(rs.stream.color)
        
        try:
            try:
                profile = pipeline.start(config)
            except Exception as start_err:
                if enable_imu:
                    # Retry without IMU streams
                    pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    profile = pipeline.start(config)
                    enable_imu = False
                    self.status_msg.emit("Khởi động camera thành công (Đã bỏ qua luồng IMU).")
                else:
                    raise start_err
                    
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            # --- CẤU HÌNH TỐI ƯU NGOÀI TRỜI ---                                                                
            if depth_sensor:                                                                                    
                # 1. Tắt Emitter để tránh nhiễu ánh sáng mặt trời                                               
                if depth_sensor.supports(rs.option.emitter_enabled):                                            
                    depth_sensor.set_option(rs.option.emitter_enabled, 0.0)                                     
                                                                                                                
                # 2. Đặt Preset về High Accuracy (3) hoặc Remove IR Pattern (6)                                 
                if depth_sensor.supports(rs.option.visual_preset):                                              
                    depth_sensor.set_option(rs.option.visual_preset, 3)                                         
            # ----------------------------------                     
            self.status_msg.emit("Đang kết nối. Pipeline bắt đầu hoạt động.")
        except Exception as e:
            self.error_occurred.emit(f"Không thể khởi động camera: {str(e)}")
            self.running = False
            return
            
        try:
            # Retrieve camera intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            cam_matrix = np.array([
                [intrinsics.fx, 0, intrinsics.ppx],
                [0, intrinsics.fy, intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float32)
            dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)
            
            # Initialize AprilTag detector (using Aruco module in OpenCV 5)
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            
            # Initialize RealSense SDK post-processing filters
            decimate_filter = rs.decimation_filter()
            spatial_filter = rs.spatial_filter()
            temporal_filter = rs.temporal_filter()
            hole_filling_filter = rs.hole_filling_filter()
            threshold_filter = rs.threshold_filter()
            colorizer = rs.colorizer()
            
            # Wait for sensors to warm up/auto-exposure to stabilize
            for _ in range(10):
                if not self.running:
                    break
                pipeline.wait_for_frames()
                
            self.status_msg.emit("Camera đang phát truyền hình trực tiếp.")
            
            # Reset trackers and history
            self._trackers = {}
            self._distance_history.clear()
            self._filtered_R_A = None
            self._filtered_P_A = None
            last_time = time.time()
            
            while self.running:
                frames = pipeline.wait_for_frames()
                
                # Align depth frame to color frame
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                    
                # Get IMU data if enabled
                gyro_data = None
                if enable_imu:
                    try:
                        gyro_frame = frames.first_or_default(rs.stream.gyro)
                        if gyro_frame:
                            motion_data = gyro_frame.as_motion_frame().get_motion_data()
                            gyro_data = [motion_data.x, motion_data.y, motion_data.z]
                    except Exception:
                        pass
                        
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                if dt <= 0 or dt > 0.5:
                    dt = 0.033
                    
                # Convert frames to numpy arrays
                color_image = np.asanyarray(color_frame.get_data()).copy()
                
                # Read latest config variables in a thread-safe way
                with QMutexLocker(self._mutex):
                    tag_size = self._tag_size
                    source_id = self._source_id
                    target_ids_str = self._target_ids_str
                    coord_mode = self._coord_mode
                    is_logging = self._is_logging
                    filter_alpha = self._filter_alpha
                    max_lost_frames = self._max_lost_frames
                    enable_smoothing = self._enable_smoothing
                    enable_keep_alive = self._enable_keep_alive
                    enable_decimation = self._enable_decimation
                    enable_hole_filling = self._enable_hole_filling
                    enable_spatial = self._enable_spatial
                    enable_temporal = self._enable_temporal
                    enable_threshold = self._enable_threshold
                    threshold_min = self._threshold_min
                    threshold_max = self._threshold_max
                    laser_power = self._laser_power
                    use_ref_map = self._use_ref_map
                    reference_map = self._reference_map
                    use_imu = self._use_imu
                
                # Apply Laser Power dynamically
                if depth_sensor and depth_sensor.supports(rs.option.laser_power):
                    try:
                        current_val = depth_sensor.get_option(rs.option.laser_power)
                        if abs(current_val - laser_power) > 0.01:
                            depth_sensor.set_option(rs.option.laser_power, float(laser_power))
                    except Exception:
                        pass
                
                # Apply RealSense SDK post-processing filters to depth_frame
                if enable_threshold:
                    threshold_filter.set_option(rs.option.min_distance, threshold_min)
                    threshold_filter.set_option(rs.option.max_distance, threshold_max)
                    depth_frame = threshold_filter.process(depth_frame)
                if enable_decimation:
                    depth_frame = decimate_filter.process(depth_frame)
                if enable_spatial:
                    depth_frame = spatial_filter.process(depth_frame)
                if enable_temporal:
                    depth_frame = temporal_filter.process(depth_frame)
                if enable_hole_filling:
                    depth_frame = hole_filling_filter.process(depth_frame)
                
                # Cast the processed frame back to a depth_frame to expose get_width(), get_height(), get_distance()
                if depth_frame:
                    depth_frame = depth_frame.as_depth_frame()
                if not depth_frame:
                    continue
                
                # Grayscale for AprilTag detection
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = detector.detectMarkers(gray)
                
                # Setup 3D points for pose estimation (PnP)
                s = tag_size
                obj_points = np.array([
                    [-s/2, -s/2, 0], # Top-Left
                    [ s/2, -s/2, 0], # Top-Right
                    [ s/2,  s/2, 0], # Bottom-Right
                    [-s/2,  s/2, 0]  # Bottom-Left
                ], dtype=np.float32)
                
                detected_this_frame = set()
                detected_tags_raw = {}
                
                if ids is not None:
                    flat_ids = np.ravel(ids)
                    for i, tag_id_val in enumerate(flat_ids):
                        tag_id = int(tag_id_val)
                        corners_i = np.array(corners[i], dtype=np.float32).reshape(4, 2)
                        
                        # Calculate 2D center
                        u_c = float(np.mean(corners_i[:, 0]))
                        v_c = float(np.mean(corners_i[:, 1]))
                        
                        # Solve PnP
                        success, rvec, tvec = cv2.solvePnP(obj_points, corners_i, cam_matrix, dist_coeffs)
                        if success:
                            p_3d_pnp = tvec.flatten()
                        else:
                            p_3d_pnp = np.array([0.0, 0.0, 0.0])
                            rvec, tvec = None, None
                        
                        # Retrieve 3D point via Depth sensor
                        depth_val = 0.0
                        depth_w = depth_frame.get_width()
                        depth_h = depth_frame.get_height()
                        color_h, color_w = color_image.shape[:2]
                        
                        u_c_depth = int(u_c * depth_w / color_w)
                        v_c_depth = int(v_c * depth_h / color_h)
                        
                        depths = []
                        for dy in range(-2, 3):
                            for dx in range(-2, 3):
                                nu, nv = u_c_depth + dx, v_c_depth + dy
                                if 0 <= nu < depth_w and 0 <= nv < depth_h:
                                    d = depth_frame.get_distance(nu, nv)
                                    if d > 0.0:
                                        depths.append(d)
                        if depths:
                            depth_val = float(np.median(depths))
                            
                        # Deproject depth-based center
                        if depth_val > 0.0:
                            p_3d_depth = np.array(rs.rs2_deproject_pixel_to_point(intrinsics, [u_c, v_c], depth_val))
                        else:
                            p_3d_depth = np.array([0.0, 0.0, 0.0])
                            
                        detected_tags_raw[tag_id] = {
                            'corners': corners_i,
                            'center_2d': (u_c, v_c),
                            'rvec': rvec,
                            'tvec': tvec,
                            'pos_pnp': p_3d_pnp,
                            'pos_depth': p_3d_depth,
                            'depth': depth_val
                        }
                        
                        # Update or create tracker
                        if tag_id not in self._trackers:
                            self._trackers[tag_id] = TagTracker(tag_id, alpha=filter_alpha, max_lost_frames=max_lost_frames)
                        
                        self._trackers[tag_id].update(corners_i, p_3d_pnp, p_3d_depth, rvec, tvec, 
                                                      alpha=filter_alpha, max_lost_frames=max_lost_frames)
                        detected_this_frame.add(tag_id)
                
                # Predict for trackers not detected in this frame
                for tid, tracker in list(self._trackers.items()):
                    if tid not in detected_this_frame:
                        tracker.predict(max_lost_frames=max_lost_frames)
                        
                # Filter active tags for calculations and drawing
                active_tags = {}
                for tid, tracker in self._trackers.items():
                    if not tracker.is_tracked:
                        continue
                    if not enable_keep_alive and tracker.lost_frames > 0:
                        continue
                        
                    # Choose raw or filtered values
                    if enable_smoothing:
                        pos_pnp = tracker.pos_pnp_filtered
                        pos_depth = tracker.pos_depth_filtered
                        corners_i = tracker.corners_filtered
                        rvec = tracker.rvec_filtered
                        tvec = tracker.tvec_filtered
                    else:
                        if tracker.lost_frames == 0 and tid in detected_tags_raw:
                            raw = detected_tags_raw[tid]
                            pos_pnp = raw['pos_pnp']
                            pos_depth = raw['pos_depth']
                            corners_i = raw['corners']
                            rvec = raw['rvec']
                            tvec = raw['tvec']
                        else:
                            pos_pnp = tracker.pos_pnp_filtered
                            pos_depth = tracker.pos_depth_filtered
                            corners_i = tracker.corners_filtered
                            rvec = tracker.rvec_filtered
                            tvec = tracker.tvec_filtered
                            
                    active_tags[tid] = {
                        'corners': corners_i,
                        'center_2d': (float(np.mean(corners_i[:, 0])), float(np.mean(corners_i[:, 1]))),
                        'rvec': rvec,
                        'tvec': tvec,
                        'pos_pnp': pos_pnp,
                        'pos_depth': pos_depth,
                        'lost_frames': tracker.lost_frames
                    }
                
                # 1. Reference Map Calibration Accumulation (Cuốn Chiếu)
                if self._calibration_active:
                    if target_ids_str.strip() == "":
                        target_ids = [tid for tid in sorted(active_tags.keys()) if tid != source_id]
                    else:
                        try:
                            parsed_ids = [int(x.strip()) for x in target_ids_str.split(",") if x.strip().isdigit()]
                            target_ids = [tid for tid in parsed_ids if tid in active_tags]
                        except Exception:
                            target_ids = [tid for tid in sorted(active_tags.keys()) if tid != source_id]
                            
                    detected_targets = [tid for tid in target_ids if tid in active_tags and not active_tags[tid]['lost_frames'] > 0]
                    if len(detected_targets) > 0:
                        # Dùng tag đầu tiên quét được làm mỏ neo tạm thời cho bản đồ cục bộ
                        temp_anchor = detected_targets[0]
                        
                        if temp_anchor in active_tags and active_tags[temp_anchor]['rvec'] is not None:
                            anchor_tag = active_tags[temp_anchor]
                            pos_a = anchor_tag['pos_pnp'] if coord_mode == 'pnp' else anchor_tag['pos_depth']
                            R_a, _ = cv2.Rodrigues(anchor_tag['rvec'])
                            
                            for tid in detected_targets:
                                tag = active_tags[tid]
                                if tag['rvec'] is not None:
                                    pos_t = tag['pos_pnp'] if coord_mode == 'pnp' else tag['pos_depth']
                                    R_t, _ = cv2.Rodrigues(tag['rvec'])
                                    
                                    # Tính toán tọa độ cục bộ (Local coordinates)
                                    p_local = R_a.T @ (pos_t - pos_a)
                                    r_local = R_a.T @ R_t
                                    
                                    if tid not in self._calibration_data_p:
                                        self._calibration_data_p[tid] = []
                                        self._calibration_data_r[tid] = []
                                    self._calibration_data_p[tid].append(p_local)
                                    self._calibration_data_r[tid].append(r_local)
                                    
                            self._calibration_frames_collected += 1
                            self.status_msg.emit(f"Đang quét bản đồ cục bộ... Khung hình {self._calibration_frames_collected}/60")
                            
                            if self._calibration_frames_collected >= 60:
                                local_map = {}
                                for tid, plist in self._calibration_data_p.items():
                                    if len(plist) > 0:
                                        p_avg = np.mean(plist, axis=0)
                                        rlist = self._calibration_data_r[tid]
                                        M = np.mean(rlist, axis=0)
                                        U, S, Vt = np.linalg.svd(M)
                                        r_avg = U @ Vt
                                        if np.linalg.det(r_avg) < 0:
                                            r_avg = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
                                            
                                        local_map[str(tid)] = {
                                            'p_local': p_avg.tolist(),
                                            'r_local': r_avg.tolist()
                                        }
                                        
                                # LƯU TRỮ VÀ GHÉP BẢN ĐỒ (MAP STITCHING)
                                if getattr(self, '_reference_map', None) is None:
                                    self._reference_map = {}
                                    
                                if not self._reference_map:
                                    # Nếu bản đồ tổng trống, lấy bản đồ cục bộ làm bản đồ tổng
                                    self._reference_map = local_map
                                    self._calibration_anchor_id = temp_anchor
                                else:
                                    # Tìm các tag trùng lặp giữa bản đồ hiện tại và bản đồ tổng
                                    common_ids = [tid for tid in local_map.keys() if tid in self._reference_map.keys()]
                                    
                                    if len(common_ids) > 0:
                                        # Tính ma trận chuyển đổi từ Local Map sang Global Map
                                        if len(common_ids) >= 3:
                                            # Dùng thuật toán Kabsch nếu có từ 3 tag trùng trở lên (chính xác nhất)
                                            P_global = [self._reference_map[cid]['p_local'] for cid in common_ids]
                                            Q_local = [local_map[cid]['p_local'] for cid in common_ids]
                                            R_L2G, T_L2G = self._kabsch_alignment(np.array(Q_local), np.array(P_global))
                                        else:
                                            # Dùng 1 hoặc 2 tag trùng lặp (Dựa vào thông tin Rotation của AprilTag)
                                            cid = common_ids[0]
                                            p_g = np.array(self._reference_map[cid]['p_local'])
                                            R_g = np.array(self._reference_map[cid]['r_local'])
                                            p_l = np.array(local_map[cid]['p_local'])
                                            R_l = np.array(local_map[cid]['r_local'])
                                            
                                            R_L2G = R_g @ R_l.T
                                            T_L2G = p_g - R_L2G @ p_l
                                            
                                        # Biến đổi các tag MỚI từ Local sang Global và thêm vào Reference Map
                                        for tid, data in local_map.items():
                                            if tid not in self._reference_map:
                                                p_l_new = np.array(data['p_local'])
                                                R_l_new = np.array(data['r_local'])
                                                
                                                p_g_new = R_L2G @ p_l_new + T_L2G
                                                R_g_new = R_L2G @ R_l_new
                                                
                                                self._reference_map[tid] = {
                                                    'p_local': p_g_new.tolist(),
                                                    'r_local': R_g_new.tolist()
                                                }
                                    else:
                                        self.status_msg.emit("Không tìm thấy tag trùng lặp! Không thể nối bản đồ.")
                                        self._calibration_active = False
                                        continue

                                calibration_result = {
                                    'anchor_id': int(self._calibration_anchor_id) if getattr(self, '_calibration_anchor_id', None) else int(temp_anchor),
                                    'map': self._reference_map
                                }
                                self.calibration_complete.emit(calibration_result)
                                self._calibration_active = False
                                self.status_msg.emit(f"Cập nhật thành công! Tổng cộng {len(self._reference_map)} Tag trong bản đồ.")
                
                # 2. Reference Map Optimization & Virtual Tag Reconstruction (Handheld Mode with IMU complementary filter)
                if use_ref_map and reference_map is not None:
                    detected_mapped_ids = []
                    for tid_str, ref_data in reference_map.items():
                        tid = int(tid_str)
                        if tid in active_tags and not active_tags[tid]['lost_frames'] > 0:
                            if active_tags[tid]['rvec'] is not None:
                                detected_mapped_ids.append(tid)
                                
                    if len(detected_mapped_ids) > 0:
                        # 1. Estimate measured pose
                        if len(detected_mapped_ids) >= 3:
                            P_pts = []
                            Q_pts = []
                            for tid in detected_mapped_ids:
                                ref_data = reference_map[str(tid)]
                                P_pts.append(ref_data['p_local'])
                                tag = active_tags[tid]
                                pos_t = tag['pos_pnp'] if coord_mode == 'pnp' else tag['pos_depth']
                                Q_pts.append(pos_t)
                                
                            R_A_meas, P_A_meas = self._kabsch_alignment(np.array(P_pts), np.array(Q_pts))
                        else:
                            tid = detected_mapped_ids[0]
                            tag = active_tags[tid]
                            pos_t = tag['pos_pnp'] if coord_mode == 'pnp' else tag['pos_depth']
                            R_t, _ = cv2.Rodrigues(tag['rvec'])
                            
                            ref_data = reference_map[str(tid)]
                            p_local = np.array(ref_data['p_local'])
                            r_local = np.array(ref_data['r_local'])
                            
                            R_A_meas = R_t @ r_local.T
                            P_A_meas = pos_t - R_A_meas @ p_local
                            
                        # 2. IMU complementary filter / temporal fusion
                        if self._filtered_R_A is None:
                            self._filtered_R_A = R_A_meas
                            self._filtered_P_A = P_A_meas
                        else:
                            # Apply IMU gyro prediction if available and enabled
                            if use_imu and gyro_data is not None:
                                w = np.array(gyro_data)
                                theta = np.linalg.norm(w) * dt
                                if theta > 1e-5:
                                    axis = w / np.linalg.norm(w)
                                    rvec_gyro = axis * theta
                                    dR, _ = cv2.Rodrigues(rvec_gyro)
                                    R_pred = dR @ self._filtered_R_A
                                    P_pred = dR @ self._filtered_P_A
                                else:
                                    R_pred = self._filtered_R_A
                                    P_pred = self._filtered_P_A
                            else:
                                R_pred = self._filtered_R_A
                                P_pred = self._filtered_P_A
                                
                            # Fusion weights
                            alpha_rot = 0.9   # responsive rotation
                            alpha_trans = 0.9 # responsive translation
                            
                            # Fuse translations
                            self._filtered_P_A = (1 - alpha_trans) * P_pred + alpha_trans * P_A_meas
                            
                            # Fuse rotations using SVD
                            M_R = (1 - alpha_rot) * R_pred + alpha_rot * R_A_meas
                            U, S, Vt = np.linalg.svd(M_R)
                            R_smooth = U @ Vt
                            if np.linalg.det(R_smooth) < 0:
                                R_smooth = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
                            self._filtered_R_A = R_smooth
                            
                        R_A_opt = self._filtered_R_A
                        P_A_opt = self._filtered_P_A
                        
                        # 3. Reconstruct all tags in the reference map
                        for tid_str, ref_data in reference_map.items():
                            tid = int(tid_str)
                            p_local = np.array(ref_data['p_local'])
                            r_local = np.array(ref_data['r_local'])
                            
                            pos_reconstructed = R_A_opt @ p_local + P_A_opt
                            R_reconstructed = R_A_opt @ r_local
                            rvec_reconstructed, _ = cv2.Rodrigues(R_reconstructed)
                            
                            if tid in active_tags:
                                active_tags[tid]['pos_pnp'] = pos_reconstructed
                                active_tags[tid]['pos_depth'] = pos_reconstructed
                                active_tags[tid]['rvec'] = rvec_reconstructed
                                active_tags[tid]['tvec'] = pos_reconstructed.reshape(3, 1)
                            else:
                                s = tag_size
                                local_corners = np.array([
                                    [-s/2, -s/2, 0],
                                    [ s/2, -s/2, 0],
                                    [ s/2,  s/2, 0],
                                    [-s/2,  s/2, 0]
                                ], dtype=np.float32)
                                cam_corners = (R_reconstructed @ local_corners.T).T + pos_reconstructed
                                pts_2d = self._project_points_3d_to_2d(cam_corners, cam_matrix, dist_coeffs)
                                
                                active_tags[tid] = {
                                    'corners': pts_2d,
                                    'center_2d': (float(np.mean(pts_2d[:, 0])), float(np.mean(pts_2d[:, 1]))),
                                    'rvec': rvec_reconstructed,
                                    'tvec': pos_reconstructed.reshape(3, 1),
                                    'pos_pnp': pos_reconstructed,
                                    'pos_depth': pos_reconstructed,
                                    'lost_frames': 1,
                                    'is_virtual': True
                                }
                    else:
                        # Reset filter if no tags are detected
                        self._filtered_R_A = None
                        self._filtered_P_A = None
                
                # Process measurements
                results = {
                    'detected_tags': {},
                    'source_detected': False,
                    'num_targets': 0,
                    'polyline_dist': None,
                    'fitted_line_dist': None,
                    'selected_mode': coord_mode
                }
                
                # Check for source tag
                source_pos = None
                if source_id in active_tags:
                    results['source_detected'] = True
                    src_tag = active_tags[source_id]
                    source_pos = src_tag['pos_pnp'] if coord_mode == 'pnp' else src_tag['pos_depth']
                
                # Parse target IDs
                target_ids = []
                if target_ids_str.strip() == "":
                    target_ids = [tid for tid in sorted(active_tags.keys()) if tid != source_id]
                else:
                    try:
                        parsed_ids = [int(x.strip()) for x in target_ids_str.split(",") if x.strip().isdigit()]
                        target_ids = [tid for tid in parsed_ids if tid in active_tags]
                    except Exception:
                        target_ids = [tid for tid in sorted(active_tags.keys()) if tid != source_id]

                # Map detected tag details to results
                for tid, tag in active_tags.items():
                    role = "Nguồn" if tid == source_id else ("Đường mục tiêu" if tid in target_ids else "Không sử dụng")
                    status_text = "Đang phát hiện" if tag['lost_frames'] == 0 else f"Bám vết (mất {tag['lost_frames']}f)"
                    results['detected_tags'][tid] = {
                        'pos_pnp': tag['pos_pnp'] * 1000.0,
                        'pos_depth': tag['pos_depth'] * 1000.0,
                        'role': role,
                        'status': status_text,
                        'lost_frames': tag['lost_frames']
                    }

                # Extract 3D points of targets
                target_pts = []
                target_map = {}
                for tid in target_ids:
                    tag = active_tags[tid]
                    pos = tag['pos_pnp'] if coord_mode == 'pnp' else tag['pos_depth']
                    target_pts.append(pos)
                    target_map[tid] = pos
                
                results['num_targets'] = len(target_pts)
                
                # Draw tag boundaries and info
                for tid, tag in active_tags.items():
                    corners_i = tag['corners'].astype(np.int32)
                    is_source = (tid == source_id)
                    is_lost = tag['lost_frames'] > 0
                    
                    if is_lost:
                        color = (0, 165, 255) # Orange for lost-but-tracked
                        thickness = 1
                    else:
                        color = (0, 0, 255) if is_source else ((0, 255, 255) if tid in target_ids else (128, 128, 128))
                        thickness = 3 if is_source or tid in target_ids else 1
                    
                    # Draw boundary
                    if is_lost:
                        self._draw_dashed_polygon(color_image, tag['corners'], color, thickness=1, dash_len=5)
                    else:
                        for k in range(4):
                            cv2.line(color_image, tuple(corners_i[k]), tuple(corners_i[(k+1)%4]), color, thickness)
                        
                    # Write ID text
                    cx, cy = int(tag['center_2d'][0]), int(tag['center_2d'][1])
                    label_text = f"ID:{tid} (Lost)" if is_lost else f"ID:{tid}"
                    cv2.putText(color_image, label_text, (cx - 25, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 if is_lost else 2, cv2.LINE_AA)
                    
                    # Draw 3D axis
                    if not is_lost and tag['rvec'] is not None and tag['tvec'] is not None:
                        cv2.drawFrameAxes(color_image, cam_matrix, dist_coeffs, 
                                          tag['rvec'], tag['tvec'], length=tag_size * 0.5, thickness=2)
                
                # Compute distance to plane of targets if source and targets are detected
                if source_pos is not None and len(target_pts) > 0:
                    # Perpendicular distance to the fitted plane of target tags
                    poly_dist, poly_closest_3d = self._distance_point_to_fitted_plane(source_pos, target_pts)
                    if poly_dist is not None:
                        raw_dist = poly_dist * 1000.0
                        
                        with QMutexLocker(self._mutex):
                            w_size = self._window_size
                        
                        # Append raw distance to history
                        self._distance_history.append(raw_dist)
                        if len(self._distance_history) > w_size:
                            self._distance_history.pop(0)
                        
                        # Compute moving average
                        filtered_dist = sum(self._distance_history) / len(self._distance_history)
                        results['polyline_dist'] = filtered_dist
                        
                        # Split target IDs into even and odd to represent the surface mesh
                        even_ids = [tid for tid in target_ids if tid % 2 == 0]
                        odd_ids = [tid for tid in target_ids if tid % 2 != 0]
                        
                        # Draw surface mesh connecting even and odd target tags
                        self._draw_surface_mesh(color_image, target_map, even_ids, odd_ids, cam_matrix, dist_coeffs)
                        # Draw shortest path vector perpendicular to the fitted plane
                        self._draw_distance_vector(color_image, source_pos, poly_closest_3d, 
                                                   cam_matrix, dist_coeffs, (0, 255, 0), "Khoảng cách", filtered_dist)
                else:
                    self._distance_history.clear()
                
                # Log data if active
                if is_logging:
                    self._write_log_row(results, source_id, source_pos, target_ids)
                
                # Generate colorized depth image to send to GUI
                colorized_depth = colorizer.colorize(depth_frame)
                depth_image = np.asanyarray(colorized_depth.get_data()).copy()
                results['depth_image'] = depth_image
                
                # Send frame and results to GUI
                self.frame_ready.emit(color_image, results)
                
            pipeline.stop()
            self.status_msg.emit("Đã dừng truyền hình. Camera ở trạng thái nghỉ.")
            
        except Exception as e:
            self.error_occurred.emit(f"Lỗi trong vòng lặp xử lý: {str(e)}")
            try:
                pipeline.stop()
            except Exception:
                pass
            self.running = False

    def _draw_dashed_polygon(self, image, corners, color, thickness=1, dash_len=5):
        # corners is shape (4, 2)
        for k in range(4):
            pt1 = corners[k]
            pt2 = corners[(k+1)%4]
            dist = np.linalg.norm(pt1 - pt2)
            if dist == 0:
                continue
            direction = (pt2 - pt1) / dist
            num_dashes = int(dist / (2 * dash_len))
            for i in range(num_dashes):
                s_pt = pt1 + direction * (i * 2 * dash_len)
                e_pt = pt1 + direction * ((i * 2 + 1) * dash_len)
                cv2.line(image, (int(s_pt[0]), int(s_pt[1])), (int(e_pt[0]), int(e_pt[1])), color, thickness, cv2.LINE_AA)

    # Mathematical algorithms for 3D distances
    def _distance_point_to_segment(self, p, a, b):
        ab = b - a
        ap = p - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq == 0.0:
            return np.linalg.norm(ap), a
            
        t = np.dot(ap, ab) / ab_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest_point = a + t * ab
        dist = np.linalg.norm(p - closest_point)
        return dist, closest_point

    def _distance_point_to_polyline(self, p, qs):
        if len(qs) == 0:
            return None, None
        if len(qs) == 1:
            dist = np.linalg.norm(p - qs[0])
            return dist, qs[0]
            
        min_dist = float('inf')
        best_closest = None
        for i in range(len(qs) - 1):
            dist, closest = self._distance_point_to_segment(p, qs[i], qs[i+1])
            if dist < min_dist:
                min_dist = dist
                best_closest = closest
        return min_dist, best_closest

    def _distance_point_to_fitted_plane(self, p, qs):
        """
        Fits a plane to a set of 3D points qs: ax + by + cz + d = 0
        and returns the perpendicular distance from point p to the plane, 
        along with the projected point on the plane.
        """
        N = len(qs)
        if N == 0:
            return None, None
        if N == 1:
            dist = np.linalg.norm(p - qs[0])
            return dist, qs[0]
        if N == 2:
            return self._distance_point_to_segment(p, qs[0], qs[1])
            
        pts = np.array(qs)
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        
        try:
            _, _, vh = np.linalg.svd(centered)
            normal = vh[2]
            normal = normal / np.linalg.norm(normal)
            
            # Perpendicular distance from p to the plane
            dist = abs(np.dot(normal, p - centroid))
            # Projection of p onto the plane
            p_proj = p - np.dot(normal, p - centroid) * normal
            return dist, p_proj
        except Exception:
            # Fallback to segment in case SVD has issues
            return self._distance_point_to_segment(p, qs[0], qs[-1])

    def _kabsch_alignment(self, P, Q):
        """
        Finds the optimal rotation R and translation T such that R @ P_i + T approx = Q_i.
        P: shape (N, 3) - reference points (local coordinates)
        Q: shape (N, 3) - detected points (camera coordinates)
        """
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)
        
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q
        
        H = P_centered.T @ Q_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
            
        T = centroid_Q - R @ centroid_P
        return R, T

    # Drawing helpers
    def _project_points_3d_to_2d(self, pts_3d, cam_matrix, dist_coeffs):
        pts_3d = np.array(pts_3d, dtype=np.float32).reshape(-1, 3)
        rvec = np.zeros(3, dtype=np.float32)
        tvec = np.zeros(3, dtype=np.float32)
        img_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, cam_matrix, dist_coeffs)
        return img_pts.reshape(-1, 2)

    def _draw_polyline(self, image, target_pts, cam_matrix, dist_coeffs):
        if len(target_pts) < 1:
            return
        pts_2d = self._project_points_3d_to_2d(target_pts, cam_matrix, dist_coeffs)
        
        # Draw polyline segments
        for k in range(len(pts_2d) - 1):
            pt1 = (int(pts_2d[k][0]), int(pts_2d[k][1]))
            pt2 = (int(pts_2d[k+1][0]), int(pts_2d[k+1][1]))
            cv2.line(image, pt1, pt2, (255, 144, 30), 3, cv2.LINE_AA) # Rich light blue
            
        # Draw vertices
        for pt in pts_2d:
            cv2.circle(image, (int(pt[0]), int(pt[1])), 6, (0, 255, 255), -1)

    def _draw_surface_mesh(self, image, target_map, even_ids, odd_ids, cam_matrix, dist_coeffs):
        # Find detected even and odd targets
        even_detected = [tid for tid in even_ids if tid in target_map]
        odd_detected = [tid for tid in odd_ids if tid in target_map]
        
        # 1. Fill the surface patches with transparency (60% transparency, 40% opacity)
        if len(even_detected) > 1 and len(odd_detected) > 1:
            overlay = image.copy()
            num_patches = min(len(even_detected), len(odd_detected)) - 1
            for i in range(num_patches):
                p1 = target_map[even_detected[i]]
                p2 = target_map[even_detected[i+1]]
                p3 = target_map[odd_detected[i+1]]
                p4 = target_map[odd_detected[i]]
                
                # Project 3D quad vertices to 2D
                pts_2d = self._project_points_3d_to_2d([p1, p2, p3, p4], cam_matrix, dist_coeffs)
                pts_2d_int = pts_2d.astype(np.int32).reshape((-1, 1, 2))
                
                # Fill polygon with a modern holographic cyan-blue color (BGR: 255, 128, 0)
                cv2.fillPoly(overlay, [pts_2d_int], (255, 128, 0))
                
            # Alpha blending: 0.4 opacity (60% transparency)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)
        
        # 2. Draw even boundary line (yellow)
        if len(even_detected) > 1:
            pts_even = [target_map[tid] for tid in even_detected]
            pts_2d = self._project_points_3d_to_2d(pts_even, cam_matrix, dist_coeffs)
            for i in range(len(pts_2d) - 1):
                cv2.line(image, tuple(pts_2d[i].astype(int)), tuple(pts_2d[i+1].astype(int)), (255, 255, 0), 2, cv2.LINE_AA)
            for pt in pts_2d:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 6, (255, 255, 0), -1)
                
        # 3. Draw odd boundary line (cyan)
        if len(odd_detected) > 1:
            pts_odd = [target_map[tid] for tid in odd_detected]
            pts_2d = self._project_points_3d_to_2d(pts_odd, cam_matrix, dist_coeffs)
            for i in range(len(pts_2d) - 1):
                cv2.line(image, tuple(pts_2d[i].astype(int)), tuple(pts_2d[i+1].astype(int)), (0, 229, 255), 2, cv2.LINE_AA)
            for pt in pts_2d:
                cv2.circle(image, (int(pt[0]), int(pt[1])), 6, (0, 229, 255), -1)
                
        # 4. Draw cross lines connecting them (orange)
        min_len = min(len(even_detected), len(odd_detected))
        for i in range(min_len):
            p_even = target_map[even_detected[i]]
            p_odd = target_map[odd_detected[i]]
            pts_2d = self._project_points_3d_to_2d([p_even, p_odd], cam_matrix, dist_coeffs)
            cv2.line(image, tuple(pts_2d[0].astype(int)), tuple(pts_2d[1].astype(int)), (255, 165, 0), 1, cv2.LINE_AA)

    def _draw_distance_vector(self, image, p_src, p_dst, cam_matrix, dist_coeffs, color, label, distance_mm):
        proj_pts = self._project_points_3d_to_2d([p_src, p_dst], cam_matrix, dist_coeffs)
        pt_src = (int(proj_pts[0][0]), int(proj_pts[0][1]))
        pt_dst = (int(proj_pts[1][0]), int(proj_pts[1][1]))
        
        # Draw vector line with arrows/circles
        cv2.line(image, pt_src, pt_dst, color, 2, cv2.LINE_AA)
        cv2.circle(image, pt_src, 4, color, -1)
        cv2.circle(image, pt_dst, 4, color, -1)
        
        # Overlay value text
        mid_pt = (int((pt_src[0] + pt_dst[0]) / 2) + 15, int((pt_src[1] + pt_dst[1]) / 2) - 5)
        cv2.putText(image, f"{label}: {distance_mm:.1f}mm", mid_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Logger logic
    def _write_log_row(self, results, source_id, source_pos, target_ids):
        if not self._log_writer:
            return
        
        elapsed = time.time() - self._log_start_time
        timestamp_formatted = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        src_coords = [f"{x*1000.0:.2f}" for x in source_pos] if source_pos is not None else ["N/A", "N/A", "N/A"]
        
        num_targets = results['num_targets']
        poly_dist = f"{results['polyline_dist']:.2f}" if results['polyline_dist'] is not None else "N/A"
        targets_str = ";".join(str(tid) for tid in target_ids) if target_ids else "None"
        
        try:
            self._log_writer.writerow([
                f"{elapsed:.4f}", timestamp_formatted,
                source_id, src_coords[0], src_coords[1], src_coords[2],
                num_targets, poly_dist,
                targets_str, results['selected_mode']
            ])
            self._log_file.flush()
        except Exception:
            pass # ignore logger errors during thread run to keep camera stream active


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài Đặt Tham Số Hệ Thống")
        self.setMinimumWidth(500)
        self.init_ui(parent)
        
    def init_ui(self, parent):
        layout = QVBoxLayout(self)
        
        # 1. Config parameters group
        config_group = QGroupBox("CẤU HÌNH PHÁT HIỆN & THỐNG KÊ")
        config_grid = QGridLayout(config_group)
        
        # Tag size in mm
        config_grid.addWidget(QLabel("Kích thước AprilTag (mm):"), 0, 0)
        self.sb_tag_size = QDoubleSpinBox()
        self.sb_tag_size.setRange(1.0, 1000.0)
        self.sb_tag_size.setValue(150.0) # default 150mm
        self.sb_tag_size.setSuffix(" mm")
        config_grid.addWidget(self.sb_tag_size, 0, 1)
        
        # Source tag ID (ID 1)
        config_grid.addWidget(QLabel("AprilTag ID Nguồn (Gốc):"), 1, 0)
        self.sb_source_id = QSpinBox()
        self.sb_source_id.setRange(0, 1000)
        self.sb_source_id.setValue(1) # Default ID 1
        config_grid.addWidget(self.sb_source_id, 1, 1)
        
        # Target tag IDs
        config_grid.addWidget(QLabel("ID Đường Mục Tiêu (Ngăn cách bằng dấu phẩy):"), 2, 0)
        self.txt_target_ids = QLineEdit()
        self.txt_target_ids.setPlaceholderText("Trống: Dùng tất cả tag khác")
        config_grid.addWidget(self.txt_target_ids, 2, 1)
        
        # Coordinate calculation mode
        config_grid.addWidget(QLabel("Thuật toán tọa độ 3D:"), 3, 0)
        self.cb_coord_mode = QComboBox()
        self.cb_coord_mode.addItem("Sử dụng Cảm biến Depth RealSense trực tiếp (Mặc định)", "depth")
        self.cb_coord_mode.addItem("Sử dụng SolvePnP hình học", "pnp")
        config_grid.addWidget(self.cb_coord_mode, 3, 1)
        
        layout.addWidget(config_group)
        
        # 2. Tracking & Filtering configuration
        tracking_group = QGroupBox("BÁM VẾT & BỘ LỌC MƯỢT 3D (TRACKING)")
        tracking_grid = QGridLayout(tracking_group)
        
        self.chk_enable_smoothing = QCheckBox("Kích hoạt bộ lọc mượt 3D (EMA)")
        self.chk_enable_smoothing.setChecked(False)
        tracking_grid.addWidget(self.chk_enable_smoothing, 0, 0, 1, 2)
        
        tracking_grid.addWidget(QLabel("Độ phản hồi bộ lọc (Alpha):"), 1, 0)
        self.sb_filter_alpha = QDoubleSpinBox()
        self.sb_filter_alpha.setRange(0.01, 1.0)
        self.sb_filter_alpha.setValue(0.70)
        self.sb_filter_alpha.setSingleStep(0.05)
        tracking_grid.addWidget(self.sb_filter_alpha, 1, 1)
        
        self.chk_enable_keep_alive = QCheckBox("Duy trì trạng thái khi mất dấu tạm thời")
        self.chk_enable_keep_alive.setChecked(False)
        tracking_grid.addWidget(self.chk_enable_keep_alive, 2, 0, 1, 2)
        
        tracking_grid.addWidget(QLabel("Khung hình duy trì tối đa (Max Lost):"), 3, 0)
        self.sb_max_lost_frames = QSpinBox()
        self.sb_max_lost_frames.setRange(1, 100)
        self.sb_max_lost_frames.setValue(15)
        tracking_grid.addWidget(self.sb_max_lost_frames, 3, 1)
        
        tracking_grid.addWidget(QLabel("Độ dài bộ lọc trung bình (khung hình):"), 4, 0)
        self.sb_window_size = QSpinBox()
        self.sb_window_size.setRange(1, 1000)
        self.sb_window_size.setValue(10) # default window size of 10 frames (highly responsive)
        tracking_grid.addWidget(self.sb_window_size, 4, 1)
        
        layout.addWidget(tracking_group)
        
        # 2.5 RealSense SDK Post-Processing filters
        realsense_group = QGroupBox("BỘ LỌC CHIỀU SÂU REALSENSE (POST-PROCESSING)")
        realsense_grid = QGridLayout(realsense_group)
        
        self.chk_decimation = QCheckBox("Decimation Filter (Giảm độ phân giải)")
        self.chk_decimation.setChecked(False)
        realsense_grid.addWidget(self.chk_decimation, 0, 0)
        
        self.chk_hole_filling = QCheckBox("Hole Filling Filter (Vá lỗ thủng)")
        self.chk_hole_filling.setChecked(True)
        realsense_grid.addWidget(self.chk_hole_filling, 0, 1)
        
        self.chk_spatial = QCheckBox("Spatial Filter (Làm mịn không gian)")
        self.chk_spatial.setChecked(True)
        realsense_grid.addWidget(self.chk_spatial, 1, 0)
        
        self.chk_temporal = QCheckBox("Temporal Filter (Làm mịn theo thời gian)")
        self.chk_temporal.setChecked(True)
        realsense_grid.addWidget(self.chk_temporal, 1, 1)
        
        self.chk_threshold = QCheckBox("Threshold Filter (Bộ lọc ngưỡng)")
        self.chk_threshold.setChecked(True)
        realsense_grid.addWidget(self.chk_threshold, 2, 0, 1, 2)
        
        realsense_grid.addWidget(QLabel("Ngưỡng Min (mét):"), 3, 0)
        self.sb_threshold_min = QDoubleSpinBox()
        self.sb_threshold_min.setRange(0.1, 10.0)
        self.sb_threshold_min.setValue(1.0)
        self.sb_threshold_min.setSingleStep(0.05)
        realsense_grid.addWidget(self.sb_threshold_min, 3, 1)
        
        realsense_grid.addWidget(QLabel("Ngưỡng Max (mét):"), 4, 0)
        self.sb_threshold_max = QDoubleSpinBox()
        self.sb_threshold_max.setRange(0.1, 10.0)
        self.sb_threshold_max.setValue(4.0)
        self.sb_threshold_max.setSingleStep(0.1)
        realsense_grid.addWidget(self.sb_threshold_max, 4, 1)
        
        realsense_grid.addWidget(QLabel("Công suất phát Laser (mW):"), 5, 0)
        self.sb_laser_power = QSpinBox()
        self.sb_laser_power.setRange(0, 360)
        self.sb_laser_power.setValue(150)
        self.sb_laser_power.setSingleStep(10)
        self.sb_laser_power.setSuffix(" mW")
        realsense_grid.addWidget(self.sb_laser_power, 5, 1)
        
        layout.addWidget(realsense_group)
        
        # 3. Dialog buttons
        buttons_layout = QHBoxLayout()
        self.btn_ok = QPushButton("Đóng")
        self.btn_ok.setObjectName("btn_start")
        self.btn_ok.clicked.connect(self.accept)
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.btn_ok)
        layout.addLayout(buttons_layout)

        # Style setting dialog similarly to main app widgets
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a1a;
            }
            QLabel {
                color: #d1d1d1;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #2d2d2d;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                background-color: #222222;
                color: #00e5ff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 6px;
                color: #00e5ff;
                font-size: 12px;
                text-transform: uppercase;
            }
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
                background-color: #2c2c2c;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
                border: 1px solid #00e5ff;
            }
            QPushButton {
                background-color: #00c853;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00e676;
            }
        """)

        # Connect signals for real-time update
        if parent and hasattr(parent, 'push_config_to_worker'):
            self.sb_tag_size.valueChanged.connect(parent.push_config_to_worker)
            self.sb_source_id.valueChanged.connect(parent.push_config_to_worker)
            self.txt_target_ids.textChanged.connect(parent.push_config_to_worker)
            self.cb_coord_mode.currentIndexChanged.connect(parent.push_config_to_worker)
            self.chk_enable_smoothing.stateChanged.connect(parent.push_config_to_worker)
            self.sb_filter_alpha.valueChanged.connect(parent.push_config_to_worker)
            self.chk_enable_keep_alive.stateChanged.connect(parent.push_config_to_worker)
            self.sb_max_lost_frames.valueChanged.connect(parent.push_config_to_worker)
            self.sb_window_size.valueChanged.connect(parent.push_config_to_worker)
            self.chk_decimation.stateChanged.connect(parent.push_config_to_worker)
            self.chk_hole_filling.stateChanged.connect(parent.push_config_to_worker)
            self.chk_spatial.stateChanged.connect(parent.push_config_to_worker)
            self.chk_temporal.stateChanged.connect(parent.push_config_to_worker)
            self.chk_threshold.stateChanged.connect(parent.push_config_to_worker)
            self.sb_threshold_min.valueChanged.connect(parent.push_config_to_worker)
            self.sb_threshold_max.valueChanged.connect(parent.push_config_to_worker)
            self.sb_laser_power.valueChanged.connect(parent.push_config_to_worker)


class GroundTruthApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Hệ Thống Đo Khoảng Cách AprilTag 3D - Intel RealSense D455")
        
        # Set Window Icon
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "diving.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            
        self.setMinimumSize(1200, 850) # Increased height to accommodate the plot comfortably
        
        # Initialize settings dialog
        self.settings_dialog = SettingsDialog(self)
        
        # Reference Map state
        self.reference_map = None
        self.reference_anchor_id = None
        ref_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_map.json")
        if os.path.exists(ref_map_path):
            try:
                with open(ref_map_path, "r", encoding="utf-8") as f:
                    ref_data = json.load(f)
                    self.reference_map = ref_data.get('map')
                    self.reference_anchor_id = ref_data.get('anchor_id')
            except Exception as e:
                print(f"Lỗi tải reference_map.json: {e}")
        
        # Initialize thread
        self.worker = CameraWorker()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.error_occurred.connect(self.handle_camera_error)
        self.worker.status_msg.connect(self.update_status)
        self.worker.calibration_complete.connect(self.on_calibration_complete)
        
        # Plot data history
        self.plot_time_history = []
        self.plot_poly_history = []
        self.plot_start_time = None
        self.max_history_points = 300 # scrolling window size (e.g. 10s at 30fps)
        
        # FPS tracker
        self.fps_timer = time.time()
        self.fps_counter = 0
        
        self.init_ui()
        self.apply_stylesheet()
        
    def init_ui(self):
        # Tạo Menu Bar
        menu_bar = self.menuBar()
        settings_menu = menu_bar.addMenu("Cài đặt")
        
        settings_action = settings_menu.addAction("Cấu hình hệ thống...")
        settings_action.triggered.connect(self.show_settings_dialog)

        # Main central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # Main layout (Horizontal split: Left is view, Right is control panel)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Left Area: Title + Camera Feed + Real-time Plot
        left_layout = QVBoxLayout()
        
        # Title bar
        title_label = QLabel("CAMERA MONITOR & 3D DETECTIONS")
        title_label.setObjectName("title_label")
        left_layout.addWidget(title_label)
        
        # Camera monitor container for dual streams
        self.camera_container = QWidget()
        self.camera_layout = QHBoxLayout(self.camera_container)
        self.camera_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_layout.setSpacing(10)
        
        # Label to display color video frames
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Vui lòng nhấn 'Run' để kết nối camera.")
        self.video_label.setObjectName("lbl_video")
        self.video_label.setMinimumSize(480, 360)
        self.camera_layout.addWidget(self.video_label)
        
        # Label to display depth video frames
        self.depth_label = QLabel()
        self.depth_label.setAlignment(Qt.AlignCenter)
        self.depth_label.setText("Luồng Depth đang ẩn.")
        self.depth_label.setObjectName("lbl_video")
        self.depth_label.setMinimumSize(480, 360)
        self.depth_label.setVisible(False) # Default hidden
        self.camera_layout.addWidget(self.depth_label)
        
        left_layout.addWidget(self.camera_container, 3) # Stretch factor 3
        
        # Real-time Plot Widget using pyqtgraph
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#1a1a1a')
        self.plot_widget.setTitle("Depth(MM)", color='#00e5ff', size='10pt')
        self.plot_widget.setLabel('left', 'Depth', units='mm', color='#d1d1d1')
        self.plot_widget.setLabel('bottom', 'Time', units='s', color='#d1d1d1')
        self.plot_widget.setYRange(200, 500, padding=0)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        
        # Plot curve
        self.curve_poly = self.plot_widget.plot(pen=pg.mkPen(color='#00e676', width=2.5))
        left_layout.addWidget(self.plot_widget, 2) # Stretch factor 2
        
        # Quick status bar inside left layout
        status_bar_layout = QHBoxLayout()
        
        self.status_bar_lbl = QLabel("Hệ thống đã sẵn sàng.")
        self.status_bar_lbl.setObjectName("lbl_status_bar")
        
        self.fps_lbl = QLabel("FPS: N/A")
        self.fps_lbl.setObjectName("lbl_status_bar")
        self.fps_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        status_bar_layout.addWidget(self.status_bar_lbl)
        status_bar_layout.addWidget(self.fps_lbl)
        
        left_layout.addLayout(status_bar_layout)
        
        main_layout.addLayout(left_layout, 2)
        
        # Right Area: Control Panel & Tables
        right_layout = QVBoxLayout()
        
        # 1. Connection and Stream controls
        stream_group = QGroupBox("ĐIỀU KHIỂN THIẾT BỊ")
        stream_grid = QGridLayout(stream_group)
        
        self.btn_scan = QPushButton("Quét Thiết Bị (Scan)")
        self.btn_scan.setObjectName("btn_normal")
        self.btn_scan.clicked.connect(self.scan_devices)
        stream_grid.addWidget(self.btn_scan, 0, 0, 1, 2)
        
        self.btn_toggle_stream = QPushButton("Run")
        self.btn_toggle_stream.setObjectName("btn_start")
        self.btn_toggle_stream.clicked.connect(self.toggle_camera_stream)
        stream_grid.addWidget(self.btn_toggle_stream, 1, 0, 1, 2)
        
        # Checkbox to toggle depth view
        self.chk_show_depth = QCheckBox("Hiển thị camera Depth chiều sâu")
        self.chk_show_depth.setChecked(False)
        self.chk_show_depth.stateChanged.connect(self.toggle_depth_visibility)
        stream_grid.addWidget(self.chk_show_depth, 2, 0, 1, 2)
        
        # Checkbox for using reference map
        self.chk_use_ref_map = QCheckBox("Sử dụng bản đồ tham chiếu (Giảm nhiễu)")
        self.chk_use_ref_map.setChecked(False)
        self.chk_use_ref_map.setEnabled(False)
        self.chk_use_ref_map.stateChanged.connect(self.push_config_to_worker)
        stream_grid.addWidget(self.chk_use_ref_map, 3, 0, 1, 2)
        
        # Checkbox for using IMU fusion
        self.chk_use_imu = QCheckBox("Sử dụng cảm biến IMU (D455)")
        self.chk_use_imu.setChecked(True)
        self.chk_use_imu.stateChanged.connect(self.push_config_to_worker)
        stream_grid.addWidget(self.chk_use_imu, 4, 0, 1, 2)
        
        # Button to reset/start new map
        self.btn_reset_map = QPushButton("Tạo bản đồ mới (Xóa dữ liệu cũ)")
        self.btn_reset_map.setObjectName("btn_stop")
        self.btn_reset_map.setEnabled(False)
        self.btn_reset_map.clicked.connect(self.reset_map_calibration)
        stream_grid.addWidget(self.btn_reset_map, 5, 0, 1, 2)
        
        # Button to append to map (Cuốn chiếu)
        self.btn_calibrate_map = QPushButton("Quét & Nối Tag (Cuốn chiếu)")
        self.btn_calibrate_map.setObjectName("btn_normal")
        self.btn_calibrate_map.setEnabled(False)
        self.btn_calibrate_map.clicked.connect(self.start_map_calibration)
        stream_grid.addWidget(self.btn_calibrate_map, 6, 0, 1, 2)

        self.btn_save_map = QPushButton("Lưu file (Save)")
        self.btn_save_map.setObjectName("btn_normal")
        self.btn_save_map.clicked.connect(self.save_map_file)
        stream_grid.addWidget(self.btn_save_map, 7, 0, 1, 1)
        
        self.btn_load_map = QPushButton("Tải file (Load)")
        self.btn_load_map.setObjectName("btn_normal")
        self.btn_load_map.clicked.connect(self.load_map_file)
        stream_grid.addWidget(self.btn_load_map, 7, 1, 1, 1)
        
        right_layout.addWidget(stream_group)
        
        # 3. Measurements cards (large display)
        measure_group = QGroupBox("KẾT QUẢ ĐO KHOẢNG CÁCH")
        measure_vbox = QVBoxLayout(measure_group)
        
        # Card: Polyline distance
        card_poly = QFrameCard(self)
        card_poly_layout = QVBoxLayout(card_poly)
        card_poly_title = QLabel("KHOẢNG CÁCH AprilTag ID1 ĐẾN ĐƯỜNG MỤC TIÊU")
        card_poly_title.setObjectName("card_title")
        self.lbl_poly_val = QLabel("N/A")
        self.lbl_poly_val.setObjectName("lbl_poly_val")
        self.lbl_poly_val.setAlignment(Qt.AlignCenter)
        card_poly_layout.addWidget(card_poly_title)
        card_poly_layout.addWidget(self.lbl_poly_val)
        
        measure_vbox.addWidget(card_poly)
        right_layout.addWidget(measure_group)
        
        # 4. Detected tags list table
        table_group = QGroupBox("DANH SÁCH TAG ĐANG PHÁT HIỆN")
        table_vbox = QVBoxLayout(table_group)
        
        self.table_tags = QTableWidget()
        self.table_tags.setColumnCount(6)
        self.table_tags.setHorizontalHeaderLabels(["ID", "Vai trò", "Trạng thái", "X (mm)", "Y (mm)", "Z (mm)"])
        self.table_tags.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_tags.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_tags.setObjectName("table_tags")
        table_vbox.addWidget(self.table_tags)
        
        right_layout.addWidget(table_group, 1) # table gets remaining vertical stretch
        
        # 5. Data logger panel
        logger_group = QGroupBox("GHI NHẬT KÝ ĐO LƯỜNG (REAL-TIME LOGGER)")
        logger_layout = QHBoxLayout(logger_group)
        
        self.chk_logging = QCheckBox("Bật ghi nhật ký trực tiếp")
        self.chk_logging.setEnabled(False)
        self.chk_logging.stateChanged.connect(self.toggle_logger)
        logger_layout.addWidget(self.chk_logging)
        
        self.lbl_logger_status = QLabel("Chưa mở file nhật ký.")
        self.lbl_logger_status.setObjectName("lbl_logger_status")
        logger_layout.addWidget(self.lbl_logger_status, 1)
        
        self.btn_select_log = QPushButton("Chọn File Lưu...")
        self.btn_select_log.setObjectName("btn_normal")
        self.btn_select_log.clicked.connect(self.select_log_file)
        logger_layout.addWidget(self.btn_select_log)
        
        right_layout.addWidget(logger_group)
        
        main_layout.addLayout(right_layout, 1)
        
        # Load reference map if exists
        if self.reference_map is not None:
            self.chk_use_ref_map.setEnabled(True)
            self.chk_use_ref_map.setChecked(True)

    def show_settings_dialog(self):
        self.settings_dialog.exec()
        
    def apply_stylesheet(self):
        qss = """
        QMainWindow {
            background-color: #1a1a1a;
        }
        
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
            color: #d1d1d1;
            font-size: 13px;
        }
        
        #title_label {
            font-weight: bold;
            font-size: 15px;
            color: #00e5ff;
            padding: 5px;
            background-color: #232323;
            border-radius: 4px;
            border-left: 4px solid #00e5ff;
        }
        
        QLabel#lbl_video {
            background-color: #0d0d0d;
            border: 2px solid #282828;
            border-radius: 8px;
            color: #757575;
            font-weight: bold;
            font-size: 15px;
        }
        
        QLabel#lbl_status_bar {
            color: #888888;
            font-size: 11px;
            padding: 4px;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 1px solid #2d2d2d;
            border-radius: 8px;
            margin-top: 15px;
            padding-top: 15px;
            background-color: #222222;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 6px;
            color: #00e5ff;
            font-size: 12px;
            text-transform: uppercase;
        }
        
        QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
            background-color: #2c2c2c;
            border: 1px solid #3c3c3c;
            border-radius: 4px;
            padding: 5px;
            color: #ffffff;
        }
        
        QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
            border: 1px solid #00e5ff;
        }
        
        QPushButton {
            background-color: #2d2d2d;
            border: 1px solid #444444;
            border-radius: 4px;
            padding: 8px 12px;
            color: #ffffff;
            font-weight: bold;
        }
        
        QPushButton:hover {
            background-color: #3d3d3d;
            border-color: #555555;
        }
        
        QPushButton#btn_start {
            background-color: #00c853;
            color: #ffffff;
            border: none;
        }
        
        QPushButton#btn_start:hover {
            background-color: #00e676;
        }
        
        QPushButton#btn_start:disabled {
            background-color: #1b5e20;
            color: #7f7f7f;
        }
        
        QPushButton#btn_stop {
            background-color: #d50000;
            color: #ffffff;
            border: none;
        }
        
        QPushButton#btn_stop:hover {
            background-color: #ff1744;
        }
        
        QPushButton#btn_stop:disabled {
            background-color: #b71c1c;
            color: #7f7f7f;
        }
        
        QPushButton#btn_normal {
            background-color: #0091ea;
            border: none;
            color: white;
        }
        
        QPushButton#btn_normal:hover {
            background-color: #00b0ff;
        }
        
        QTableWidget {
            background-color: #1e1e1e;
            alternate-background-color: #252525;
            border: 1px solid #2d2d2d;
            gridline-color: #2d2d2d;
            border-radius: 6px;
        }
        
        QHeaderView::section {
            background-color: #2c2c2c;
            color: #00e5ff;
            padding: 5px;
            border: none;
            border-bottom: 1px solid #3d3d3d;
            font-weight: bold;
        }
        
        QTableWidget::item {
            padding: 4px;
        }
        
        QCheckBox {
            spacing: 5px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
        }
        """
        self.setStyleSheet(qss)
        
    def push_config_to_worker(self):
        tag_size = self.settings_dialog.sb_tag_size.value() / 1000.0 # Convert mm to meters
        source_id = self.settings_dialog.sb_source_id.value()
        target_ids = self.settings_dialog.txt_target_ids.text()
        coord_mode = self.settings_dialog.cb_coord_mode.currentData()
        
        # Read tracking settings
        filter_alpha = self.settings_dialog.sb_filter_alpha.value()
        max_lost_frames = self.settings_dialog.sb_max_lost_frames.value()
        enable_smoothing = self.settings_dialog.chk_enable_smoothing.isChecked()
        enable_keep_alive = self.settings_dialog.chk_enable_keep_alive.isChecked()
        window_size = self.settings_dialog.sb_window_size.value()
        
        # Read RealSense SDK filters
        enable_decimation = self.settings_dialog.chk_decimation.isChecked()
        enable_hole_filling = self.settings_dialog.chk_hole_filling.isChecked()
        enable_spatial = self.settings_dialog.chk_spatial.isChecked()
        enable_temporal = self.settings_dialog.chk_temporal.isChecked()
        enable_threshold = self.settings_dialog.chk_threshold.isChecked()
        threshold_min = self.settings_dialog.sb_threshold_min.value()
        threshold_max = self.settings_dialog.sb_threshold_max.value()
        laser_power = self.settings_dialog.sb_laser_power.value()
        use_ref_map = self.chk_use_ref_map.isChecked()
        reference_map = self.reference_map
        use_imu = self.chk_use_imu.isChecked()
        
        self.worker.update_config(tag_size, source_id, target_ids, coord_mode,
                                  filter_alpha, max_lost_frames, enable_smoothing, enable_keep_alive, window_size,
                                  enable_decimation, enable_hole_filling, enable_spatial, enable_temporal, enable_threshold,
                                  threshold_min, threshold_max, laser_power, use_ref_map, reference_map, use_imu)
        
    def scan_devices(self):
        self.status_bar_lbl.setText("Đang quét thiết bị camera...")
        QApplication.processEvents()
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                self.status_bar_lbl.setText("Không tìm thấy camera RealSense.")
                QMessageBox.warning(self, "Quét Thiết Bị", 
                                    "Không phát hiện thấy camera Intel RealSense nào đang kết nối.\n"
                                    "Vui lòng kiểm tra lại cáp kết nối USB 3.0 và đảm bảo camera đã được cắm.")
            else:
                dev_list = []
                for i, dev in enumerate(devices):
                    name = dev.get_info(rs.camera_info.name)
                    sn = dev.get_info(rs.camera_info.serial_number)
                    dev_list.append(f"• {name} (S/N: {sn})")
                
                self.status_bar_lbl.setText(f"Đã tìm thấy {len(devices)} thiết bị.")
                devs_text = "\n".join(dev_list)
                QMessageBox.information(self, "Quét Thiết Bị", 
                                        f"Đã phát hiện thấy {len(devices)} thiết bị RealSense kết nối:\n\n{devs_text}")
        except Exception as e:
            self.status_bar_lbl.setText("Lỗi khi quét thiết bị.")
            QMessageBox.critical(self, "Lỗi Quét Thiết Bị", f"Lỗi xảy ra trong quá trình quét: {str(e)}")

    def toggle_camera_stream(self):
        if self.worker.isRunning():
            self.stop_camera_stream()
        else:
            self.start_camera_stream()

    def start_camera_stream(self):
        # Kiểm tra thiết bị RealSense trước để tránh treo app
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                QMessageBox.warning(self, "Không tìm thấy camera", 
                                    "Không thể bắt đầu. Không phát hiện thấy camera Intel RealSense nào đang kết nối.\n"
                                    "Vui lòng kiểm tra lại cáp cắm và thử lại.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Lỗi kiểm tra camera", f"Có lỗi xảy ra khi kiểm tra camera: {str(e)}")
            return

        self.push_config_to_worker()
        self.btn_toggle_stream.setText("Stop")
        self.btn_toggle_stream.setObjectName("btn_stop")
        self.btn_toggle_stream.style().unpolish(self.btn_toggle_stream)
        self.btn_toggle_stream.style().polish(self.btn_toggle_stream)
        self.btn_scan.setEnabled(False)
        self.settings_dialog.sb_source_id.setEnabled(False)
        self.btn_calibrate_map.setEnabled(True)
        
        # Reset FPS calculation variables
        self.fps_timer = time.time()
        self.fps_counter = 0
        
        self.worker.start()
        
    def stop_camera_stream(self):
        self.chk_logging.setChecked(False) # Turn off logger if camera is stopped
        self.worker.stop()
        
        # Reset UI display
        self.video_label.clear()
        self.video_label.setText("Truyền hình camera đã dừng. Nhấn 'Run' để kết nối lại.")
        self.lbl_poly_val.setText("N/A")
        self.lbl_poly_val.setStyleSheet("")
        self.table_tags.setRowCount(0)
        
        # Reset plot history
        self.plot_start_time = None
        self.plot_time_history.clear()
        self.plot_poly_history.clear()
        self.curve_poly.setData([], [])
        
        self.btn_toggle_stream.setText("Run")
        self.btn_toggle_stream.setObjectName("btn_start")
        self.btn_toggle_stream.style().unpolish(self.btn_toggle_stream)
        self.btn_toggle_stream.style().polish(self.btn_toggle_stream)
        self.btn_scan.setEnabled(True)
        self.settings_dialog.sb_source_id.setEnabled(True)
        self.btn_calibrate_map.setEnabled(False)
        
        # Reset FPS
        self.fps_lbl.setText("FPS: N/A")
        self.fps_counter = 0
        
    def reset_map_calibration(self):
        self.worker.start_calibration(reset=True)
        self.status_bar_lbl.setText("Đã xóa bản đồ cũ trong bộ nhớ. Vui lòng nhấn 'Quét & Nối Tag' để bắt đầu tạo mốc mới.")
        
    def start_map_calibration(self):
        self.worker.start_calibration(reset=False)
        self.status_bar_lbl.setText("Đang quét và tính toán nội suy... Vui lòng giữ yên camera!")

    def save_map_file(self):
        if not self.reference_map:
            QMessageBox.warning(self, "Lưu bản đồ", "Hiện chưa có bản đồ nào trong bộ nhớ để lưu!")
            return
            
        # Tạo tên file mặc định dựa trên thời gian thực
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"reference_map_{timestamp}.json"
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), default_name)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu file Bản đồ", default_path, "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                result = {
                    'anchor_id': self.reference_anchor_id,
                    'map': self.reference_map
                }
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=4)
                self.status_bar_lbl.setText(f"Đã lưu bản đồ thành công tại: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Thành công", f"Đã lưu bản đồ vào file:\n{os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu file: {str(e)}")

    def load_map_file(self):
        default_dir = os.path.dirname(os.path.abspath(__file__))
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file Bản đồ", default_dir, "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    ref_data = json.load(f)
                    
                    if 'map' not in ref_data:
                        raise ValueError("File không đúng định dạng bản đồ Reference Map!")
                        
                    self.reference_map = ref_data.get('map')
                    self.reference_anchor_id = ref_data.get('anchor_id')
                    
                self.chk_use_ref_map.setEnabled(True)
                self.chk_use_ref_map.setChecked(True)
                self.push_config_to_worker()
                
                self.status_bar_lbl.setText(f"Đã tải bản đồ: {os.path.basename(file_path)} (Anchor ID: {self.reference_anchor_id})")
                QMessageBox.information(self, "Thành công", f"Đã tải bản đồ chứa {len(self.reference_map)} Tag thành công.")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Không thể đọc file: {str(e)}")

    def on_calibration_complete(self, result):
        self.reference_map = result['map']
        self.reference_anchor_id = result['anchor_id']
        
        # Đổi cơ chế lưu tự động: Luôn đính kèm thời gian (Timestamp) để chống mất dữ liệu do nhấn nhầm
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reference_map_autosave_{timestamp}.json"
        ref_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        try:
            with open(ref_map_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
            self.status_bar_lbl.setText(f"Cập nhật bản đồ xong! Đã tự động sao lưu tại: {filename}")
        except Exception as e:
            self.status_bar_lbl.setText("Cập nhật bản đồ xong nhưng bị lỗi khi tự động lưu file.")
            QMessageBox.warning(self, "Lỗi tự động lưu", f"Không thể tự động lưu bản đồ: {e}")
            
        self.chk_use_ref_map.setEnabled(True)
        self.chk_use_ref_map.setChecked(True)
        self.push_config_to_worker()
        
    def toggle_depth_visibility(self, state):
        if state == Qt.Checked.value or state is True:
            self.depth_label.setVisible(True)
        else:
            self.depth_label.setVisible(False)
            self.depth_label.clear()
            self.depth_label.setText("Luồng Depth đang ẩn.")
        
    def select_log_file(self):
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Chọn file lưu Nhật Ký CSV", desktop_path, "CSV Files (*.csv)"
        )
        if file_path:
            self.lbl_logger_status.setText(f"Lưu tại: {os.path.basename(file_path)}")
            self.lbl_logger_status.setToolTip(file_path)
            self._saved_file_path = file_path
            self.chk_logging.setEnabled(True)
            
    def toggle_logger(self, state):
        if state == Qt.Checked.value or state is True:
            if hasattr(self, '_saved_file_path') and self._saved_file_path:
                try:
                    self.worker.start_logging(self._saved_file_path)
                    self.lbl_logger_status.setText("Đang ghi dữ liệu...")
                    self.btn_select_log.setEnabled(False)
                except Exception as e:
                    QMessageBox.critical(self, "Lỗi Logger", f"Không thể tạo hoặc ghi file: {str(e)}")
                    self.chk_logging.setChecked(False)
            else:
                QMessageBox.warning(self, "Chưa chọn file", "Vui lòng chọn file lưu nhật ký trước.")
                self.chk_logging.setChecked(False)
        else:
            self.worker.stop_logging()
            self.btn_select_log.setEnabled(True)
            if hasattr(self, '_saved_file_path') and self._saved_file_path:
                self.lbl_logger_status.setText(f"Tạm dừng. File: {os.path.basename(self._saved_file_path)}")
            else:
                self.lbl_logger_status.setText("Chưa chọn file lưu.")
 
    @Slot(np.ndarray, dict)
    def update_frame(self, frame_bgr, results):
        # Calculate FPS
        self.fps_counter += 1
        now = time.time()
        if now - self.fps_timer >= 1.0:
            current_fps = self.fps_counter / (now - self.fps_timer)
            self.fps_lbl.setText(f"FPS: {current_fps:.1f}")
            self.fps_counter = 0
            self.fps_timer = now
            
        # Directly wrap BGR frame in QImage without conversion and copy
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale pixmap using FastTransformation for maximum rendering speed
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
        # Display depth image if the checkbox is checked and depth data is available
        if self.chk_show_depth.isChecked() and 'depth_image' in results:
            depth_bgr = results['depth_image']
            dh, dw, dch = depth_bgr.shape
            d_bytes_per_line = dch * dw
            
            dq_image = QImage(depth_bgr.data, dw, dh, d_bytes_per_line, QImage.Format_BGR888)
            dpixmap = QPixmap.fromImage(dq_image)
            dscaled_pixmap = dpixmap.scaled(self.depth_label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
            self.depth_label.setPixmap(dscaled_pixmap)
        
        # Update real-time plot
        if self.plot_start_time is None:
            self.plot_start_time = time.time()
        elapsed = time.time() - self.plot_start_time
        
        poly_val = results['polyline_dist']
        
        self.plot_time_history.append(elapsed)
        self.plot_poly_history.append(poly_val if poly_val is not None else float('nan'))
        
        if len(self.plot_time_history) > self.max_history_points:
            self.plot_time_history.pop(0)
            self.plot_poly_history.pop(0)
            
        self.curve_poly.setData(self.plot_time_history, self.plot_poly_history)
        
        # Update metrics cards
        # Polyline distance display
        if results['polyline_dist'] is not None:
            self.lbl_poly_val.setText(f"{results['polyline_dist']:.1f} mm")
            self.lbl_poly_val.setStyleSheet("color: #00e676; font-size: 28px; font-weight: bold;")
        else:
            self.lbl_poly_val.setText("N/A")
            self.lbl_poly_val.setStyleSheet("color: #757575; font-size: 28px; font-weight: bold;")
            
        # Update detected tags table
        detected = results['detected_tags']
        self.table_tags.setRowCount(len(detected))
        
        for idx, (tid, data) in enumerate(sorted(detected.items())):
            # Table columns: ID, Role, Status, X (mm), Y (mm), Z (mm)
            # Select values depending on coordinate mode
            selected_pos = data['pos_pnp'] if results['selected_mode'] == 'pnp' else data['pos_depth']
            
            item_id = QTableWidgetItem(str(tid))
            item_role = QTableWidgetItem(data['role'])
            item_status = QTableWidgetItem(data['status'])
            item_x = QTableWidgetItem(f"{selected_pos[0]:.1f}")
            item_y = QTableWidgetItem(f"{selected_pos[1]:.1f}")
            item_z = QTableWidgetItem(f"{selected_pos[2]:.1f}")
            
            # Format text colors for status
            if data['lost_frames'] == 0:
                item_status.setForeground(QColor("#00e676")) # bright green
            else:
                item_status.setForeground(QColor("#ffaa00")) # orange/yellow
            
            # Format text colors for role
            if "Nguồn" in data['role']:
                item_role.setForeground(QColor("#ff1744"))
            elif "Mục tiêu" in data['role'] or "Đường" in data['role']:
                item_role.setForeground(QColor("#00e5ff"))
                
            self.table_tags.setItem(idx, 0, item_id)
            self.table_tags.setItem(idx, 1, item_role)
            self.table_tags.setItem(idx, 2, item_status)
            self.table_tags.setItem(idx, 3, item_x)
            self.table_tags.setItem(idx, 4, item_y)
            self.table_tags.setItem(idx, 5, item_z)

    @Slot(str)
    def handle_camera_error(self, error_msg):
        QMessageBox.critical(self, "Lỗi Camera RealSense", error_msg)
        self.stop_camera_stream()
        
    @Slot(str)
    def update_status(self, msg):
        self.status_bar_lbl.setText(msg)
        
    def closeEvent(self, event):
        # Shut down worker thread properly to unlock realsense camera
        self.chk_logging.setChecked(False)
        if self.worker.isRunning():
            self.worker.stop()
        event.accept()


class QFrameCard(QWidget):
    """Custom premium UI component representing a metric card with a dark background and border."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget {
                background-color: #242424;
                border: 1px solid #353535;
                border-radius: 8px;
            }
            #card_title {
                color: #888888;
                font-size: 11px;
                font-weight: bold;
                border: none;
                background-color: transparent;
            }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GroundTruthApp()
    window.show()
    sys.exit(app.exec())
