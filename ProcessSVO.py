import os
import sys
import time
import csv
import argparse
import numpy as np
import cv2

# Add ZED SDK and CUDA bin directories to PATH for loading DLLs on Windows
if sys.platform == "win32":
    zed_sdk_paths = [
        r"C:\Program Files\ZED SDK\bin",
        r"C:\Program Files (x86)\ZED SDK\bin"
    ]
    for path in zed_sdk_paths:
        if os.path.exists(path):
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(path)
                except Exception:
                    pass

try:
    import pyzed.sl as sl
except ImportError:
    print("[ERROR] Thư viện 'pyzed' (ZED SDK Python wrapper) chưa được cài đặt.")
    print("Vui lòng cài đặt ZED SDK để tiếp tục.")

# PySide6 GUI imports
try:
    from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                                 QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, 
                                 QLineEdit, QGroupBox, QComboBox, QFileDialog, 
                                 QMessageBox, QDoubleSpinBox, QSpinBox, QProgressBar, QTextEdit, QCheckBox)
    from PySide6.QtGui import QFont, QColor
    HAS_GUI_LIBS = True
except ImportError:
    HAS_GUI_LIBS = False

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

def distance_point_to_segment(p, a, b):
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

def distance_point_to_fitted_plane(p, qs):
    N = len(qs)
    if N == 0:
        return None, None
    if N == 1:
        dist = np.linalg.norm(p - qs[0])
        return dist, qs[0]
    if N == 2:
        return distance_point_to_segment(p, qs[0], qs[1])
        
    pts = np.array(qs)
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    
    try:
        _, _, vh = np.linalg.svd(centered)
        normal = vh[2]
        normal = normal / np.linalg.norm(normal)
        
        dist = abs(np.dot(normal, p - centroid))
        p_proj = p - np.dot(normal, p - centroid) * normal
        return dist, p_proj
    except Exception:
        return distance_point_to_segment(p, qs[0], qs[-1])

def kabsch_alignment(P, Q):
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

def project_points_3d_to_2d(pts_3d, cam_matrix, dist_coeffs):
    pts_3d = np.array(pts_3d, dtype=np.float32).reshape(-1, 3)
    rvec = np.zeros(3, dtype=np.float32)
    tvec = np.zeros(3, dtype=np.float32)
    img_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, cam_matrix, dist_coeffs)
    return img_pts.reshape(-1, 2)

def process_svo(args, log_cb=None, progress_cb=None, cancel_check=None):
    def print_msg(msg):
        if log_cb:
            log_cb(msg)
        else:
            print(msg)

    print_msg(f"=======================================================")
    print_msg(f"BẮT ĐẦU XỬ LÝ OFFLINE SVO: {args.svo}")
    print_msg(f"=======================================================")
    
    # Nạp bản đồ cấu trúc (Reference Map) nếu có cấu hình
    reference_map = None
    if args.map:
        if not os.path.exists(args.map):
            print_msg(f"[ERROR] Không tìm thấy tệp bản đồ cấu trúc tại: {args.map}")
            return False, "Không tìm thấy file bản đồ cấu trúc."
        try:
            import json
            with open(args.map, 'r', encoding='utf-8') as f:
                map_data = json.load(f)
                if isinstance(map_data, dict) and 'map' in map_data:
                    reference_map = map_data['map']
                else:
                    reference_map = map_data
            print_msg(f"[*] Nạp bản đồ cấu trúc thành công từ: {args.map} ({len(reference_map)} thẻ tag)")
        except Exception as e:
            print_msg(f"[WARNING] Không thể tải bản đồ cấu trúc: {e}")
    
    # 1. Cấu hình ZED SDK đọc tệp SVO
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(args.svo)
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.METER
    
    depth_mode_map = {
        "NONE": sl.DEPTH_MODE.NONE,
        "NEURAL": getattr(sl.DEPTH_MODE, "NEURAL", sl.DEPTH_MODE.QUALITY),
        "NEURAL_PLUS": getattr(sl.DEPTH_MODE, "NEURAL_PLUS", sl.DEPTH_MODE.QUALITY),
        "ULTRA": sl.DEPTH_MODE.ULTRA,
        "QUALITY": sl.DEPTH_MODE.QUALITY,
        "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE
    }
    
    selected_depth_mode = depth_mode_map.get(args.depth_mode.upper(), sl.DEPTH_MODE.PERFORMANCE)
    init_params.depth_mode = selected_depth_mode
    print_msg(f"[*] Chế độ Depth: {args.depth_mode.upper()}")
    
    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        msg = f"Không thể mở tệp SVO: {err}"
        print_msg(f"[ERROR] {msg}")
        return False, msg
        
    nb_frames = zed.get_svo_number_of_frames()
    print_msg(f"[*] Tổng số khung hình: {nb_frames}")
    
    # 2. Cấu hình bộ nhận diện AprilTag
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.minMarkerPerimeterRate = 0.03
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    trackers = {}
    csv_file = open(args.csv, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Frame_Index", "Timestamp_MS", 
        "Source_ID", "Source_X_mm", "Source_Y_mm", "Source_Z_mm",
        "Num_Targets", "Polyline_Dist_mm", "Target_IDs"
    ])
    
    video_writer = None
    if args.output_video:
        cam_info = zed.get_camera_information()
        w = cam_info.camera_configuration.resolution.width
        h = cam_info.camera_configuration.resolution.height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output_video, fourcc, 30.0, (w, h))
        print_msg(f"[*] Ghi video kết quả ra: {args.output_video}")
    
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    
    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    cam_matrix = np.array([
        [calib.fx, 0, calib.cx],
        [0, calib.fy, calib.cy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)
    
    s = args.tag_size / 1000.0
    obj_points = np.array([
        [-s/2, -s/2, 0],
        [ s/2, -s/2, 0],
        [ s/2,  s/2, 0],
        [-s/2,  s/2, 0]
    ], dtype=np.float32)
    
    target_ids = []
    if args.target_ids:
        try:
            target_ids = [int(x.strip()) for x in args.target_ids.split(",") if x.strip().isdigit()]
        except Exception:
            pass
            
    print_msg(f"[*] Thẻ Nguồn (Source): ID {args.source_id}")
    print_msg(f"[*] Danh sách Thẻ Đích (Targets): {target_ids}")
    
    start_time = time.time()
    processed_count = 0
    
    print_msg("\n--- BẮT ĐẦU CHẠY PHÂN TÍCH (FPS TỐI ĐA) ---")
    
    for f_idx in range(nb_frames):
        # Kiểm tra lệnh huỷ từ giao diện
        if cancel_check and cancel_check():
            print_msg("\n[CANCEL] Đã yêu cầu hủy bỏ xử lý từ người dùng.")
            break
            
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            break
            
        zed.retrieve_image(image_mat, sl.VIEW.LEFT)
        bgra_image = image_mat.get_data()
        color_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        depth_data = None
        if selected_depth_mode != sl.DEPTH_MODE.NONE:
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth_data = depth_mat.get_data()
            
        # ROI Tracking
        active_tracker_tids = [tid for tid, tracker in trackers.items() if tracker.is_tracked]
        detected_corners = []
        detected_ids = []
        
        if active_tracker_tids and not args.disable_roi:
            for tid in active_tracker_tids:
                tracker = trackers[tid]
                cx = int(np.mean(tracker.corners_filtered[:, 0]))
                cy = int(np.mean(tracker.corners_filtered[:, 1]))
                
                half_size = args.roi_size // 2
                x1 = max(0, cx - half_size)
                y1 = max(0, cy - half_size)
                x2 = min(gray.shape[1], cx + half_size)
                y2 = min(gray.shape[0], cy + half_size)
                
                roi = gray[y1:y2, x1:x2]
                corners_roi, ids_roi, _ = detector.detectMarkers(roi)
                
                if ids_roi is not None:
                    for i, tag_id_val in enumerate(np.ravel(ids_roi)):
                        tag_id = int(tag_id_val)
                        c_global = corners_roi[i] + np.array([x1, y1], dtype=np.float32)
                        detected_corners.append(c_global)
                        detected_ids.append([tag_id])
            
            if len(detected_ids) > 0:
                corners = detected_corners
                ids = np.array(detected_ids)
            else:
                corners, ids, rejected = detector.detectMarkers(gray)
        else:
            corners, ids, rejected = detector.detectMarkers(gray)
            
        detected_this_frame = set()
        active_tags = {}
        
        if ids is not None:
            flat_ids = np.ravel(ids)
            for i, tag_id_val in enumerate(flat_ids):
                tag_id = int(tag_id_val)
                corners_i = np.array(corners[i], dtype=np.float32).reshape(4, 2)
                
                success, rvec, tvec = cv2.solvePnP(obj_points, corners_i, cam_matrix, dist_coeffs)
                p_3d_pnp = tvec.flatten() if success else np.array([0.0, 0.0, 0.0])
                
                p_3d_depth = p_3d_pnp.copy()
                depth_val = 0.0
                if depth_data is not None:
                    u_c = float(np.mean(corners_i[:, 0]))
                    v_c = float(np.mean(corners_i[:, 1]))
                    
                    depth_w = depth_data.shape[1]
                    depth_h = depth_data.shape[0]
                    u_c_depth = int(u_c * depth_w / color_image.shape[1])
                    v_c_depth = int(v_c * depth_h / color_image.shape[0])
                    
                    depths = []
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            nu, nv = u_c_depth + dx, v_c_depth + dy
                            if 0 <= nu < depth_w and 0 <= nv < depth_h:
                                d = depth_data[nv, nu]
                                if not np.isnan(d) and not np.isinf(d) and d > 0.0:
                                    depths.append(d)
                    if depths:
                        depth_val = float(np.median(depths))
                        
                    if depth_val > 0.0:
                        z = depth_val
                        x = (u_c - calib.cx) * z / calib.fx
                        y = (v_c - calib.cy) * z / calib.fy
                        p_3d_depth = np.array([x, y, z])
                
                if tag_id not in trackers:
                    trackers[tag_id] = TagTracker(tag_id, alpha=args.filter_alpha, max_lost_frames=args.max_lost_frames)
                trackers[tag_id].update(corners_i, p_3d_pnp, p_3d_depth, rvec, tvec,
                                        alpha=args.filter_alpha, max_lost_frames=args.max_lost_frames)
                detected_this_frame.add(tag_id)
                
        for tid, tracker in list(trackers.items()):
            if tid not in detected_this_frame:
                tracker.predict(max_lost_frames=args.max_lost_frames)
                
        for tid, tracker in trackers.items():
            if tracker.is_tracked:
                active_tags[tid] = {
                    'pos_pnp': tracker.pos_pnp_filtered,
                    'pos_depth': tracker.pos_depth_filtered,
                    'corners': tracker.corners_filtered,
                    'rvec': tracker.rvec_filtered,
                    'tvec': tracker.tvec_filtered,
                    'lost_frames': tracker.lost_frames
                }
                
        # Tái dựng các thẻ tag ảo từ bản đồ cấu trúc (Reference Map)
        if reference_map is not None:
            detected_mapped_ids = []
            for tid_str, ref_data in reference_map.items():
                tid = int(tid_str)
                if tid in active_tags and active_tags[tid]['lost_frames'] == 0:
                    if active_tags[tid]['rvec'] is not None:
                        detected_mapped_ids.append(tid)
                        
            if len(detected_mapped_ids) > 0:
                if len(detected_mapped_ids) >= 3:
                    P_pts = []
                    Q_pts = []
                    for tid in detected_mapped_ids:
                        ref_data = reference_map[str(tid)]
                        P_pts.append(ref_data['p_local'])
                        tag = active_tags[tid]
                        pos_t = tag['pos_pnp'] if args.coord_mode == 'pnp' else tag['pos_depth']
                        Q_pts.append(pos_t)
                    R_L2C, T_L2C = kabsch_alignment(np.array(P_pts), np.array(Q_pts))
                else:
                    tid = detected_mapped_ids[0]
                    tag = active_tags[tid]
                    pos_t = tag['pos_pnp'] if args.coord_mode == 'pnp' else tag['pos_depth']
                    R_t, _ = cv2.Rodrigues(tag['rvec'])
                    
                    ref_data = reference_map[str(tid)]
                    p_local = np.array(ref_data['p_local'])
                    r_local = np.array(ref_data['r_local'])
                    
                    R_L2C = R_t @ r_local.T
                    T_L2C = pos_t - R_L2C @ p_local
                    
                for tid_str, ref_data in reference_map.items():
                    tid = int(tid_str)
                    p_local = np.array(ref_data['p_local'])
                    r_local = np.array(ref_data['r_local'])
                    
                    pos_reconstructed = R_L2C @ p_local + T_L2C
                    R_reconstructed = R_L2C @ r_local
                    rvec_reconstructed, _ = cv2.Rodrigues(R_reconstructed)
                    
                    if tid in active_tags:
                        active_tags[tid]['pos_pnp'] = pos_reconstructed
                        active_tags[tid]['pos_depth'] = pos_reconstructed
                        active_tags[tid]['rvec'] = rvec_reconstructed
                        active_tags[tid]['tvec'] = pos_reconstructed.reshape(3, 1)
                    else:
                        local_corners = np.array([
                            [-s/2, -s/2, 0],
                            [ s/2, -s/2, 0],
                            [ s/2,  s/2, 0],
                            [-s/2,  s/2, 0]
                        ], dtype=np.float32)
                        cam_corners = (R_reconstructed @ local_corners.T).T + pos_reconstructed
                        pts_2d = project_points_3d_to_2d(cam_corners, cam_matrix, dist_coeffs)
                        
                        active_tags[tid] = {
                            'pos_pnp': pos_reconstructed,
                            'pos_depth': pos_reconstructed,
                            'rvec': rvec_reconstructed,
                            'tvec': pos_reconstructed.reshape(3, 1),
                            'corners': pts_2d,
                            'lost_frames': 1,
                            'is_virtual': True
                        }
                        
        source_pos = None
        if args.source_id in active_tags:
            src_tag = active_tags[args.source_id]
            source_pos = src_tag['pos_pnp'] if args.coord_mode == 'pnp' else src_tag['pos_depth']
            
        target_pts = []
        detected_targets_list = []
        for tid in target_ids:
            if tid in active_tags:
                pos = active_tags[tid]['pos_pnp'] if args.coord_mode == 'pnp' else active_tags[tid]['pos_depth']
                target_pts.append(pos)
                detected_targets_list.append(tid)
                
        polyline_dist = None
        if source_pos is not None and len(target_pts) > 0:
            polyline_dist, _ = distance_point_to_fitted_plane(source_pos, target_pts)
            if polyline_dist is not None:
                polyline_dist = polyline_dist * 1000.0
                
        ts_ms = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        src_coords = [source_pos[0]*1000.0, source_pos[1]*1000.0, source_pos[2]*1000.0] if source_pos is not None else ["", "", ""]
        
        csv_writer.writerow([
            f_idx, ts_ms, 
            args.source_id, src_coords[0], src_coords[1], src_coords[2],
            len(target_pts), polyline_dist if polyline_dist is not None else "",
            ",".join(map(str, detected_targets_list))
        ])
        
        if video_writer is not None:
            for tid, tag in active_tags.items():
                corners_i = tag['corners'].astype(np.int32)
                color = (0, 0, 255) if tid == args.source_id else ((0, 255, 255) if tid in target_ids else (128, 128, 128))
                thickness = 3 if tid == args.source_id or tid in target_ids else 1
                for k in range(4):
                    cv2.line(color_image, tuple(corners_i[k]), tuple(corners_i[(k+1)%4]), color, thickness)
                cv2.putText(color_image, f"ID:{tid}", (int(corners_i[0][0]), int(corners_i[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(color_image, f"Frame: {f_idx}/{nb_frames}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            video_writer.write(color_image)
            
        processed_count += 1
        
        # Cập nhật GUI
        if progress_cb:
            progress_cb(processed_count, nb_frames)
            
        if processed_count % 100 == 0 or processed_count == nb_frames:
            elapsed = time.time() - start_time
            curr_fps = processed_count / elapsed if elapsed > 0 else 0
            print_msg(f" -> Đã xử lý {processed_count}/{nb_frames} khung hình ({processed_count/nb_frames*100:.1f}%). FPS hiện tại: {curr_fps:.2f}")
            
    csv_file.close()
    if video_writer is not None:
        video_writer.release()
    zed.close()
    
    total_elapsed = time.time() - start_time
    final_fps = processed_count / total_elapsed if total_elapsed > 0 else 0
    print_msg(f"\n=======================================================")
    print_msg(f"HOÀN THÀNH XỬ LÝ SVO!")
    print_msg(f"Tổng số frame đã xử lý: {processed_count}")
    print_msg(f"Tổng thời gian: {total_elapsed:.2f} giây")
    print_msg(f"FPS TRUNG BÌNH ĐẠT ĐƯỢC: {final_fps:.2f} khung hình/giây")
    print_msg(f"Dữ liệu CSV xuất ra: {args.csv}")
    if args.output_video:
        print_msg(f"Video kết quả xuất ra: {args.output_video}")
    print_msg(f"=======================================================")
    return True, f"Thành công! FPS trung bình: {final_fps:.2f}"

# ----------------- GUI IMPLEMENTATION -----------------
if HAS_GUI_LIBS:
    class SVOWorkerThread(QThread):
        progress_updated = Signal(int, int)
        log_received = Signal(str)
        finished_signal = Signal(bool, str)

        def __init__(self, args):
            super().__init__()
            self.args = args
            self.cancel_requested = False

        def run(self):
            def log_callback(msg):
                self.log_received.emit(msg)
                
            def progress_callback(curr, total):
                self.progress_updated.emit(curr, total)
                
            def cancel_callback():
                return self.cancel_requested

            try:
                success, message = process_svo(
                    self.args, 
                    log_cb=log_callback, 
                    progress_cb=progress_callback, 
                    cancel_check=cancel_callback
                )
                self.finished_signal.emit(success, message)
            except Exception as e:
                self.finished_signal.emit(False, str(e))

    class ProcessSVOGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Bộ Xử Lý Offline SVO - AprilTag 3D Ground Truth")
            self.resize(750, 650)
            self.worker = None
            self.init_ui()
            self.apply_stylesheet()
            self.auto_find_files()

        def init_ui(self):
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)
            
            # --- FILE SELECTION GROUP ---
            file_group = QGroupBox("Cấu hình Đường dẫn Tệp")
            file_layout = QGridLayout(file_group)
            
            file_layout.addWidget(QLabel("Tệp SVO Đầu vào:"), 0, 0)
            self.txt_svo = QLineEdit()
            self.btn_browse_svo = QPushButton("Duyệt...")
            self.btn_browse_svo.clicked.connect(self.browse_svo)
            file_layout.addWidget(self.txt_svo, 0, 1)
            file_layout.addWidget(self.btn_browse_svo, 0, 2)
            
            file_layout.addWidget(QLabel("Bản đồ cấu trúc (.json - Tùy chọn):"), 1, 0)
            self.txt_map = QLineEdit()
            self.btn_browse_map = QPushButton("Duyệt...")
            self.btn_browse_map.clicked.connect(self.browse_map)
            file_layout.addWidget(self.txt_map, 1, 1)
            file_layout.addWidget(self.btn_browse_map, 1, 2)
            
            file_layout.addWidget(QLabel("Tệp CSV Kết quả:"), 2, 0)
            self.txt_csv = QLineEdit("svo_report.csv")
            self.btn_browse_csv = QPushButton("Lưu tại...")
            self.btn_browse_csv.clicked.connect(self.browse_csv)
            file_layout.addWidget(self.txt_csv, 2, 1)
            file_layout.addWidget(self.btn_browse_csv, 2, 2)
            
            file_layout.addWidget(QLabel("Video kết quả (.mp4 - Tùy chọn):"), 3, 0)
            self.txt_video = QLineEdit()
            self.txt_video.setPlaceholderText("Đường dẫn lưu video kết quả để xem (FPS sẽ giảm nhẹ)")
            self.btn_browse_video = QPushButton("Lưu tại...")
            self.btn_browse_video.clicked.connect(self.browse_video)
            file_layout.addWidget(self.txt_video, 3, 1)
            file_layout.addWidget(self.btn_browse_video, 3, 2)
            
            main_layout.addWidget(file_group)
            
            # --- PARAMETERS GROUP ---
            param_group = QGroupBox("Tham số Thuật toán")
            param_layout = QGridLayout(param_group)
            
            # Chế độ depth
            param_layout.addWidget(QLabel("Chế độ Depth:"), 0, 0)
            self.cb_depth = QComboBox()
            self.cb_depth.addItems(["PERFORMANCE", "NONE", "NEURAL_PLUS", "NEURAL", "ULTRA", "QUALITY"])
            param_layout.addWidget(self.cb_depth, 0, 1)
            
            # Giải thuật khoảng cách
            param_layout.addWidget(QLabel("Thuật toán Khoảng cách:"), 0, 2)
            self.cb_coord = QComboBox()
            self.cb_coord.addItems(["depth", "pnp"])
            param_layout.addWidget(self.cb_coord, 0, 3)
            
            # Tag Source ID
            param_layout.addWidget(QLabel("ID Thẻ Nguồn (Source):"), 1, 0)
            self.sb_source_id = QSpinBox()
            self.sb_source_id.setValue(1)
            param_layout.addWidget(self.sb_source_id, 1, 1)
            
            # Tag target IDs
            param_layout.addWidget(QLabel("Danh sách ID Thẻ Đích (Targets):"), 1, 2)
            self.txt_targets = QLineEdit()
            self.txt_targets.setPlaceholderText("Ví dụ: 2, 3")
            param_layout.addWidget(self.txt_targets, 1, 3)
            
            # Tag Size
            param_layout.addWidget(QLabel("Kích thước AprilTag (mm):"), 2, 0)
            self.sb_tag_size = QDoubleSpinBox()
            self.sb_tag_size.setRange(1.0, 1000.0)
            self.sb_tag_size.setValue(150.0)
            param_layout.addWidget(self.sb_tag_size, 2, 1)
            
            # ROI Size
            param_layout.addWidget(QLabel("Kích thước Vùng ROI (px):"), 2, 2)
            self.sb_roi_size = QSpinBox()
            self.sb_roi_size.setRange(50, 1000)
            self.sb_roi_size.setValue(350)
            param_layout.addWidget(self.sb_roi_size, 2, 3)
            
            # ROI Checkbox
            self.chk_disable_roi = QCheckBox("Tắt bám vết vùng ROI (Quét toàn khung hình 2K)")
            param_layout.addWidget(self.chk_disable_roi, 3, 0, 1, 4)
            
            main_layout.addWidget(param_group)
            
            # --- CONSOLE/LOG GROUP ---
            log_group = QGroupBox("Nhật ký Xử lý")
            log_layout = QVBoxLayout(log_group)
            self.txt_log = QTextEdit()
            self.txt_log.setReadOnly(True)
            self.txt_log.setFont(QFont("Consolas", 10))
            log_layout.addWidget(self.txt_log)
            main_layout.addWidget(log_group)
            
            # --- PROGRESS BAR AND BUTTONS ---
            bottom_layout = QHBoxLayout()
            self.progress_bar = QProgressBar()
            self.progress_bar.setValue(0)
            bottom_layout.addWidget(self.progress_bar)
            
            self.btn_run = QPushButton("BẮT ĐẦU XỬ LÝ SVO")
            self.btn_run.clicked.connect(self.start_processing)
            bottom_layout.addWidget(self.btn_run)
            
            main_layout.addLayout(bottom_layout)

        def apply_stylesheet(self):
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #1a1a1a;
                }
                QLabel {
                    color: #d1d1d1;
                    font-size: 13px;
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
                QLineEdit, QComboBox {
                    background-color: #2c2c2c;
                    border: 1px solid #3c3c3c;
                    border-radius: 4px;
                    padding: 5px;
                    color: #ffffff;
                }
                QDoubleSpinBox, QSpinBox {
                    background-color: #2c2c2c;
                    border: 1px solid #3c3c3c;
                    border-radius: 4px;
                    padding: 5px;
                    color: #ffffff;
                }
                QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
                    border: 1px solid #00e5ff;
                }
                QCheckBox {
                    color: #d1d1d1;
                }
                QPushButton {
                    background-color: #00c853;
                    border: none;
                    border-radius: 4px;
                    padding: 8px 16px;
                    color: #ffffff;
                    font-weight: bold;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background-color: #00e676;
                }
                QPushButton[btnType="danger"] {
                    background-color: #d32f2f;
                }
                QPushButton[btnType="danger"]:hover {
                    background-color: #f44336;
                }
                QTextEdit {
                    background-color: #121212;
                    border: 1px solid #2d2d2d;
                    color: #ffffff;
                    border-radius: 4px;
                }
                QProgressBar {
                    border: 1px solid #2d2d2d;
                    border-radius: 4px;
                    background-color: #222222;
                    text-align: center;
                    color: #ffffff;
                    font-weight: bold;
                    height: 25px;
                }
                QProgressBar::chunk {
                    background-color: #00e5ff;
                }
            """)

        def auto_find_files(self):
            # Tự động quét tìm file svo mặc định
            default_svo = "HD2K_SN35214682_09-30-47.svo2"
            if os.path.exists(default_svo):
                self.txt_svo.setText(os.path.abspath(default_svo))
                
            # Quét tìm file bản đồ .json gần nhất trong thư mục
            files = sorted([f for f in os.listdir('.') if f.startswith('reference_map_autosave') and f.endswith('.json')])
            if files:
                self.txt_map.setText(os.path.abspath(files[-1]))

        def browse_svo(self):
            path, _ = QFileDialog.getOpenFileName(self, "Chọn file ZED SVO", "", "SVO Files (*.svo *.svo2)")
            if path:
                self.txt_svo.setText(path)

        def browse_map(self):
            path, _ = QFileDialog.getOpenFileName(self, "Chọn file bản đồ cấu trúc JSON", "", "JSON Files (*.json)")
            if path:
                self.txt_map.setText(path)

        def browse_csv(self):
            path, _ = QFileDialog.getSaveFileName(self, "Lưu tệp CSV", "", "CSV Files (*.csv)")
            if path:
                self.txt_csv.setText(path)

        def browse_video(self):
            path, _ = QFileDialog.getSaveFileName(self, "Lưu Video bám vết", "", "Video Files (*.mp4)")
            if path:
                self.txt_video.setText(path)

        def start_processing(self):
            if self.worker and self.worker.isRunning():
                # Hủy bỏ xử lý
                self.btn_run.setEnabled(False)
                self.worker.cancel_requested = True
                self.txt_log.append("\n[GUI] Đang gửi yêu cầu dừng xử lý...")
                return

            svo_path = self.txt_svo.text().strip()
            if not svo_path or not os.path.exists(svo_path):
                QMessageBox.warning(self, "Cảnh báo", "Vui lòng chọn đường dẫn tệp SVO hợp lệ.")
                return
                
            # Đọc cấu hình từ UI đóng gói làm args
            class DummyArgs:
                pass
            args = DummyArgs()
            args.svo = svo_path
            args.map = self.txt_map.text().strip() if self.txt_map.text().strip() else None
            args.csv = self.txt_csv.text().strip()
            args.output_video = self.txt_video.text().strip() if self.txt_video.text().strip() else None
            args.depth_mode = self.cb_depth.currentText()
            args.coord_mode = self.cb_coord.currentText()
            args.source_id = self.sb_source_id.value()
            args.target_ids = self.txt_targets.text().strip()
            args.tag_size = self.sb_tag_size.value()
            args.roi_size = self.sb_roi_size.value()
            args.filter_alpha = 0.7
            args.max_lost_frames = 15
            args.disable_roi = self.chk_disable_roi.isChecked()

            # Thiết lập UI hoạt động
            self.txt_log.clear()
            self.progress_bar.setValue(0)
            self.btn_run.setText("HỦY BỎ XỬ LÝ")
            self.btn_run.setProperty("btnType", "danger")
            self.apply_stylesheet() # Cập nhật màu nút đỏ

            # Khởi chạy Worker Thread
            self.worker = SVOWorkerThread(args)
            self.worker.progress_updated.connect(self.update_progress)
            self.worker.log_received.connect(self.write_log)
            self.worker.finished_signal.connect(self.processing_finished)
            self.worker.start()

        @Slot(int, int)
        def update_progress(self, curr, total):
            if total > 0:
                percent = int(curr / total * 100)
                self.progress_bar.setValue(percent)

        @Slot(str)
        def write_log(self, text):
            self.txt_log.append(text)
            # Tự động cuộn xuống cuối
            self.txt_log.ensureCursorVisible()

        @Slot(bool, str)
        def processing_finished(self, success, message):
            self.btn_run.setText("BẮT ĐẦU XỬ LÝ SVO")
            self.btn_run.setProperty("btnType", "normal")
            self.btn_run.setEnabled(True)
            self.apply_stylesheet()

            if success:
                QMessageBox.information(self, "Hoàn thành", f"Quá trình xử lý SVO hoàn tất!\n\n{message}")
            else:
                QMessageBox.critical(self, "Lỗi xảy ra", f"Xử lý thất bại:\n\n{message}")

# --- ENTRY POINT ---
if __name__ == "__main__":
    # Nhận diện tham số dòng lệnh
    if len(sys.argv) > 1:
        # Chạy chế độ dòng lệnh (CLI Mode)
        parser = argparse.ArgumentParser(description="Xử lý offline tệp ZED SVO với FPS tối đa sử dụng ROI Tracking.")
        parser.add_argument("--svo", type=str, default="HD2K_SN35214682_09-30-47.svo2", help="Đường dẫn tới tệp SVO cần xử lý.")
        parser.add_argument("--map", type=str, default=None, help="Đường dẫn tới tệp JSON bản đồ cấu trúc (được tạo bởi GroundTruth.py).")
        parser.add_argument("--csv", type=str, default="svo_report.csv", help="Đường dẫn lưu file dữ liệu CSV.")
        parser.add_argument("--output_video", type=str, default=None, help="Đường dẫn lưu video kết quả (ví dụ: result.mp4). Để trống để tắt và đạt FPS cao nhất.")
        parser.add_argument("--depth_mode", type=str, default="PERFORMANCE", choices=["NONE", "PERFORMANCE", "QUALITY", "ULTRA", "NEURAL", "NEURAL_PLUS"],
                            help="Chế độ depth của ZED SDK. Chọn 'NONE' để chỉ dùng SolvePnP (đạt tốc độ >200 FPS).")
        parser.add_argument("--coord_mode", type=str, default="depth", choices=["depth", "pnp"], help="Giải thuật tính khoảng cách.")
        parser.add_argument("--tag_size", type=float, default=150.0, help="Kích thước viền ngoài AprilTag (mm).")
        parser.add_argument("--source_id", type=int, default=1, help="ID thẻ nguồn.")
        parser.add_argument("--target_ids", type=str, default="", help="Danh sách ID thẻ mục tiêu (ngăn cách bằng dấu phẩy, ví dụ '2,3').")
        parser.add_argument("--roi_size", type=int, default=350, help="Kích thước vùng tìm kiếm ROI xung quanh tag.")
        parser.add_argument("--filter_alpha", type=float, default=0.7, help="Hệ số EMA làm mịn.")
        parser.add_argument("--max_lost_frames", type=int, default=15, help="Số khung hình tối đa giữ trạng thái khi mất dấu.")
        parser.add_argument("--disable_roi", action="store_true", help="Tắt tính năng ROI Tracking (chạy quét toàn khung hình để so sánh).")
        
        args = parser.parse_args()
        
        if not os.path.exists(args.svo):
            print(f"[ERROR] Không tìm thấy tệp SVO tại đường dẫn: {args.svo}")
            sys.exit(1)
            
        process_svo(args)
    else:
        # Chạy chế độ giao diện đồ họa (GUI Mode)
        if not HAS_GUI_LIBS:
            print("[ERROR] Không tìm thấy PySide6. Vui lòng cài đặt để chạy GUI.")
            sys.exit(1)
            
        app = QApplication(sys.argv)
        gui = ProcessSVOGUI()
        gui.show()
        sys.exit(app.exec())
