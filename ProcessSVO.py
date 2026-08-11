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
    print("Vui lòng chạy file get_python_api.py trong thư mục ZED SDK để cài đặt.")
    sys.exit(1)

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

def process_svo(args):
    print(f"\n=======================================================")
    print(f"BẮT ĐẦU XỬ LÝ OFFLINE SVO: {args.svo}")
    print(f"=======================================================")
    
    # 1. Cấu hình ZED SDK đọc tệp SVO
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(args.svo)
    
    # KHÔNG giới hạn theo thời gian thực để chạy với tốc độ cao nhất
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.METER
    
    # Lựa chọn chế độ depth tương ứng
    depth_mode_map = {
        "NONE": sl.DEPTH_MODE.NONE, # Cực nhanh, chỉ chạy PnP trên CPU
        "NEURAL": getattr(sl.DEPTH_MODE, "NEURAL", sl.DEPTH_MODE.QUALITY),
        "NEURAL_PLUS": getattr(sl.DEPTH_MODE, "NEURAL_PLUS", sl.DEPTH_MODE.QUALITY),
        "ULTRA": sl.DEPTH_MODE.ULTRA,
        "QUALITY": sl.DEPTH_MODE.QUALITY,
        "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE
    }
    
    selected_depth_mode = depth_mode_map.get(args.depth_mode.upper(), sl.DEPTH_MODE.PERFORMANCE)
    init_params.depth_mode = selected_depth_mode
    print(f"[*] Cấu hình chế độ Depth: {args.depth_mode.upper()}")
    
    zed = sl.Camera()
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] Không thể mở tệp SVO: {err}")
        return
        
    nb_frames = zed.get_svo_number_of_frames()
    print(f"[*] Tổng số khung hình trong SVO: {nb_frames}")
    
    # 2. Cấu hình bộ nhận diện AprilTag
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    
    # Tối ưu hóa tham số detector để xử lý nhanh hơn nữa
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.minMarkerPerimeterRate = 0.03
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE # Dùng ROI bám góc nên không cần refine toàn cục
    
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    
    # Cài đặt biến lưu trữ
    trackers = {}
    
    # Chuẩn bị file CSV để xuất dữ liệu
    csv_file = open(args.csv, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Frame_Index", "Timestamp_MS", 
        "Source_ID", "Source_X_mm", "Source_Y_mm", "Source_Z_mm",
        "Num_Targets", "Polyline_Dist_mm", "Target_IDs"
    ])
    
    # Nếu cấu hình ghi đè Video để xem kết quả
    video_writer = None
    if args.output_video:
        # Lấy thông số độ phân giải thực tế của camera
        cam_info = zed.get_camera_information()
        w = cam_info.camera_configuration.resolution.width
        h = cam_info.camera_configuration.resolution.height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(args.output_video, fourcc, 30.0, (w, h))
        print(f"[*] Đang bật ghi video kết quả ra: {args.output_video} (Tốc độ xử lý sẽ giảm nhẹ do encoder)")
    
    # Tạo các ma trận trung gian
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime_params = sl.RuntimeParameters()
    
    # Lấy thông số Calibration
    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    cam_matrix = np.array([
        [calib.fx, 0, calib.cx],
        [0, calib.fy, calib.cy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32) # SVO đã được Rectify sẵn bởi SDK nên coeffs bằng 0
    
    # Cấu hình đối tượng 3D AprilTag phục vụ giải PnP
    s = args.tag_size / 1000.0 # Quy đổi mm sang mét
    obj_points = np.array([
        [-s/2, -s/2, 0],
        [ s/2, -s/2, 0],
        [ s/2,  s/2, 0],
        [-s/2,  s/2, 0]
    ], dtype=np.float32)
    
    # Phân tích danh sách Target Tag IDs
    target_ids = []
    if args.target_ids:
        try:
            target_ids = [int(x.strip()) for x in args.target_ids.split(",") if x.strip().isdigit()]
        except Exception:
            pass
            
    print(f"[*] Thẻ Nguồn (Source): ID {args.source_id}")
    print(f"[*] Danh sách Thẻ Đích (Targets): {target_ids}")
    
    # Đo đạc hiệu năng FPS
    start_time = time.time()
    processed_count = 0
    
    print("\n--- BẮT ĐẦU CHẠY PHÂN TÍCH (FPS TỐI ĐA) ---")
    
    for f_idx in range(nb_frames):
        # Đọc khung hình
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            break
            
        # Lấy ảnh trái (Left RGB)
        zed.retrieve_image(image_mat, sl.VIEW.LEFT)
        bgra_image = image_mat.get_data()
        color_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # Lấy dữ liệu độ sâu (nếu chế độ depth không phải NONE)
        depth_data = None
        if selected_depth_mode != sl.DEPTH_MODE.NONE:
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth_data = depth_mat.get_data()
            
        # --- THUẬT TOÁN ROI TRACKING ---
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
            
        # Cập nhật thông tin các tag phát hiện được
        detected_this_frame = set()
        active_tags = {}
        
        if ids is not None:
            flat_ids = np.ravel(ids)
            for i, tag_id_val in enumerate(flat_ids):
                tag_id = int(tag_id_val)
                corners_i = np.array(corners[i], dtype=np.float32).reshape(4, 2)
                
                # Giải thuật PnP tính tọa độ 3D
                success, rvec, tvec = cv2.solvePnP(obj_points, corners_i, cam_matrix, dist_coeffs)
                p_3d_pnp = tvec.flatten() if success else np.array([0.0, 0.0, 0.0])
                
                # Giải thuật dùng cảm biến độ sâu (Depth)
                p_3d_depth = p_3d_pnp.copy()
                depth_val = 0.0
                if depth_data is not None:
                    u_c = float(np.mean(corners_i[:, 0]))
                    v_c = float(np.mean(corners_i[:, 1]))
                    
                    # Trích xuất giá trị độ sâu trung vị xung quanh tâm
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
                        # Deproject
                        z = depth_val
                        x = (u_c - calib.cx) * z / calib.fx
                        y = (v_c - calib.cy) * z / calib.fy
                        p_3d_depth = np.array([x, y, z])
                
                # Cập nhật bộ lọc
                if tag_id not in trackers:
                    trackers[tag_id] = TagTracker(tag_id, alpha=args.filter_alpha, max_lost_frames=args.max_lost_frames)
                trackers[tag_id].update(corners_i, p_3d_pnp, p_3d_depth, rvec, tvec,
                                        alpha=args.filter_alpha, max_lost_frames=args.max_lost_frames)
                detected_this_frame.add(tag_id)
                
        # Dự đoán cho các tag bị mất dấu tạm thời
        for tid, tracker in list(trackers.items()):
            if tid not in detected_this_frame:
                tracker.predict(max_lost_frames=args.max_lost_frames)
                
        # Tổng hợp dữ liệu các tag đang active
        for tid, tracker in trackers.items():
            if tracker.is_tracked:
                pos_pnp = tracker.pos_pnp_filtered
                pos_depth = tracker.pos_depth_filtered
                
                active_tags[tid] = {
                    'pos_pnp': pos_pnp,
                    'pos_depth': pos_depth,
                    'corners': tracker.corners_filtered,
                    'rvec': tracker.rvec_filtered,
                    'tvec': tracker.tvec_filtered,
                    'lost_frames': tracker.lost_frames
                }
                
        # Tính toán khoảng cách
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
                polyline_dist = polyline_dist * 1000.0 # Quy đổi sang mm
                
        # Ghi dữ liệu vào CSV
        ts_ms = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        src_coords = [source_pos[0]*1000.0, source_pos[1]*1000.0, source_pos[2]*1000.0] if source_pos is not None else ["", "", ""]
        
        csv_writer.writerow([
            f_idx, ts_ms, 
            args.source_id, src_coords[0], src_coords[1], src_coords[2],
            len(target_pts), polyline_dist if polyline_dist is not None else "",
            ",".join(map(str, detected_targets_list))
        ])
        
        # Vẽ đồ họa vẽ đè kết quả lên video (nếu bật)
        if video_writer is not None:
            # Vẽ viền cho các tag đang active
            for tid, tag in active_tags.items():
                corners_i = tag['corners'].astype(np.int32)
                color = (0, 0, 255) if tid == args.source_id else ((0, 255, 255) if tid in target_ids else (128, 128, 128))
                thickness = 3 if tid == args.source_id or tid in target_ids else 1
                for k in range(4):
                    cv2.line(color_image, tuple(corners_i[k]), tuple(corners_i[(k+1)%4]), color, thickness)
                cv2.putText(color_image, f"ID:{tid}", (int(corners_i[0][0]), int(corners_i[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Ghi thông tin FPS lên hình
            cv2.putText(color_image, f"Frame: {f_idx}/{nb_frames}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            video_writer.write(color_image)
            
        processed_count += 1
        
        # In tiến trình xử lý mỗi 100 khung hình
        if processed_count % 100 == 0 or processed_count == nb_frames:
            elapsed = time.time() - start_time
            curr_fps = processed_count / elapsed if elapsed > 0 else 0
            print(f" -> Đã xử lý {processed_count}/{nb_frames} khung hình ({processed_count/nb_frames*100:.1f}%). FPS hiện tại: {curr_fps:.2f}")
            
    # Dọn dẹp
    csv_file.close()
    if video_writer is not None:
        video_writer.release()
    zed.close()
    
    total_elapsed = time.time() - start_time
    final_fps = processed_count / total_elapsed if total_elapsed > 0 else 0
    print(f"\n=======================================================")
    print(f"HOÀN THÀNH XỬ LÝ SVO!")
    print(f"Tổng số frame đã xử lý: {processed_count}")
    print(f"Tổng thời gian: {total_elapsed:.2f} giây")
    print(f"FPS TRUNG BÌNH ĐẠT ĐƯỢC: {final_fps:.2f} khung hình/giây")
    print(f"Dữ liệu CSV xuất ra: {args.csv}")
    if args.output_video:
        print(f"Video kết quả xuất ra: {args.output_video}")
    print(f"=======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xử lý offline tệp ZED SVO với FPS tối đa sử dụng ROI Tracking.")
    parser.add_argument("--svo", type=str, default="HD2K_SN35214682_09-30-47.svo2", help="Đường dẫn tới tệp SVO cần xử lý.")
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
    
    # Kiểm tra sự tồn tại của tệp SVO
    if not os.path.exists(args.svo):
        print(f"[ERROR] Không tìm thấy tệp SVO tại đường dẫn: {args.svo}")
        sys.exit(1)
        
    process_svo(args)
