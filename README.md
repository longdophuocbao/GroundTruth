# Hệ Thống Đo Khoảng Cách AprilTag 3D - Intel RealSense D455 & Stereolabs ZED 2i

Dự án này cung cấp một giải pháp phần mềm hoàn chỉnh và trực quan nhằm xác định tọa độ 3D của các thẻ **AprilTag** và tính toán khoảng cách thực tế giữa một thẻ nguồn (Source) đến một đường gấp khúc mục tiêu (Polyline) tạo bởi nhiều thẻ đích (Targets). 

Hệ thống hỗ trợ đa thiết bị camera đo độ sâu tiên tiến bao gồm **Intel RealSense D455** và **Stereolabs ZED 2i** (ở cả chế độ UVC tiêu chuẩn và chế độ ZED SDK tận dụng sức mạnh GPU CUDA), được tối ưu cho các nghiên cứu khoa học, thử nghiệm robot và đo lường hệ thống phản hồi vị trí thực (Ground Truth) có độ chính xác cao.

---

## 🌟 Tính Năng Nổi Bật

*   **Hỗ trợ Đa Camera (Multi-Camera Support)**:
    *   **Intel RealSense D455**: Kết nối trực tiếp qua RealSense SDK, hỗ trợ đầy đủ các bộ lọc phần cứng và công suất laser phát xạ.
    *   **ZED 2i (ZED SDK CUDA Mode)**: Khai thác sức mạnh phần cứng GPU NVIDIA qua ZED SDK 5.0.7+, hỗ trợ đo độ sâu bằng Trí tuệ nhân tạo (Neural AI).
    *   **ZED 2i (UVC Mode)**: Sử dụng luồng video RGB tiêu chuẩn qua giao tiếp UVC, kết hợp giải thuật hình học SolvePnP để xác định khoảng cách 3D.
*   **Phát hiện AprilTag 3D thời gian thực**: Sử dụng OpenCV Aruco hỗ trợ họ thẻ `DICT_APRILTAG_36h11` với độ trễ cực thấp.
*   **Bộ lọc chiều sâu RealSense SDK (Post-Processing)**:
    *   *Threshold Filter*: Giới hạn khoảng cách đo chiều sâu Min/Max để lọc bỏ vật thể quá xa hoặc quá gần.
    *   *Decimation Filter*: Giảm độ phân giải chiều sâu giúp tăng tốc độ xử lý và giảm nhiễu hạt.
    *   *Spatial & Temporal Filters*: Làm mịn ảnh chiều sâu trong không gian (lọc giữ cạnh) và làm mịn theo thời gian (giảm hiện tượng nhấp nháy pixel).
    *   *Hole Filling Filter*: Vá các lỗ trống chiều sâu do góc khuất hoặc bề mặt phản xạ kém bằng thuật toán nội suy thông minh.
*   **Cấu hình chiều sâu ZED SDK AI (CUDA-Accelerated)**:
    *   *Chế độ AI Depth Mode*: Hỗ trợ 3 chế độ Neural mới nhất của ZED SDK 5: **NEURAL_PLUS** (Độ chính xác tối đa), **NEURAL** (Cân bằng), và **NEURAL_LIGHT** (Tốc độ cao), bên cạnh các chế độ cũ (ULTRA, QUALITY, PERFORMANCE).
    *   *Sensing Mode / Fill Mode*: Tự động áp dụng bộ lọc lấp đầy khoảng trống (Vá lỗ hổng chiều sâu) tương thích với cả ZED SDK 3.x/4.x và 5.x.
    *   *Ngưỡng lọc nhiễu*: Tinh chỉnh `Confidence` (Ngưỡng tin cậy) và `Texture Confidence` (Ngưỡng vân bề mặt) trực tiếp.
*   **Bám vết và lọc nhiễu nâng cao**:
    *   *Bộ lọc mượt EMA (Exponential Moving Average)*: Làm mịn tọa độ 3D và góc quay của thẻ tag, triệt tiêu hiện tượng rung sai số (jitter).
    *   *Cơ chế Keep-Alive*: Duy trì trạng thái và vị trí của thẻ tag khi bị che khuất tạm thời trong số khung hình định trước.
*   **Giao diện đồ họa (GUI) Hiện đại & Trực quan**:
    *   Xây dựng bằng **PySide6 (Qt for Python)** với giao diện tối (Dark Mode) cao cấp.
    *   Vẽ đồ thị khoảng cách thời gian thực (Real-time Plot) mượt mà bằng **pyqtgraph**.
    *   Bảng trạng thái chi tiết (ID, Vai trò, Trạng thái bám vết, Tọa độ X, Y, Z tính bằng mm).
*   **Ghi nhật ký thời gian thực (CSV Logger)**: Lưu dữ liệu đo lường trực tiếp ra file CSV phục vụ hậu xử lý và vẽ đồ thị báo cáo khoa học.
*   **Khả năng tự động vá lỗi DLL (ZED DLL Auto-Fix)**:
    Tự động giải quyết các lỗi nạp DLL của Python trên Windows (như không tìm thấy `sl_ai64.dll` hoặc `CORRUPTED SDK INSTALLATION`). Chương trình tự động thêm các thư mục SDK/CUDA vào DLL search path của Python và sao chép các DLL cốt lõi vào thư mục `pyzed` trong `site-packages` khi khởi chạy.

---

## 🛠️ Yêu Cầu Hệ Thống & Cài Đặt

### 1. Phần cứng
*   Camera **Intel RealSense D455** hoặc **Stereolabs ZED 2i** (kết nối qua cáp USB 3.0 chuyên dụng).
*   Card đồ họa **NVIDIA** (Khuyên dùng GTX 1650 trở lên với driver mới nhất) để hỗ trợ ZED SDK CUDA.
*   Thẻ AprilTag họ `36h11` được in đúng kích thước thực tế (ví dụ: viền đen ngoài cùng rộng 150mm).

### 2. Cài đặt Môi trường & Thư viện Python

Khuyên dùng **Python 3.10** chạy trong môi trường ảo **Conda** trên hệ điều hành Windows để đảm bảo tính ổn định tối đa.

#### Bước 2.1: Tạo và kích hoạt môi trường ảo Conda
Mở PowerShell (hoặc Command Prompt) và chạy các lệnh sau:
```powershell
# Tạo môi trường ảo với Python 3.10
conda create -n groundtruth python=3.10 -y

# Kích hoạt môi trường ảo
conda activate groundtruth
```

#### Bước 2.2: Cài đặt các thư viện phụ thuộc chính
```bash
pip install numpy opencv-python pyrealsense2 PySide6 pyqtgraph requests
```

#### Bước 2.3: Cài đặt wrapper Python cho ZED SDK (pyzed)
Đảm bảo bạn đã tải và cài đặt ZED SDK (v5.0.7 trở lên dành cho CUDA 11.8 / TensorRT 10.9) từ website chính thức của Stereolabs. 

Sau đó, tiến hành cài đặt wrapper `pyzed` vào môi trường ảo của bạn bằng cách chạy lệnh sau:
```powershell
python "C:\Program Files (x86)\ZED SDK\get_python_api.py"
```
*(Nếu cài đặt ZED SDK ở thư mục khác, hãy thay đổi đường dẫn tới tệp `get_python_api.py` tương ứng).*

---

## 🚀 Hướng Dẫn Sử Dụng

### Bước 1: Kết nối và Thiết lập Tag
1. Cắm camera đo độ sâu của bạn vào cổng **USB 3.0** của máy tính.
2. Sắp xếp các thẻ AprilTag trong tầm quan sát của camera:
    *   Đặt một thẻ làm **Tag Nguồn (Source)** (Mặc định cấu hình là ID 1).
    *   Đặt các thẻ khác làm **Tag Mục Tiêu (Targets)** (ví dụ: ID 0, 3) để tạo thành một đường dẫn (Polyline).

### Bước 2: Khởi chạy phần mềm
Trong môi trường ảo Conda đã được kích hoạt, khởi chạy tệp mã nguồn chính:
```bash
python GroundTruth.py
```

### Bước 3: Cấu hình trên giao diện
1.  **Chọn Camera**: Trên bảng điều khiển chính, chọn camera tương ứng:
    *   `Intel RealSense D455`
    *   `ZED 2i (UVC Mode)` (chỉ sử dụng giải thuật PnP để tính tọa độ 3D)
    *   `ZED 2i (ZED SDK CUDA Mode)`
2.  **Mở Hộp Thoại Cài Đặt**: Trên thanh công cụ, nhấn vào menu **Cài đặt** -> chọn **Cấu hình hệ thống...**. 
    *   Giao diện cài đặt sẽ tự động thay đổi cấu trúc bộ lọc và nhãn tham số dựa trên camera đang kết nối.
    *   **Đối với RealSense**: Điều chỉnh các bộ lọc phần cứng (Threshold Min/Max, Decimation, Hole Filling, Spatial, Temporal) và Công suất Laser.
    *   **Đối với ZED SDK**: Lựa chọn chế độ chiều sâu AI (`NEURAL_PLUS`, `NEURAL`, `NEURAL_LIGHT` - mặc định ZED SDK 5.0.7 sẽ clamp `depth_min` ở mức tối thiểu vật lý là `0.4` mét), chế độ lấp đầy `Sensing/Fill Mode`, ngưỡng `Confidence` và `Texture Confidence`.
3.  **Thuật toán tọa độ 3D**: Lựa chọn tính khoảng cách bằng cảm biến độ sâu (Depth Sensor) hoặc SolvePnP hình học.
4.  **Bấm "Run"**: Nhấn nút **Run** (màu xanh lá) để bắt đầu nhận luồng video và đo đạc thực tế.

---

## 📦 Biên Dịch Ra File Thực Thi (.exe)

Nếu bạn muốn đóng gói ứng dụng thành một tệp `.exe` chạy độc lập bằng **Nuitka**:

1. Cài đặt Nuitka:
   ```bash
   pip install nuitka
   ```
2. Chạy lệnh biên dịch (đã tích hợp biểu tượng chuyên nghiệp `diving.ico` và đóng gói thành 1 file duy nhất):
   ```bash
   python -m nuitka --standalone --onefile --enable-plugin=pyside6 --enable-plugin=numpy --windows-console-mode=disable --include-package=pyrealsense2 --windows-icon-from-ico=diving.ico --assume-yes-for-downloads GroundTruth.py
   ```

---

## 🛠️ Xử Lý Sự Cố Thường Gặp (Troubleshooting)

1.  **Lỗi: `sl_ai library loading failed / sl_ai64.dll` hoặc `CORRUPTED SDK INSTALLATION`**
    *   *Nguyên nhân*: Trình biên dịch Python 3.8+ trên Windows không tự tìm kiếm DLL trong biến môi trường `PATH`.
    *   *Khắc phục*: File mã nguồn `GroundTruth.py` đã tích hợp cơ chế **ZED DLL Auto-Fix** để tự động vá lỗi này bằng cách sao chép các DLL gốc vào thư mục của `pyzed`. Hãy đảm bảo bạn khởi chạy chương trình bằng quyền User thông thường và kiểm tra xem thư mục `pyzed` đã có các file `sl_zed64.dll` và `sl_ai64.dll` hay chưa.
2.  **Cảnh báo: `depth_minimum_distance: 0.3METER is too close, clamped to 0.4METER`**
    *   *Khắc phục*: ZED 2i có giới hạn phần cứng tiêu cự tối thiểu là 0.4m. Ứng dụng đã được cấu hình mặc định là `0.4` m. Bạn nên tránh hạ giá trị này xuống dưới 0.4m để đảm bảo dữ liệu đo đạc chính xác nhất.
3.  **Lỗi nạp camera khi dùng ZED SDK ở chế độ NEURAL**:
    *   Nếu máy tính bị lỗi driver card đồ họa hoặc TensorRT không tương thích, camera sẽ không thể mở được ở chế độ NEURAL.
    *   *Khắc phục*: Hệ thống đã tích hợp cơ chế tự động chuyển đổi dự phòng (Fallback) sang chế độ **`ULTRA`** (không dùng AI) để ứng dụng hoạt động bình thường. Bạn có thể thay đổi thiết lập này trực tiếp trong Hộp thoại cài đặt.
