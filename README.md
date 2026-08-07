# Hệ Thống Đo Khoảng Cách AprilTag 3D - Intel RealSense D455

Dự án này cung cấp một giải pháp phần mềm hoàn chỉnh và trực quan nhằm xác định tọa độ 3D của các thẻ **AprilTag** và tính toán khoảng cách thực tế giữa một thẻ nguồn (Source) đến một đường gấp khúc mục tiêu (Polyline) tạo bởi nhiều thẻ đích (Targets). Hệ thống được thiết kế đặc biệt để phục vụ cho các nghiên cứu khoa học, thử nghiệm robot và đo lường hệ thống phản hồi vị trí thực (Ground Truth) có độ chính xác cao sử dụng camera đo độ sâu **Intel RealSense D455**.

---

## 🌟 Tính Năng Nổi Bật

*   **Phát hiện AprilTag 3D thời gian thực**: Sử dụng thư viện OpenCV Aruco hỗ trợ họ thẻ `DICT_APRILTAG_36h11`.
*   **Các bộ lọc xử lý ảnh chiều sâu RealSense SDK (Post-Processing)**:
    *   **Threshold Filter**: Lọc bỏ các điểm ảnh nằm ngoài giới hạn khoảng cách Min và Max mong muốn để loại bỏ nhiễu nền hoặc các vật thể ở quá xa/quá gần.
    *   **Decimation Filter**: Giảm độ phân giải của hình ảnh độ sâu để giảm nhiễu hạt và tăng tốc độ xử lý.
    *   **Spatial Filter**: Làm mịn ảnh chiều sâu trong không gian bằng thuật toán lọc giữ biên cạnh (edge-preserving).
    *   **Temporal Filter**: Làm mịn ảnh chiều sâu theo thời gian, giảm hiện tượng nhấp nháy pixel chiều sâu giữa các frame.
    *   **Hole Filling Filter**: Vá các lỗ trống (vùng thiếu dữ liệu depth do bị khuất hoặc phản xạ kém) bằng thuật toán nội suy thông minh.
*   **Bám vết và lọc nhiễu nâng cao**:
    *   **Bộ lọc mượt EMA (Exponential Moving Average)**: Giảm thiểu hiện tượng rung sai số (jitter) trong tọa độ 3D và góc xoay.
    *   **Cơ chế Keep-Alive**: Duy trì bám vết và nội suy vị trí của thẻ tag ngay cả khi bị che khuất tạm thời (trong khoảng số lượng khung hình định trước).
*   **Hai thuật toán tính toán tọa độ 3D linh hoạt**:
    1.  **Cảm biến Depth RealSense (Mặc định)**: Lấy trực tiếp dữ liệu khoảng cách chiều sâu tại vùng tâm thẻ tag và chiếu ngược (deproject) sang hệ tọa độ 3D.
    2.  **SolvePnP hình học**: Sử dụng thông số nội tại (Camera Intrinsics) của camera màu RealSense kết hợp kích thước thực tế của tag để giải bài toán Pose Estimation.
*   **Đo khoảng cách đến đường gấp khúc (Polyline)**: Tính toán khoảng cách vuông góc ngắn nhất từ vị trí thẻ Nguồn (Source) đến đường dẫn tạo bởi chuỗi thẻ Mục tiêu (Targets) trong không gian 3D.
*   **Hiển thị luồng Depth chiều sâu song song (Tùy chọn)**: Hỗ trợ hiển thị đồng thời cả camera màu RGB (đã vẽ đè tọa độ/đo đạc) và camera chiều sâu (được tô màu colorized bằng SDK RealSense) để giám sát trực quan hơn, có thể bật/tắt linh hoạt.
*   **Giao diện đồ họa (GUI) trực quan & hiện đại**: 
    *   Xây dựng bằng **PySide6 (Qt for Python)** với giao diện tối (Dark Mode) cao cấp.
    *   Vẽ đồ thị khoảng cách thời gian thực (Real-time Plot) mượt mà bằng **pyqtgraph**.
    *   Bảng theo dõi trạng thái chi tiết của từng tag (ID, Vai trò, Trạng thái bám vết, Tọa độ X, Y, Z tính bằng mm).
*   **Ghi nhật ký đo lường (Real-time Logger)**: Hỗ trợ ghi dữ liệu trực tiếp ra file CSV phục vụ cho việc hậu xử lý, vẽ đồ thị nghiên cứu khoa học.

---

## 📁 Cấu Trúc Thư Mục

```text
ground truth/
├── GroundTruth.py            # Mã nguồn chính của ứng dụng GUI PySide6
└── README.md                 # Tài liệu hướng dẫn sử dụng (File này)
```

---

## 🛠️ Yêu Cầu Hệ Thống & Cài Đặt

### 1. Phần cứng
*   Camera đo độ sâu **Intel RealSense D455** (hoặc các phiên bản tương đương như D435, D435i).
*   Cáp kết nối USB 3.0 chuyên dụng để truyền dữ liệu băng thông rộng.
*   Các thẻ AprilTag (họ `36h11`) được in đúng kích thước thực tế (mặc định là 150mm). Có thể sử dụng các file SVG đi kèm trong thư mục này để in.

### 2. Phần mềm & Thư viện Python
Yêu cầu phiên bản **Python 3.9 trở lên**. Cài đặt các thư viện phụ thuộc bằng lệnh sau:

```bash
pip install numpy opencv-python pyrealsense2 PySide6 pyqtgraph
```

> [!IMPORTANT]
> Đảm bảo rằng Driver và SDK của Intel RealSense đã được cấu hình đúng trên máy tính để thư viện `pyrealsense2` có thể giao tiếp với thiết bị phần cứng.

---

## 🚀 Hướng Dẫn Sử Dụng

### Bước 1: Kết nối thiết bị
1. Cắm camera Intel RealSense D455 vào cổng **USB 3.0** của máy tính.
2. Sắp xếp các thẻ AprilTag trong tầm quan sát của camera:
    *   Đặt một thẻ làm **Tag Nguồn (Source)** (Mặc định là ID 1).
    *   Đặt một hoặc nhiều thẻ khác làm **Tag Mục Tiêu (Targets)** (Ví dụ: ID 0, ID 3). Các tag này sẽ tạo thành một đường dẫn (đường gấp khúc).

### Bước 2: Khởi chạy phần mềm
Chạy tệp mã nguồn chính bằng Python:

```bash
python GroundTruth.py
```

### Bước 3: Cấu hình trên giao diện
1.  **Mở Hộp Thoại Cài Đặt**: Trên thanh công cụ, nhấn vào menu **Cài đặt** -> chọn **Cấu hình hệ thống...**. Hộp thoại cài đặt sẽ xuất hiện và cho phép điều chỉnh mọi thông số của hệ thống trực tiếp (real-time).
2.  **Kích thước AprilTag**: Nhập kích thước thực tế của viền đen ngoài cùng thẻ tag (tính bằng milimet, ví dụ: `150.0 mm`).
3.  **AprilTag ID Nguồn**: Nhập ID của thẻ tag di động cần đo khoảng cách (mặc định là `1`).
4.  **ID Đường Mục Tiêu**: Nhập danh sách ID các tag cố định tạo nên đường dẫn, cách nhau bằng dấu phẩy (ví dụ: `0,3`). Nếu để trống, hệ thống tự động nhận tất cả các tag khác làm mục tiêu.
5.  **Thuật toán tọa độ 3D**: Chọn giữa `Sử dụng Cảm biến Depth RealSense trực tiếp (Mặc định)` hoặc `Sử dụng SolvePnP hình học`.
6.  **Cấu hình bộ lọc (Tracking)**:
    *   *Kích hoạt bộ lọc mượt 3D (EMA)* để giảm nhiễu giật hình ảnh.
    *   *Khung hình duy trì tối đa (Max Lost)*: Số lượng khung hình mà hệ thống vẫn duy trì vị trí cũ của tag sau khi tag bị che khuất trước khi đánh dấu là mất kết nối.
    *   *Độ dài bộ lọc trung bình (Window Size)*: Số lượng khung hình dùng để tính toán trung bình trượt cho khoảng cách đo được hiển thị trên đồ thị.
7.  **Bộ lọc chiều sâu RealSense (Post-Processing)**:
    *   Tích chọn các bộ lọc tương ứng để cải thiện chất lượng ảnh chiều sâu từ camera: `Threshold Filter` (kèm theo nhập ngưỡng khoảng cách **Min** và **Max** mong muốn tính bằng mét), `Decimation Filter`, `Hole Filling Filter`, `Spatial Filter`, `Temporal Filter`. Các bộ lọc này sẽ tự động được áp dụng trực tiếp theo thứ tự tối ưu của RealSense SDK.

### Bước 4: Vận hành và ghi nhật ký
1.  Nhấn nút **Quét Thiết Bị (Scan)** trên giao diện chính để kiểm tra số lượng và thông tin chi tiết camera Intel RealSense đang kết nối.
2.  Tích chọn **Hiển thị camera Depth chiều sâu** nếu bạn muốn hiển thị đồng thời cả luồng hình ảnh màu RGB và ảnh chiều sâu (được tô màu). Bỏ tích để ẩn luồng depth.
3.  Nhấn nút **Run** (màu xanh lá) để bắt đầu nhận luồng video từ camera RealSense. Nút sẽ tự động chuyển thành nút **Stop** (màu đỏ).
4.  **Ghi nhật ký CSV (Tùy chọn)**:
    *   Nhấn **Chọn File Lưu...** để tạo đường dẫn và tên tệp CSV (mặc định lưu ra màn hình Desktop).
    *   Tích chọn **Bật ghi nhật ký trực tiếp** để bắt đầu lưu dữ liệu. Có thể nhấn bỏ tích để tạm dừng ghi.
5.  Nhấn nút **Stop** (màu đỏ) khi kết thúc quá trình đo đạc để dừng luồng truyền hình và giải phóng thiết bị camera.

---

## 📦 Biên Dịch Ra File Thực Thi (.exe)

Nếu bạn muốn đóng gói ứng dụng thành một tệp `.exe` độc lập để chạy trên các máy tính khác mà không cần cài đặt môi trường Python, bạn có thể biên dịch bằng công cụ **Nuitka**.

### 1. Cài đặt Nuitka
Cài đặt Nuitka từ pip:
```bash
pip install nuitka
```

### 2. Lệnh biên dịch ứng dụng
Chạy một trong các lệnh sau trong thư mục chứa dự án để biên dịch sang tệp `.exe` duy nhất (đã tích hợp biểu tượng chuyên nghiệp `diving.ico`, tối ưu hóa cho PySide6, OpenCV, RealSense, và tự động ẩn cửa sổ Console màu đen khi khởi chạy):

#### Biên dịch bằng Nuitka (Khuyên dùng - tốc độ nhanh nhất, kích thước tối ưu):
```bash
python -m nuitka --standalone --onefile --enable-plugin=pyside6 --enable-plugin=numpy --windows-console-mode=disable --include-package=pyrealsense2 --windows-icon-from-ico=diving.ico --assume-yes-for-downloads GroundTruth.py
```

#### Biên dịch bằng PyInstaller (Phương án thay thế):
```bash
pyinstaller --onefile --windowed --icon=diving.ico GroundTruth.py
```

Sau khi quá trình biên dịch hoàn thành, tệp thực thi `GroundTruth.exe` sẽ được tạo ra sẵn sàng sử dụng.

---

## 📐 Thuật Toán & Cơ Sở Toán Học

### 1. Bộ lọc Exponential Moving Average (EMA)
Để làm mịn tọa độ $P_{filtered}$ tại khung hình hiện tại $t$ từ tọa độ đo thô $P_{raw}$, công thức được áp dụng là:

$$P_{filtered}^{(t)} = \alpha \cdot P_{raw}^{(t)} + (1 - \alpha) \cdot P_{filtered}^{(t-1)}$$

Trong đó:
*   $\alpha$ (Độ phản hồi bộ lọc): Giá trị từ $0.01$ đến $1.0$. Giá trị càng nhỏ thì đường đi càng mượt nhưng sẽ có độ trễ lớn hơn so với thực tế.

### 2. Khoảng cách ngắn nhất từ một điểm đến đường gấp khúc (Polyline)
Đường gấp khúc mục tiêu được tạo bởi chuỗi các điểm 3D của các Target Tags: $Q = \{q_1, q_2, \dots, q_n\}$.
Để tính khoảng cách từ điểm nguồn $P$ đến đường gấp khúc, hệ thống tính khoảng cách từ $P$ đến từng đoạn thẳng $S_i = [q_i, q_{i+1}]$:

1.  Chiếu điểm $P$ xuống đoạn thẳng $S_i$ để tìm điểm tiệm cận nhất $C_i$:
    $$t = \frac{(P - q_i) \cdot (q_{i+1} - q_i)}{\|q_{i+1} - q_i\|^2}$$
    $$t_{clipped} = \max(0, \min(1, t))$$
    $$C_i = q_i + t_{clipped} \cdot (q_{i+1} - q_i)$$
2.  Tính khoảng cách Euclidean giữa $P$ và $C_i$:
    $$d_i = \|P - C_i\|$$
3.  Khoảng cách ngắn nhất đến toàn bộ đường gấp khúc là:
    $$d = \min(d_1, d_2, \dots, d_{n-1})$$

---

## 📊 Định Dạng Nhật Ký CSV Đầu Ra

Tệp nhật ký được xuất dưới định dạng `.csv` chuẩn hóa với các trường thông tin sau:

| Tên Cột | Kiểu Dữ Liệu | Mô Tả |
| :--- | :--- | :--- |
| `Timestamp_Sec` | Float | Thời gian chạy chương trình tính bằng giây (s). |
| `Time_Formatted` | String | Mốc thời gian thực tế chi tiết dạng `YYYY-MM-DD HH:MM:SS.mmm`. |
| `Source_ID` | Integer | ID của thẻ tag nguồn (Source Tag). |
| `Source_X_mm` | Float | Tọa độ X của thẻ nguồn so với camera (mm). |
| `Source_Y_mm` | Float | Tọa độ Y của thẻ nguồn so với camera (mm). |
| `Source_Z_mm` | Float | Tọa độ Z của thẻ nguồn so với camera (mm). |
| `Num_Targets` | Integer | Số lượng thẻ mục tiêu đang phát hiện được. |
| `Polyline_Dist_mm`| Float | Khoảng cách ngắn nhất từ thẻ nguồn đến đường mục tiêu (mm). |
| `Target_IDs` | String | Danh sách các ID thẻ mục tiêu đang hoạt động (phân cách bằng dấu `;`). |
| `Selected_Coord_Mode`| String | Thuật toán tọa độ đang chọn (`pnp` hoặc `depth`). |

---

## 🛠️ Xử Lý Sự Cố Thường Gặp (Troubleshooting)

1.  **Lỗi: "Không thể khởi động camera..."**
    *   *Nguyên nhân*: Camera chưa được kết nối đúng cách hoặc đang bị một ứng dụng khác chiếm dụng luồng (như ứng dụng Intel RealSense Viewer).
    *   *Khắc phục*: Kiểm tra lại cáp kết nối USB 3.0, tắt tất cả ứng dụng camera khác và khởi động lại phần mềm.
2.  **Khoảng cách đo đạc bị sai số lớn**
    *   *Nguyên nhân*: Kích thước thực tế của thẻ AprilTag in ra không đúng với thông số nhập trên giao diện (ví dụ in tag nhỏ nhưng cấu hình 150mm).
    *   *Khắc phục*: Dùng thước đo chính xác kích thước viền đen bên ngoài của thẻ tag đã in và cập nhật lại ô `Kích thước AprilTag (mm)`.
3.  **Tọa độ các thẻ tag bị nhảy/rung lắc liên tục**
    *   *Khắc phục*: Bật tính năng *Kích hoạt bộ lọc mượt 3D (EMA)* và giảm chỉ số *Alpha* xuống khoảng `0.15` - `0.3`.
