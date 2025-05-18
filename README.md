<h1 align="center"> DỰ ÁN MÔ PHỎNG THUẬT TOÁN 8-PUZZLE-SOLVER</h1>

**TỔNG QUAN**
- Dự án này là một ứng dụng Python giải bài toán 8-puzzle cổ điển, phát triển như một dự án về trí tuệ nhân tạo (AI). Ứng dụng sử dụng 20 thuật toán tìm kiếm để tìm đường từ trạng thái ban đầu đến trạng thái mục tiêu, bao gồm BFS, A*, Q-Learning, và các phương pháp tìm kiếm cục bộ. Giao diện Pygame trực quan cho phép người dùng theo dõi quá trình giải, phân tích hiệu suất thuật toán qua biểu đồ cột ngang, và xuất kết quả dưới dạng GIF. Dự án còn ghi lại nhật ký hiệu suất và tạo biểu đồ so sánh, hỗ trợ nghiên cứu và giảng dạy AI.

**NGUỒN GỐC CỦA 8-PUZZLE**

- 8-Puzzle là một bài toán trượt ô (sliding puzzle) được phổ biến vào cuối thế kỷ 19 bởi Sam Loyd. Trò chơi gồm lưới 3x3 với 8 ô số (1-8) và một ô trống, mục tiêu là di chuyển các ô để đạt trạng thái mục tiêu (thường là 1 -> 2 -> 3, 4 -> 5 -> 6, 7 -> 8 -> 0). 8-Puzzle được sử dụng rộng rãi trong nghiên cứu AI để minh họa các thuật toán tìm kiếm như BFS, A*, và các phương pháp heuristic. Với không gian trạng thái lớn (362,880 trạng thái khả thi), nó là bài toán lý tưởng để so sánh hiệu quả thuật toán.

**TÍNH NĂNG CHÍNH**

**Giao Diện Tương Tác:** 
- Điều khiển quá trình giải qua Pygame với các nút Tiến (Step Forward), Lùi (Step Back), Đặt lại (Reset), tạm dừng (Pause, Continue), và thoát (Back).
AI Điều Khiển Giải Pháp: 20 thuật toán tìm kiếm:
**Uninformed Search:**
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Uniform-Cost Search (UCS)
- Iterative Deepening Search (IDS)
**Heuristic-Based Search:**
- A* Search (A*)
- Iterative Deepening A* Search (IDA*)
- Greedy Search
**Local Search:**
- Simple Hill Climbing
- Steepest-Ascent Hill Climbing
- Stochastic Hill Climbing
**Advanced Search:**
- Simulated Annealing
- Genetic Algorithm
- Q-Learning
- Beam Search
**Specialized Search:**
- Nondeterministic Search
- Partial Observation Search
**Constraint Satisfaction Problems (CSP):** Backtracking, Forward Checking, Min-Conflicts
- Phân Tích Hiệu Suất: Ghi lại số bước, thời gian, và bộ nhớ, tạo biểu đồ cột ngang (lưu trong charts/).
- Xuất GIF: Tạo GIF động minh họa quá trình giải (lưu trong gifs/).
- Kiểm Tra Tính Khả Thi: Xác định puzzle có giải được bằng số đảo ngược.
- Hiệu Ứng Hình Ảnh: Bảng màu hiện đại, hiệu ứng hover, chuyển tiếp mượt mà.
- Nhật Ký Hiệu Suất: Lưu thống kê thuật toán để phân tích sau.

**CÀI ĐẶT VÀ CHẠY TRÒ CHƠI**

**Yêu Cầu:**
- Python 3.6 trở lên   
- Môi trường ảo (khuyến nghị sử dụng)
- Hệ điều hành: Đã thử nghiệm trên Windows; có thể cần điều chỉnh cho macOS/Linux
- Quyền ghi cho thư mục charts/, gifs/, temp_puzzle_frames/

**HƯỚNG DẪN CÀI ĐẶT**
_1. TẢI DỰ ÁN_

Tải toàn bộ file dự án hoặc sao chép từ kho lưu trữ (repository) về máy của bạn.

_2. TẠO MÔI TRƯỜNG ẢO_
* **Bước 1:** Tạo môi trường ảo
  ````bash
  python -m venv venv
  ````
* **Bước 2:** Kích hoạt môi trường ảo
  * _**Trên Windows:**_
   ````bash
  venv\Scripts\activate
  ````
* _**Trên macOS và Linux:**_
  ````bash
  source venv/bin/activate
  ````
* **Bước 3:** Cài đặt các thư viện cần thiết
  ````bash
  pip install -r requirements.txt
  ````
_3.CÀI ĐẶT MÔI TRƯỜNG_
- Cài đặt các thư viện cần thiết được liệt kê trong file requirements.txt:
   ````bash
  pip install -r requirements.txt
  ````

**Cài đặt các thư viện trong requirements.txt:**
- contourpy==1.3.2
- cycler==0.12.1
- fonttools==4.58.0
- imageio==2.37.0
- kiwisolver==1.4.8
- matplotlib==3.10.3
- numpy==2.2.4
- packaging==25.0
- pandas==2.2.3
- pillow==11.2.1
- pygame==2.6.1
- pyparsing==3.2.3
- python-dateutil==2.9.0.post0
- pytz==2025.2
- scipy==1.15.2
- seaborn==0.13.2
- six==1.17.0
- tzdata==2025.2

**Cài đặt thủ công nếu cần:**
   ````bash
   pip install pygame==2.6.1 numpy==2.2.4 matplotlib==3.10.3 imageio==2.37.0 seaborn==0.13.2
  ````

- Các thư viện chuẩn Python: os, sys, time, random (không cần cài).
**Lưu ý:** Đảm bảo đúng phiên bản thư viện để tránh xung đột.

_4. CHẠY TRÒ CHƠI_

Khởi động bằng file chính: 
  ````bash
  python 8_Puzzle_Solver_old.py
  ````

**5. HƯỚNG DẪN TRÒ CHƠI**

**BẮT ĐẦU TRÒ CHƠI**
1. Chạy **python 8_Puzzle_Solver_old.py** để mở menu chính.
2. Menu hiển thị danh sách thuật toán và nút "Play All".
- Giao diện 20 thuật toán: người chơi có thể chọn thuật toán bất kỳ để xem tên thuật toán, mô tả, hướng giải quyết, ưu và nhược điểm của thuật toán đó, bên dưới có nút "Back" để quay lại menu hiển thị.
- nút "Play All": tiến hành vào giao diện chính của trò chơi, tại đây người chơi sẽ chọn 1 trong 20 thuật toán để chơi.

**CÁCH ĐIỀU KHIỂN**
- Nút thuật toán: Người chơi chọn thuật toán bất kỳ (BFS, A*, v.v.) trong khung, trước khi thuật toán bắt đầu chạy, thuật toán sẽ hiển thị số bước di chuyển (Steps), thời gian thực hiện (Times), bộ nhớ sử dụng (KB) và lưu biểu đồ "ggplot" vào  thư mục (charts), người chơi có thể sử dụng thanh kéo để xem hết tất cả thuật toán.
- Tạm Dừng/Tiếp Tục: Nhấp "Pause/Continue" để điều khiển quá trình giải của thuật toán giúp người xem hiểu rõ tuần tự các bước giải.
- Tiến Bước/lùi bước: Nhấp nút "Step Forward" để xem bước tiếp theo và nút "Step Back" để xem lại bước trước đó sau khi thuật toán đang trong tình trạng tạm dừng (Pause).
- Đặt Lại: Nhấp "Reset" để trở về trạng thái ban đầu sau khi thuật toán đã đến đích cuối hoặc trong quá trình giải.
- Quay Lại: Nhấp "Back" để về menu hiển thị.
- Phím Escape: Thoát chương trình (lưu biểu đồ và GIF).

**Lưu ý:** Nếu trong quá trình giải thuật toán mà người chơi nhấn thuật toán khác để chạy thì thuật toán đang giải sẽ dừng lại và hiển thị thông báo "Algorithm running" và hỏi người chơi có muốn dừng lại không, nếu nhấn "Stop" thì thuật toán đang chạy sẽ dừng lại và người chơi chỉ cần nhấn thuật toán mới để chạy, còn nếu nhấn "Continue" thì thuật toán đang chạy sẽ tiếp tục chạy.

**MỤC TIÊU**
- Giải Puzzle: Di chuyển các ô từ trạng thái ban đầu (ví dụ: ((2, 6, 5), (8, 0, 7), (4, 3, 1))) đến trạng thái mục tiêu (((1, 2, 3), (4, 5, 6), (7, 8, 0))).
- Phân Tích: Xem thống kê (số bước, thời gian, bộ nhớ) và biểu đồ sau khi giải.
GIF: Xuất quá trình giải để trình bày hoặc nghiên cứu.

**CÁC MÀN HÌNH TRÒ CHƠI**
- Menu Chính: Chọn thuật toán, xem mô tả và "Play All" vào màn hình giải.
- Màn Hình Giải: Hiển thị puzzle, nút điều khiển, trạng thái hiện tại.
- Thống Kê: Pop-up sau khi giải, hiển thị số bước, thời gian, bộ nhớ, tệp GIF.

**PHÂN TÍCH HIỆU SUẤT THUẬT TOÁN**
Ứng dụng ghi lại hiệu suất từng thuật toán và phân tích dựa trên:
- Số bước di chuyển đến mục tiêu.
- Thời gian thực thi (giây).
- Dung lượng bộ nhớ đỉnh (KB).

**KẾT QUẢ PHÂN TÍCH**
- Biểu Đồ Trực Quan: Biểu đồ cột ngang (PNG) lưu trong charts/ (ví dụ: charts/bfs_performance.png), so sánh:
  - Số bước (xanh dương)
  - Thời gian (cam)
  - Bộ nhớ (xanh lá)
- GIF Động: Lưu trong gifs/ (ví dụ: gifs/bfs_solution.gif), minh họa từng bước giải.
- Khi thoát (Escape hoặc Back), chương trình lưu biểu đồ và GIF tự động.

**CẤU TRÚC DỰ ÁN**
- 8_Puzzle_Solver_old.py: File chính, chứa giao diện, thuật toán, biểu đồ, và GIF.
- charts/: Lưu biểu đồ cột ngang (PNG).
- gifs/: Lưu GIF quá trình giải.
- temp_puzzle_frames/: Lưu ảnh tạm cho GIF (tự xóa).
- README.md: Tài liệu dự án.

***
Kết quả thu được

Màn hình hiển thị

![Màn hình hiển thị](./image/Màn%20hình%20hiển%20thị.png)

Màn hình mô tả thuật toán

![Mô tả thuật toán](./image/Màn%20hình%20mô%20tả.png)

Màn hình chính

![Màn hình chính](./image/Màn%20hình%20chính.png)

Màn hình đích

![Màn hình đích](./image/Màn%20hình%20đích.png)

Màn hình thống kế (vd: BFS)

![Màn hình thống kê](./image/Màn%20hình%20thống%20kê.png)

***
**THÔNG TIN LIÊN HỆ**

Nếu bạn có câu hỏi hoặc cần hỗ trợ, liên hệ:

Tên: Lưu Trần Kim Phú

Email: luutrankimphu2021@gmail.com

Số điện thoại: +84 981744757

GitHub: https://github.com/l-tkphu01

Link download trò chơi: https://drive.google.com/drive/u/0/folders/1_MmIGTVQUxrvP7cTkr_oc1KqRuT3og_m

Tôi rất mong nhận được phản hồi và đóng góp từ bạn để cải thiện dự án này.
