# Xử lý dữ liệu SCADA của trạm biến áp 110Kv, 220Kv, 500Kv ,đánh nhãn tự động bất thường hay không bất thường của trạm biến áp trên bộ dữ liệu Instant_MaiDong_All.xlsx
Bộ dữ liệu Instant_MaiDong_All.xlsx là bộ dữ liệu SCADA được đo ở trạm biến áp phường Mai Động, Hà Nội, dữ liệu được thu từ ngày 01/05/2022 đến ngày 31/05/2022, SCADA ở trạm biến áp mỗi 30p trả kết quả về một lần.
<img src=".\images\chuthich.jpg">
SCADA (Supervisory Control and Data Acquisition) là một hệ thống giám sát, điều khiển và thu thập dữ liệu từ xa được sử dụng rộng rãi trong các ngành công nghiệp, bao gồm cả ngành sản xuất và phân phối điện năng. 
Việc lắp đặt hệ thống SCADA tại trạm biến áp 110kV mang lại nhiều lợi ích:

- Tối ưu hóa hoạt động: SCADA cung cấp khả năng giám sát và điều khiển trạm biến áp từ xa, giúp giảm thời gian và công sức cần thiết để kiểm tra và điều chỉnh các thiết bị tại chỗ.

- Nâng cao chất lượng điện: SCADA có thể giám sát và điều chỉnh các thông số vận hành như dòng điện, điện áp, tần số, và pha, giúp cải thiện chất lượng nguồn điện và giảm thiểu các sự cố.

- Phát hiện sự cố nhanh chóng: SCADA có thể giúp phát hiện và cảnh báo sự cố trong thời gian thực, cho phép nhân viên kỹ thuật phản ứng nhanh chóng và giảm thiểu thời gian ngừng cung cấp.

- Thu thập và phân tích dữ liệu: SCADA thu thập dữ liệu từ các cảm biến và thiết bị, giúp theo dõi hiệu suất và xu hướng hoạt động theo thời gian. Điều này hỗ trợ việc ra quyết định về bảo dưỡng và nâng cấp.

Các thuật toán được áp dụng để tiền xử lý với bộ dữ liệu Instant_MaiDong_All.xlsx  là :

- Xử lý Minssing value :
	Thay thế bằng giá trị trung bình (Mean or median imputation),
	Xóa dòng giá trị bị trống (Drop line),
	Nội suy các giá trị bị thiếu bằng phương pháp tuyến tính (Linear interpolation),
	Ước tính giá trị bị thiếu bằng K điểm gần nhất (KNNImputer)…
	
- Phát hiện và loại bỏ dữ liệu ngoại lai:
	Interquartile range,
	Z-score,
	Percentile,
	HDBSCAN,
	Local Outlier Factor (LOF),
	One-Class SVM,
	isolationForest….

- Chuẩn hóa dữ liệu:
	Minmaxsaler,
	Standardscaler,
	Normalizer…
# Thực nghiệm
- Dữ liệu ban đầu
<img src=".\images\dulieubandau.jpg">
- Dữ liệu missing
<img src=".\images\datamissing.png">
Ta thấy được dữ liệu missing chủ yếu ở các cột Cos phi, phi a, phi b, phi c.
- Dữ liệu sau khi loại bỏ giá trị missing
<img src=".\images\datanomissing.png">
- Dữ liệu trước và sau khi loại bỏ giá trị ngoại lai
<img src=".\images\F(Hz).jpg">
<img src=".\images\Pt(kW).jpg">
<img src=".\images\Qt(kVAR).jpg">
<img src=".\images\Ia(A).jpg">
<img src=".\images\Cos phi.jpg">
<img src=".\images\phi a.jpg">
<img src=".\images\Ua(V).jpg">
- Dữ liệu lúc này
<img src=".\images\dulieusausuly.jpg">
- Dữ liệu sau khi áp dụng thuật toán chuẩn hóa
<img src=".\images\chuanhoa1.jpg">
<img src=".\images\chuan2.jpg">
<img src=".\images\chuan3.jpg">

## Gán nhãn tự động
Ta có thể dự vào các yếu tố sau của SCADA để góp phần đánh giá trạm biến áp tại thời điểm đó là bình thường hay bất thường.
- Tần số (F) từ 49 đến 50 Hz, phù hợp với tần số lưới điện.
- Công suất hoạt động (Pt) từ khoảng 500 kW đến hơn 2000 kW, phù hợp với công suất đo tại trạm biến áp.
- Các giá trị dòng điện (I) từ khoảng 15A đến hơn 60A, phù hợp với dòng điện trung bình của trạm biến áp.

**Kết quả**
<img src=".\images\datadanhnhan.jpg">
