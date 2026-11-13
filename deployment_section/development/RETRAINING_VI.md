# GROUP 9 — Hướng dẫn tự retrain model (`retrain_pipeline.py`)

Script chạy **offline trên máy bạn**, không chạy trong Streamlit. Thí nghiệm retraining đã thực hiện trong nghiên cứu sử dụng **BRFSS 2025**; quy trình dưới đây dành cho các lần retraining tương lai bằng dữ liệu mới đã có nhãn. Script làm theo `deployment_bundle/retraining_config.json` và **không tự thay model đang chạy** nếu không có `--promote` và mọi tiêu chí đều đạt.

```
BRFSS 2015 (lịch sử) ─┐                         ┌─ locked test 50,736 dòng (tái tạo y hệt, không bao giờ train)
                      ├─ validate ─ kiểm policy ─┤
Dữ liệu mới đã gán nhãn┘   (loại dòng lỗi)      └─ train cũ + dòng mới (trừ holdout) ─► model ứng viên
                                                                                       │
                     so sánh ứng viên vs production trên CÙNG các dòng (PR-AUC, Brier, Precision@K)
                                                                                       │
                      đạt hết tiêu chí + có --promote ─► thay deployment_bundle (bản cũ được lưu archive)
```

## 1. Cài đặt (một lần)

Trong thư mục `deployment_section/development`, Terminal VS Code:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-retrain.txt
.\.venv\Scripts\python.exe test_retrain_offline.py
```

Phải thấy 5 dòng `PASS`. Dùng Python 3.13 như app để kết quả tái lập chính xác.

## 2. Chuẩn bị dữ liệu

1. File lịch sử `diabetes_binary_health_indicators_BRFSS2015.csv` (253,680 dòng, không chỉnh sửa) đã có sẵn ở thư mục gốc repository; không cần copy vào `development\data\`.
2. Khi chạy từ `deployment_section\development\`, truyền đường dẫn file gốc bằng `--historical ..\..\diabetes_binary_health_indicators_BRFSS2015.csv`. Có thể tạo thư mục `data\` (đã nằm trong `.gitignore`) để lưu các file đầu vào cục bộ khác.
3. Dữ liệu mới có nhãn — một trong hai dạng:
   - **Export Google Sheet cho demo**: chỉ dùng dữ liệu giả lập theo `LOGGING_SETUP_VI.md`; điền cột `actual_diabetes_outcome` (0 hoặc 1) và `outcome_recorded_at` (ISO) → **File › Download › CSV** → lưu `data\sheet_export.csv`. Dòng nào chưa có outcome sẽ được bỏ qua (pending), không bị coi là lỗi. Không xem đây là dữ liệu bệnh nhân thật hoặc bằng chứng model học từ người dùng.
   - **Dataset mới đã recode**: CSV có đủ 21 cột feature + cột `Diabetes_binary`, cùng cách mã hoá như 2015. Thí nghiệm nghiên cứu đã dùng BRFSS 2025; các lần chạy trong tương lai cần dữ liệu mới có nhãn và phải kiểm tra tính tương thích trước khi dùng.

Khi **chạy thủ công để thử pipeline** bằng Sheet demo, có thể export lại cả Sheet (file dài thêm). Script tự nhận biết dòng nào model hiện tại **đã dùng rồi** (qua `deployment_bundle/new_data_manifest.json`) và chỉ đếm dòng mới; Streamlit không tự chạy bước này.

## 3. Chạy

**Chạy thử, không đụng vào model đang chạy** (nên làm trước; dữ liệu Sheet giả lập chỉ phù hợp cho bước thử này):

```powershell
.\.venv\Scripts\python.exe retrain_pipeline.py --historical ..\..\diabetes_binary_health_indicators_BRFSS2015.csv --new-data data\sheet_export.csv
```

**Chỉ khi có dữ liệu mới phù hợp đã gán nhãn và muốn thay model** (tắt Streamlit trước, Ctrl+C; không dùng dữ liệu Sheet giả lập):

```powershell
.\.venv\Scripts\python.exe retrain_pipeline.py --historical ..\..\diabetes_binary_health_indicators_BRFSS2015.csv --new-data data\new_labeled_cohort.csv --promote
```

Tuỳ chọn hay dùng:

| Tuỳ chọn                 | Tác dụng                                                                               |
| ------------------------ | -------------------------------------------------------------------------------------- |
| `--new-data A.csv B.csv` | Nhiều file cùng lúc (Sheet export + dataset)                                           |
| `--ignore-schedule`      | Chạy sớm hơn lịch 30 ngày (vẫn được promote)                                           |
| `--skip-cv`              | Bỏ 5-fold CV để chạy nhanh hơn khoảng 2 phút (OOF PR-AUC sẽ để trống)                  |
| `--force`                | Bỏ qua yêu cầu số lượng dữ liệu — **chỉ để test**, ứng viên không bao giờ được promote |
| `--historical PATH`      | Đường dẫn khác cho file BRFSS 2015                                                     |

## 4. Đọc kết quả

| Exit code / STATUS               | Nghĩa                                                                     |
| -------------------------------- | ------------------------------------------------------------------------- |
| `2` · `SKIPPED_POLICY_NOT_MET`   | Chưa đủ 30 ngày hoặc chưa đủ dòng có nhãn mới → không train               |
| `0` · `CANDIDATE_READY`          | Ứng viên đạt tiêu chí, nằm ở `model_registry\candidates\v…` — chưa deploy |
| `0` · `PROMOTED`                 | Đã thay `deployment_bundle\`; bản cũ ở `model_registry\archive\`          |
| `3` · `REJECTED_CRITERIA_FAILED` | Ứng viên kém hơn → giữ model cũ                                           |
| `1` · `ERROR`                    | Dữ liệu sai / không tái tạo được locked test → không làm gì cả            |

Mỗi lần chạy tạo `retraining_runs\<run_id>\`: `report.md` (bảng so sánh để chép vào báo cáo), `report.json`, và `rejected_rows.csv` (dòng bị loại + lý do, số dòng khớp với số dòng trong Sheet).

Kiểm tra một ứng viên bằng smoke test: `.\.venv\Scripts\python.exe smoke_test.py model_registry\candidates\<thư_mục>`

## 5. Tiêu chí (trong `retraining_config.json`)

- **Đủ dữ liệu:** ≥ 500 dòng có nhãn chưa dùng, ≥ 50 dương tính, ≥ 50 âm tính; ≥ 30 ngày từ lần train trước.
- **Được promote khi:** PR-AUC ứng viên ≥ production trên holdout dữ liệu mới (hoặc locked test nếu holdout quá ít); PR-AUC locked test không giảm quá 0.005; Brier không tăng quá 0.002; Precision@K ở cả 6 mức capacity không giảm quá 0.02.

Các con số này là **ngưỡng cấu hình của research prototype** theo `retraining_config.json`, không phải ngưỡng đã được xác nhận cho ứng dụng lâm sàng. Quy trình yêu cầu phê duyệt thủ công trước khi promote; các ngưỡng cần được đánh giá độc lập trước mọi mục đích sử dụng lâm sàng.

## 6. Rollback

Tắt Streamlit, xoá `deployment_bundle\`, copy thư mục `model_registry\archive\v<phiên_bản_cũ>_…` về và đổi tên thành `deployment_bundle`, chạy `smoke_test.py`.

## 7. Sau khi promote (làm tay)

- README §3 và caption trong `app.py` còn ghi **1.03%** → sửa theo `sample_output.json` mới (report.md ghi sẵn số mới).
- Đọc lại đoạn mô tả calibration trong `app.py` (“predicted scores are higher…”) so với đường cong mới.
- Commit `deployment_bundle\` mới lên GitHub để Streamlit Cloud dùng; báo người viết Methodology/Results/5.1.

## 8. Lưu ý khi viết báo cáo

- **Đã kiểm chứng:** trên dữ liệu gốc, pipeline tái tạo **chính xác** model v1.0.0 (chênh lệch điểm = 0), Brier 0.0987, OOF PR-AUC 0.4384 / ROC-AUC 0.8316, toàn bộ `risk_reference`, `capacity_config` và bảng calibration.
- Google Sheets theo `LOGGING_SETUP_VI.md` hiện chỉ phục vụ **demo với dữ liệu giả lập**, không phải nguồn dữ liệu bệnh nhân thực tế. Không retrain hoặc báo cáo đó là “model học từ người dùng”. Việc retraining tương lai cần dữ liệu mới có nhãn được thu thập và kiểm duyệt riêng; Streamlit không tự học từ dự đoán người dùng.
- `make_demo_prediction_log.py` tạo file `*_SIMULATED.csv` chỉ để thử pipeline; script **chặn promote** với file có chữ SIMULATED trong tên.
- Người dùng app không phải mẫu ngẫu nhiên của dân số (selection bias); outcome tự báo cáo. Đây là giới hạn cần nêu ở phần Discussion.
- Hyperparameter được **giữ nguyên** từ lần tối ưu Optuna của v1.0.0, không tuning lại.
