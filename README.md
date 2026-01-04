# voice_pipeline (UI + Train/Test)

## Cấu trúc data
- Train: `data/train/<label>/*.wav`
- Test: `data/test/*.wav`


## Chạy UI
```bash
python gui.py
```

### Hướng dẫn nhanh
1. Data (train): ADD để tạo label, chọn label -> ADD để thêm file wav vào label.
2. TRAIN để train model (lưu `models/speaker_model.npz`)
3. Data TEST: ADD để thêm wav test.
4. Chọn test item và bấm **RUN TEST** (hoặc double-click item) để chạy đúng item đang chọn.
5. save để xuất txt (mặc định `<datatest>_result.txt`)

## Chạy CLI
```bash
python main.py --test data/test/<file>.wav
```
