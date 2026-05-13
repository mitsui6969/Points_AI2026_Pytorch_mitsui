# Points_AI2026_Pytorch_mitsui
郭研究室のニューラルネットワークコードのPytorch版

# 使い方
### 学習開始
```
python train.py
```
※教師データはあらかじめ`data/Teacher_Data/*`に入れておく

### テスト実行
1. 
```
python test.py
```
2. `data/result_Data/result_segmentation.png` に保存された画像で確認

# ディレクトリ構成
```
.
├── data
│   ├── result_Data
│   │   └── result_segmentation.png
│   └── Teacher_Data
│       ├── ans_img~.png
│       ├── ans_text~.txt
│       ├── img~.png
├── PointsAI.mlpackage
│   ├── Data
│   │   └── com.apple.CoreML
│   │       ├── weights
│   │       │   └── weight.bin
│   │       └── model.mlmodel
│   └── Manifest.json
├── .gitignore
├── dataset.py
├── export_coreml.py
├── model_weights.pth
├── model.py
├── README.md
├── test.py
└── train.py
```
