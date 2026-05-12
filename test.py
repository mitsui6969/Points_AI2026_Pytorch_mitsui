import torch
import numpy as np
from PIL import Image

# 作成した設計図を読み込む
from model import SegmentationNN

def main():
    print("=== AIによるセグメンテーション（画像判定）を開始します ===")

    # 1. AIの準備と、学習した「脳（重み）」の読み込み
    model = SegmentationNN()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval() # 学習モードから「推論（テスト）モード」に切り替え

    # 2. テスト用画像の読み込み
    # ※学習に使った画像とは別の画像を指定して、本当に賢いかテストします。
    img_path = "data/Teacher_Data/img250512_141838.png" 
    print(f"テスト画像を読み込み中: {img_path}")
    image = Image.open(img_path).convert('RGB')
    width, height = image.size
    img_data = np.array(image)

    # 結果を保存するための真っ黒なキャンバスを用意
    output_img_data = np.zeros((height, width, 3), dtype=np.uint8)

    # ----------------------------------------------------
    # クラス(0〜4)に対応する「塗りつぶす色」の設定 (RGB)
    # ※研究室の仕様（正解画像の色）に合わせて自由に変更してください！
    # ----------------------------------------------------
    color_map = {
        0: [255, 0, 0],     # クラス0: 赤 (人・車)
        1: [255, 255, 0],   # クラス1: 黄（道・地面）
        2: [255, 192, 203], # クラス2: ピンク（建物・人工物）
        3: [0, 255, 0],     # クラス3: 緑（木・植物）
        4: [0, 0, 255]      # クラス4: 青 (空・雲)
    }

    # 入力データ用リストと、色を塗るための座標リスト
    inputs = []
    coords = [] 

    print("画像全体から特徴量を抽出中...（10秒ほどかかる場合があります）")
    # 画像の全ピクセルをスキャン
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            r, g, b = img_data[y, x]
            brightness = max(r, g, b)

            # dataset.py と全く同じ独自の特徴量計算ルール
            map_val = 0.05 + (brightness / 255.0) * 0.95
            siki_green = 60 * map_val
            rf, gf, bf = float(r), float(g), float(b)

            if brightness < 100 or (gf - bf) > siki_green or (gf - rf) > siki_green:
                mygreen = gf / 255.0
            else:
                mygreen = 0.0

            siki_blue = 10 * map_val
            if brightness > 178 or ((bf - gf) > siki_blue and (bf - rf) > siki_blue):
                myblue = bf / 255.0
            else:
                myblue = 0.0

            patch = img_data[y-1:y+2, x-1:x+2]
            patch_brightness = np.max(patch, axis=2)
            median_val = np.median(patch_brightness) / 255.0

            x_norm = x / width
            y_norm = y / height
            r_norm = r / 255.0
            g_norm = g / 255.0
            b_norm = b / 255.0

            feature = [x_norm, y_norm, r_norm, g_norm, b_norm, mygreen, myblue, median_val]
            inputs.append(feature)
            coords.append((x, y))

    print("AIが全ピクセルの判定を行っています...")
    # PyTorchのテンソルに変換
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

    # 「with torch.no_grad():」は、推論時にメモリを節約するためのおまじないです
    with torch.no_grad():
        outputs = model(inputs_tensor)
        # 5つのクラスの中で、一番確率が高いもの(argmax)を選ぶ
        predicted_classes = torch.argmax(outputs, dim=1).numpy()

    print("推論結果を新しい画像に描き込んでいます...")
    for i in range(len(coords)):
        x, y = coords[i]
        pred_class = predicted_classes[i]
        output_img_data[y, x] = color_map[pred_class]

    # 画像ファイルとして保存
    result_image = Image.fromarray(output_img_data)
    save_path = "data/result_Data/result_segmentation.png"
    result_image.save(save_path)
    print(f"=== 完了！推論結果を {save_path} に保存しました ===")

if __name__ == "__main__":
    main()