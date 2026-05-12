import torch
import coremltools as ct
from model import SegmentationNN

def main():
    print("=== PyTorchモデルをCore MLに変換します ===")

    # 1. 学習済みのPyTorchモデルを読み込む
    model = SegmentationNN()
    model.load_state_dict(torch.load("model_weights.pth"))
    model.eval() # 変換時は必ず推論モードにする

    # 2. ダミーの入力データを作成する（AIの入力の「形」をAppleに教えるため）
    # 私たちのAIは「1ピクセルあたり8次元(x, y, r, g, b, mygreen, myblue, median)」のデータを受け取ります
    dummy_input = torch.rand(1, 8) 

    # 3. PyTorchのモデルをトレース（処理の流れを記録）する
    traced_model = torch.jit.trace(model, dummy_input)

    # 4. Core ML形式に変換！
    print("Core ML形式への変換中...")
    mlmodel = ct.convert(
        traced_model,
        # 入力データの名前と形を定義（Swift側でこの名前を使います）
        inputs=[ct.TensorType(name="pixel_features", shape=dummy_input.shape)]
    )

    # 5. 保存
    save_path = "PointsAI.mlpackage"
    mlmodel.save(save_path)
    print(f"=== 完了！ {save_path} が生成されました ===")
    print("これをMacからiPadの開発環境（Xcode）にドラッグ＆ドロップしてください！")

if __name__ == "__main__":
    main()