import torch
import torch.nn as nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
import coremltools as ct

# 【新規追加】AIの出力を「箱（辞書）」から「中身（純粋な配列）」だけに取り出すためのラッパークラス
class SegmentationWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # AIは辞書を返すので、'out' というキーに入っている塗り絵の推論結果だけを抽出して返します
        return self.model(x)['out']

def create_ios_friendly_model(num_classes):
    # 【修正】警告を消すため、pretrained=False ではなく weights=None を使用します
    model = lraspp_mobilenet_v3_large(weights=None, num_classes=num_classes)
    return model

def main():
    print("=== 新しい画像セグメンテーションモデルの構築と変換 ===")
    
    # クラス数を定義（0=背景, 1=道, 2=点字ブロック の3クラス）
    num_classes = 3
    base_model = create_ios_friendly_model(num_classes)
    
    # 【新規追加】元のモデルをラッパーで包み、出力が純粋なデータの塊になるようにします
    model = SegmentationWrapper(base_model)
    model.eval()

    # ダミー画像を作成（バッチサイズ1, 3チャンネル(RGB), 縦512 x 横512）
    dummy_input = torch.rand(1, 3, 512, 512)

    # 推論のテスト実行
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"モデルの入力サイズ: {dummy_input.shape}")
    print(f"モデルの出力サイズ: {output.shape}") 

    print("\nCore ML形式への変換中...")
    
    # 【修正】万が一の内部エラーを防ぐため、念のため strict=False も付与しておきます
    traced_model = torch.jit.trace(model, dummy_input, strict=False)
    
    # Core MLモデルへの変換（入力を画像として明示）
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input_image", shape=dummy_input.shape, color_layout=ct.colorlayout.RGB)]
    )

    save_path = "ClearPathSegmentation.mlpackage"
    mlmodel.save(save_path)
    print(f"=== 完了！ {save_path} が生成されました ===")

if __name__ == "__main__":
    main()