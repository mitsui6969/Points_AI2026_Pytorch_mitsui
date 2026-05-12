import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SegmentationNN
from dataset import CustomSegmentationDataset

def main():
    print("=== AI学習プログラムを開始します ===")
    
    # 1. フォルダのパスを指定してデータを一括読み込み！
    data_folder = "data/Teacher_Data"
    
    dataset = CustomSegmentationDataset(data_dir=data_folder) 
    
    # データが多いので、batch_sizeを 64 または 128 に増やして高速化します
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 2. モデルと学習設定の準備
    model = SegmentationNN()
    
    # ----------------------------------------------------
    # クラスごとの「間違えた時のペナルティ（重み）」を設定
    # 人・車(0)を2倍、建物(2)を3倍のペナルティにして重点的に学習させる
    # ----------------------------------------------------
    class_weights = torch.tensor([2.0, 1.0, 3.0, 1.0, 1.0]) # 0: 人・車, 1: 道・地面, 2: 建物・人工物, 3: 木・植物, 4: 空・雲
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)
    
    # 3. 学習ループ
    epochs = 20
    for epoch in range(epochs):
        total_loss = 0.0
        correct_pixels = 0  # 正解したピクセル数
        total_pixels = 0    # 全ピクセル数
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()             
            outputs = model(inputs)           
            loss = criterion(outputs, labels) 
            loss.backward()                   
            optimizer.step()                  
            
            total_loss += loss.item()
            
            # --- 正解率(%)を計算するための処理を追加 ---
            with torch.no_grad():
                predicted = torch.argmax(outputs, dim=1) # AIの答え
                correct_pixels += (predicted == labels).sum().item()
                total_pixels += labels.size(0)
            
        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_pixels / total_pixels) * 100.0 # %に変換
        
        print(f"エポック {epoch+1:2d}/{epochs} | 平均誤差(Loss): {avg_loss:.4f} | 正解率: {accuracy:.2f}%")
        
    # 4. 学習済みモデルの保存
    save_path = "model_weights.pth"
    torch.save(model.state_dict(), save_path)
    print(f"=== 学習完了！AIの重みを {save_path} に保存しました ===")

if __name__ == "__main__":
    main()