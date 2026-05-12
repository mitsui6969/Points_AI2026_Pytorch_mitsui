import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class CustomSegmentationDataset(Dataset):
    def __init__(self, data_dir):
        print(f"フォルダ [{data_dir}] からデータを一括読み込みします...")
        self.inputs = []
        self.labels = []
        
        # フォルダ内の "ans_text" から始まるテキストファイルをすべて検索
        txt_files = glob.glob(os.path.join(data_dir, "ans_text*.txt"))
        
        if len(txt_files) == 0:
            print("エラー: 学習用のテキストファイルが見つかりません。")
            return
            
        for txt_path in txt_files:
            # ファイル名から時間IDを抽出 (例: ans_text250512_141634.txt -> 250512_141634)
            basename = os.path.basename(txt_path)
            file_id = basename.replace("ans_text", "").replace(".txt", "")
            
            # 対応する画像ファイルのパスを作成
            img_path = os.path.join(data_dir, f"img{file_id}.png")
            
            # 画像が存在しなければスキップ
            if not os.path.exists(img_path):
                print(f"警告: 画像が見つからないためスキップします -> {img_path}")
                continue
                
            print(f"読み込み中: {file_id}")
            
            # 1枚分の読み込み処理
            image = Image.open(img_path).convert('RGB')
            width, height = image.size
            img_data = np.array(image) 
            data = np.loadtxt(txt_path)
            
            for row in data:
                x_coord, y_coord = int(row[1]), int(row[2])
                label = int(row[3])
                
                if x_coord < 1 or x_coord >= width - 1 or y_coord < 1 or y_coord >= height - 1:
                    continue 
                    
                r, g, b = img_data[y_coord, x_coord]
                brightness = max(r, g, b)
                
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
                    
                patch = img_data[y_coord-1:y_coord+2, x_coord-1:x_coord+2]
                patch_brightness = np.max(patch, axis=2)
                median_val = np.median(patch_brightness) / 255.0
                
                x_norm = x_coord / width
                y_norm = y_coord / height
                r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
                
                feature = [x_norm, y_norm, r_norm, g_norm, b_norm, mygreen, myblue, median_val]
                self.inputs.append(feature)
                self.labels.append(label)
                
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        print(f"=== すべての読み込みが完了しました！総データ数: {len(self.inputs)} 件 ===")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]