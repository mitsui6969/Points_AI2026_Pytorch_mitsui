import torch
import torch.nn as nn

class SegmentationNN(nn.Module):
    def __init__(self, input_n=8, mid1=30, mid2=10, mid3=10, output_n=5):
        super(SegmentationNN, self).__init__()
        # 層の定義
        self.fc1 = nn.Linear(input_n, mid1)
        self.fc2 = nn.Linear(mid1, mid2)
        self.fc3 = nn.Linear(mid2, mid3)
        self.fc4 = nn.Linear(mid3, output_n)
        
        # 活性化関数 (現代の標準であるReLUを使用)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # データの流れ方
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x