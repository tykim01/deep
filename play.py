# ==============================
# 🔹 기본 라이브러리
# ==============================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import numpy as np

# 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 🌟 혼동 행렬 시각화 함수 
# ==========================================
def plot_confusion_matrix(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for ts, hm, y in loader:
            ts, hm, y = ts.to(device), hm.to(device), y.to(device)
            out = model(ts, hm)
            pred = out.argmax(1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    class_names = ['Flat', 'Rough', 'Slope', 'Stair']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted Label (AI 예측값)')
    plt.ylabel('True Label (실제 정답)')
    plt.title('Terrain Classification Confusion Matrix (Validation)')
    plt.show() 

# ==========================================
# 1. 시계열 특징 추출 브랜치
# ==========================================
class Inception1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.b1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.b2 = nn.Conv1d(in_channels, 16, kernel_size=5, padding=2)
        self.b3 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)

    def forward(self, x):
        return torch.cat([
            torch.relu(self.b1(x)),
            torch.relu(self.b2(x)),
            torch.relu(self.b3(x))
        ], dim=1) 

class TimeBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Inception1D(18),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Conv1d(48, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten() 
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. 히트맵 기반 특징 추출 브랜치
# ==========================================
class HeatmapBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.fc = nn.Identity() 

        self.proj = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)
        return x

# ==========================================
# 3. 최종 지형 분류기
# ==========================================
class TerrainClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.branch_a = TimeBranch()
        self.branch_b = HeatmapBranch()

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes) 
        )

    def forward(self, ts, heatmap):
        a = self.branch_a(ts)
        b = self.branch_b(heatmap)
        return self.classifier(torch.cat([a, b], dim=1))

# ==========================================
# 4. 데이터 전처리
# ==========================================
def ts_to_heatmap(ts):
    ts = ts.T 
    ts = (ts - np.mean(ts)) / (np.std(ts) + 1e-6) 
    ts = np.stack([ts, ts, ts], axis=0) 
    ts = torch.tensor(ts, dtype=torch.float32)

    ts = torch.nn.functional.interpolate(
        ts.unsqueeze(0),
        size=(224, 224),
        mode='bilinear'
    ).squeeze(0)

    return ts

# ==========================================
# 5. 데이터셋 정의
# ==========================================
class TerrainDataset(Dataset):
    def __init__(self):
        self.files = [
            ("flat.npy", 0),
            ("rough.npy", 1),
            ("slope.npy", 2),
            ("stair.npy", 3)
        ]
        self.samples = []

        for file, label in self.files:
            data = np.load(file) 
            for ts in data:
                for start in range(0, 75, 2):
                    window = ts[start:start+25]
                    self.samples.append((window, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ts_np, label = self.samples[idx]
        heatmap = ts_to_heatmap(ts_np) 
        ts = torch.tensor(ts_np, dtype=torch.float32).permute(1, 0)
        ts += 0.00 * torch.randn_like(ts) 
        return ts, heatmap, torch.tensor(label)

# ==========================================
# 6. 평가 함수
# ==========================================
def evaluate(model, loader, criterion, device):
    model.eval() 
    total, correct = 0, 0
    val_loss_sum = 0.0 

    with torch.no_grad():
        for ts, hm, y in loader:
            ts, hm, y = ts.to(device), hm.to(device), y.to(device)
            out = model(ts, hm) 
            loss = criterion(out, y) 
            val_loss_sum += loss.item()
            
            pred = out.argmax(1) 
            total += y.size(0)
            correct += (pred == y).sum().item()

    avg_loss = val_loss_sum / len(loader) 
    accuracy = 100 * correct / total
    return accuracy, avg_loss 

# ==========================================
# 7. 학습 및 시각화 프로세스
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TerrainDataset()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size 

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8)

    model = TerrainClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience = 40 
    counter = 0
    
    # 기록용 리스트
    history_train_acc, history_val_acc = [], []
    history_train_loss, history_val_loss = [], []

    for epoch in range(50): 
        model.train() 
        total, correct, loss_sum = 0, 0, 0

        for ts, hm, y in train_loader:
            ts, hm, y = ts.to(device), hm.to(device), y.to(device)

            optimizer.zero_grad() 
            out = model(ts, hm) 
            loss = criterion(out, y) 
            loss.backward() 
            optimizer.step() 

            loss_sum += loss.item()
            pred = out.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()

        avg_train_loss = loss_sum / len(train_loader)
        train_acc = 100 * correct / total
        val_acc, val_loss = evaluate(model, val_loader, criterion, device)

        # 콘솔 출력에는 정확도를 남겨두었습니다 (학습 과정 모니터링용)
        print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:6.2f}% (Loss: {avg_train_loss:.4f}) | Val Acc: {val_acc:6.2f}% (Loss: {val_loss:.4f})")
        
        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)
        history_train_loss.append(avg_train_loss)
        history_val_loss.append(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc 
            counter = 0
            torch.save(model.state_dict(), "best_model.pth") 
        else:
            counter += 1 

        if counter >= patience:
            print("🔥 Early stopping triggered!")
            break

    # ----------------------------------------
    # 📈 학습 곡선 시각화 (🌟 Train Loss / Val Loss 따로따로!)
    # ----------------------------------------
    print("\n📈 학습 곡선(Train Loss & Val Loss)을 그립니다...")
    plt.figure(figsize=(12, 5)) 

    # 1. 좌측 그래프: Train Loss
    plt.subplot(1, 2, 1) 
    plt.plot(history_train_loss, label='Train Loss', marker='o', color='blue')
    plt.title('Learning Curve (Train Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. 우측 그래프: Validation Loss
    plt.subplot(1, 2, 2) 
    plt.plot(history_val_loss, label='Validation Loss', marker='s', color='orange')
    plt.title('Learning Curve (Validation Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show() 

    # ----------------------------------------
    # 📊 최종 평가 및 혼동 행렬 시각화 
    # ----------------------------------------
    print("\n📊 최고 성능 모델을 불러와 검증(Validation) 혼동 행렬을 그립니다...")
    model.load_state_dict(torch.load("best_model.pth", weights_only=True)) 
    
    final_val_acc, final_val_loss = evaluate(model, val_loader, criterion, device)
    print(f"🔥 Final Validation Accuracy: {final_val_acc:.2f}% (Loss: {final_val_loss:.4f})")
    
    plot_confusion_matrix(model, val_loader, device)

# ==========================================
# 실행
# ==========================================
if __name__ == "__main__":
    train()