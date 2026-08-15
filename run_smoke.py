import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from train import train_pose_model

# Create small synthetic dataset
N = 64
csi = np.random.randn(N, 540, 20).astype(np.float32)
# Baseline endpoints [x1,y1,x2,y2]
targets = np.random.randn(N, 4).astype(np.float32)

# Split
train_n = int(N * 0.7)
val_n = int(N * 0.15)
test_n = N - train_n - val_n

train_ds = TensorDataset(torch.from_numpy(csi[:train_n]), torch.from_numpy(targets[:train_n]))
val_ds = TensorDataset(torch.from_numpy(csi[train_n:train_n+val_n]), torch.from_numpy(targets[train_n:train_n+val_n]))
test_ds = TensorDataset(torch.from_numpy(csi[train_n+val_n:]), torch.from_numpy(targets[train_n+val_n:]))

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

# Run one epoch smoke test
model, history, test_loss, test_pck20, test_mpe, pck_details = train_pose_model(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    batch_size=8,
    n_epochs=1,
    patience=1,
    lr=1e-4,
    weight_decay=1e-5,
    keypoint_scale=1000.0,
    gpu_config='0',
    output_dir='smoke_test_output',
    use_augmentation=False
)

print('Smoke test finished')
print('Test loss:', test_loss)
print('Test PCK@0.2:', test_pck20)
print('Test MPE:', test_mpe)
