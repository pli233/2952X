import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.cluster import DBSCAN

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
model.eval()

cluster_centers = np.load('cluster_centers.npy')  # 假设已存储

cluster_centers = torch.tensor(cluster_centers, dtype=torch.float32)

input_image = torch.randn((1, 3, 224, 224), requires_grad=True)

optimizer = optim.Adam([input_image], lr=0.01)

def clamp_image(img):
    return img.clamp(0, 1)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

target_center = cluster_centers[0]  # 假设选取第 0 个聚类中心

for step in range(1000):
    optimizer.zero_grad()
    
    normalized_image = normalize(input_image.squeeze(0)).unsqueeze(0)
    output = model(normalized_image)
    
    loss = torch.nn.functional.mse_loss(output, target_center)
    print(f'Step {step}, Loss: {loss.item()}')
    
    loss.backward()
    optimizer.step()
    
    input_image.data = clamp_image(input_image.data)

# # 将生成的图像保存为文件
# generated_image = input_image.detach().squeeze(0).permute(1, 2, 0).numpy()
# generated_image = (generated_image * 255).astype(np.uint8)  # 转为 0-255 范围
# Image.fromarray(generated_image).save('generated_image.png')