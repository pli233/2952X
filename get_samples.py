import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.cluster import DBSCAN


def get_projected_data():
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    imagenet_path = '/CSCI2952X/datasets/origin_ImageNet-1k/imagenet/Data/train'
    dataset = datasets.ImageFolder(root=imagenet_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=10000, shuffle=False, num_workers=4)

    feature_matrix = []

    with torch.no_grad(): 
        for images, _ in tqdm(dataloader):
            outputs = model(images)
            # print(outputs)
            feature_matrix.append(outputs.cpu().numpy())

    feature_matrix = np.concatenate(feature_matrix, axis=0)
    print(feature_matrix)

    np.save('imagenet_features.npy', feature_matrix)
    
def density_cluster():
    feature_matrix = np.load('imagenet_features.npy')
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # 可以根据需要调整 eps 和 min_samples 参数
    labels = dbscan.fit_predict(feature_matrix)

    # 打印聚类结果
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'发现的簇数量: {num_clusters}')
    print(f'每个数据点的标签: {labels}')

    # 统计每个簇的大小
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    print(f'每个簇的大小: {cluster_sizes}')
    
def main():
    get_projected_data()
    density_cluster()
    


if __name__ == "__main__":
    main()
    
