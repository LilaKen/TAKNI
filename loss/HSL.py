import torch
import torch.nn as nn
import torch.nn.functional as F


class HypersphereLocalizationLoss(nn.Module):
    def __init__(self, radius=1.0, num_classes=10, feature_dim=128):
        """
        初始化超球平面计算损失类。

        参数:
            radius (float): 超球的半径，默认为1.0。
            num_classes (int): 已知类别的数量。
            feature_dim (int): 特征向量的维度。
        """
        super(HypersphereLocalizationLoss, self).__init__()
        self.radius = radius
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # 初始化类别中心特征和超球中心
        self.class_centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.hypersphere_center = nn.Parameter(torch.zeros(feature_dim))

    def to(self, device):
        """
        将模型参数移动到指定设备。

        参数:
            device (torch.device): 目标设备（如 'cpu' 或 'cuda'）。
        """
        self.class_centers = nn.Parameter(self.class_centers.to(device))
        self.hypersphere_center = nn.Parameter(self.hypersphere_center.to(device))
        return super().to(device)

    def forward(self, features, label):
        """
        计算超球平面损失。

        参数:
            features (torch.Tensor): 输入样本的特征向量，形状为 [feature_dim]。
            label (int or torch.Tensor): 输入样本的真实标签，标量值。

        返回:
            loss (torch.Tensor): 总损失值。
        """
        # 确保模型参数与输入数据位于同一设备
        device = self.class_centers.device
        features = features.to(device)

        if not isinstance(label, torch.Tensor):
            # 如果 label 是 Python 标量，转换为张量并确保类型为 long
            label = torch.tensor(label, dtype=torch.long, device=device)
        else:
            # 如果 label 已经是张量，确保其类型为 long 并移动到正确设备
            label = label.to(device).long()

        # 归一化特征向量
        features = F.normalize(features, p=2, dim=0)

        # 1. 计算超球半径约束损失 Ln
        class_distances = torch.norm(self.class_centers - self.hypersphere_center.unsqueeze(0), dim=1)
        radius_loss = torch.abs(class_distances - self.radius).mean()

        # 2. 计算样本到类别中心的聚类损失 Lc
        target_center = self.class_centers[label]  # 根据标签选择对应的类别中心
        target_center = F.normalize(target_center.squeeze(0), p=2, dim=0)  # 归一化类别中心

        # 确保 shapes 匹配
        assert features.shape == target_center.shape, f"Shapes mismatch: features {features.shape}, target_center {target_center.shape}"

        clustering_loss = torch.norm(features - target_center, dim=0)

        # 3. 总损失
        total_loss = radius_loss + clustering_loss

        return total_loss