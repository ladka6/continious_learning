import torch
from torch import nn


class FeatureStatsCollector:
    def __init__(self, feature_dim, min_variance=1e-6):
        self.feature_dim = feature_dim
        self.min_variance = min_variance
        self._class_sums = {}
        self._class_sumsq = {}
        self._class_counts = {}

    def update(self, features, labels):
        features = features.detach().cpu().float()
        labels = labels.detach().cpu().long()

        unique_labels = labels.unique()
        for class_id in unique_labels:
            class_idx = int(class_id.item())
            mask = labels == class_id
            class_features = features[mask]

            class_sum = class_features.sum(dim=0)
            class_sumsq = torch.sum(class_features * class_features, dim=0)
            class_count = class_features.size(0)

            if class_idx not in self._class_sums:
                self._class_sums[class_idx] = torch.zeros(self.feature_dim)
                self._class_sumsq[class_idx] = torch.zeros(self.feature_dim)
                self._class_counts[class_idx] = 0

            self._class_sums[class_idx] += class_sum
            self._class_sumsq[class_idx] += class_sumsq
            self._class_counts[class_idx] += class_count

    def compute_mean_variance(self):
        class_stats = {}
        for class_idx in sorted(self._class_counts.keys()):
            count = max(self._class_counts[class_idx], 1)
            mean = self._class_sums[class_idx] / count
            variance = self._class_sumsq[class_idx] / count - mean * mean
            variance = torch.clamp(variance, min=self.min_variance)
            class_stats[class_idx] = {
                "mean": mean.float(),
                "variance": variance.float(),
                "count": int(self._class_counts[class_idx]),
            }
        return class_stats


def generate_samples(class_stats, n_samples, min_variance=1e-6, device="cpu"):
    features = []
    labels = []
    for class_idx, stats in class_stats.items():
        mean = stats["mean"].to(device)
        variance = torch.clamp(stats["variance"].to(device), min=min_variance)
        std = torch.sqrt(variance)

        noise = torch.randn(n_samples, mean.numel(), device=device)
        sampled_features = mean.unsqueeze(0) + noise * std.unsqueeze(0)

        features.append(sampled_features)
        labels.append(
            torch.full((n_samples,), int(class_idx), dtype=torch.long, device=device)
        )

    if len(features) == 0:
        return (
            torch.empty(0, 0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


class TaskGate(nn.Module):
    def __init__(self, input_dim, num_tasks=1, hidden_dim=0):
        super().__init__()
        self.hidden_dim = hidden_dim

        if hidden_dim > 0:
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            classifier_in_dim = hidden_dim
        else:
            self.backbone = nn.Identity()
            classifier_in_dim = input_dim

        self.classifier = nn.Linear(classifier_in_dim, num_tasks)

    @property
    def num_tasks(self):
        return self.classifier.out_features

    def extend(self, num_tasks):
        if num_tasks <= self.num_tasks:
            return

        old_classifier = self.classifier
        new_classifier = nn.Linear(old_classifier.in_features, num_tasks)
        new_classifier = new_classifier.to(old_classifier.weight.device)

        with torch.no_grad():
            new_classifier.weight[: old_classifier.out_features].copy_(
                old_classifier.weight
            )
            new_classifier.bias[: old_classifier.out_features].copy_(
                old_classifier.bias
            )

        self.classifier = new_classifier

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

class TaskGateWithRandomProjection(nn.Module):
    def __init__(self, input_dim, num_tasks=1, hidden_dim=0, projection_dim=10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.random_projection = nn.Parameter(torch.randn(input_dim, self.projection_dim)).requires_grad_(False)
        

        if hidden_dim > 0:
            self.backbone = nn.Sequential(
                nn.Linear(projection_dim if projection_dim else input_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            classifier_in_dim = hidden_dim
        else:
            self.backbone = nn.Identity()
            classifier_in_dim = projection_dim if projection_dim else input_dim

        self.classifier = nn.Linear(classifier_in_dim, num_tasks)

    @property
    def num_tasks(self):
        return self.classifier.out_features

    def extend(self, num_tasks):
        if num_tasks <= self.num_tasks:
            return

        old_classifier = self.classifier
        new_classifier = nn.Linear(old_classifier.in_features, num_tasks)
        new_classifier = new_classifier.to(old_classifier.weight.device)

        with torch.no_grad():
            new_classifier.weight[: old_classifier.out_features].copy_(
                old_classifier.weight
            )
            new_classifier.bias[: old_classifier.out_features].copy_(
                old_classifier.bias
            )

        self.classifier = new_classifier

    def forward(self, x):
        x = torch.relu(x @ self.random_projection)
        x = self.backbone(x)
        return self.classifier(x)
