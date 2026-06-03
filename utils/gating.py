import numpy as np
import torch
from scipy.cluster.vq import kmeans2
from torch import nn


class FeatureStatsCollector:
    def __init__(
        self,
        feature_dim,
        min_variance=1e-6,
        stats_mode="diag",
        n_centroids=10,
    ):
        self.feature_dim = feature_dim
        self.min_variance = min_variance
        self.stats_mode = stats_mode
        self.n_centroids = max(1, int(n_centroids))
        self._class_sums = {}
        self._class_sumsq = {}
        self._class_counts = {}
        self._class_outer = {}
        self._class_features = {}

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
            class_outer = None
            if self.stats_mode == "covariance":
                class_outer = class_features.transpose(0, 1) @ class_features
            elif self.stats_mode == "multi-centroid":
                class_features = class_features.clone()

            if class_idx not in self._class_sums:
                self._class_sums[class_idx] = torch.zeros(self.feature_dim)
                self._class_sumsq[class_idx] = torch.zeros(self.feature_dim)
                self._class_counts[class_idx] = 0
                if self.stats_mode == "covariance":
                    self._class_outer[class_idx] = torch.zeros(
                        self.feature_dim, self.feature_dim
                    )
                elif self.stats_mode == "multi-centroid":
                    self._class_features[class_idx] = []

            self._class_sums[class_idx] += class_sum
            self._class_sumsq[class_idx] += class_sumsq
            self._class_counts[class_idx] += class_count
            if self.stats_mode == "covariance":
                self._class_outer[class_idx] += class_outer
            elif self.stats_mode == "multi-centroid":
                self._class_features[class_idx].append(class_features)

    def compute_mean_variance(self):
        class_stats = {}
        for class_idx in sorted(self._class_counts.keys()):
            count = max(self._class_counts[class_idx], 1)
            mean = self._class_sums[class_idx] / count
            variance = self._class_sumsq[class_idx] / count - mean * mean
            variance = torch.clamp(variance, min=self.min_variance)
            stats = {
                "mean": mean.float(),
                "variance": variance.float(),
                "count": int(self._class_counts[class_idx]),
            }
            if self.stats_mode == "covariance":
                covariance = self._class_outer[class_idx] / count - torch.outer(
                    mean, mean
                )
                covariance = 0.5 * (covariance + covariance.transpose(0, 1))
                covariance.diagonal().copy_(
                    torch.clamp(covariance.diagonal(), min=self.min_variance)
                )
                stats["covariance"] = covariance.float()
            elif self.stats_mode == "multi-centroid":
                stats = self._compute_multi_centroid_stats(class_idx)
            class_stats[class_idx] = stats
        return class_stats

    def _compute_multi_centroid_stats(self, class_idx):
        class_features = self._class_features.get(class_idx, [])
        if not class_features:
            return {"components": [], "count": 0}

        features = torch.cat(class_features, dim=0).float()
        num_points = features.size(0)
        num_centroids = min(self.n_centroids, num_points)

        if num_centroids <= 1:
            return {
                "components": [self._build_component_stats(features)],
                "count": int(num_points),
            }

        features_np = features.numpy().astype(np.float64, copy=False)
        rng = np.random.default_rng(0)
        init_idx = rng.choice(num_points, size=num_centroids, replace=False)
        init_centroids = features_np[init_idx]

        try:
            _, cluster_labels = kmeans2(
                features_np,
                init_centroids,
                iter=30,
                minit="matrix",
            )
        except Exception:
            return {
                "components": [self._build_component_stats(features)],
                "count": int(num_points),
            }

        components = []
        for centroid_idx in range(num_centroids):
            cluster_mask = cluster_labels == centroid_idx
            if not np.any(cluster_mask):
                continue
            cluster_features = features[torch.from_numpy(cluster_mask)]
            components.append(self._build_component_stats(cluster_features))

        if not components:
            components = [self._build_component_stats(features)]

        return {"components": components, "count": int(num_points)}

    def _build_component_stats(self, features):
        mean = features.mean(dim=0)
        variance = torch.var(features, dim=0, unbiased=False)
        variance = torch.clamp(variance, min=self.min_variance)
        return {
            "mean": mean.float(),
            "variance": variance.float(),
            "count": int(features.size(0)),
        }


def _allocate_component_samples(total_samples, component_counts):
    if total_samples <= 0 or not component_counts:
        return [0 for _ in component_counts]

    weights = np.asarray(component_counts, dtype=np.float64)
    if np.all(weights <= 0):
        weights = np.ones_like(weights)
    weights = weights / weights.sum()

    raw = weights * total_samples
    allocated = np.floor(raw).astype(np.int64)
    remainder = int(total_samples - allocated.sum())

    if remainder > 0:
        order = np.argsort(-(raw - allocated))
        allocated[order[:remainder]] += 1

    return allocated.tolist()


def generate_samples(class_stats, n_samples, min_variance=1e-6, device="cpu"):
    features = []
    labels = []
    for class_idx, stats in class_stats.items():
        if "components" in stats:
            component_counts = [component.get("count", 0) for component in stats["components"]]
            component_samples = _allocate_component_samples(n_samples, component_counts)
            for component, samples_for_component in zip(
                stats["components"], component_samples
            ):
                if samples_for_component <= 0:
                    continue

                mean = component["mean"].to(device)
                variance = torch.clamp(
                    component["variance"].to(device), min=min_variance
                )
                std = torch.sqrt(variance)
                noise = torch.randn(
                    samples_for_component, mean.numel(), device=device
                )
                sampled_features = mean.unsqueeze(0) + noise * std.unsqueeze(0)

                features.append(sampled_features)
                labels.append(
                    torch.full(
                        (samples_for_component,),
                        int(class_idx),
                        dtype=torch.long,
                        device=device,
                    )
                )
            continue

        mean = stats["mean"].to(device)
        covariance = stats.get("covariance")
        if covariance is not None:
            covariance = covariance.to(device)
            eye = torch.eye(mean.numel(), device=device, dtype=covariance.dtype)
            jitter = min_variance
            sampled_features = None
            for _ in range(5):
                try:
                    chol = torch.linalg.cholesky(covariance + jitter * eye)
                    noise = torch.randn(n_samples, mean.numel(), device=device)
                    sampled_features = mean.unsqueeze(0) + noise @ chol.transpose(0, 1)
                    break
                except RuntimeError:
                    jitter *= 10.0

            if sampled_features is None:
                variance = torch.clamp(stats["variance"].to(device), min=min_variance)
                std = torch.sqrt(variance)
                noise = torch.randn(n_samples, mean.numel(), device=device)
                sampled_features = mean.unsqueeze(0) + noise * std.unsqueeze(0)
        else:
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
