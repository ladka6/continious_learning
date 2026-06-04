import torch
from torch import nn


class FeatureStatsCollector:
    def __init__(self, feature_dim, min_variance=1e-6, stats_mode="diag"):
        self.feature_dim = feature_dim
        self.min_variance = min_variance
        self.stats_mode = stats_mode
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

            if class_idx not in self._class_sums:
                self._class_sums[class_idx] = torch.zeros(self.feature_dim)
                self._class_sumsq[class_idx] = torch.zeros(self.feature_dim)
                self._class_counts[class_idx] = 0
                if self.stats_mode == "covariance":
                    self._class_outer[class_idx] = torch.zeros(
                        self.feature_dim, self.feature_dim
                    )
                if self.stats_mode == "fetril":
                    self._class_features[class_idx] = []

            self._class_sums[class_idx] += class_sum
            self._class_sumsq[class_idx] += class_sumsq
            self._class_counts[class_idx] += class_count
            if self.stats_mode == "covariance":
                self._class_outer[class_idx] += class_outer
            if self.stats_mode == "fetril":
                self._class_features[class_idx].append(class_features.clone())

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
                covariance.diagonal().copy_(torch.clamp(covariance.diagonal(), min=self.min_variance))
                stats["covariance"] = covariance.float()
            if self.stats_mode == "fetril":
                stats["features"] = torch.cat(self._class_features[class_idx], dim=0)
            class_stats[class_idx] = stats
        return class_stats


def generate_samples(class_stats, n_samples, min_variance=1e-6, device="cpu"):
    features = []
    labels = []
    for class_idx, stats in class_stats.items():
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


def _sample_rows(features, n_samples):
    if features.size(0) == 0 or n_samples <= 0:
        return features[:0]

    if features.size(0) >= n_samples:
        indices = torch.randperm(features.size(0))[:n_samples]
    else:
        indices = torch.randint(0, features.size(0), (n_samples,))
    return features[indices]


def _select_fetril_source_class(
    target_mean,
    current_means,
    current_class_ids,
    strategy="nearest",
    topk=1,
):
    if strategy == "random":
        random_idx = torch.randint(0, len(current_class_ids), (1,)).item()
        return current_class_ids[random_idx]

    target_mean = torch.nn.functional.normalize(target_mean.unsqueeze(0), p=2, dim=1)
    source_means = torch.nn.functional.normalize(current_means, p=2, dim=1)
    similarities = torch.matmul(target_mean, source_means.transpose(0, 1)).squeeze(0)
    order = torch.argsort(similarities, descending=True)
    rank = min(max(int(topk), 1), len(current_class_ids)) - 1
    return current_class_ids[int(order[rank].item())]


def generate_fetril_features(
    task_feature_stats,
    current_task,
    n_samples,
    device="cpu",
    source_strategy="nearest",
    topk=1,
):
    current_class_stats = task_feature_stats.get(current_task, {})
    if len(current_class_stats) == 0:
        return (
            torch.empty(0, 0, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    current_class_ids = sorted(current_class_stats.keys())
    current_means = torch.stack(
        [current_class_stats[class_idx]["mean"].float() for class_idx in current_class_ids],
        dim=0,
    )

    features = []
    labels = []

    for class_idx in current_class_ids:
        class_features = current_class_stats[class_idx].get("features")
        if class_features is None or class_features.numel() == 0:
            continue
        sampled = _sample_rows(class_features.float(), n_samples).to(device)
        features.append(sampled)
        labels.append(
            torch.full((sampled.size(0),), int(class_idx), dtype=torch.long, device=device)
        )

    for task_idx, class_stats in task_feature_stats.items():
        if task_idx == current_task:
            continue
        for class_idx, stats in class_stats.items():
            source_class_idx = _select_fetril_source_class(
                stats["mean"].float(),
                current_means,
                current_class_ids,
                strategy=source_strategy,
                topk=topk,
            )
            source_stats = current_class_stats[source_class_idx]
            source_features = source_stats.get("features")
            if source_features is None or source_features.numel() == 0:
                continue

            sampled_source = _sample_rows(source_features.float(), n_samples)
            translated = (
                sampled_source
                + stats["mean"].float().unsqueeze(0)
                - source_stats["mean"].float().unsqueeze(0)
            ).to(device)
            features.append(translated)
            labels.append(
                torch.full(
                    (translated.size(0),),
                    int(class_idx),
                    dtype=torch.long,
                    device=device,
                )
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
