import torch
from torch.nn import functional as F


class WeightedCrossEntropy(torch.nn.Module):
    def __init__(self, ignore_index: int, distribution: list[float]) -> None:
        super(WeightedCrossEntropy, self).__init__()
        # Initialize the weights based on the given distribution
        self.weights = [1 / w if w!=0 else 0 for w in distribution]

        # Convert weights to a tensor and move to CUDA
        loss_weights = torch.Tensor(self.weights).to("cuda")
        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=loss_weights
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute the weighted cross-entropy loss
        return self.loss(logits, target)


class DICELoss(torch.nn.Module):
    def __init__(self, ignore_index: int) -> None:
        super(DICELoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        num_classes = logits.shape[1]

        # Convert logits to probabilities using softmax or sigmoid
        if num_classes == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)

        # Create a mask to ignore the specified index
        mask = target != self.ignore_index
        target = target.clone()
        target[~mask] = 0

        # Convert target to one-hot encoding if necessary
        if num_classes == 1:
            target = target.unsqueeze(1)
        else:
            target = F.one_hot(target, num_classes=num_classes)
            target = target.permute(0, 3, 1, 2)

        # Apply the mask to the target
        target = target.float() * mask.unsqueeze(1).float()
        intersection = torch.sum(probs * target, dim=(2, 3))
        union = torch.sum(probs + target, dim=(2, 3))

        # Compute the Dice score
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        valid_dice = dice_score[mask.any(dim=1).any(dim=1)]
        dice_loss = 1 - valid_dice.mean()  # Dice loss is 1 minus the Dice score

        return dice_loss


class FocalLoss(torch.nn.Module):
    def __init__(self, ignore_index: int, distribution: list[float], gamma: float = 2.0) -> None:
        super(FocalLoss, self).__init__()
        # Initialize the weights based on the given distribution
        #self.weights = [1 / w if w!=0 else 0 for w in distribution]

        # Convert weights to a tensor and move to CUDA
        #loss_weights = torch.Tensor(self.weights).to("cuda")
        self.gamma = gamma
        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction='none', # weight=loss_weights, 
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute the cross-entropy loss
        ce_loss = self.loss(logits, target)
        
        # Get the predicted probabilities
        probs = F.softmax(logits, dim=1)
        
        # Select the probabilities for the true class
        p_t = probs.gather(1, target.unsqueeze(1))  # Shape (B, 1)

        # Compute the focal loss component
        focal_loss = ce_loss * (1 - p_t) ** self.gamma
        
        # Return the mean loss over the batch
        return focal_loss.mean()


class VICReg(torch.nn.Module):

    def __init__(self, vic_weights: list[float], inv_loss: str = "mse", ignore_index = None):
        super().__init__()

        self.variance_loss_epsilon = 1e-08
        
        self.variance_loss_weight = vic_weights[0]
        self.invariance_loss_weight = vic_weights[1]
        self.covariance_loss_weight = vic_weights[2]

        if inv_loss == "mse":
            self.inv = torch.nn.MSELoss()
        elif inv_loss == "cca":
            self.inv = CCALoss()
        elif inv_loss == "ntxent":
            self.inv = NTXentLoss()

    def forward(self, z_a, z_b, each_comp=False):

        loss_inv = self.inv(z_a, z_b)

        std_z_a = torch.sqrt(
            z_a.var(dim=0) + self.variance_loss_epsilon
        )
        std_z_b = torch.sqrt(
            z_b.var(dim=0) + self.variance_loss_epsilon
        )
        loss_v_a = torch.mean(F.relu(1 - std_z_a))
        loss_v_b = torch.mean(F.relu(1 - std_z_b))
        loss_var = loss_v_a + loss_v_b

        N, D = z_a.shape

        z_a = z_a - z_a.mean(dim=0)
        z_b = z_b - z_b.mean(dim=0)

        cov_z_a = ((z_a.T @ z_a) / (N - 1)).square()  # DxD
        cov_z_b = ((z_b.T @ z_b) / (N - 1)).square()  # DxD
        loss_c_a = (cov_z_a.sum() - cov_z_a.diagonal().sum()) / D
        loss_c_b = (cov_z_b.sum() - cov_z_b.diagonal().sum()) / D
        loss_cov = loss_c_a + loss_c_b

        
        weighted_var = loss_var * self.variance_loss_weight
        weighted_cov = loss_cov * self.covariance_loss_weight

        weighted_inv = loss_inv * self.invariance_loss_weight

        loss = weighted_inv + weighted_var + weighted_cov
        if each_comp: return loss.mean(), loss_var, loss_inv, loss_cov
        else: return loss.mean()


class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits_1, logits_2):
        """
        :param logits_1: Tensor of shape (batch_size, feature_dim)
        :param logits_2: Tensor of shape (batch_size, feature_dim)
        :return: NT-Xent loss
        """
        batch_size = logits_1.shape[0]

        # Normalize logits
        logits_1 = F.normalize(logits_1, p=2, dim=1)
        logits_2 = F.normalize(logits_2, p=2, dim=1)

        # Compute similarity between all pairs (batch_size x batch_size)
        logits = torch.matmul(logits_1, logits_2.T) / self.temperature

        # Extract positive pairs (batch_size, 1)
        positive_logits = torch.diag(logits)

        # Remove diagonal elements from logits (mask self-comparisons)
        mask = torch.eye(batch_size, dtype=torch.bool).to(logits.device)
        logits = logits[~mask].view(batch_size, -1)  # Remove the diagonal and reshape

        # Concatenate the positive logits with the remaining logits
        logits = torch.cat([positive_logits.unsqueeze(1), logits], dim=1)

        # Labels are always 0 (first column has the positive pair)
        labels = torch.zeros(batch_size, dtype=torch.long).to(logits.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


class CCALoss(torch.nn.Module):
    def __init__(self, epsilon=1e-4):
        """
        Canonical Correlation Analysis (CCA) Loss.
        
        Args:
            epsilon (float): Small constant for numerical stability.
        """
        super(CCALoss, self).__init__()
        self.epsilon = epsilon

    import torch

    def forward(self, H1, H2):
        """
        Canonical Correlation Analysis (CCA) Loss with regularization for small batch sizes.

        Args:
            H1: (batch_size, feature_dim) Tensor for view 1.
            H2: (batch_size, feature_dim) Tensor for view 2.
            reg: Regularization coefficient.

        Returns:
            -cca_loss: Negative correlation between the views.
        """
        batch_size = H1.shape[0]

        # Compute covariance matrices
        H1_mean = H1 - H1.mean(dim=0, keepdim=True)
        H2_mean = H2 - H2.mean(dim=0, keepdim=True)

        Sigma_11 = (H1_mean.T @ H1_mean) / (batch_size - 1) + self.epsilon * torch.eye(H1.shape[1], device=H1.device)
        Sigma_22 = (H2_mean.T @ H2_mean) / (batch_size - 1) + self.epsilon * torch.eye(H2.shape[1], device=H2.device)
        Sigma_12 = (H1_mean.T @ H2_mean) / (batch_size - 1)

        # Compute square root inverse using SVD (stable alternative to Cholesky)
        U1, S1, V1 = torch.linalg.svd(Sigma_11)
        Sigma_11_inv_sqrt = U1 @ torch.diag(1.0 / torch.sqrt(S1)) @ V1.T

        U2, S2, V2 = torch.linalg.svd(Sigma_22)
        Sigma_22_inv_sqrt = U2 @ torch.diag(1.0 / torch.sqrt(S2)) @ V2.T

        # Compute correlation matrix
        C = Sigma_11_inv_sqrt @ Sigma_12 @ Sigma_22_inv_sqrt

        # Maximize sum of singular values (canonical correlations)
        cca_corr = torch.linalg.svdvals(C)
        loss = -cca_corr.sum()  # Negative because we want to maximize correlation

        return loss
