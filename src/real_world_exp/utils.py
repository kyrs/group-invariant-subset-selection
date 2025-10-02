import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def nt_xent_loss(z1: torch.Tensor,
                 z2: torch.Tensor,
                 temperature: float = 0.5,
                 eps: float = 1e-8) -> torch.Tensor:
    """
    Symmetric NT-Xent (SimCLR) for a batch of positive pairs (z1, z2).

    Key upgrades vs. the bare version:
    • ε-normalisation to avoid div-by-zero
    • fp32 maths even under AMP
    • self-similarities masked to −∞
    • row-max subtraction to prevent exp() overflow
    """
    B = z1.size(0)

    # 1. L2-normalise (fp32 for safety)
    z1 = F.normalize(z1, dim=1, eps=eps).float()
    z2 = F.normalize(z2, dim=1, eps=eps).float()

    # 2. Full similarity matrix, τ-scaled
    z   = torch.cat([z1, z2], dim=0)            # [2B, D]
    sim = z @ z.T                               # cosine similarities
    sim.div_(temperature)

    # 3. Mask self-similarities
    diag = torch.eye(2 * B, dtype=torch.bool, device=sim.device)
    sim.masked_fill_(diag, -float('inf'))

    # 4. Row-max stabilisation (helps when τ is small / batch large)
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

    # 5. Cross-entropy targets
    targets = torch.arange(B, device=sim.device)
    loss_a  = F.cross_entropy(sim[:B], targets + B)
    loss_b  = F.cross_entropy(sim[B:], targets)

    return 0.5 * (loss_a + loss_b)

def rotate_batch(x, angle_range=(-180, 180), interpolation=TF.InterpolationMode.BILINEAR):
    """
    Rotate each grayscale image in a batch by a random angle.
    
    Args:
        x (Tensor): Tensor of shape [B, H, W]
        angle_range (tuple): Min and max angle (in degrees)
        interpolation: Interpolation mode (default: bilinear)
        
    Returns:
        Tensor: Rotated batch of same shape [B, H, W]
    """
    assert x.dim() == 3, "Input must be 3D (B, H, W)"
    
    angles = (angle_range[1] - angle_range[0]) * torch.rand(x.size(0), device=x.device) + angle_range[0]
    
    return torch.stack([
        TF.rotate(img.unsqueeze(0), angle=float(angle), interpolation=interpolation).squeeze(0)
        for img, angle in zip(x, angles)
    ])
def rotation_contrastive_loss(embeddings1, embeddings2, temperature=0.5):
    """
    embeddings1: [B, D] - original embeddings
    embeddings2: [B, D] - rotated embeddings (positive pair)
    """
    embeddings1 = F.normalize(embeddings1, dim=1)
    embeddings2 = F.normalize(embeddings2, dim=1)

    batch_size = embeddings1.size(0)
    representations = torch.cat([embeddings1, embeddings2], dim=0)  # [2B, D]

    similarity_matrix = torch.matmul(representations, representations.T)  # [2B, 2B]
    similarity_matrix = similarity_matrix / temperature

    # Mask out self-comparisons
    mask = torch.eye(2 * batch_size, device=embeddings1.device).bool()
    similarity_matrix.masked_fill_(mask, -float("inf"))

    # Create positive pair indices
    targets = torch.arange(batch_size, device=embeddings1.device)
    positives = torch.cat([targets + batch_size, targets], dim=0)

    # Contrastive loss
    loss = F.cross_entropy(similarity_matrix, positives)
    return loss



def grouped_nt_xent_loss(embeddings: torch.Tensor,
                         labels:      torch.Tensor,
                         temperature: float = 0.5,
                         eps: float   = 1e-8) -> torch.Tensor:
    """
    Supervised NT-Xent.
    ──────────────────
    • For every label that appears ≥ 2 times in the batch,
      we create pairs (z1, z2) by a simple cyclic shift
      within the group:  [a, b, c] → (a↔b), (b↔c), (c↔a).
    • Each anchor therefore has exactly one positive (as in SimCLR),
      while all *other* embeddings across the batch serve as negatives.
    """
    device = embeddings.device
    anchor_idx, pos_idx = [], []

    # build index lists group-wise
    for lbl in labels.unique():
        idx = torch.nonzero(labels == lbl, as_tuple=True)[0]
        if idx.numel() < 2:                       # skip singletons
            continue
        anchor_idx.append(idx)
        pos_idx.append(idx.roll(shifts=-1, dims=0))  # cyclic shift

    if not anchor_idx:
        raise ValueError("No positive pairs could be formed "
                         "(each label appears only once in this batch).")

    anchor_idx = torch.cat(anchor_idx)
    pos_idx    = torch.cat(pos_idx)

    z1 = embeddings[anchor_idx]   # [P, D]
    z2 = embeddings[pos_idx]      # [P, D]

    return nt_xent_loss(z1, z2, temperature=temperature, eps=eps)

