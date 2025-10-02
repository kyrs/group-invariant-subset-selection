import torch

# Load the old checkpoint
old_ckpt = torch.load("../../../models/so2/rot_mnist/so2_eqv_model.pth", map_location="cpu")

# Create a new checkpoint dict
new_ckpt = {}

for k, v in old_ckpt.items():
    # Rename weight keys for R2Conv layers (blockX.0.weight → blockX.0.filter)
    if k.startswith("encoder.block") and ".0.weights" in k:
        new_key = k.replace(".0.weights", ".0.filter")
        print(f"Renaming {k} → {new_key}")
    else:
        new_key = k
    new_ckpt[new_key] = v

# Save the new checkpoint
torch.save(new_ckpt, "../../../models/so2/rot_mnist/model_e2cnn_so2_renamed.pth")
