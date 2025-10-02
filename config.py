class RotMnistConfig:
    path = "src/real_world_exp/data/ROTMNIST/mnist_all_rotation_normalized_float_combined.amat"
    model_eqv = "models/so2/rot_mnist/so2_eqv_model_epoch_19.pth"
    model_contrast = "models/so2/rot_mnist/so2_classical_model_epoch_19.pth"

class TetrisConfig: 
    num_elements = 10000
    model_eqv = "models/se3/tetris/se3_eqv_model_tetris_epoch_19.pth"
    model_contrast = "models/se3/tetris/se3_classical_model_tetris_ec_epoch19.pth"


def configSelector(args):
    if args.data == "rot_mnist":
        return RotMnistConfig()
    elif args.data == "tetris":
        return TetrisConfig()
    else:
        raise ValueError(f"Configuration for dataset {args.data} is not defined.")