from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms
import torch 
import random


class RandomGridPlacementTransform:
    def __init__(self, grid_size = 3, channel = 1):
        self.grid_size = grid_size
        self.channel = channel

    def __call__(self, img_tensor):
        # Convert the image to a tensor
        # img_tensor = F.to_tensor(img)
        _, h, w = img_tensor.shape
        
        # Create a blank background with 3x the original size
        blank_background = torch.zeros((self.channel, self.grid_size * h, self.grid_size * w))  # Blank background of 3*height, 3*width
        
        # Randomly select a grid position to place the image
        grid_h_pos = random.randint(0, self.grid_size-1)  # Grid position in vertical direction (0, 1, 2)
        grid_w_pos = random.randint(0, self.grid_size-1)  # Grid position in horizontal direction (0, 1, 2)

        # Compute the position where the image will be placed
        top = grid_h_pos * h
        left = grid_w_pos * w
        
        # Place the image on the blank background
        blank_background[:, top:top + h, left:left + w] = img_tensor

        return blank_background




def get_dataset(args, seed_gen):
    assert(args.val_ratio<1)

    dataset = args.data
    data_path =args.data_dir
    val_ratio = args.val_ratio
    min_degree = 0
    max_degree = 360
    grid_size = 2
    group_type = args.group_type 
    if dataset == 'MNIST':
        filter_class = [6,9]
        if group_type =="unitary":
            raise NotImplementedError
        elif group_type == "permutation":
            channel = 1
            im_size = (28, 28) # For LeNet
            num_classes = 10
            mean = [0.5]
            std = [0.5]
            transform = transforms.Compose([transforms.ToTensor(), RandomGridPlacementTransform(grid_size = grid_size, channel = channel), transforms.Normalize(mean=mean, std=std), transforms.Resize(im_size)])
        else:
            raise Exception("Given group type is not supported.")
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        print(dst_train.targets)
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        
        class_names = [str(c) for c in range(num_classes)]

        
        dst_train_len = len(dst_train)
        valid_subset_len = int(val_ratio*dst_train_len) 
        train_subset_len = dst_train_len - valid_subset_len
        train_subset, val_subset = torch.utils.data.random_split(
        dst_train, [train_subset_len, valid_subset_len], generator=seed_gen)

    else:
        exit('unknown dataset: %s'%dataset)


    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader



class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling

if __name__ == "__main__":
    gen_seeding = torch.Generator()
    gen_seeding.manual_seed(42)
    get_dataset("MNIST", "../datatset", val_ratio = 0.2, seed_gen=gen_seeding)