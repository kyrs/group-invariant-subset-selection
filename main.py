import torch 
import random 
import numpy as np  
import argparse
from src.synthetic_data import orthogonalGrpDataset,PermutationGrpDataset
from src.real_world_exp.dataloader import MNISTDataset as RotMNISTDataset
from src.real_world_exp.dataloader import TetrisDataGenerator, TetrisDatasetPyG
from src.tsne_plot import plot_and_save_umap
from sklearn.model_selection import train_test_split 
from src.noise import flip_label_noise 
from src.subset_selection import get_cutstat_subset, tracker_dict_based_selection
from src.model_with_baselines import classifier
from collections import Counter
import os
import json
from src.logging import TensorBoardLogger
import sys
from config import configSelector
from src.metric import compute_pairwise_distances, identity_rep, ortho_grp_rep, permute_grp_rep, lorentz_grp_rep, get_embeddings_from_config, fetch_embedding_model
def seed_fn(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def label_info(y_true, y_pred):
    assert(len(y_true)== len(y_pred))

    equal_value = (y_true==y_pred).sum()
    acc = (equal_value/len(y_true))*100
    #y_bal = y_pred.sum()/ len(y_pred)
    

    y_pred_list = y_pred.tolist()
    y_true_list = y_true.tolist()
    print(f"y_pred : {Counter(y_pred_list)}")
    print(f"y_true : {Counter(y_true_list)}")

    smallest_y_pred = min(Counter(y_pred_list).values())
    y_pred_bal = smallest_y_pred / len(y_pred_list)

    smallest_true_y = min(Counter(y_true_list).values())
    y_true_bal = smallest_true_y / len(y_true_list)

    if acc.item() < 0:
        acc = torch.tensor(0.0)
    if y_pred_bal < 0:
        y_pred_bal = torch.tensor(0.0)
    if y_true_bal < 0:
        y_true_bal = torch.tensor(0.0)
    return acc.item(), y_pred_bal, y_true_bal


def get_scores(args, subset_x_train, subset_y_train, subset_y_orig, test_data, identifier="euclid"):
    seed_fn(args.seed)
   
    acc, y_hat, _ = classifier(train_data=(subset_x_train,subset_y_train) , test_data= test_data, input_size= args.vec_dim, learning_rate = args.lr, batch_size=args.batch_size, device = args.device, args=args)
    acc_y_label , y_pred_bal, y_true_bal = label_info(y_true = subset_y_orig, y_pred = subset_y_train)
    scores = { "acc": acc, "label_corrupt_acc": acc_y_label,
             "label_selected_bal": y_pred_bal, "label_noisy_bal": y_true_bal}
    return scores

def main(args):
    ## fn code (model) ## daloader ## network
    ## fn (conrastive learning, group based fn)
    dataClassDict = {"ortho": orthogonalGrpDataset, "perm": PermutationGrpDataset, "rot_mnist": RotMNISTDataset, "tetris": TetrisDataGenerator, "svhn":SVHNDataset, "cifar10": CIFAR10Dataset} 
    LOGGER = TensorBoardLogger(dataset = args.data, logDir = args.log_dir)
    hparams = vars(args)
    seed_fn(args.seed)

    print(f"Dataset: {args.data}, vec_dim: {args.vec_dim}, num_samples: {args.num_samples}, test_size: {args.test_size}, lr: {args.lr}, noise_prob: {args.noise_prob}, seed: {args.seed}, NN: {args.NN}, subset_size_percent: {args.subset_size_percent}, sampling_type: {args.sampling_type}, method : {args.method}")
    classDataset = dataClassDict[args.data]
    if args.data in ["ortho", "perm"]:
        grpDataObj = classDataset(num_samples =args.num_samples, dim = args.vec_dim, sampling_type = args.sampling_type)
        data_fet, data_label = grpDataObj.generate_data()
        classNum = 2
        priors = 1/classNum
    elif args.data == "rot_mnist":
        config = configSelector(args)
        dataObj = classDataset(path = config.path)
        data_fet, data_label = dataObj.generate_data()
        classNum = 10
        priors = None
    elif args.data == "tetris":
        config = configSelector(args)
        dataObj = classDataset()
        data_fet, data_label = dataObj.generate_data(num_elements = config.num_elements)
        classNum = 8
        priors = None
    else:
        raise NotImplementedError(f"Dataset {args.data} not implemented")

    x_train, x_test, y_orig, y_test = train_test_split(data_fet, data_label, test_size = args.test_size, random_state = args.seed)

    assert(len(x_train)== len(y_orig))
    assert(len(x_test) == len(y_test))
    ### introduce noise

    y_train = flip_label_noise(y = y_orig, p = args.noise_prob, classNum=classNum)
    x_train = torch.tensor(x_train, dtype= torch.float32)
    x_test = torch.tensor(x_test, dtype= torch.float32)

    if not isinstance(y_train, torch.Tensor):
        y_train = torch.tensor(y_train)
    
    if not isinstance(y_test, torch.Tensor):
        y_test = torch.tensor(y_test)

    if not isinstance(y_orig, torch.Tensor):
        y_orig = torch.tensor(y_orig)

    test_dat = (x_test, y_test)

    args.classNum = classNum
    if args.method == "noiseless_complete":
        print("noiseless_complete method selected, using entire training set")
        score = get_scores(args=args, subset_x_train= x_train, subset_y_train = y_orig, subset_y_orig = y_orig, test_data = test_dat, identifier= args.method)
        print( f"Scores: {score}")
    elif args.method == "noiseless_random":
        N_select = int(args.subset_size_percent * len(x_train))
        indices_to_keep = torch.randperm(len(x_train))[:N_select]
        score = get_scores(args=args, subset_x_train= x_train[indices_to_keep], subset_y_train = y_orig[indices_to_keep], subset_y_orig = y_orig[indices_to_keep], test_data = test_dat, identifier= args.method)
        print( f"Scores: {score}")
    else:
        if args.method == "vanilla_cutstats":
            print("cutstats_vanilla method selected, using cutstats based selection")
            if args.data in ["rot_mnist", "svhn", "cifar10"]:
                fet = x_train.reshape(len(x_train), -1)
           
            else:
                fet = x_train
            pairwise_dists_euclid =  compute_pairwise_distances(arr = fet, fn = identity_rep) 
            indices_to_keep = get_cutstat_subset(labels = y_train, pairwise_dists = pairwise_dists_euclid, coverage=args.subset_size_percent, K=args.NN, device='cpu', priors = priors)
        
        elif args.method == "noisy_complete":
            print("noisy_complete method selected, using entire training set with noisy labels")
            indices_to_keep = torch.arange(len(x_train))
        
        elif args.method in ["forget", "grand", "entropy", "herding"]:
            print(f"{args.method} method selected, using tracker_dict based selection")
            # if args.data not in ["tetris"]:
            _, _, tracker_dict = classifier(train_data=(x_train,y_train) , test_data= (x_test,y_test), input_size= args.vec_dim, learning_rate = args.lr, device = args.device, batch_size=args.batch_size, returnTracker = True, args=args)
            # else:
                # raise NotImplementedError("Methods like forget, grand, el2n, entropy, herding are not implemented for Rotated MNIST dataset")
            indices_to_keep = tracker_dict_based_selection(tracker_dict, method=args.method, coverage=args.subset_size_percent)
        
        elif args.method== "random":
            N_select = int(args.subset_size_percent * len(x_train))
            indices_to_keep = torch.randperm(len(x_train))[:N_select]
        elif args.method == "contrastive_cutstats":
            print("contrastive_cutstats method selected, using contrastive learning based selection")
            if args.data in ["rot_mnist", "tetris"]:
                model = fetch_embedding_model(args=args, method="classical")

                embeddings = get_embeddings_from_config(model_path = config.model_contrast, model = model, dataset = x_train, method="classical", args=args)
                pairwise_dists_euclid =  compute_pairwise_distances(arr = embeddings, fn=identity_rep) 
                indices_to_keep = get_cutstat_subset(labels = y_train, pairwise_dists = pairwise_dists_euclid, coverage=args.subset_size_percent, K=args.NN, device='cpu', priors = priors)
            else:
                raise NotImplementedError("Contrastive method not implemented yet")
        
        elif args.method == "ours":
            if args.data in ["rot_mnist", "tetris"]:
                print(f"Using our method with eqv learning for {args.data} dataset")
                model = fetch_embedding_model(args=args, method="eqv")

                embeddings = get_embeddings_from_config(model_path = config.model_eqv, model = model,method="eqv", dataset = x_train, args=args)
                pairwise_dists_euclid =  compute_pairwise_distances(arr = embeddings, fn=identity_rep) 
                indices_to_keep = get_cutstat_subset(labels = y_train, pairwise_dists = pairwise_dists_euclid, coverage=args.subset_size_percent, K=args.NN, device='cpu', priors = priors)
            else:    
                fn_grp = {"ortho": ortho_grp_rep, "perm": permute_grp_rep}
                pairwise_dists_euclid =  compute_pairwise_distances(arr = x_train, fn = fn_grp[args.data]) 
                indices_to_keep = get_cutstat_subset(labels = y_train, pairwise_dists = pairwise_dists_euclid, coverage=args.subset_size_percent, K=args.NN, device='cpu', priors = priors)
 
            
    
        else:
            raise NotImplementedError(f"Method {args.method} not implemented")
        
        
        subset_x_train = x_train[indices_to_keep]
        subset_y_train = y_train[indices_to_keep]
        subset_y_orig = y_orig[indices_to_keep]
        score = get_scores(args=args, subset_x_train= subset_x_train, subset_y_train = subset_y_train, subset_y_orig = subset_y_orig, test_data = test_dat, identifier= args.method)

    ##############################
    # assert(np.isfinite(x_train).all())
    # assert(np.isfinite(y_train).all())
    
    # plot_and_save_umap(X =x_train, y= y_train, filename=args.method, highlight_idx = indices_to_keep)
    print(f"Scores: {score}")
    LOGGER.log_hypm(hparams= hparams, metrics = score)
    LOGGER.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vec_dim", type = int, default= 100)
    parser.add_argument("--num_samples", type = int, default= 10000)
    parser.add_argument("--test_size", type = float, default= 0.2)
    parser.add_argument("--lr", type = float, default= 0.0001)
    parser.add_argument("--batch_size", type = int, default= 1024)
    parser.add_argument("--noise_prob", type = float, default= 0.2)
    parser.add_argument("--seed", type = int, default = 42)
    parser.add_argument("--NN", type = int, default= 20)
    parser.add_argument("--subset_size_percent", type = float, required = True)
    parser.add_argument("--sampling_type", type = str, choices = ["uniform", "normal"], default = "uniform")
    parser.add_argument("--device", type = str, default = "cuda:0")
    parser.add_argument("--log_dir", type = str, default = "./logs_tx/")
    parser.add_argument("--data", type = str, default = "ortho", choices = ["ortho", "perm", "rot_mnist", "tetris"], help = "Dataset to use for the experiment")
    parser.add_argument("--method", type = str, default = "noiseless_complete", choices = ["noiseless_complete", "noisy_complete", "vanilla_cutstats", "forget", "entropy", "herding", "random", "ours", "vanilla_cutstats", "contrastive_cutstats", "noiseless_random"], help = "Method to use for subset selection")
    args = parser.parse_args()
    vals = main(args)
            


