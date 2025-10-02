from tensorboardX import SummaryWriter 
import datetime 
from os.path import join 
import numpy as  np
import os 
import PIL 
import json 

class TensorBoardLogger(object):
    def __init__(self, dataset, logDir, folderName = None):
        self.dataset  = dataset 
        self.logDir = logDir 
        
        if folderName is None :
            folderName = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self.log_dir = f"{logDir}/{dataset}/{folderName}"
        self.writer = SummaryWriter(self.log_dir)
 
        os.makedirs(self.log_dir, exist_ok=True) 

    def write_opt_step(self, key, value, info, step):
        indexName = f"opt/{info}/{key}" 
        self.writer.add_scalar(indexName, value, step)
    

    def log_hypm(self, hparams, metrics):
        
        self.writer.add_hparams(hparams, metrics)
        json_path = os.path.join(self.log_dir, "hyperparams_and_metrics.json")
        with open(json_path, "w") as f:
            json.dump({"hyperparameters": hparams, "metrics": metrics}, f, indent=4)

    def close(self):
        self.writer.close()