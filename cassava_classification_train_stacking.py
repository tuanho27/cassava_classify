from collections import defaultdict
import copy
import random
import numpy as np
import os
import shutil
from urllib.request import urlretrieve
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
cudnn.benchmark = True
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import timm
from timm.loss import JsdCrossEntropy
from utils import Mixup, RandAugment, AsymmetricLossSingleLabel, SCELoss, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from PIL import Image
from utils import merge_data, balance_data, TrainDataset, TestDataset

SEED = 42

os.environ['CUDA_VISIBLE_DEVICES'] ="1"

def seed_everything(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(SEED)

class CNNStackModel(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(CNNStackModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 256, kernel_size=(1,3), stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(5,1), stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(512, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 1024, bias=True)
        self.last_linear = nn.Linear(1024, num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x).view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.last_linear(x)
        return x
    
def calculate_accuracy(output, target):
#     return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()
    if params["mix_up"]:
        output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        return accuracy_score(output.cpu(), target.argmax(1).cpu())
    
    output = torch.softmax(output, dim=1)
    return accuracy_score(output.argmax(1).cpu(), target.cpu())

class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()
        self.curr_acc = 0.
    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]
        self.curr_acc = metric["avg"]
    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

def declare_pred_model(name, load_pretrained=False, weight=None):
    model = timm.create_model(name,
            pretrained=False,
            num_classes=params["num_classes"],
            drop_rate=params["drop_rate"])
    model = model.to(params["device"])
    
    if params["distributed"]:
        assert ValueError("No need to implement in a single machine")
    else:
        model = torch.nn.DataParallel(model) 
        
    if load_pretrained:
        state_dict = torch.load(weight)
        print(f"Load pretrained model: {name} ",state_dict["preds"])
        model.load_state_dict(state_dict["model"])
        # best_acc = state_dict["preds"]   
    return model   

def tta_stack_validate(loader, model, params, fold_idx):
    model.eval()
    stream = tqdm(loader)
    outputs = {
                "preds":[],
                "gts":[]
                "image_ids":[]
                }
    with torch.no_grad():
        for i, data in enumerate(stream, start=1):
            tta_output = []   
            for i, image in enumerate(data["images"]):
                # out = torch.softmax(model(image), dim=1)
                logit = model(image)
                tta_output.append(logit)
                outputs["preds"].extend(torch.stack(tta_output, dim=0))
            outputs["gts"].extend(data["labels"])
            outputs["image_ids"].extend(data["image_ids"])
            
    return outputs


if __name__ == "__main__":
    
    root = os.path.join(os.environ["HOME"], "Workspace/datasets/taiyoyuden/cassava")
    train = pd.read_csv(f'{root}/train.csv')
    train_external = pd.read_csv(f'{root}/external/train_external.csv')
    test_external = pd.read_csv(f'{root}/external/test_external.csv')
    test_external_pseudo = pd.read_csv(f'{root}/external/test_external_pseudo.csv')
    test = pd.read_csv(f'{root}/sample_submission.csv')
    label_map = pd.read_json(f'{root}/label_num_to_disease_map.json', 
                            orient='index')
    
    models_name = ["resnest26d","resnest50d", "skresnet50", "tf_efficientnet_b3_ns", "vit_base_patch16_384"]
    WEIGHTS = [

            "./weights/resnest26d/resnest26d_fold0_best_epoch_4_final_2nd.pth",
            "./weights/resnest26d/resnest26d_fold1_best_epoch_7_final_2nd.pth",
            "./weights/resnest26d/resnest26d_fold2_best_epoch_4_final_2nd.pth",
            "./weights/resnest26d/resnest26d_fold3_best_epoch_15_final_2nd.pth",
            "./weights/resnest26d/resnest26d_fold4_best_epoch_21_final_2nd.pth",
    #         #
            
            # 2*"./weights/tf_efficientnet_b3_ns/tf_efficientnet_b3_ns_fold1_best_epoch_19_external.pth", 
            # 3*"./weights/tf_efficientnet_b3_ns/tf_efficientnet_b3_ns_fold1_best_epoch_1_final_512.pth",
    #         "./weights/tf_efficientnet_b3_ns/tf_efficientnet_b3_ns_fold1_best_epoch_26_512.pth",

        
            #"./weights/resnest50d/resnest50d_fold0_best_epoch_30_final_1st.pth",
            # "./weights/resnest50d/resnest50d_fold0_best_epoch_13_final_2nd.pth",
            "./weights/resnest50d/resnest50d_fold0_best_epoch_10_final_3rd.pth",
            # "./weights/resnest50d/resnest50d_fold1_best_epoch_95_final_1st.pth",
            # "./weights/resnest50d/resnest50d_fold1_best_epoch_17_final_2nd.pth",
            "./weights/resnest50d/resnest50d_fold1_best_epoch_8_final_5th_pseudo.pth",
            # "./weights/resnest50d/resnest50d_fold2_best_epoch_50_final_1st.pth",
            "./weights/resnest50d/resnest50d_fold2_best_epoch_22_final_2nd.pth",
            # "./weights/resnest50d/resnest50d_fold3_best_epoch_2_final_2nd.pth",
            "./weights/resnest50d/resnest50d_fold3_best_epoch_1_final_3rd.pth",
            # "./weights/resnest50d/resnest50d_fold4_best_epoch_10_final_2nd.pth",
            # "./weights/resnest50d/resnest50d_fold4_best_epoch_15_final_3rd.pth" ,
            "./weights/resnest50d/resnest50d_fold4_best_epoch_1_final_5th_pseudo.pth",
            # "./weights/resnest50d/resnest50d_fold4_best_epoch_10_final-4th.pth"
            #
            # "./weights/tf_efficientnet_b4_ns_fold0_best_epoch_14_final_2nd_loss.pth",
            # "./weights/tf_efficientnet_b4_ns_fold0_best_epoch_24_final_2nd.pth",
            "./weights/tf_efficientnet_b4_ns_fold0_best_epoch_25_final_3rd.pth",
            # "./weights/tf_efficientnet_b4_ns_fold1_best_epoch_20_final_2nd.pth",
            "./weights/tf_efficientnet_b4_ns_fold1_best_epoch_30_final_3rd.pth",
            # "./weights/tf_efficientnet_b4_ns_fold2_best_epoch_20_final_2nd.pth",
            "./weights/tf_efficientnet_b4_ns_fold2_best_epoch_7_final_3rd.pth",
            "./weights/tf_efficientnet_b4_ns_fold3_best_epoch_16_final_3rd.pth",
            # "./weights/tf_efficientnet_b4_ns_fold3_best_epoch_27_final_2nd.pth",
            "./weights/tf_efficientnet_b4_ns_fold4_best_epoch_20_final_3rd.pth",
            # "./weights/tf_efficientnet_b4_ns_fold4_best_epoch_23_final_2nd.pth",

    ]

    model_index = 1
    ckpt_index = 1

    # ensemble_models_name = list(5*["resnest26d"] + 5*["tf_efficientnet_b3_ns"] + 7*["resnest50d"])
    ensemble_models_name = list(5*["resnest26d"] + 5*["resnest50d"])
    
    ensemble_ckpt_index = [i for i in range(len(ensemble_models_name))]
    
    params = {
        "visualize": False,
        "fold": [0,1,2,3,4],
        "load_pretrained": True,
        "image_size": 512,
        "num_classes": 5,
        "model": models_name[model_index],
        "device": "cuda",
        "batch_size": 2,
        "num_workers": 2,
        "drop_block": 0.2,
        "drop_rate": 0.2,
        "local_rank":0,
        "distributed": False,
        "tta": True,
        "crops_tta":False,
        "balance_data":False,
        "kfold_pred":True,
        "ensemble": True,
        "error_analysis":False,
    }
    val_transform = A.Compose(
        [
            A.CenterCrop(height=params["image_size"], width=params["image_size"], p=1),
            A.Resize(params["image_size"],params["image_size"]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    transform_tta0 = A.Compose(
        [
            A.CenterCrop(height=params["image_size"], width=params["image_size"], p=1),    
            A.Resize(height=params["image_size"], width=params["image_size"], p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),   
            ToTensorV2()
        ]
        )

    transform_tta1 = A.Compose(
        [
            A.CenterCrop(height=params["image_size"], width=params["image_size"], p=1),
            A.Resize(height=params["image_size"], width=params["image_size"], p=1),
            A.HorizontalFlip(p=1.),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),   
            ToTensorV2()
        ]
    )
    transform_tta2 = A.Compose(
        [
            A.CenterCrop(height=params["image_size"], width=params["image_size"], p=1),
            A.Resize(height=params["image_size"], width=params["image_size"], p=1),
            A.VerticalFlip(p=1.),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    transform_tta3 = A.Compose(
        [
            # A.CenterCrop(height=params["image_size"], width=params["image_size"], p=1),
            A.Resize(height=params["image_size"], width=params["image_size"], p=1),
            A.RandomRotate90(p=1.),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),     
            ToTensorV2(),
        ]
    )

    test_transform_tta = [transform_tta0, transform_tta1, transform_tta2, transform_tta3]
  
    folds = train.copy()
    Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['label'])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    cv_acc = 0.
    
    ## Pred Resnet 26d
    r26_logit_preds = []
    for name, fold_idx in zip(ensemble_models_name[:5], ensemble_ckpt_index[:5]):
        print(f"Validate Fold: {fold_idx}")
        fold = fold_idx
        train_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

        train_folds = folds.loc[train_idx].reset_index(drop=True)
        val_folds = folds.loc[val_idx].reset_index(drop=True)

        if params["tta"]:
            val_pred_dataset = TestDataset(val_folds, root, transform=test_transform_tta, valid_test=True)
        else:
            val_pred_dataset = TestDataset(val_folds, root, transform=val_transform, valid_test=True)

        val_pred_loader = DataLoader(
            val_pred_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=2, pin_memory=True,
        )
        model  = declare_pred_model(name, load_pretrained=params["load_pretrained"], weight=WEIGHTS[fold_idx])
        r26_logit_preds.append(tta_stack_validate(val_pred_loader, model, params, fold_idx))
    import ipdb; ipdb.set_trace()
        
  
    gts = torch.cat(r26_logit_preds["gts"]).data.cpu().numpy()
    preds = torch.cat(r26_logit_preds["preds"])

    opt_acc, opt_weight = optimize_weight(gts, preds, stype=args.stype)
    print('optimal accuracy: {}'.format(opt_acc), 'optimal weight ', opt_weight)
    import ipdb; ipdb.set_trace()

    torch.save({
        'preds': preds,
        'gts': gts,
        'image_paths': r26_logit_preds["images_id"],
        'opt_acc': opt_acc,
        'opt_weight': opt_weight
    }, 'prediction/{}_{}_pred_fold{}.pth'.format(args.backbone, args.image_size, '_'.join(str(x) for x in args.folds)))
    


    ## Pred Resnet 50d
    r50_logit_preds = []
    for name, fold_idx in zip(ensemble_models_name[5:10], ensemble_ckpt_index[5:10]):
        print(f"Validate Fold: {fold_idx}")
        fold = fold_idx
        train_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

        train_folds = folds.loc[train_idx].reset_index(drop=True)
        val_folds = folds.loc[val_idx].reset_index(drop=True)

        if params["tta"]:
            val_pred_dataset = TestDataset(val_folds, root, transform=test_transform_tta, valid_test=True)
        else:
            val_pred_dataset = TestDataset(val_folds, root, transform=val_transform, valid_test=True)

        val_pred_loader = DataLoader(
            val_pred_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=2, pin_memory=True,
        )
        model  = declare_pred_model(name, load_pretrained=params["load_pretrained"], weight=WEIGHTS[fold_idx])
        r50_logit_preds.append(tta_stack_validate(val_pred_loader, model, params, fold_idx))


    import ipdb; ipdb.set_trace()
    image_paths = np.array(image_paths)
    gts = torch.cat(gts).data.cpu().numpy()
    preds = torch.cat(preds)

    opt_acc, opt_weight = optimize_weight(gts, preds, stype=args.stype)
    print('optimal accuracy: {}'.format(opt_acc), 'optimal weight ', opt_weight)

    torch.save({
        'preds': preds,
        'gts': gts,
        'image_paths': image_paths,
        'opt_acc': opt_acc,
        'opt_weight': opt_weight
    }, 'prediction/{}_{}_pred_fold{}.pth'.format(args.backbone, args.image_size, '_'.join(str(x) for x in args.folds)))
    
    ## Pred efficientnet-b3
