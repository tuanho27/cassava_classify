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
from  torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import timm
from timm.loss import JsdCrossEntropy
from utils import Mixup, RandAugment, AsymmetricLossSingleLabel, SCELoss, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, fmix, RAdam
from utils import merge_data, balance_data, TrainDataset, TestDataset
from PIL import Image
from torchcontrib.optim import SWA
from apex import amp

cudnn.benchmark = True
SEED = 42

def seed_everything(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(SEED)

os.environ['CUDA_VISIBLE_DEVICES'] ="0"

def calculate_accuracy(output, target):
#     return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()
    if params["mix_up"]:
        output = torch.argmax(torch.softmax(output, dim=1), dim=1)
#         return accuracy_score(output.cpu(), target.argmax(1).cpu())
        return accuracy_score(output.cpu(), target.cpu())
    
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

def update_hard_sample(train_loader, model, val_criterion, thres):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    train_loss_list = {'image_id':[],
                       'label':[],
                       'loss':[],
                       'fold':[]}
    model.eval()
    stream = tqdm(train_loader)
    with torch.no_grad():
        for i, (images, target, name) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True)#.view(-1,params['batch_size'])
            output = model(images)
            loss = val_criterion(output, target)
            if loss > thres:
                train_loss_list['image_id'].append(name[0])
                train_loss_list['label'].append(int(target[0].cpu().numpy()))
                train_loss_list['loss'].append(loss)
                train_loss_list['fold'].append(params['fold'])
    print("Number hard samples:",len(train_loss_list["loss"]))
        
    return dict(image_id=train_loss_list['image_id'],
                label=train_loss_list['label'],
                fold=train_loss_list['fold'])


def declare_model(params, load_pretrained=False, weight=None):
    if "efficientnet" in params["model"]:   
        model = timm.create_model(
                params["model"],
                pretrained=False,
                num_classes=params["num_classes"], 
                drop_rate=params["drop_rate"], 
                drop_path_rate=0.2)
    else:
        model = timm.create_model(params["model"],
                pretrained=False,
                num_classes=params["num_classes"],
                drop_rate=params["drop_rate"])
    model = model.to(params["device"]) 
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    # optimizer = RAdam(model.parameters(), lr=params["lr"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=params["lr_min"], last_epoch=-1)
           
    if params["fp16"]:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)    
        
    if params["distributed"]:
        assert ValueError("No need to implement in a single machine")
    else:
        model = torch.nn.DataParallel(model) 
        
    if load_pretrained:
        state_dict = torch.load(weight)
        name = params["model"]
        try:
            print(f"Load pretrained model: {name} ",state_dict["preds"], state_dict["loss"])
        except:
            print(f"Load pretrained model: {name} ",state_dict["preds"])
        model.load_state_dict(state_dict["model"])
        if params["fp16"]:
            optimizer.load_state_dict(state_dict['optimizer'])
            amp.load_state_dict(state_dict['amp'])
        best_acc = state_dict["preds"]
    else:
        best_acc = 0.85  

    return model, optimizer, scheduler, best_acc


def train_epoch(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    if params["hard_negative_sample"]:
        stream = tqdm(update_train_loader)
    else:
        stream = tqdm(train_loader)
    for i, (images, target, _, _) in enumerate(stream, start=1):
#         with autocast():
        images = images.to(params["device"]) #, non_blocking=True)
        target = target.to(params["device"]) #, non_blocking=True) #.view(-1,params['batch_size'])
        if params["mix_up"]:
            images , mtarget = mixup_fn(images, target)
        if epoch > 10 and params["fmix"]:
            images , ftarget = fmix(images, target, alpha=1., decay_power=5.,
                        shape=(params["image_size"],params["image_size"]),
                        device=params["device"])      
            
            
        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
            
        if epoch > 10 and params["fmix"]:
            loss = criterion_fmix(output, ftarget[0]) * ftarget[2] + criterion_fmix(output, ftarget[1]) * (1. - ftarget[2])
            
        else:
            loss = criterion(output, mtarget)
            
#         loss = criterion(output, target)
            
        if params['gradient_accumulation_steps'] > 1:
            loss = loss / params['gradient_accumulation_steps']
    
        accuracy = calculate_accuracy(output, target)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        optimizer.zero_grad()
        loss.backward()
#         with amp.scale_loss(loss, optimizer) as scaled_loss:
#             scaled_loss.backward()
            
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )
            
def validate(val_loader, model, criterion, optimizer, epoch, params, fold, best_acc):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target, _,_) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True)#.view(-1,params['batch_size'])
            output = model(images)
            loss = val_criterion(output, target)
            output = torch.softmax(output, dim = 1)
            accuracy = accuracy_score(output.argmax(1).cpu(), target.cpu())

            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )           
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            
        #to save weight
        if (metric_monitor.curr_acc > best_acc): # or epoch == params["epochs"]:
            print(f"Save best weight at acc {round(metric_monitor.curr_acc,4)}, epoch: {epoch}")
            best_acc = metric_monitor.curr_acc
            
            directory = f'weights/{params["model"]}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            if params["fp16"]:
                torch.save({'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                    'loss': loss,
                    'preds': round(metric_monitor.curr_acc,4)},
                     f'weights/{params["model"]}/{params["model"]}_fold{fold}_best_epoch_{epoch}.pth')
            else:
                torch.save({'model': model.state_dict(), 
                    'loss': loss,
                    'optimizer': optimizer.state_dict(),
                    'preds': round(metric_monitor.curr_acc,4)},
                     f'weights/{params["model"]}/{params["model"]}_fold{fold}_best_epoch_{epoch}.pth')  
    return best_acc

def train_epoch_bak(train_loader, model, criterion, optimizer, epoch, params, scaler=None, criterion_fmix=None):
    metric_monitor = MetricMonitor()
    model.train()
    if params["hard_negative_sample"]:
        stream = tqdm(update_train_loader)
    else:
        stream = tqdm(train_loader)
    for i, (images, target, _, _) in enumerate(stream, start=1):
        mix_decision = np.random.rand()
        with autocast():
            images = images.to(params["device"]) #, non_blocking=True)
            target = target.to(params["device"]) #, non_blocking=True) #.view(-1,params['batch_size'])
            if params["mix_up"]:
                images , mtarget = mixup_fn(images, target)
                if params["distill_soft_label"]:
                    mtarget = mtarget*0.7 + soft_target.to(params['device']) * 0.3
            
            # if epoch > 15 and params["fmix"] and not params["fp16"]:
            #     images , ftarget = fmix(images, target, alpha=1., decay_power=5.,
            #                         shape=(params["image_size"],params["image_size"]),
            #                         device=params["device"])

            output = model(images)
        
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = criterion(output, mtarget)
            # if epoch > 70 and params["fmix"]:
                # loss = criterion_fmix(output, ftarget[0]) * ftarget[2] + criterion_fmix(output, ftarget[1]) * (1. - ftarget[2])
            # else:
                # loss = criterion(output, mtarget)
            # loss1 = symetric_criterion(output, target)
            # loss2 = asymetric_criterion(output, target)
            
            if params['gradient_accumulation_steps'] > 1:
                loss = loss / params['gradient_accumulation_steps']
        
            accuracy = calculate_accuracy(output, target)
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            optimizer.zero_grad()
            loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
                # scaled_loss.backward()
            optimizer.step()
            # optimizer.update_swa()
            # scaler.step(optimizer)
  
            stream.set_description(
                "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

if __name__ == "__main__":

    root = os.path.join(os.environ["HOME"], "Workspace/datasets/taiyoyuden/cassava")
    train = pd.read_csv(f'{root}/train.csv')
    train_external = pd.read_csv(f'{root}/external/train_external.csv')
    test_external = pd.read_csv(f'{root}/external/test_external.csv')
    # test_external_pseudo = pd.read_csv(f'{root}/external/test_external_pseudo_0.8.csv')
    test_external_pseudo = pd.read_csv(f'{root}/external/test_external_pseudo_0.8_round2.csv')
    test = pd.read_csv(f'{root}/sample_submission.csv')
    label_map = pd.read_json(f'{root}/label_num_to_disease_map.json', orient='index')

    models_name = ["resnest26d","resnest50d","tf_efficientnet_b3_ns" ,"tf_efficientnet_b4_ns","legacy_seresnext26_32x4d", "vit_base_patch16_384"]
    WEIGHTS = [
    
        # #"./weights/resnest50d/resnest50d_fold0_best_epoch_30_final_1st.pth",
        # #"./weights/resnest50d/resnest50d_fold0_best_epoch_13_final_2nd.pth",
        # "./weights/resnest50d/resnest50d_fold0_best_epoch_10_final_3rd.pth",
        # #"./weights/resnest50d/resnest50d_fold1_best_epoch_95_final_1st.pth",
        # #"./weights/resnest50d/resnest50d_fold1_best_epoch_17_final_2nd.pth",
        # "./weights/resnest50d/resnest50d_fold1_best_epoch_8_final_5th_pseudo.pth",
        # #"./weights/resnest50d/resnest50d_fold2_best_epoch_50_final_1st.pth",
        # "./weights/resnest50d/resnest50d_fold2_best_epoch_22_final_2nd.pth",
        # #"./weights/resnest50d/resnest50d_fold3_best_epoch_2_final_2nd.pth",
        # "./weights/resnest50d/resnest50d_fold3_best_epoch_1_final_3rd.pth",
        # #"./weights/resnest50d/resnest50d_fold4_best_epoch_10_final_2nd.pth",
        # #"./weights/resnest50d/resnest50d_fold4_best_epoch_15_final_3rd.pth" ,
        # #"./weights/resnest50d/resnest50d_fold4_best_epoch_10_final-4th.pth"
        # "./weights/resnest50d/resnest50d_fold4_best_epoch_1_final_5th_pseudo.pth",
        #
        # # "./weights/resnest26d/resnest26d_fold0_best_epoch_4_final_2nd.pth",
        # "weights/resnest26d/resnest26d_fold0_best_epoch_19_final_3rd.pth",
        # "./weights/resnest26d/resnest26d_fold1_best_epoch_7_final_2nd.pth",
        # "./weights/resnest26d/resnest26d_fold2_best_epoch_4_final_2nd.pth",
        # # "./weights/resnest26d/resnest26d_fold3_best_epoch_15_final_2nd.pth",
        # "weights/resnest26d/resnest26d_fold3_best_epoch_10_final_3rd.pth",
        # # "./weights/resnest26d/resnest26d_fold4_best_epoch_21_final_2nd.pth",
        # "weights/resnest26d/resnest26d_fold4_best_epoch_6_final_3rd.pth"
        #
        # "weights/tf_efficientnet_b4_ns/tf_efficientnet_b4_ns_fold0_best_epoch_75_final_1st.pth",
        # "weights/tf_efficientnet_b4_ns/tf_efficientnet_b4_ns_fold1_best_epoch_53_final_1st.pth",
        # "weights/tf_efficientnet_b4_ns/tf_efficientnet_b4_ns_fold2_best_epoch_75_final_1st.pth",
        # "weights/tf_efficientnet_b4_ns/tf_efficientnet_b4_ns_fold3_best_epoch_84_final_1st.pth",
        # "weights/tf_efficientnet_b4_ns/tf_efficientnet_b4_ns_fold4_best_epoch_66_final_1st.pth"
        "weights/tf_efficientnet_b4_ns/tf_efficientnet_b4_ns_fold1_best_epoch_65_final_1st.pth",
        "weights/legacy_seresnext26_32x4d/legacy_seresnext26_32x4d_fold2_best_epoch_92_final_1st.pth",
        "weights/legacy_seresnext26_32x4d/legacy_seresnext26_32x4d_fold2_best_epoch_28_final_2nd.pth"
        
    ]
    
    model_index = 4
    ckpt_index = -1
    params = {
        "visualize": False,
        "fold": [2],
        "distill_soft_label":False,
        "train_external": True,
        "train_clean_only": False,
        "test_external": False,
        "load_pretrained": True,
        "fp16": False,
        "resume": False,
        "image_size": 512,
        "num_classes": 5,
        "model": models_name[model_index],
        "device": "cuda",
        "lr": 1e-4,
        "lr_min":1e-7,
        "batch_size": 8,
        "num_workers": 8,
        "epochs": 30,
        "gradient_accumulation_steps": 1,
        "drop_block": 0.2,
        "drop_rate": 0.2,
        "mix_up": True,
        "cutmix":True,
        "fmix":True,
        "smooth_label": 0.1,
        "rand_aug": False,
        "local_rank":0,
        "distributed": False,
        "hard_negative_sample": False,
        "tta": True,
        "train_phase":True,
        "balance_data":False,
        "kfold_pred":False
    }
    scaler = GradScaler()   

        
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(height=params["image_size"], width=params["image_size"], p=1),
            A.OneOf([
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),], p=1.
            ),
    #         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.IAAAffine(rotate=0.2, shear=0.2,p=0.5),
            A.CoarseDropout(max_holes=20, max_height=int(params["image_size"]/15), max_width=int(params["image_size"]/15), p=0.5),
    #         A.IAAAdditiveGaussianNoise(p=1.),
            A.MedianBlur(p=0.5),
            A.Equalize(p=0.2),
            A.GridDistortion(p=0.2),
    #         A.RandomGridShuffle(grid=(100, 100), p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.CenterCrop(height=params["image_size"], width=params["image_size"], p=1),
            A.Resize(params["image_size"],params["image_size"]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    if params["cutmix"]:
        mixup_fn = Mixup(mixup_alpha=1., cutmix_alpha=1., label_smoothing=params["smooth_label"], num_classes=params["num_classes"])
    else:
        mixup_fn = Mixup(mixup_alpha=1., label_smoothing=params["smooth_label"], num_classes=params["num_classes"])

        
    val_criterion = nn.CrossEntropyLoss().to(params["device"])
    criterion_fmix = nn.CrossEntropyLoss().to(params["device"])
    criterion = LabelSmoothingCrossEntropy().to(params["device"])
    if params["mix_up"]:
        criterion = SoftTargetCrossEntropy().to(params["device"])
    asymetric_criterion = AsymmetricLossSingleLabel().to(params["device"])
    symetric_criterion = SCELoss(smooth_label=params["smooth_label"]).to(params["device"])
        
     # model = getattr(models, params["model"])(pretrained=False, num_classes=5)
    folds = train.copy()
    Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['label'])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    
    for i, fold_idx in enumerate(params["fold"]):
        print(f"Train Fold: {fold_idx}")
        # dataset train/test split
        fold = fold_idx
        train_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

        train_folds = folds.loc[train_idx].reset_index(drop=True)
        val_folds = folds.loc[val_idx].reset_index(drop=True)
        
        for idx,label in enumerate(train_external['label']):
            train_external.loc[idx,'fold'] = fold
        train_external['fold'] = train_external['fold'].astype(int)
        
        if params["train_external"]:
            train_folds = merge_data(train_folds, train_external)
            params["distill_soft_label"] = False
            
        if params["train_clean_only"]:
            train_folds = train_external
            params["distill_soft_label"] = False
            
        if params["test_external"]:
            train_folds = merge_data(train_folds, test_external_pseudo)
            params["distill_soft_label"] = False
            
        if params["balance_data"]:
            train_folds = balance_data(train_folds, mode="undersampling")    
            val_folds = balance_data(val_folds, mode="undersampling", val=True)
            params["distill_soft_label"] = False
        
        # for distillation using soft label    
        if params["distill_soft_label"]:
            soft_target_list = set(params["fold"]) - set([fold_idx])
            for k, f in enumerate(soft_target_list):
                if k == 0:
                    soft_target_distill = pd.read_csv(f'./error_analysis/val_{params["model"]}_{f}_pred.csv')
                else:
                    soft_target_distill = merge_data(soft_target_distill,  pd.read_csv(f'./error_analysis/val_{params["model"]}_{f}_pred.csv'))
            soft_target_distill = soft_target_distill.reset_index(drop=True)
            soft_target_distill = soft_target_distill.set_index('image_id').sort_index().values
            train_dataset = TrainDataset(train_folds, root, transform=train_transform, soft_df=soft_target_distill)
        else:
            train_dataset = TrainDataset(train_folds, root, transform=train_transform)            
        val_dataset = TrainDataset(val_folds, root, transform=val_transform)

        if params["hard_negative_sample"]:
            train_loader = DataLoader(
                train_dataset, batch_size=1, shuffle=True, num_workers=params["num_workers"], pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"], pin_memory=True,
            )
        val_loader = DataLoader(
            val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"], pin_memory=True,
        )
    
        # model declaration
        if "efficientnet" in params["model"]:
            model = timm.create_model(
                params["model"],
                pretrained=True,
                num_classes=params["num_classes"], 
                drop_rate=params["drop_rate"], 
                drop_path_rate=0.3)     
        else:
            model = timm.create_model(
                params["model"],
                pretrained=True,
                num_classes=params["num_classes"])
                #drop_block_rate=params["drop_block"])    
    
        model = model.to(params["device"])
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=params["lr_min"], last_epoch=-1)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=params["lr_min"], last_epoch=-1)

        if params["fp16"]:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

        if params["distributed"]:
            assert ValueError("No need to implement in a single machine")
        else:
            model = torch.nn.DataParallel(model)    
        if params["load_pretrained"]:
            state_dict = torch.load(WEIGHTS[ckpt_index])
            print("Load pretrained model: ",state_dict["preds"])
            model.load_state_dict(state_dict["model"])
            if params["fp16"]:
                optimizer.load_state_dict(state_dict['optimizer'])
                amp.load_state_dict(state_dict['amp'])

            best_acc = state_dict["preds"]
            # Hard negative mining based on train data and pretrained model on that data
            if params["hard_negative_sample"]:
                update_train_data = update_hard_sample(train_loader, model, val_criterion, thres=0.2)
                update_train_folds = pd.DataFrame(data=update_train_data)
                update_train_folds = pd.concat(5*[update_train_folds])
                #check the update distribution when filter data
                print("Class distribution for the new data")
                visualize_class_dis(update_train_folds, params["fold"])

                #update the training set
                update_train_dataset = TrainDataset(update_train_folds, transform=train_transform)
                update_train_loader = DataLoader(
                    update_train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"], pin_memory=True,
                )                
        else:
            best_acc = 0.83   
        
        # trainning process    
        for epoch in range(1, params["epochs"] + 1):
            train_epoch(train_loader, model, criterion, optimizer, epoch, params)
            best_acc = validate(val_loader, model, criterion, optimizer ,epoch, params, fold, best_acc)
        
        del model
            