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
from utils import merge_data, balance_data, TrainDataset
import h5py
import torch.nn.functional as F

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

class CNNStackModel(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(CNNStackModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channels, 256, kernel_size=(1,3), stride=1, padding=0),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3,1), stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(512, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 1024, bias=True)
        self.last_linear = nn.Linear(1024, num_classes, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.init_layer()

    def init_layer(self):
        for layer in self.conv1:
            if "Conv2d" in str(layer):
                torch.nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.last_linear(x)
        return x

# Dataset

class TestDataset(Dataset):
    def __init__(self, df, root, transform=None, valid_test=False):
        self.df = df
        self.root = root
        self.file_names = df['image_id'].values
        self.transform = transform
        self.valid_test = valid_test
        if self.valid_test:
            self.labels = df['label'].values  
        else:
            assert ValueError("Test data does not have annotation, plz check!")
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        if self.valid_test:
            file_path = f'{self.root}/train_images/{file_name}'
        else:
            file_path = f'{self.root}/test_images/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if isinstance(self.transform, list):
            outputs = {'images':[],
                       'labels':[],
                       'image_ids':[]}
            for trans in self.transform:
                augmented = trans(image=image)
                image_aug = augmented['image']
                outputs["images"].append(image_aug)
                del image_aug
            if self.valid_test:
                label = torch.tensor(self.labels[idx]).long()
                outputs['labels'] = len(self.transform)*[label]
                outputs['image_ids'].append(file_name)
            else:
                outputs['labels'] = len(self.transform)*[-1]                
            return outputs
        else:
            augmented = self.transform(image=image)
            image = augmented['image'] 
            if self.valid_test:
                label = torch.tensor(self.labels[idx]).long()
            return dict(images=image, 
                        labels=label,
                        image_ids=file_name)
        return image

class StackTrainDataset(Dataset):
    def __init__(self, df, labels, transform=None):
        self.df = df
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.df[idx]
        label =self.labels[idx]
        if self.transform:
            augmented = self.transform(image=image.cpu().numpy().transpose(1,2,0))
            image = augmented['image']
        return image, label

def calculate_accuracy(output, target):
#     return torch.true_divide((target == output).sum(dim=0), output.size(0)).item()
    if params["mix_up"]:
        output = torch.argmax(torch.softmax(output, dim=1), dim=1)
        return accuracy_score(output.cpu(), target.cpu())
        # return accuracy_score(output.cpu(), target.argmax(1).cpu())

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

def declare_pred_model(name, weight):
    if "efficientnet" in name:
        model = timm.create_model(
                name,
                pretrained=False,
                num_classes=5, 
                drop_rate=0.2, 
                drop_path_rate=0.3)
    else:
        model = timm.create_model(
                name,
                pretrained=False,
                num_classes=5,
                drop_rate=0.2)

    model = model.to(params["device"])
    model = torch.nn.DataParallel(model) 
    state_dict = torch.load(weight)
    print(f"Load pretrained model: {name} ",state_dict["preds"])
    model.load_state_dict(state_dict["model"])
    best_acc = state_dict["preds"]   
    return model.eval() 


def gmean(input_x, dim):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))     

def tta_stack_validate(loader, model, params, fold_idx, gen_prob, tta):
    model.eval()
    stream = tqdm(loader)
    preds = []
    gts = []
    image_ids = []
    probs = []            
    with torch.no_grad():
        for i, data in enumerate(stream, start=1):
            tta_output = []
            tta_prob_output = []
            if tta:
                for i, image in enumerate(data["images"]):
                    image = image.to(params["device"], non_blocking=True)
                    logit = model(image)
                    tta_output.append(logit)
                    if gen_prob:
                        tta_prob_output.append(torch.softmax(logit, dim=1))
                # tta_output = torch.stack(tta_output, dim=0).mean(dim=0)
                prob_output = torch.stack(tta_prob_output, dim=0).mean(dim=0)
                output = torch.cat(tta_output, dim=0)
                label = data["labels"][0]
                img_id = data["image_ids"][0]
            else:
                logit = model(data["images"].to(params["device"], non_blocking=True))
                prob_output = torch.softmax(logit, dim=1)
                output = logit
                label = data["labels"]
                img_id = data["image_ids"]
                
            probs.append(prob_output)
            preds.append(output)
            gts.append(label)
            image_ids.extend(img_id)
    return torch.stack(preds, dim=0), torch.cat(gts), image_ids, torch.cat(probs).cpu().numpy()


def train_epoch(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
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
        elif params["mix_up"]:
            loss = criterion(output, mtarget)
        else:
            loss = criterion(output, target)

        if params['gradient_accumulation_steps'] > 1:
            loss = loss / params['gradient_accumulation_steps']
    
        accuracy = calculate_accuracy(output, target)
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )
            
def validate(val_loader, model, criterion, optimizer, epoch, params, fold, best_acc):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
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

if __name__ == "__main__":
    
    root = os.path.join(os.environ["HOME"], "Workspace/datasets/taiyoyuden/cassava")
    train = pd.read_csv(f'{root}/train.csv')
    train_external = pd.read_csv(f'{root}/external/train_external.csv')
    test_external = pd.read_csv(f'{root}/external/test_external.csv')
    test_external_pseudo = pd.read_csv(f'{root}/external/test_external_pseudo.csv')
    test = pd.read_csv(f'{root}/sample_submission.csv')
    label_map = pd.read_json(f'{root}/label_num_to_disease_map.json', 
                            orient='index')
    
    models_name = ["resnest26d","resnest50d", "tf_efficientnet_b4_ns", "legacy_seresnext26_32x4d"]
    WEIGHTS_26 = [
            "/home/cybercore/.tuan_temp/traffic_cassava/data/resnest26/resnest26d_fold0_best_epoch_19_final_3rd.pth",
            "/home/cybercore/.tuan_temp/traffic_cassava/data/resnest26/resnest26d_fold1_best_epoch_7_final_2nd.pth",
            "/home/cybercore/.tuan_temp/traffic_cassava/data/resnest26/resnest26d_fold2_best_epoch_4_final_2nd.pth",
            "/home/cybercore/.tuan_temp/traffic_cassava/data/resnest26/resnest26d_fold3_best_epoch_10_final_3rd.pth",
            "/home/cybercore/.tuan_temp/traffic_cassava/data/resnest26/resnest26d_fold4_best_epoch_6_final_3rd.pth"]
    WEIGHTS_50 = [
            "/home/cybercore/.tuan_temp/traffic_cassava/data/resnest50/resnest50d_fold0_best_epoch_10_final_3rd.pth",
            "/home/cybercore/.tuan_temp/traffic_cassava/data/resnest50/resnest50d_fold1_best_epoch_8_final_5th_pseudo.pth",
            "/home/cybercore/.tuan_temp/traffic_cassava/data/resnest50/resnest50d_fold2_best_epoch_22_final_2nd.pth",
            "/home/cybercore/.tuan_temp/traffic_cassava/data/resnest50/resnest50d_fold3_best_epoch_1_final_3rd.pth",
            "/home/cybercore/.tuan_temp/traffic_cassava/data/resnest50/resnest50d_fold4_best_epoch_1_final_5th_pseudo.pth"]
    WEIGHTS_b4 = [
            "/home/cybercore/.tuan_temp/traffic_cassava/data/effb4/tf_efficientnet_b4_ns_fold0_best_epoch_25_final_3rd.pth",
            "/home/cybercore/.tuan_temp/traffic_cassava/data/effb4/tf_efficientnet_b4_ns_fold1_best_epoch_26_final_5th.pth",
            "/home/cybercore/.tuan_temp/traffic_cassava/data/effb4/tf_efficientnet_b4_ns_fold2_best_epoch_29_final_3rd.pth",
            "/home/cybercore/.tuan_temp/traffic_cassava/data/effb4/tf_efficientnet_b4_ns_fold3_best_epoch_14_final_2nd.pth",
            "/home/cybercore/.tuan_temp/traffic_cassava/data/effb4/tf_efficientnet_b4_ns_fold4_best_epoch_20_final_3rd.pth"]
    
    WEIGHTS_se26 = [
         "/home/cybercore/.tuan_temp/traffic_cassava/data/seresnext26/legacy_seresnext26_32x4d_fold0_best_epoch_29_final_4th.pth",
         "/home/cybercore/.tuan_temp/traffic_cassava/data/seresnext26/legacy_seresnext26_32x4d_fold1_best_epoch_21_final_5th.pth",
         "/home/cybercore/.tuan_temp/traffic_cassava/data/seresnext26/legacy_seresnext26_32x4d_fold2_best_epoch_26_final_5th.pth",
         "/home/cybercore/.tuan_temp/traffic_cassava/data/seresnext26/legacy_seresnext26_32x4d_fold3_best_epoch_4_final_5th.pth",
         "/home/cybercore/.tuan_temp/traffic_cassava/data/seresnext26/legacy_seresnext26_32x4d_fold4_best_epoch_17_final_5th.pth"]

    params = {
        "visualize": False,
        "fold": [0,1,2,3,4],
        "model": 'cnn-stack',
        "image_size": 512,
        "num_classes": 5,
        "device": "cuda",
        "fp16": False,
        "batch_size": 1,
        "lr": 1e-3,
        "lr_min": 1e-8,
        "epochs": 70,
        "num_workers": 2,
        "mix_up":True,
        "fmix":False,
        "drop_block": 0.2,
        "drop_rate": 0.2,
        "tta": False,
        "device":"cuda:0",
        "create_data": False,
        "gen_prob": True,
        "smooth_label": 0.1,
        "gradient_accumulation_steps":1
    }
    ## stack model transform 
    train_transform = A.Compose(
    [
        A.IAAAdditiveGaussianNoise(p=0.5),
        ToTensorV2(),
    ])
    val_transform = A.Compose(
    [
        ToTensorV2(),
    ])

    ## Level 1 model prediction transform
    pred_transform = A.Compose(
    [
        A.CenterCrop(height=params["image_size"], width=params["image_size"], p=1),    
        A.Resize(height=params["image_size"], width=params["image_size"], p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),   
        ToTensorV2(),
    ])
    
    transform_tta0 = A.Compose(
        [
            A.CenterCrop(height=params["image_size"], width=params["image_size"], p=1),    
            A.Resize(height=params["image_size"], width=params["image_size"], p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),   
            ToTensorV2()
        ])

    transform_tta1 = A.Compose(
        [
            A.CenterCrop(height=params["image_size"], width=params["image_size"], p=1),
            A.Resize(height=params["image_size"], width=params["image_size"], p=1),
            A.HorizontalFlip(p=1.),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),   
            ToTensorV2()
        ])
    transform_tta2 = A.Compose(
        [
            A.CenterCrop(height=params["image_size"], width=params["image_size"], p=1),
            A.Resize(height=params["image_size"], width=params["image_size"], p=1),
            A.VerticalFlip(p=1.),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    transform_tta3 = A.Compose(
        [
            A.CenterCrop(height=params["image_size"], width=params["image_size"], p=1),
            A.Resize(height=params["image_size"], width=params["image_size"], p=1),
            A.RandomRotate90(p=1.),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),     
            ToTensorV2(),
        ])
    test_transform_tta = [transform_tta0, transform_tta1, transform_tta2, transform_tta3]
    mixup_fn = Mixup(mixup_alpha=1.,label_smoothing=params["smooth_label"], num_classes=params["num_classes"])

    folds = train.copy()
    Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['label'])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    cv_acc = 0.

    # outputs_r26_h5f = h5py.File('../data/result_r26.h5', 'w')
    # outputs_r50_h5f = h5py.File('../data/result_r50.h5', 'w')
    # outputs_eb4_h5f = h5py.File('../data/result_eb4.h5', 'w')

    ## Preds all models
    r26_logit_preds = {
        "logits":[],
        "targets": [],
        "image_ids":[]
    }
    r50_logit_preds = {
        "logits":[],
        "targets": [],
        "image_ids":[]
    }
    eb4_logit_preds = {
        "logits":[],
        "targets": [],
        "image_ids":[]
    }
    se26_logit_preds = {
        "logits":[],
        "targets": [],
        "image_ids":[]
    }
    stack_probs = {
        "image_ids":[],
        "targets": [],
        "m1":[],
        "m2":[],
        "m3":[],
        "m4":[],
    }

    for fold_idx in range(5):
        fold = fold_idx
        if params["create_data"]:
            # train_idx = folds[folds['fold'] != fold].index
            val_idx = folds[folds['fold'] == fold].index
            # train_folds = folds.loc[train_idx].reset_index(drop=True)
            val_folds = folds.loc[val_idx].reset_index(drop=True)
            print(f"************** Create stacking data on valid Fold: {fold_idx} **************\n")
            if params["tta"]:
                val_pred_dataset = TestDataset(val_folds, root, transform=test_transform_tta, valid_test=True)
            else:
                val_pred_dataset = TestDataset(val_folds, root, transform=pred_transform, valid_test=True)
            val_pred_loader = DataLoader(
                val_pred_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=2, pin_memory=True,
            )
            r26_model = declare_pred_model(models_name[0], WEIGHTS_26[fold_idx])
            r50_model = declare_pred_model(models_name[1], WEIGHTS_50[fold_idx])
            eb4_model = declare_pred_model(models_name[2], WEIGHTS_b4[fold_idx])
            se26_model = declare_pred_model(models_name[3], WEIGHTS_se26[fold_idx])

            r26_outputs = tta_stack_validate(val_pred_loader, r26_model, params, fold_idx, params["gen_prob"], params["tta"])
            r26_logit_preds["logits"].append(r26_outputs[0])
            r26_logit_preds["targets"].append(r26_outputs[1])
            r26_logit_preds["image_ids"].append(r26_outputs[2])
            print(f"Check target r26 fold {fold_idx}: {accuracy_score(val_folds['label'],r26_outputs[1].cpu().numpy())}")
            #
            stack_probs["m1"].append(r26_outputs[3])
            stack_probs["image_ids"].extend(r26_outputs[2])
            stack_probs["targets"].append(r26_outputs[1])
            if params["tta"]:
                print(f"TTA ACC r26 fold {fold_idx}: {accuracy_score(val_folds['label'],r26_outputs[3].argmax(1))}")
            else:
                print(f"ACC r26 fold {fold_idx}: {accuracy_score(val_folds['label'],r26_outputs[3].argmax(1))}")

            r50_outputs = tta_stack_validate(val_pred_loader, r50_model, params, fold_idx, params["gen_prob"], params["tta"])
            r50_logit_preds["logits"].append(r50_outputs[0])
            r50_logit_preds["targets"].append(r50_outputs[1])
            r50_logit_preds["image_ids"].append(r50_outputs[2])
            print(f"Check ACC r50 fold {fold_idx}: {accuracy_score(val_folds['label'],r50_outputs[3].cpu().numpy())}")
            #
            stack_probs["m2"].append(r50_outputs[3])

            eb4_outputs = tta_stack_validate(val_pred_loader, eb4_model, params, fold_idx, params["gen_prob"], params["tta"])
            eb4_logit_preds["logits"].append(eb4_outputs[0])
            eb4_logit_preds["targets"].append(eb4_outputs[1])
            eb4_logit_preds["image_ids"].append(eb4_outputs[2])
            print(f"Check ACC eb4 fold {fold_idx}: {accuracy_score(val_folds['label'],eb4_outputs[3].cpu().numpy())}")
            #
            stack_probs["m3"].append(eb4_outputs[3])

            se26_outputs = tta_stack_validate(val_pred_loader, se26_model, params, fold_idx, params["gen_prob"], params["tta"])
            se26_logit_preds["logits"].append(se26_outputs[0])
            se26_logit_preds["targets"].append(se26_outputs[1])
            se26_logit_preds["image_ids"].append(se26_outputs[2])
            print(f"Check ACC se26 fold {fold_idx}: {accuracy_score(val_folds['label'],se26_outputs[3].cpu().numpy())}")
            #
            stack_probs["m4"].append(se26_outputs[3])

        else:
            r26_data = torch.load(f'results/result_r26_5folds.pth')
            r50_data = torch.load(f'results/result_r50_5folds.pth')  
            eb4_data = torch.load(f'results/result_eb4_5folds.pth')  
            se26_data =torch.load(f'results/result_se26_5folds.pth')  

            print(f"************** Start Training Stacking model Fold: {fold_idx} **************\n")
            stack_model = CNNStackModel(params["num_classes"], len(models_name))
            stack_model = stack_model.to(params["device"])
            stack_model = torch.nn.DataParallel(stack_model) 

            train_list = set(list([0,1,2,3,4])) - set([fold_idx])
            train_data = torch.stack([torch.cat([r26_data["preds"]["logits"][f] for f in train_list]),
                                     torch.cat([r50_data["preds"]["logits"][f] for f in train_list]),
                                     torch.cat([eb4_data["preds"]["logits"][f] for f in train_list]),
                                     torch.cat([se26_data["preds"]["logits"][f] for f in train_list]),
                                    ])
            val_data = torch.stack([r26_data["preds"]["logits"][fold_idx],
                                    r50_data["preds"]["logits"][fold_idx],
                                    eb4_data["preds"]["logits"][fold_idx],
                                    se26_data["preds"]["logits"][fold_idx]
                                    ])
            train_data = train_data.transpose(1,0)
            val_data = val_data.transpose(1,0)
            # since the target is the same, just take the infomation of the first model
            train_labels = torch.cat([r26_data["preds"]["targets"][f] for f in train_list])
            val_labels = torch.cat([r26_data["preds"]["targets"][fold_idx]])

            optimizer = torch.optim.AdamW(stack_model.parameters(), lr=params["lr"])
            # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=params["lr_min"], last_epoch=-1)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=params["lr_min"], last_epoch=-1) 
            train_dataset = StackTrainDataset(train_data, train_labels, transform=train_transform)
            val_dataset = StackTrainDataset(val_data, val_labels, transform=val_transform)

            train_loader = DataLoader(
                train_dataset, batch_size=128, shuffle=True, pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=128, shuffle=False, pin_memory=True,
            )
            criterion = LabelSmoothingCrossEntropy().to(params["device"])
            val_criterion = nn.CrossEntropyLoss().to(params["device"])
            if params["mix_up"]:
                criterion = SoftTargetCrossEntropy().to(params["device"])
            best_acc = 0.88
            for epoch in range(1, params["epochs"] + 1):
                train_epoch(train_loader, stack_model, criterion, optimizer, epoch, params)
                best_acc = validate(val_loader, stack_model, val_criterion, optimizer ,epoch, params, fold_idx, best_acc)

    if params["create_data"]:
        # outputs_r26_h5f.create_dataset(f'r26{}', data=r26_logit_preds) 
        # outputs_r50_h5f.create_dataset(f'r50{}', data=r50_logit_preds) 
        # outputs_eb4_h5f.create_dataset(f'eb4{}', data=eb4_logit_preds) 
        torch.save({'preds': r26_logit_preds}, f'results_notta/result_r26_5folds.pth') 
        torch.save({'preds': r50_logit_preds}, f'results_notta/result_r50_5folds.pth')  
        torch.save({'preds': eb4_logit_preds}, f'results_notta/result_eb4_5folds.pth')
        torch.save({'preds': se26_logit_preds}, f'results_notta/result_se26_5folds.pth')
        stack_prob_out = pd.DataFrame()
        stack_prob_out['image_id'] = np.array(stack_probs["image_ids"])
        stack_prob_out['label'] = np.array(torch.cat(stack_probs["targets"]), dtype=int)

        stack_prob_out['m1_0'] = np.concatenate(stack_probs["m1"])[:, 0]
        stack_prob_out['m1_1'] = np.concatenate(stack_probs["m1"])[:, 1]
        stack_prob_out['m1_2'] = np.concatenate(stack_probs["m1"])[:, 2]
        stack_prob_out['m1_3'] = np.concatenate(stack_probs["m1"])[:, 3]
        stack_prob_out['m1_4'] = np.concatenate(stack_probs["m1"])[:, 4]

        stack_prob_out['m2_0'] = np.concatenate(stack_probs["m2"])[:, 0]
        stack_prob_out['m2_1'] = np.concatenate(stack_probs["m2"])[:, 1]
        stack_prob_out['m2_2'] = np.concatenate(stack_probs["m2"])[:, 2]
        stack_prob_out['m2_3'] = np.concatenate(stack_probs["m2"])[:, 3]
        stack_prob_out['m2_4'] = np.concatenate(stack_probs["m2"])[:, 4]
        
        stack_prob_out['m3_0'] = np.concatenate(stack_probs["m3"])[:, 0]
        stack_prob_out['m3_1'] = np.concatenate(stack_probs["m3"])[:, 1]
        stack_prob_out['m3_2'] = np.concatenate(stack_probs["m3"])[:, 2]
        stack_prob_out['m3_3'] = np.concatenate(stack_probs["m3"])[:, 3]
        stack_prob_out['m3_4'] = np.concatenate(stack_probs["m3"])[:, 4]

        stack_prob_out['m4_0'] = np.concatenate(stack_probs["m4"])[:, 0]
        stack_prob_out['m4_1'] = np.concatenate(stack_probs["m4"])[:, 1]
        stack_prob_out['m4_2'] = np.concatenate(stack_probs["m4"])[:, 2]
        stack_prob_out['m4_3'] = np.concatenate(stack_probs["m4"])[:, 3]
        stack_prob_out['m4_4'] = np.concatenate(stack_probs["m4"])[:, 4]

        stack_prob_out.to_csv(f'train_stack_{len(models_name)}_models.csv', index=False)

    # opt_acc, opt_weight = optimize_weight(gts, preds, stype=args.stype)