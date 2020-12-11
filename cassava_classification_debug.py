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
cudnn.benchmark = True
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import timm
from timm.loss import JsdCrossEntropy
from utils import Mixup, RandAugment, AsymmetricLossSingleLabel, SCELoss, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from PIL import Image
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


def merge_data(df1, df2):
    merge_df = pd.concat([df1, df2], axis=0) #,how ='outer', on ='image_id')
    return merge_df


def balance_data(df, mode="undersampling", val=False):
    class_0 = df[df.label==0]
    class_1 = df[df.label==1]
    class_2 = df[df.label==2]
    class_3 = df[df.label==3]
    class_4 = df[df.label==4]
    if mode == "undersampling":
        # upsample minority
        class_3_downsampled = resample(class_3,
                                  replace=True, # sample with replacement
                                  n_samples=int(len(class_3)*2/3), # match number in majority class
                                  random_state=27) # reproducible results
        if val:
            return  pd.concat([class_0, class_1, class_2, class_3_downsampled, class_4]) 
    
        class_1_downsampled = resample(class_1,
                          replace=True, # sample with replacement
                          n_samples=int(len(class_1)*0.7), # match number in majority class
                          random_state=27) # reproducible results
        class_4_upsampled = resample(class_4,
                          replace=True, # sample with replacement
                          n_samples=int(len(class_4)*1.3), # match number in majority class
                          random_state=27) # reproducible results
        return pd.concat([class_0, class_1_downsampled, class_2, class_3_downsampled, class_4_upsampled]) 
    else:
        class_0_upsampled = resample(class_0,
                          replace=True, # sample with replacement
                          n_samples=int(len(class_3)/4), # match number in majority class
                          random_state=27) # reproducible results
        class_1_upsampled = resample(class_1,
                          replace=True, # sample with replacement
                          n_samples=int(len(class_3)/3), # match number in majority class
                          random_state=27) # reproducible results
        class_2_upsampled = resample(class_2,
                          replace=True, # sample with replacement
                          n_samples=int(len(class_3)/3), # match number in majority class
                          random_state=27) # reproducible results
        class_4_upsampled = resample(class_4,
                          replace=True, # sample with replacement
                          n_samples=int(len(class_3)/3), # match number in majority class
                          random_state=27) # reproducible results
        return pd.concat([class_0_upsampled, class_1_upsampled, class_2_upsampled, class_3, class_4_upsampled]) 


# Dataset
class TrainDataset(Dataset):
    def __init__(self, df, transform=None, mosaic_mix = False):
        self.df = df
        self.file_names = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform
        self.mosaic_mix = mosaic_mix
        self.rand_aug_fn = RandAugment()
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{root}/train_images/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = torch.tensor(self.labels[idx]).long()
        if params["rand_aug"]:
            image = np.array(self.rand_aug_fn(Image.fromarray(image)))
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label, file_name
    

class TestDataset(Dataset):
    def __init__(self, df, transform=None, valid_test=False, fcrops=False):
        self.df = df
        self.file_names = df['image_id'].values
        self.transform = transform
        self.valid_test = valid_test
        self.fcrops = fcrops
        if self.valid_test:
            self.labels = df['label'].values  
        else:
            assert ValueError("Test data does not have annotation, plz check!")
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        if self.valid_test:
            file_path = f'{root}/train_images/{file_name}'
            #file_path = f'{root}/external/extraimages/{file_name}'
        else:
            file_path = f'{root}/test_images/{file_name}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if isinstance(self.transform, list):
            outputs = {'images':[],
                       'labels':[],
                       'image_ids':[]}
            if self.fcrops:
                for trans in self.transform:
                    image_aug = transforms.ToPILImage()(image)
                    image_aug = trans(image_aug)
                    outputs["images"].append(image_aug)
                    del image_aug
            else:
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
        return image

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

def train_epoch(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    if params["hard_negative_sample"]:
        stream = tqdm(update_train_loader)
    else:
        stream = tqdm(train_loader)
    for i, (images, target, _) in enumerate(stream, start=1):
        images = images.to(params["device"]) #, non_blocking=True)
        target = target.to(params["device"]) #, non_blocking=True) #.view(-1,params['batch_size'])
        if params["mix_up"]:
            images , target = mixup_fn(images, target)
        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = criterion(output, target)
        # loss1 = symetric_criterion(output, target)
        # loss2 = asymetric_criterion(output, target)
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


def validate(val_loader, model, criterion, epoch, params, fold, best_acc):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target, _) in enumerate(stream, start=1):
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
            if not os.path.exists(params["model"]):
                os.makedirs(params["model"])
            torch.save({'model': model.state_dict(), 
                'loss': loss,
                'preds': round(metric_monitor.curr_acc,4)},
                 f'weights/{params["model"]}/{params["model"]}_fold{fold}_best_epoch_{epoch}.pth')

            best_acc = metric_monitor.curr_acc
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

    models_name = ["resnest26d","resnest50d","tf_efficientnet_b3_ns", "skresnet34" ,"cspresnet50", "vit_base_patch16_384"]
    weights = [
        "weights/resnest26d/resnest26d_fold0_best_epoch_28_final_1st.pth",
        "weights/resnest26d/resnest26d_fold0_best_epoch_13_final_mixup.pth",
        "weights/resnest26d/resnest26d_fold2_best_epoch_3_final_hnm.pth",
        "weights/resnest26d/resnest26d_fold4_best_epoch_29_1st.pth",
        "weights/resnest26d/resnest26d_fold4_best_epoch_26_mix.pth",
        "weights/resnest26d/resnest26d_fold4_best_epoch_12_cutmix.pth",
        "weights/resnest26d/resnest26d_fold4_best_epoch_3_external.pth",
        "weights/resnest26d/resnest26d_fold4_best_epoch_21_final_512.pth",
        "weights/tf_efficientnet_b3_ns/tf_efficientnet_b3_ns_fold1_best_epoch_19_external.pth",
        "weights/tf_efficientnet_b3_ns/tf_efficientnet_b3_ns_fold1_best_epoch_26_512.pth",
        "weights/tf_efficientnet_b3_ns/tf_efficientnet_b3_ns_fold1_best_epoch_1_final_512.pth",
        "weights/resnest50d/resnest50d_fold1_best_epoch_95_final_1st.pth",
    #     "weights/resnest50d/resnest50d_fold1_best_epoch_9_final_512.pth"
        "weights/resnest50d/resnest50d_fold1_best_epoch_38_clean_1st.pth"
            
    ]
    model_index = 0
    ckpt_index = 1
    fold_ckpt_index = [11,12]
    fold_ckpt_weight = [1,1]

    params = {
        "visualize": False,
        "fold": [0,1,2,3,4],
        "train_external": False,
        "train_clean_only": False,
        "test_external": False,
        "load_pretrained": False,
        "resume": False,
        "image_size": 320,
        "num_classes": 5,
        "model": models_name[model_index],
        "device": "cuda",
        "lr": 1e-3,
        "lr_min":1e-7,
        "batch_size": 2,
        "num_workers": 1,
        "epochs": 10,
        "gradient_accumulation_steps": 1,
        "drop_block": 0.1,
        "drop_rate": 0.3,
        "mix_up": False,
        "cutmix":False,
        "smooth_label": 0,
        "rand_aug": False,
        "local_rank":0,
        "distributed": False,
        "hard_negative_sample": False,
        "tta": True,
        "train_phase":True,
        "balance_data":False,
        "kfold_pred":False
    }

    if "efficientnet" in params["model"]:
        model = timm.create_model(
                params["model"],
                pretrained=False,
                num_classes=params["num_classes"], 
                drop_rate=params["drop_rate"], 
                drop_path_rate=0.3)
        
    elif "skresnet" in params["model"]:
        model = timm.create_model(
                params["model"],
                pretrained=True,
                num_classes=params["num_classes"],
                drop_block_rate=params["drop_block"],
                drop_path_rate=0.2)
    else:
        model = timm.create_model(
                params["model"],
                pretrained=True,
                num_classes=params["num_classes"],
                drop_block_rate=params["drop_block"])

    model = model.to(params["device"])
    val_criterion = nn.CrossEntropyLoss().to(params["device"])
    # criterion = nn.CrossEntropyLoss().to(params["device"])
    criterion = LabelSmoothingCrossEntropy().to(params["device"])
    if params["mix_up"]:
        criterion = SoftTargetCrossEntropy().to(params["device"])
    asymetric_criterion = AsymmetricLossSingleLabel().to(params["device"])
    symetric_criterion = SCELoss(smooth_label=params["smooth_label"]).to(params["device"])

    if params["distributed"]:
        assert ValueError("No need to implement in a single machine")
    else:
        model = torch.nn.DataParallel(model)    
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    # scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=params["lr_min"], last_epoch=-1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=params["lr_min"], last_epoch=-1)
            
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


    folds = train.copy()
    Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['label'])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    for i in params["fold"]:
        print(f"Train Fold: {i}")
        fold = params["fold"][i]
        train_idx = folds[folds['fold'] != fold].index
        val_idx = folds[folds['fold'] == fold].index

        train_folds = folds.loc[train_idx].reset_index(drop=True)
        val_folds = folds.loc[val_idx].reset_index(drop=True)

        for idx,label in enumerate(train_external['label']):
            train_external.loc[idx,'fold'] = fold
        train_external['fold'] = train_external['fold'].astype(int)
        if params["train_external"]:
            train_folds = merge_data(train_folds, train_external)
        if params["train_clean_only"]:
            train_folds = train_external
        #     train_folds = pd.read_csv(f'{root}/train_{params["fold"]}_pseudo.csv')
        #     val_folds = pd.read_csv(f'{root}/val_{params["fold"]}_pseudo.csv')    

        if params["test_external"]:
            train_folds = merge_data(train_folds, test_external_pseudo)
            
        if params["balance_data"]:
            train_folds = balance_data(train_folds, mode="undersampling")    
            val_folds = balance_data(val_folds, mode="undersampling", val=True)

        train_dataset = TrainDataset(train_folds, transform=train_transform)
        val_dataset = TrainDataset(val_folds, transform=val_transform)


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

        if params["load_pretrained"]:
            state_dict = torch.load(weights[ckpt_index])
            print("Load pretrained model: ",state_dict["preds"])
            model.load_state_dict(state_dict["model"])
            best_acc = state_dict["preds"]
            # Hard negative mining based on train data and pretrained model on that data
            if params["hard_negative_sample"]:
                update_train_data = update_hard_sample(train_loader, model, val_criterion, thres=0.2)
                update_train_folds = pd.DataFrame(data=update_train_data)
                update_train_folds = pd.concat(5*[update_train_folds])
                #update the training set
                update_train_dataset = TrainDataset(update_train_folds, transform=train_transform)
                update_train_loader = DataLoader(
                    update_train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"], pin_memory=True,
                )                
        else:
            best_acc = 0.
            
        for epoch in range(1, params["epochs"] + 1):
            train_epoch(train_loader, model, criterion, optimizer, epoch, params)
            best_acc = validate(val_loader, model, criterion, epoch, params, fold, best_acc)