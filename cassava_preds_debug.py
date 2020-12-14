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

os.environ['CUDA_VISIBLE_DEVICES'] ="1"


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
def gmean(input_x, dim):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim))

def tta_validate(loader, model, params, fold_idx):
    num_tta = len(test_transform_tta)
    pred_list = {'image_id':[],
                 'prob':[],}

    incorrect_pred_list = {'image_id':[],
                       'label':[],
                       'pred':[],
                       'prob':[]}
    correct_pred_list = {'image_id':[],
                       'label':[],
                       'pred':[],
                       'prob':[]}    
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(loader)
    count_change = 0
    with torch.no_grad():
        for i, data in enumerate(stream, start=1):
            if params["visualize"]:
                visualize_tta(data, pred)
            tta_output = []   
            for i, image in enumerate(data["images"]):
                out = torch.softmax(model(image), dim=1)
                tta_output.append(out)
            output = gmean(torch.stack(tta_output, dim=0), dim = 0)
#             output = torch.softmax(tta_output, dim=1)
            pred = output.argmax(1)

            topk_output, topk_ids = torch.topk(output, params["num_classes"])
            for i in range(len(data["labels"][0])):
                if params["distill_soft_label"]:
                    pred_list['image_id'].append(data["image_ids"][0][i])
                    pred_list['prob'].append(output[i].cpu().numpy())
                    
                if params["error_analysis"]:
                    ## adjust the output prediction
                    max_1st = topk_ids[i][0]
                    max_2nd = topk_ids[i][1]
                    if  max_1st == 0 and max_2nd == 4 and output[i][max_2nd] > 0.2:
                        pred[i] = max_2nd
                        count_change+=1
                    if max_1st == 3 and max_2nd == 2 and output[i][max_2nd] > 0.2:
                        pred[i] = max_2nd
                        count_change+=1

                    if output[i][max_1st] < 0.45 and output[i][max_2nd] > 0.25:
                        pred[i] = max_2nd
                        count_change+=1

                if data["labels"][0][i] != output.argmax(1).cpu()[i]:
                    incorrect_pred_list['image_id'].append(data["image_ids"][0][i])                        
                    incorrect_pred_list['label'].append(data["labels"][0][i].cpu().numpy())
                    incorrect_pred_list['pred'].append(output.argmax(1).cpu()[i].cpu().numpy())
                    incorrect_pred_list['prob'].append(output[i].cpu().numpy())
                else:
                    correct_pred_list['image_id'].append(data["image_ids"][0][i])                        
                    correct_pred_list['label'].append(data["labels"][0][i].cpu().numpy())
                    correct_pred_list['pred'].append(output.argmax(1).cpu()[i].cpu().numpy())
                    correct_pred_list['prob'].append(output[i].cpu().numpy())   
            accuracy = accuracy_score(pred.cpu(), data["labels"][0].cpu())
            metric_monitor.update("Accuracy", accuracy)            
            stream.set_description(
                "TTA Validation. {metric_monitor}".format(metric_monitor=metric_monitor)
            )
        if params["distill_soft_label"]:
            pred_val = pd.DataFrame(pred_list)
            pred_val.to_csv(f'./error_analysis/val_{params["model"]}_{fold_idx}_pred.csv' ,index=False)
        best_acc = metric_monitor.curr_acc
        
    # print(f"Total output change: {count_change}")
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

    models_name = ["resnest26d","resnest50d","tf_efficientnet_b3_ns"]
    WEIGHTS = [
        #"./weights/resnest50d/resnest50d_fold0_best_epoch_30_final_1st.pth",
        # "./weights/resnest50d/resnest50d_fold0_best_epoch_13_final_2nd.pth",
        # "./weights/resnest50d/resnest50d_fold0_best_epoch_10_final_3rd.pth",
        "./weights/resnest50d/resnest50d_fold0_best_epoch_16_final_4th.pth",
        # "./weights/resnest50d/resnest50d_fold1_best_epoch_95_final_1st.pth",
        "./weights/resnest50d/resnest50d_fold1_best_epoch_17_final_2nd.pth",
        # "./weights/resnest50d/resnest50d_fold2_best_epoch_50_final_1st.pth",
        "./weights/resnest50d/resnest50d_fold2_best_epoch_22_final_2nd.pth",
        # "./weights/resnest50d/resnest50d_fold3_best_epoch_2_final_2nd.pth",
        # "./weights/resnest50d/resnest50d_fold3_best_epoch_1_final_3rd.pth",
        "./weights/resnest50d/resnest50d_fold3_best_epoch_2_final_4th.pth",
        # "./weights/resnest50d/resnest50d_fold4_best_epoch_10_final_2nd.pth",
        "./weights/resnest50d/resnest50d_fold4_best_epoch_15_final_3rd.pth"
        
        # "./weights/resnest26d/resnest26d_fold0_best_epoch_4_final_2nd.pth",
        # # "./weights/resnest26d/resnest26d_fold0_best_epoch_19_final_3rd.pth",
        # "./weights/resnest26d/resnest26d_fold1_best_epoch_7_final_2nd.pth",
        # "./weights/resnest26d/resnest26d_fold2_best_epoch_4_final_2nd.pth",
        # "./weights/resnest26d/resnest26d_fold3_best_epoch_15_final_2nd.pth",
        # # "./weights/resnest26d/resnest26d_fold3_best_epoch_10_final_3rd.pth",
        # "./weights/resnest26d/resnest26d_fold4_best_epoch_21_final_2nd.pth",
        # # "./weights/resnest26d/resnest26d_fold4_best_epoch_6_final_3rd.pth",
    ]
    model_index = 1
    ckpt_index = 1

    params = {
        "visualize": False,
        "fold": [0,1,2,3,4],
        "distill_soft_label":False,
        "train_external": True,
        "test_external": False,
        "load_pretrained": True,
        "resume": False,
        "image_size": 512,
        "num_classes": 5,
        "model": models_name[model_index],
        "device": "cuda",
        "lr": 5e-5,
        "lr_min":1e-6,
        "batch_size": 16,
        "num_workers": 8,
        "epochs": 100,
        "gradient_accumulation_steps": 8,
        "drop_block": 0.2,
        "drop_rate": 0.2,
        "mix_up": True,
        "cutmix":True,
        "rand_aug": False,
        "local_rank":0,
        "distributed": False,
        "hard_negative_sample": False,
        "tta": True,
        "crops_tta":False,
        "train_phase":False,
        "balance_data":False,
        "kfold_pred":True,
        "ensemble": True,
        "error_analysis":False,
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

    if params["distributed"]:
        assert ValueError("No need to implement in a single machine")
    else:
        model = torch.nn.DataParallel(model)    
        
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

    ##### Test TTA with Five Crops
    transform_crop_tta0 = T.Compose([
                    T.FiveCrop(params["image_size"]),
                    T.Lambda(lambda crops: ([T.ToTensor()(crop) for crop in crops])),
                    T.Lambda(lambda norms: torch.stack([T.Normalize(mean=[0.5], std=[0.5])(norm) for norm in norms]))
            ]
    )
        
    transform_crop_tta1 = T.Compose([
                    T.FiveCrop(params["image_size"]),
                    T.Lambda(lambda crops: ([T.ToTensor()(crop) for crop in crops])),
                    T.Lambda(lambda flips: ([T.RandomHorizontalFlip(p=1.)(flip) for flip in flips])),
                    T.Lambda(lambda norms: torch.stack([T.Normalize(mean=[0.5], std=[0.5])(norm) for norm in norms]))
        ]
    )
    transform_crop_tta2 = T.Compose([
                    T.FiveCrop(params["image_size"]),
                    T.Lambda(lambda crops: ([T.ToTensor()(crop) for crop in crops])),
                    T.Lambda(lambda flips: ([T.RandomVerticalFlip(p=1.)(flip) for flip in flips])),
                    T.Lambda(lambda norms: torch.stack([T.Normalize(mean=[0.5], std=[0.5])(norm) for norm in norms]))
        ]
    )
    test_transform_tta = [transform_tta0, transform_tta1, transform_tta2, transform_tta3]
    test_transform_tta_crops = [transform_crop_tta0, transform_crop_tta1, transform_crop_tta2]
  
    folds = train.copy()
    Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds['label'])):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    cv_acc = 0.
    for i, fold_idx in enumerate(params["fold"]):
        print(f"Validate Fold: {fold_idx}")
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
        #     train_folds = pd.read_csv(f'{root}/train_{params["fold"]}_pseudo.csv')
        #     val_folds = pd.read_csv(f'{root}/val_{params["fold"]}_pseudo.csv')    

        if params["test_external"]:
            train_folds = merge_data(train_folds, test_external_pseudo)
            
        if params["balance_data"]:
            train_folds = balance_data(train_folds, mode="undersampling")    
            val_folds = balance_data(val_folds, mode="undersampling", val=True)

        train_dataset = TestDataset(train_folds, transform=test_transform_tta, valid_test=True)
        train_loader = DataLoader(
            train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"], pin_memory=True,
        )
        if params["tta"]:
            val_pred_dataset = TestDataset(val_folds, transform=test_transform_tta, valid_test=True)
            test_pred_dataset = TestDataset(test, transform=test_transform_tta)
        else:
            val_pred_dataset = TestDataset(val_folds, transform=val_transform, valid_test=True)
            test_pred_dataset = TestDataset(test, transform=val_transform)

        val_pred_dataset_crops = TestDataset(val_folds, transform=test_transform_tta_crops,valid_test=True, fcrops=True)

        val_pred_loader = DataLoader(
            val_pred_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=2, pin_memory=True,
        )
        val_pred_loader_crop = DataLoader(
            val_pred_dataset_crops, batch_size=params['batch_size'], shuffle=False, num_workers=2, pin_memory=True,
        )
        test_pred_loader = DataLoader(
            test_pred_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=2, pin_memory=True,
        )

        state_dict = torch.load(WEIGHTS[i])
        fold_acc = state_dict["preds"]
        print(f"Load pretrained model: {WEIGHTS[i]} with acc: {fold_acc}")
        model.load_state_dict(state_dict["model"])

        cv_acc += tta_validate(val_pred_loader, model, params, fold_idx)
    num_fold_train = len(params["fold"])            
    print(f"Done CV validation with  {num_fold_train} folds, Accuracy: {round(cv_acc/num_fold_train,4)}")