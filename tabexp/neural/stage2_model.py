"""Training, validation, prediction and self-contained model bundles."""
import copy
from dataclasses import dataclass
from itertools import count
import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .stage2_contract import validate_plan, plan_hash
from .stage2_data import Stage2Executor
from .stage2_network import Stage2Network


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Arrays(Dataset):
    def __init__(self, arrays, labels=None):
        self.arrays = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in arrays.items()}
        self.labels = torch.tensor(np.array(labels,copy=True), dtype=torch.float32) if labels is not None else None
        self.size = len(next(iter(self.arrays.values())))
        if any(len(v) != self.size for v in self.arrays.values()):
            raise ValueError("Misaligned input arrays")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        record = {k: v[index] for k, v in self.arrays.items()}
        return (record, self.labels[index]) if self.labels is not None else record


def array_loader(arrays,labels=None,batch_size=256,shuffle=False,generator=None):
    if hasattr(arrays,'make_loader'):
        return arrays.make_loader(labels,batch_size,shuffle,generator)
    return DataLoader(Arrays(arrays,labels),batch_size=batch_size,shuffle=shuffle,generator=generator,num_workers=0)


def predict(model, arrays, batch_size=256, device="cpu"):
    model.eval()
    output = []
    with torch.no_grad():
        for batch in array_loader(arrays,batch_size=batch_size):
            logits = model({k: v.to(device) for k, v in batch.items()})
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite prediction logits")
            output.append(logits.sigmoid().cpu().numpy())
    return np.concatenate(output)


def train(model, train_arrays, labels, val_arrays, val_labels, *, epochs=50,
          patience=5, batch_size=128, learning_rate=1e-3, weight_decay=1e-4,
          device="cpu", seed=42, checkpoint=None, logger=None,
          residual_penalty=0., include_initial=False, fusion_learning_rate=None, ranking_weight=0.):
    if epochs<0 or (epochs==0 and patience<1):
        raise ValueError('epochs must be >= 0; unlimited epochs require positive patience')
    if set(np.unique(labels)) != {0, 1} or set(np.unique(val_labels)) != {0, 1}:
        raise ValueError("Both train and validation need binary labels and both classes")
    seed_everything(seed)
    started = time.perf_counter()
    model.to(device)
    parameters = model.parameters()
    if fusion_learning_rate is not None:
        fusion_parameters = list(model.fusion_parameters())
        fusion_ids = {id(p) for p in fusion_parameters}
        parameters = [{'params':[p for p in model.parameters() if id(p) not in fusion_ids]},
                      {'params':fusion_parameters,'lr':fusion_learning_rate,'weight_decay':0.}]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    batches = array_loader(train_arrays,labels,batch_size,True,generator)
    best_score, best_state, best_epoch, stale = -float("inf"), None, 0, 0
    history = []
    if include_initial:
        initial = predict(model,val_arrays,batch_size,device)
        best_score = float(roc_auc_score(val_labels,initial))
        best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append({"epoch":0,"train_bce":None,"validation_auc":best_score,"seconds":0.})
        if checkpoint:
            torch.save({"state_dict":best_state,"epoch":0,"validation_auc":best_score},checkpoint)
        if logger:
            logger.info("epoch=0 frozen_base_validation_auc=%.6f",best_score)
    for epoch in (count(1) if epochs==0 else range(1, epochs + 1)):
        epoch_started = time.perf_counter()
        model.train()
        total = 0.
        objective_total = 0.
        for batch, target in batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            bce = loss_fn(logits, target)
            loss = bce
            if ranking_weight:
                positive,negative = logits[target==1],logits[target==0]
                if positive.numel() and negative.numel():
                    loss = loss + ranking_weight * torch.nn.functional.softplus(negative[:,None]-positive[None,:]).mean()
            if residual_penalty:
                loss = loss + residual_penalty * model.correction_penalty()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += bce.item() * len(target)
            objective_total += loss.item() * len(target)
        probabilities = predict(model, val_arrays, batch_size, device)
        score = float(roc_auc_score(val_labels, probabilities))
        history.append({"epoch": epoch, "train_bce": total / len(labels),
                        "train_objective": objective_total / len(labels),
                        "validation_auc": score, "seconds": time.perf_counter() - epoch_started})
        if logger:
            logger.info("epoch=%d train_bce=%.5f validation_auc=%.6f", epoch, total / len(labels), score)
        if score > best_score + 1e-6:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint:
                torch.save({"state_dict": best_state, "epoch": epoch, "validation_auc": score}, checkpoint)
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "validation_auc": best_score,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "seconds": time.perf_counter() - started, "history": history}


@dataclass(frozen=True)
class RegressionTarget:
    """Train-only affine scaling; bounds are task constraints, not fitted on test."""
    center: float
    scale: float
    lower: float | None = None
    upper: float | None = None

    @classmethod
    def fit(cls, labels, lower=None, upper=None):
        y=np.asarray(labels,dtype=np.float64)
        if y.ndim!=1 or not len(y) or not np.isfinite(y).all():
            raise ValueError('Regression labels must be a finite nonempty vector')
        if any(x is not None and not np.isfinite(x) for x in (lower,upper)) or lower is not None and upper is not None and lower>upper:
            raise ValueError('Invalid regression target bounds')
        if lower is not None and (y<lower).any() or upper is not None and (y>upper).any():
            raise ValueError('Training labels violate declared target bounds')
        center,scale=float(y.mean()),max(float(y.std()),1e-6)
        if not np.isfinite(center+scale):raise ValueError('Regression scaling overflow')
        return cls(center,scale,lower,upper)

    def encode(self,labels):
        return ((np.asarray(labels,dtype=np.float64)-self.center)/self.scale).astype(np.float32)

    def decode(self,values):
        output=np.asarray(values,dtype=np.float64)*self.scale+self.center
        return np.clip(output,self.lower,self.upper) if self.lower is not None or self.upper is not None else output


def predict_regression(model,arrays,target,batch_size=512,device='cpu'):
    model.eval();output=[]
    with torch.no_grad():
        for batch in array_loader(arrays,batch_size=batch_size):
            value=model({k:v.to(device) for k,v in batch.items()})
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite regression predictions')
            output.append(value.cpu().numpy())
    return target.decode(np.concatenate(output))


def train_regression(model,train_arrays,labels,val_arrays,val_labels,*,epochs=10,patience=2,
                     batch_size=512,learning_rate=.001,weight_decay=1e-5,device='cpu',seed=1,
                     checkpoint=None,logger=None,lower=None,upper=None):
    """Direct L1 in standardized target units; select by decoded validation MAE."""
    if epochs<0 or patience<1:raise ValueError('epochs >= 0 and positive patience required')
    target=RegressionTarget.fit(labels,lower,upper)
    val_labels=np.asarray(val_labels,dtype=np.float64)
    if val_labels.ndim!=1 or not len(val_labels) or not np.isfinite(val_labels).all():
        raise ValueError('Validation regression labels must be finite')
    seed_everything(seed);started=time.perf_counter();model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    batches=array_loader(train_arrays,target.encode(labels),batch_size,True,torch.Generator().manual_seed(seed))
    best=float('inf');state=None;best_epoch=0;stale=0;history=[]
    for epoch in (count(1) if epochs==0 else range(1,epochs+1)):
        phase=time.perf_counter();model.train();total=0.
        for batch,y in batches:
            batch={k:v.to(device) for k,v in batch.items()};y=y.to(device)
            optimizer.zero_grad(set_to_none=True);value=model(batch)
            loss=torch.nn.functional.l1_loss(value,y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite regression loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(y)
        p=predict_regression(model,val_arrays,target,batch_size,device)
        mae=float(np.abs(p-val_labels).mean())
        history.append({'epoch':epoch,'train_scaled_l1':total/len(labels),'validation_mae':mae,'seconds':time.perf_counter()-phase})
        if logger:logger.info('epoch=%d scaled_l1=%.5f validation_mae=%.6f',epoch,total/len(labels),mae)
        if mae<best-max(1e-8,abs(best)*1e-6 if np.isfinite(best) else 0):
            best=mae;best_epoch=epoch;stale=0;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if checkpoint:torch.save({'state_dict':state,'epoch':epoch,'validation_mae':best,'target_transform':vars(target),'prediction_kind':'regression'},checkpoint)
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(state)
    return {'validation_mae':best,'best_epoch':best_epoch,'target_transform':vars(target),
            'parameter_count':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-started,'history':history}


class Stage2Model:
    def __init__(self, plan, preprocessing=None):
        self.plan = validate_plan(plan)
        self.inputs = Stage2Executor(self.plan, preprocessing)
        self.network = None
        self.training_result = None
        self.target_transform = None

    def prepare(self, conn, rows, *, fit=False):
        store = self.inputs.prepare(conn, rows, fit=fit)
        if self.network is not None and self.network.model_spec != self.inputs.model_spec:
            store.close()
            raise ValueError('Saved model input schema mismatch')
        return store

    def fit(self, train_inputs, labels, val_inputs, val_labels, *, seed=1, device='cpu',
            epochs=10, patience=2, batch_size=512, learning_rate=.001, weight_decay=1e-5,
            checkpoint=None, logger=None):
        if self.network is not None:
            raise ValueError('Use a fresh model for training')
        seed_everything(seed)
        self.network = Stage2Network(**self.inputs.model_spec)
        regression = self.plan['context']['task']['kind'] == 'regression'
        trainer = train_regression if regression else train
        bounds = self.plan['context']['task'].get('target_bounds', {}) if regression else {}
        self.training_result = trainer(self.network, train_inputs, labels, val_inputs, val_labels,
            seed=seed, device=device, epochs=epochs, patience=patience, batch_size=batch_size,
            learning_rate=learning_rate, weight_decay=weight_decay, checkpoint=checkpoint, logger=logger, **bounds)
        if regression:
            self.target_transform = RegressionTarget(**self.training_result['target_transform'])
        return self.training_result

    def predict(self, inputs, *, batch_size=512, device='cpu'):
        if self.network is None:
            raise ValueError('Fit or load a model before prediction')
        self.network.to(device)
        if self.target_transform is not None:
            return predict_regression(self.network, inputs, self.target_transform, batch_size, device)
        return predict(self.network, inputs, batch_size, device)

    def save(self, path):
        if self.network is None:
            raise ValueError('Fit before saving')
        torch.save({'format': 'stage2_model_v1', 'plan': self.plan, 'plan_hash': plan_hash(self.plan),
            'preprocessing': self.inputs.state_dict(), 'model_spec': copy.deepcopy(self.network.model_spec),
            'state_dict': {k:v.detach().cpu() for k,v in self.network.state_dict().items()},
            'training_result': self.training_result,
            'target_transform': vars(self.target_transform) if self.target_transform else None}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('format') != 'stage2_model_v1' or plan_hash(saved['plan']) != saved['plan_hash']:
            raise ValueError('Invalid stage-two model bundle')
        model = cls(saved['plan'], saved['preprocessing'])
        model.network = Stage2Network(**saved['model_spec'])
        model.network.load_state_dict(saved['state_dict']); model.network.eval()
        model.training_result = saved['training_result']
        if saved['target_transform'] is not None:
            model.target_transform = RegressionTarget(**saved['target_transform'])
        return model
