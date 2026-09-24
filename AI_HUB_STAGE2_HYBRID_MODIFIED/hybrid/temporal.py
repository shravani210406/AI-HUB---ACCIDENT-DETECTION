"""Small trainable temporal encoder and learned motion/semantic fusion."""
import numpy as np
import torch
import hashlib
from torch import nn
from .config import resolve, feature_signature

class TemporalFusion(nn.Module):
    def __init__(self,input_dim,hidden=32,kind='gru',dropout=.3):
        super().__init__()
        self.kind = kind
        self.projection = nn.Sequential(nn.Linear(input_dim,hidden),nn.ReLU(),nn.Dropout(dropout))
        if kind == 'gru':
            self.sequence = nn.GRU(hidden,hidden,batch_first=True)
        elif kind == 'tcn':
            self.sequence = nn.Sequential(nn.Conv1d(hidden,hidden,3,padding=2,dilation=1),nn.ReLU(),
                                          nn.Dropout(dropout),nn.Conv1d(hidden,hidden,3,padding=4,dilation=2),nn.ReLU())
        else:
            raise ValueError('temporal_type must be gru or tcn')
        self.head = nn.Sequential(nn.Linear(hidden+input_dim,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))

    def forward(self,x):
        z = self.projection(x)
        if self.kind == 'gru':
            z,_ = self.sequence(z)
            h = z[:,-1]
        else:
            h = self.sequence(z.transpose(1,2))[:,:,:x.shape[1]][:,:,-1]
        return self.head(torch.cat([h,x[:,-1]],dim=-1)).squeeze(-1)

def mask_features(x,ablation):
    y = x.copy()
    if ablation in ('motion','yolo_temporal'):
        y[...,16:] = 0
    elif ablation == 'clip':
        y[...,:16] = 0
    elif ablation not in ('full','unoptimized'):
        raise ValueError('Unknown ablation')
    return y

def windows(times,features,c):
    """Resample causal windows on a time grid. Never interpolate from future frames."""
    if len(times)<2:
        return np.empty((0,0,features.shape[-1]),np.float32),np.array([])
    evaluated=[]
    next_step=times[0]+c['window_seconds']
    for t in times:
        if t+1e-8>=next_step:
            evaluated.append(t)
            next_step=t+c['stride_seconds']
    end=np.asarray(evaluated,dtype=np.float64)
    length = max(2,round(c['window_seconds']*c['target_process_fps']))
    offsets = np.linspace(-c['window_seconds'],0,length)
    indices = np.searchsorted(times,end[:,None]+offsets,side='right')-1
    return features[np.maximum(indices,0)],end

class TemporalPredictor:
    def __init__(self,c):
        torch.set_num_threads(c.get('torch_threads',4))
        path = resolve(c['temporal_checkpoint'])
        if not path.exists():
            raise FileNotFoundError(f'{path}: train the temporal model before inference; no untrained fallback is used')
        self.artifact = torch.load(path,map_location='cpu',weights_only=False)
        if self.artifact['feature_signature'] != feature_signature(c):
            raise ValueError('Feature configuration differs from training checkpoint')
        if self.artifact['ablation'] != c['ablation']:
            raise ValueError('Train a separate checkpoint for each ablation')
        for key in ('window_seconds','stride_seconds'):
            if self.artifact['config'][key]!=c[key]:
                raise ValueError(f'{key} differs from the trained temporal protocol')
        self.c = c
        self.model = TemporalFusion(**self.artifact['model_args'])
        self.model.load_state_dict(self.artifact['state_dict']); self.model.eval()
        self.compiled = None
        if c.get('temporal_openvino'):
            import json
            def sha256(artifact_path):
                digest = hashlib.sha256()
                with open(artifact_path, 'rb') as artifact:
                    for block in iter(lambda: artifact.read(1024 * 1024), b''):
                        digest.update(block)
                return digest.hexdigest()
            metadata=resolve(c['temporal_openvino']).with_suffix('.metadata.json')
            if not metadata.exists() or json.loads(metadata.read_text())['checkpoint_sha256']!=sha256(path):
                raise ValueError('Temporal OpenVINO export does not match the supplied Hybrid checkpoint')
            from .backends import compile_openvino
            self.compiled = compile_openvino(c['temporal_openvino'],c)

    def predict(self,x):
        if not len(x):
            return np.array([])
        motion_support = np.sum(np.max(x[:,:,[2,3,8,10]],axis=-1) >= self.c.get('minimum_motion_evidence',.002),axis=1) >= self.c.get('minimum_dynamic_steps',2)
        x = mask_features(x,self.c['ablation'])
        x = ((x-self.artifact['mean'])/self.artifact['std']).astype(np.float32)
        if self.c['ablation']=='motion':
            x = np.repeat(x[:,-1:,:],x.shape[1],axis=1)
        with torch.inference_mode():
            if self.compiled is None:
                logits = self.model(torch.from_numpy(x)).numpy()
            else:
                logits = np.concatenate([self.compiled([row[None]])[self.compiled.output(0)].reshape(-1) for row in x])
        probabilities = 1/(1+np.exp(-np.clip(logits,-50,50)))
        if self.c['ablation']!='clip':
            probabilities = np.where(motion_support,probabilities,0.)
        return probabilities
