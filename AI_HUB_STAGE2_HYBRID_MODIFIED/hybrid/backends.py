"""Reusable detector/tracker and OpenVINO device selection."""
import logging
import platform
from functools import partial
from time import perf_counter
import numpy as np
from .config import resolve

def openvino_device(requested):
    import openvino as ov
    devices = ov.Core().available_devices
    selected = requested.upper()
    if selected != 'AUTO' and not any(d.split('.')[0] == selected for d in devices):
        logging.warning('OpenVINO %s unavailable; using CPU', selected)
        selected = 'CPU'
    logging.info('OpenVINO devices=%s selected=%s',devices,selected)
    return selected

def compile_openvino(path,c):
    """Compile an auxiliary encoder with the same accelerator-to-CPU fallback."""
    import openvino as ov
    device=openvino_device(c['openvino_device'])
    def compile_for(target):
        properties={'INFERENCE_NUM_THREADS':c['openvino_threads']} if target=='CPU' and c.get('openvino_threads',0)>0 else {}
        return ov.Core().compile_model(str(resolve(path)),target,properties)
    try:
        return compile_for(device)
    except Exception:
        if device=='CPU':raise
        logging.warning('OpenVINO %s initialization failed for %s; retrying CPU',device,path,exc_info=True)
        return compile_for('CPU')

def hardware(c):
    import psutil
    import torch
    info = {'cpu':platform.processor(), 'ram_gb':round(psutil.virtual_memory().total/2**30,2),
            'cuda':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'backend':c['backend']}
    try:
        import openvino as ov
        info['openvino_devices'] = ov.Core().available_devices
    except ImportError:
        info['openvino_devices'] = []
    logging.info('Hardware: %s',info)
    return info

class RoadDetector:
    def __init__(self,c):
        from ultralytics import YOLO
        self.c = c
        path = resolve(c.get('detector_override') or c['yolo_model'])
        if not path.exists():
            raise FileNotFoundError(f'Required Hybrid detector artifact is missing: {path}')
        # The pinned Ultralytics backend identifies OpenVINO by directory suffix.
        if c['backend']=='openvino':
            if path.suffix=='.xml':path=path.parent
            if not path.is_dir() or not path.name.endswith('_openvino_model'):
                raise ValueError('OpenVINO detector directory must end in _openvino_model; rename the exported directory and update detector_override')
        self.model = YOLO(str(path),task='detect')
        self.device = 'intel:'+openvino_device(c['openvino_device']).lower() if c['backend']=='openvino' else c['device']
        self.elapsed = 0.
        self.inference_ms = 0.
        self.count = 0
        dummy=np.zeros((c['input_size'],c['input_size'],3),dtype=np.uint8)
        try:
            self.model.predict(dummy,imgsz=c['input_size'],device=self.device,verbose=False)
        except Exception:
            if c['backend']!='openvino' or self.device=='intel:cpu':raise
            logging.warning('OpenVINO accelerator initialization failed; retrying CPU',exc_info=True)
            self.device='intel:cpu';self.model=YOLO(str(path),task='detect')
            self.model.predict(dummy,imgsz=c['input_size'],device=self.device,verbose=False)
        if c['backend']=='openvino' and self.device=='intel:cpu' and c.get('openvino_threads',0)>0:
            import openvino as ov
            backend=self.model.predictor.model.backend
            xml=path if path.is_file() else next(path.glob('*.xml'))
            core=ov.Core()
            backend.compile_model=partial(core.compile_model,device_name='CPU',config={
                'PERFORMANCE_HINT':'LATENCY','INFERENCE_NUM_THREADS':c['openvino_threads']})
            backend.ov_compiled_model=backend.compile_model(core.read_model(str(xml)))
            backend.input_name=backend.ov_compiled_model.input().get_any_name()
            self.model.predict(dummy,imgsz=c['input_size'],device=self.device,verbose=False)

    def reset(self):
        if getattr(self.model,'predictor',None) is not None:
            for tracker in getattr(self.model.predictor,'trackers',[]):
                tracker.reset()
        self.elapsed, self.count = 0., 0
        self.inference_ms = 0.

    def detect(self,frame):
        start = perf_counter()
        result = self.model.track(frame,persist=True,tracker=self.c['tracker_type'],
                                  conf=self.c['yolo_confidence'],iou=self.c['nms_iou'],
                                  imgsz=self.c['input_size'],device=self.device,verbose=False)[0]
        self.elapsed += perf_counter()-start
        self.inference_ms += float(result.speed.get('inference',0.))
        self.count += 1
        tracks = []
        if result.boxes is not None:
            for b in result.boxes:
                name = result.names[int(b.cls.item())]
                if name not in {'car','truck','bus','motorcycle','bicycle','person','vehicle'}:
                    continue
                tracks.append({'class_name':name,'confidence':float(b.conf.item()),
                               'box':b.xyxy[0].cpu().tolist(),
                               'track_id':int(b.id.item()) if b.id is not None else None})
        return tracks
