"""Small orchestration layer; decoding, models, motion and decisions stay independent."""
from dataclasses import asdict
from time import perf_counter
from collections import deque
import numpy as np
from .video import VideoReader
from .motion import MotionFeatures
from .temporal import TemporalPredictor,windows
from .decision import DecisionEngine

class FeatureExtractor:
    def __init__(self,c,lazy_semantic=False):
        from .backends import RoadDetector
        from .semantic import SceneEncoder
        self.c = c
        self.detector = RoadDetector(c) if c['ablation']!='clip' else None
        self.semantic = SceneEncoder(c) if not lazy_semantic and c['ablation'] not in ('motion','yolo_temporal') else None

    def stream(self,path,semantic_cache=None):
        if self.detector is not None:self.detector.reset()
        motion = MotionFeatures(self.c)
        self.reader = VideoReader(path,self.c['target_process_fps'])
        if semantic_cache is None and self.semantic is None and self.c['ablation'] not in ('motion','yolo_temporal'):
            from .semantic import SceneEncoder
            self.semantic=SceneEncoder(self.c)
        semantic = np.zeros(522,dtype=np.float32)
        for index,(t,frame) in enumerate(self.reader):
            tracks = self.detector.detect(frame) if self.detector is not None else []
            m = motion.update(tracks,t,frame) if self.detector is not None else np.zeros(16,np.float32)
            if semantic_cache is not None:
                if index>=len(semantic_cache[0]) or abs(t-semantic_cache[0][index])>1e-6:
                    raise ValueError('Semantic cache sampling timestamps differ')
                semantic=semantic_cache[1][index]
            elif self.semantic is not None and index%self.c['clip_frame_interval']==0:
                semantic = self.semantic.encode(frame)
            yield t,np.concatenate([m,semantic]),frame,tracks

    def extract(self,path,semantic_cache=None):
        start = perf_counter()
        times,features = [],[]
        for t,f,_,_ in self.stream(path,semantic_cache):
            times.append(t); features.append(f)
        return np.asarray(times),np.asarray(features),{**asdict(self.reader.info),
            'processing_seconds':perf_counter()-start,'detector_seconds':self.detector.elapsed if self.detector else 0.}

class HybridPipeline:
    def __init__(self,c):
        self.c = c
        self.predictor = TemporalPredictor(c)
        self.extractor = FeatureExtractor(c)

    def run(self,path,save_video=None):
        import cv2
        import psutil
        process=psutil.Process()
        peak_rss=process.memory_info().rss
        start = perf_counter()
        engine = DecisionEngine(self.c)
        history = deque()
        trace = []
        writer = None
        next_step,probability = self.c['window_seconds'],0.
        encoding_seconds = 0.
        last_sample_wall=perf_counter()
        timing_interruptions=[]
        detection_count=vehicle_detection_count=0
        try:
            for t,f,frame,tracks in self.extractor.stream(path):
                now=perf_counter()
                if now-last_sample_wall>self.c.get('max_sample_wall_seconds',30.):
                    timing_interruptions.append({'video_seconds':t,'wall_gap_seconds':now-last_sample_wall})
                last_sample_wall=now
                detection_count+=len(tracks)
                vehicle_detection_count+=sum(d['class_name']!='person' for d in tracks)
                history.append((t,f))
                while len(history)>2 and history[1][0] < t-self.c['window_seconds']:
                    history.popleft()
                if t+1e-8 >= next_step:
                    peak_rss=max(peak_rss,process.memory_info().rss)
                    tt = np.array([a for a,_ in history]); ff = np.array([a for _,a in history])
                    grid = np.linspace(t-self.c['window_seconds'],t,max(2,round(self.c['window_seconds']*self.c['target_process_fps'])))
                    xx = ff[np.maximum(0,np.searchsorted(tt,grid,side='right')-1)][None]
                    probability = float(self.predictor.predict(xx)[0])
                    engine.update(t,probability,float(f[15]))
                    trace.append({'seconds':t,'probability':probability,'impact':float(f[15]),
                                  'accident_state':engine.active is not None,
                                  'motion_features':f[:16].tolist(),'semantic_similarities':f[-10:].tolist(),
                                  'tracked_objects':tracks})
                    next_step = t+self.c['stride_seconds']
                if save_video:
                    enc_start = perf_counter()
                    if writer is None:
                        writer = cv2.VideoWriter(str(save_video),cv2.VideoWriter_fourcc(*'mp4v'),self.c['target_process_fps'],(frame.shape[1],frame.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError('Cannot open annotated output')
                    for d in tracks:
                        x1,y1,x2,y2 = map(int,d['box'])
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(frame,str(d['track_id']),(x1,y1),0,.6,(0,255,0),2)
                    cv2.putText(frame,f'{t:.3f}s p={probability:.3f} '+('ACCIDENT' if engine.active else 'NORMAL'),(15,30),0,.7,(0,0,255),2)
                    if engine.active:
                        cv2.putText(frame,f'Onset {engine.active.onset_seconds:.3f}s',(15,60),0,.7,(0,0,255),2)
                    writer.write(frame)
                    encoding_seconds += perf_counter()-enc_start
        finally:
            if writer:
                writer.release()
        events = engine.finish()
        if not trace:
            raise ValueError('Video shorter than configured temporal window; insufficient evidence for classification')
        info = self.extractor.reader.info
        elapsed = perf_counter()-start
        return {'video':str(path),'status':'OK','prediction':'ACCIDENT' if events else 'NORMAL',
                'confidence':max((v['probability'] for v in trace),default=0.),'events':events,'trace':trace,
                'detection_count':detection_count,'vehicle_detection_count':vehicle_detection_count,
                'timing_valid':not timing_interruptions,'timing_interruptions':timing_interruptions,
                'processing_seconds':elapsed,'core_seconds':elapsed-encoding_seconds,'encoding_seconds':encoding_seconds,
                'detector_ms_per_frame':1000*self.extractor.detector.elapsed/max(info.sampled,1) if self.extractor.detector else 0.,
                'detector_inference_ms_per_frame':self.extractor.detector.inference_ms/max(info.sampled,1) if self.extractor.detector else 0.,
                'peak_rss_mb':peak_rss/2**20,
                'pipeline_ms_per_frame':1000*elapsed/max(info.sampled,1),
                'fps':info.sampled/max(elapsed,1e-9),'rtf':elapsed/max(info.duration,1e-9),
                'speedup':info.duration/max(elapsed,1e-9),
                **{k:v for k,v in asdict(info).items() if k!='fps'},'source_fps':info.fps,
                'insufficient_temporal_context':not bool(trace)}
