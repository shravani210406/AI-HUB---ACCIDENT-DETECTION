"""Smoothed image-space trajectories and pairwise interaction evidence."""
from dataclasses import dataclass
import numpy as np

MOTION_NAMES = ['objects','mean_confidence','max_speed','max_acceleration',
                'max_deceleration','direction_change','max_iou','min_distance',
                'max_closing','min_ttc','relative_speed','abrupt_stop',
                'post_motion_stop','camera_motion','track_fraction','impact']

def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:]-a[:2]).prod() + np.maximum(0, b[2:]-b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.0

def closing_speed(ci, cj, vi, vj):
    r = np.asarray(cj) - ci
    return float(-np.dot(np.asarray(vj)-vi, r) / max(np.linalg.norm(r), 1e-8))

@dataclass
class TrackState:
    time: float
    center: np.ndarray
    velocity: np.ndarray
    speed_peak: float

class MotionFeatures:
    def __init__(self, config):
        self.c = config
        self.history = {}
        self.previous_gray = None

    def global_motion(self, frame):
        import cv2
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))
        shift = np.zeros(2)
        if self.previous_gray is not None and self.c['camera_compensation']:
            pts = cv2.goodFeaturesToTrack(self.previous_gray, 150, .02, 8)
            if pts is not None and len(pts) >= 8:
                new, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, pts, None)
                if new is not None and status is not None:
                    old, new = pts[status.ravel()==1], new[status.ravel()==1]
                    if len(old) >= 8:
                        matrix, inliers = cv2.estimateAffinePartial2D(old, new, method=cv2.RANSAC)
                        if matrix is not None and inliers.mean() >= .5:
                            shift = matrix[:, 2] / [320, 180]
        self.previous_gray = gray
        return shift

    def update(self, tracks, t, frame):
        h, w = frame.shape[:2]
        shift = self.global_motion(frame)
        current, speeds, accelerations, decels, angles, stops = [], [], [], [], [], []
        alpha = self.c['motion_ema']
        self.history = {k:v for k,v in self.history.items() if t-v.time <= self.c['track_max_gap']}
        for d in tracks:
            if d.get('track_id') is None:
                continue
            box = np.asarray(d['box']) / [w,h,w,h]
            center = (box[:2]+box[2:])/2
            old = self.history.get(d['track_id'])
            velocity, acceleration, decel, angle, stop = np.zeros(2), 0., 0., 0., 0.
            peak = 0.
            if old is not None and t > old.time:
                dt = t-old.time
                center = alpha*center + (1-alpha)*(old.center + shift)
                velocity = alpha*((center-old.center-shift)/dt)+(1-alpha)*old.velocity
                acceleration = np.linalg.norm(velocity-old.velocity)/dt
                speed, old_speed = np.linalg.norm(velocity), np.linalg.norm(old.velocity)
                decel = max(0., old_speed-speed)/dt
                if speed > .005 and old_speed > .005:
                    angle = float(1 - np.clip(np.dot(velocity,old.velocity)/(speed*old_speed),-1,1))
                peak = max(old.speed_peak*np.exp(-dt/2), speed)
                stop = float(speed < .01 and peak > .04)
            self.history[d['track_id']] = TrackState(t, center, velocity, peak)
            speeds.append(float(np.linalg.norm(velocity)))
            accelerations.append(float(acceleration)); decels.append(float(decel)); angles.append(angle); stops.append(stop)
            current.append((box, center, velocity))
        overlaps, distances, closing, relative, ttcs = [], [], [], [], []
        for i, (bi, ci, vi) in enumerate(current):
            for bj, cj, vj in current[i+1:]:
                dist = float(np.linalg.norm(cj-ci))
                cs = closing_speed(ci,cj,vi,vj)
                overlaps.append(iou(bi,bj)); distances.append(dist)
                closing.append(max(cs,0)); relative.append(float(np.linalg.norm(vj-vi)))
                ttcs.append(min(10.,dist/cs) if cs > .001 else 10.)
        maximum = lambda a: max(a, default=0.)
        # A feature, never a standalone accident decision. No overlap-only trigger.
        impact = maximum(overlaps) * (maximum(decels)+maximum(closing)+maximum(stops)*.1)
        values = [len(tracks)/20, np.mean([d['confidence'] for d in tracks]) if tracks else 0,
                  maximum(speeds),maximum(accelerations),maximum(decels),maximum(angles),
                  maximum(overlaps),min(distances,default=1.),maximum(closing),min(ttcs,default=10.),
                  maximum(relative),maximum(stops),sum(stops)/max(len(stops),1),np.linalg.norm(shift),
                  len(current)/max(len(tracks),1),impact]
        return np.asarray(values,dtype=np.float32)
