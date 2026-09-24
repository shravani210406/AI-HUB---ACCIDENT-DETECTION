from collections import deque
from dataclasses import dataclass, asdict
import logging
from .video import format_time

@dataclass
class Event:
    onset_seconds: float
    confirmation_seconds: float
    end_seconds: float
    confidence: float

    def to_dict(self):
        return {**asdict(self), 'onset_timestamp':format_time(self.onset_seconds),
                'confirmation_timestamp':format_time(self.confirmation_seconds)}

class DecisionEngine:
    """K-of-N entry, sustained low exit, evidence-backed onset and event merging."""
    def __init__(self,c):
        self.c = c
        self.votes = deque(maxlen=c['vote_window'])
        self.history = deque()
        self.events = []
        self.active = None
        self.low_count = 0

    def update(self,t,p,impact=0.):
        if self.history and t-self.history[-1][0] > self.c.get('max_evidence_gap_seconds',max(1.,3*self.c['stride_seconds'])):
            self._close()
            self.votes.clear()
            self.history.clear()
        self.history.append((t,p,impact))
        while self.history and self.history[0][0] < t-self.c['lookback_seconds']:
            self.history.popleft()
        self.votes.append((t,p >= self.c['accident_threshold_high']))
        if self.active is None:
            positive = [s for s,v in self.votes if v]
            if len(positive) >= self.c['vote_min_positive'] and t-positive[0] >= self.c['minimum_event_duration']:
                # ``t`` is the source video's presentation time, not a sampled
                # frame number.  Report the first actual model-alert crossing,
                # rather than backdating to weaker motion evidence.  Interpolating
                # between the two neighbouring model outputs removes the fixed
                # sampling-step rounding error without changing any decision rule.
                first_positive = positive[0]
                high = self.c['accident_threshold_high']
                prior = [(s, q) for s, q, _ in self.history if s < first_positive]
                if prior:
                    previous_t, previous_p = prior[-1]
                    if previous_p < high:
                        first_high = next((q for s, q, _ in self.history if s == first_positive), high)
                        rise = first_high - previous_p
                        onset = previous_t + (first_positive - previous_t) * (high - previous_p) / rise if rise > 0 else first_positive
                    else:
                        onset = first_positive
                else:
                    onset = first_positive
                self.active = Event(onset,t,t,p)
                logging.info('Event confirmed at %.3fs; onset estimate %.3fs; probability %.4f',t,onset,p)
                self.low_count = 0
        else:
            self.active.end_seconds = t
            self.active.confidence = max(self.active.confidence,p)
            self.low_count = self.low_count+1 if p < self.c['accident_threshold_low'] else 0
            if self.low_count >= self.c['exit_steps']:
                self._close()
                self.votes.clear()
        return self.active is not None

    def _close(self):
        event = self.active
        if event is None:
            return
        if self.events and event.onset_seconds-self.events[-1].end_seconds <= self.c['merge_gap_seconds']:
            self.events[-1].end_seconds = event.end_seconds
            self.events[-1].confidence = max(self.events[-1].confidence,event.confidence)
        else:
            self.events.append(event)
        self.active = None

    def finish(self):
        self._close()
        return [e.to_dict() for e in self.events]
