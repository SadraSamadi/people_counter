from datetime import timedelta
from math import sqrt
from time import time

import cv2 as cv
import dlib
import numpy as np


class PeopleCounter:

    def __init__(self, input_video, output_video, prototxt, caffemodel, skip_frames, confidence, distance):
        self._input_video = input_video
        self._output_video = output_video
        self._prototxt = prototxt
        self._caffemodel = caffemodel
        self._skip_frames = skip_frames
        self._confidence = confidence
        self._distance = distance
        self._net = None
        self._video = None
        self._width = None
        self._height = None
        self._frames = None
        self._fps = None
        self._writer = None
        self._image = None
        self._status = None
        self._trackers = []
        self._people = {}
        self._counter = 0
        self._total_up = 0
        self._total_down = 0

    def init(self):
        self._net = cv.dnn.readNetFromCaffe(self._prototxt, self._caffemodel)
        self._video = cv.VideoCapture(self._input_video)
        width = self._video.get(cv.CAP_PROP_FRAME_WIDTH)
        self._width = int(width)
        height = self._video.get(cv.CAP_PROP_FRAME_HEIGHT)
        self._height = int(height)
        frames = self._video.get(cv.CAP_PROP_FRAME_COUNT)
        self._frames = int(frames)
        fps = self._video.get(cv.CAP_PROP_FPS)
        self._fps = int(fps)
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        self._writer = cv.VideoWriter(self._output_video, fourcc, self._fps, (self._width, self._height), True)

    def start(self):
        for frame in range(self._frames):
            start = time()
            self._update(frame)
            self._render(frame)
            self._writer.write(self._image)
            finish = time()
            delay = int(1000 / self._fps - (finish - start) * 1000)
            delay = max(delay, 1)
            key = cv.waitKey(delay)
            if key == 27:
                break
        self._stop()

    def _update(self, frame):
        self._status = 'Waiting'
        _, self._image = self._video.read()
        rgb = cv.cvtColor(self._image, cv.COLOR_BGR2RGB)
        if frame % self._skip_frames == 0:
            self._detect(rgb)
        else:
            for tracker in self._trackers:
                self._status = 'Tracking'
                tracker.update(rgb)
        self._track()

    def _detect(self, rgb):
        self._status = 'Detecting'
        self._trackers = []
        blob = cv.dnn.blobFromImage(self._image, 0.007843, (self._width, self._height), 127.5)
        self._net.setInput(blob)
        detections = self._net.forward()
        for detection in detections[0, 0]:
            category = int(detection[1])
            confidence = detection[2]
            if category == 15 and confidence >= self._confidence:
                bounds = np.array([self._width, self._height, self._width, self._height])
                box = detection[3:7] * bounds
                box = box.astype(int)
                rect = dlib.rectangle(*box)
                tracker = dlib.correlation_tracker()
                tracker.start_track(rgb, rect)
                self._trackers.append(tracker)

    def _track(self):
        trackers = self._trackers.copy()
        disappeared = []
        for pid, positions in self._people.items():
            tracker = self._nearest(pid, trackers)
            if tracker:
                position = self._position(tracker)
                positions.append(position)
                trackers.remove(tracker)
            else:
                disappeared.append(pid)
        for pid in disappeared:
            positions = self._people[pid]
            first = positions[0]
            last = positions[-1]
            _, first = self._center(first)
            _, last = self._center(last)
            if first < last:
                self._total_down += 1
            else:
                self._total_up += 1
            del self._people[pid]
        for tracker in trackers:
            position = self._position(tracker)
            self._people[self._counter] = [position]
            self._counter += 1

    def _nearest(self, pid, trackers):
        positions = self._people[pid]
        for tracker in trackers:
            position = self._position(tracker)
            last = positions[-1]
            dist = self._dist(position, last)
            if dist <= self._distance:
                return tracker

    def _render(self, frame):
        for pid, positions in self._people.items():
            text = str(pid)
            last = positions[-1]
            start, end = last
            x, y = self._center(last)
            cv.putText(self._image, text, (x + 8, y + 4), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv.circle(self._image, (x, y), 2, (0, 255, 255), 2)
            cv.rectangle(self._image, start, end, (255, 0, 0))
            for position in positions:
                center = self._center(position)
                cv.circle(self._image, center, 1, (0, 255, 0))
        elapsed = timedelta(milliseconds=frame * 1000 / self._fps)
        info = [
            ('Time', elapsed),
            ('Status', self._status),
            ('Up', self._total_up),
            ('Down', self._total_down),
            ('Total', self._counter)
        ]
        for i, (label, value) in enumerate(info):
            text = f'{label}: {value}'
            org = 10, (i * 20) + 20
            cv.putText(self._image, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv.imshow('Frame', self._image)

    def _position(self, tracker):
        position = tracker.get_position()
        left = position.left()
        top = position.top()
        right = position.right()
        bottom = position.bottom()
        start = int(left), int(top)
        end = int(right), int(bottom)
        return start, end

    def _center(self, position):
        (left, top), (right, bottom) = position
        return (left + right) // 2, (top + bottom) // 2

    def _dist(self, a, b):
        ac = self._center(a)
        bc = self._center(b)
        dx = ac[0] - bc[0]
        dy = ac[1] - bc[1]
        return sqrt(dx ** 2 + dy ** 2)

    def _stop(self):
        cv.destroyAllWindows()
        self._writer.release()
        self._video.release()
