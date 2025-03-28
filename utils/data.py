from pyannote.core import Annotation, Segment
import os


def rttm_to_annotations(path):
    d = load_rttm_by_video(path)
    annotations = {}
    for videoId in d:
        ann = Annotation()
        for seg in d[videoId]:
            ann[Segment(start=seg["startTime"], end=seg["endTime"])] = seg["speakerId"]
        annotations[videoId] = ann
    return annotations


def load_rttm_by_video_from_folder(path):
    data = {}
    for videoName in os.listdir(path):
        videoPath = os.path.join(path, videoName)
        data.update(load_rttm_by_video(videoPath))
    return data


def load_rttm_by_video(path):
    data = {}
    print(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 10 and fields[0] == "SPEAKER":
                file_id, start, duration, speaker = (
                    fields[1],
                    float(fields[3]),
                    float(fields[4]),
                    fields[7],
                )
                if file_id not in data:
                    data[file_id] = []
                data[file_id].append(
                    {
                        "speakerId": speaker,
                        "startTime": start,
                        "endTime": start + duration,
                        "duration": duration,
                    }
                )
    return data
