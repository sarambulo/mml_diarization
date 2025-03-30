from typing import Dict
from .data import rttm_to_annotations
from pyannote.metrics.diarization import (
    GreedyDiarizationErrorRate,
    JaccardErrorRate,
)
import pandas as pd
import json

PATH_TO_PREDS = "aws_transcribe_rttms"
PATH_TO_TARGETS = "data/few.val.rttm"


def calculate_metrics_for_video(preds, targets, der=None, jer=None):
    if not der:
        der = GreedyDiarizationErrorRate()
    if not jer:
        jer = JaccardErrorRate()
    videoMetrics = {}
    der_detail = der(preds, targets, detailed=True)

    videoMetrics["totalDuration"] = der_detail["total"]

    videoMetrics["DER"] = der_detail["diarization error rate"]

    videoMetrics["MS"] = der_detail["missed detection"]
    videoMetrics["MSR"] = der_detail["missed detection"] / der_detail["total"]

    videoMetrics["SE"] = der_detail["confusion"]
    videoMetrics["SER"] = der_detail["confusion"] / der_detail["total"]

    videoMetrics["FA"] = der_detail["false alarm"]
    videoMetrics["FAR"] = der_detail["false alarm"] / der_detail["total"]

    videoMetrics["JER"] = jer(preds, targets)
    return videoMetrics


def calculate_metrics_for_dataset(preds_dict, targets_dict):
    if len(preds_dict) == 0:
        metrics = {}
        metrics["DER"] = 1.0
        metrics["JER"] = 1.0
        metrics["MS"] = None
        metrics["MSR"] = 1.0
        metrics["FA"] = None
        metrics["FAR"] = 1.0
        metrics["SE"] = None
        metrics["SER"] = 1.0
        metrics["metricsByVideo"] = {}
        return metrics

    metricsByVideo = {}
    der = GreedyDiarizationErrorRate(skip_overlap=False)
    jer = JaccardErrorRate()
    for videoId in preds_dict:
        metricsByVideo[videoId] = calculate_metrics_for_video(
            preds=preds_dict[videoId],
            targets=targets_dict[videoId],
            der=der,
            jer=jer,
        )

    df = pd.DataFrame.from_records(list(metricsByVideo.values()))
    metrics = {}

    metrics["DER"] = float(df.DER.mean())
    metrics["JER"] = float(df.JER.mean())
    metrics["MS"] = float(df.MS.mean())
    metrics["MSR"] = float(df.MS.sum() / df.totalDuration.sum())
    metrics["FA"] = float(df.FA.mean())
    metrics["FAR"] = float(df.FA.sum() / df.totalDuration.sum())
    metrics["SE"] = float(df.SE.mean())
    metrics["SER"] = float(df.SE.sum() / df.totalDuration.sum())

    # metrics["metricsByVideo"] = metricsByVideo

    return metrics


if __name__ == "__main__":
    preds = rttm_to_annotations(PATH_TO_PREDS)
    targets = rttm_to_annotations(PATH_TO_TARGETS)
    m = calculate_metrics_for_dataset(preds, targets)
    print(json.dumps(m, indent=2))
