from typing import Dict
from utils.rttm import rttm_to_annotations
from pyannote.metrics.diarization import (
    GreedyDiarizationErrorRate,
    JaccardErrorRate,
)
import pandas as pd
import json
from pyannote.core import Annotation
from sklearn.metrics import root_mean_squared_error

def calculate_metrics_for_video(preds: Annotation, targets: Annotation, der=None, jer=None):
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

    videoMetrics['Predicted Num Speakers'] = len(preds.labels())
    videoMetrics['Ground Truth Num Speakers'] = len(targets.labels())
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

    num_speakers_rmse = root_mean_squared_error(
        y_true=df['Ground Truth Num Speakers'],
        y_pred=df['Predicted Num Speakers'],
    )

    metrics["DER"] = float(df.DER.mean())
    metrics["JER"] = float(df.JER.mean())
    metrics["MS"] = float(df.MS.mean())
    metrics["MSR"] = float(df.MSR.mean())
    metrics["FA"] = float(df.FA.mean())
    metrics["FAR"] = float(df.FAR.mean())
    metrics["SE"] = float(df.SE.mean())
    metrics["SER"] = float(df.SER.mean())
    metrics["Duration"] = float(df['totalDuration'].mean())
    metrics["Num Speakers RMSE"] = num_speakers_rmse

    return metrics
