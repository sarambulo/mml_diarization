from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

def rttm_to_annotation(path):
    ann = Annotation()
    with open(path) as f:
        for line in f:
            if line.startswith("SPEAKER"):
                parts = line.strip().split()
                file_id = parts[1]
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                ann[Segment(start, start + duration)] = speaker
    return ann

def evaluate_rttms(reference_rttm, hypothesis_rttm):
    reference = rttm_to_annotation(reference_rttm)
    hypothesis = rttm_to_annotation(hypothesis_rttm)

    der = DiarizationErrorRate()
    score = der(reference, hypothesis)
    print(f"🎯 DER: {score:.4f}")

