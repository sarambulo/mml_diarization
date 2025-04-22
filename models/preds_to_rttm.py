def gather_predictions(self, predicted_batch, metadata_batch):
    """
    Combine the model's is_speaking (0/1) with the metadata
    that has speaker_id, frame_id, chunk_id, etc.
    Return a list of dicts.
    """
    # predicted_batch is shape [B], each 0/1
    # metadata_batch is a list of length B with dicts:
    #   { 'video_id': ..., 'chunk_id': ..., 'frame_id': ..., 'speaker_id': ...}
    records = []
    for i in range(len(predicted_batch)):
        record = {
            "video_id": metadata_batch[i]["video_id"],
            "chunk_id": metadata_batch[i]["chunk_id"],
            "frame_id": metadata_batch[i]["frame_id"],
            "speaker_id": metadata_batch[i]["speaker_id"],
            "is_active": int(predicted_batch[i]),
            # compute timestamp if you know fps
            "timestamp": metadata_batch[i]["frame_id"] / 25.0,
        }
        records.append(record)
    return records


def utterances_to_rttm(self, utterances, file_id):
    rttm_lines = []
    if len(utterances) == 0:
        return []

    for speaker_id, speaker_utterances in utterances.items():
        for utt in speaker_utterances:
            start_time = f"{utt['start_time']:.6f}"
            duration = f"{utt['duration']:.6f}"
            # The speaker_id may be a string or int, up to you
            # Usually it’s the cluster index or face ID

            # Basic example:
            rttm_line = (
                f"SPEAKER {file_id:05d} 0 {start_time} {duration} <NA> <NA> "
                f"{speaker_id} <NA> <NA>"
            )
            rttm_lines.append(rttm_line)

    # Sort lines by start_time (4th column)
    rttm_lines.sort(key=lambda x: float(x.split()[3]))

    return rttm_lines


def form_rttm(self, X, file_id):
    results = self.predict_video(X)
    active_frames = self.active_frames_by_speaker_id(results)
    utterances = self.create_utterances(active_frames)
    rttm_lines = self.utterances_to_rttm(utterances, file_id)
    return rttm_lines
