from pairs.utils import *
from config import *
from collections import defaultdict
import numpy as np
import pandas as pd
import random
import os

def create_lookups(df):
    speaker_frames = defaultdict(list)  #speaker -> list of (chunk_id, frame_id, is_speaking) tuples
    chunk_speakers = defaultdict(list)  #chunk id -> list of (speaker_id, frame_id, is_speaking) tuples  
    
    for _, row in df.iterrows():
        spk = row['speaker_id']
        ch = row['chunk_id']
        fr = row['frame_id']
        is_speaking = row['is_speaking']
        speaker_frames[spk].append((ch, fr, is_speaking))
        chunk_speakers[ch].append((spk, fr, is_speaking))
    
    return speaker_frames, chunk_speakers

def get_positive_pair(current_chunk, current_frame, anchor_frames):
    same_chunk_frames = [(ch, fr) for ch, fr, _ in anchor_frames 
                               if ch == current_chunk and fr != current_frame]
    if same_chunk_frames:
        pos_chunk_id, pos_frame_id = random.choice(same_chunk_frames)
    else:
        diff_chunk_frames = [(ch, fr) for ch, fr in anchor_frames if ch != current_chunk]
        pos_chunk_id, pos_frame_id = random.choice(diff_chunk_frames)
        
    return pos_chunk_id, pos_frame_id
        
def get_visual_negative_pair(anchor_speaker_id, current_chunk, chunk_speakers):
    speakers_by_chunk = chunk_speakers[current_chunk] #list of (speaker, frame) tuples
    other_speaker_frames = [(spk, fr) for spk, fr, _ in speakers_by_chunk 
                               if spk != anchor_speaker_id]
    if other_speaker_frames: #check if found in same chunk
        neg_face_id, neg_frame_id =  random.choice(other_speaker_frames)
        return current_chunk, neg_face_id, neg_frame_id
    else:
        remaining_chunk_ids = [ch for ch in chunk_speakers.keys() if ch != current_chunk]
        for chunk_id in remaining_chunk_ids:
            speakers_by_chunk = chunk_speakers[chunk_id]
            other_speaker_frames = [(spk, fr) for spk, fr in speakers_by_chunk 
                            if spk != anchor_speaker_id]
            if other_speaker_frames:
                neg_face_id, neg_frame_id = random.choice(other_speaker_frames)
                return chunk_id, neg_face_id, neg_frame_id
    return None

def get_combined_negative_pair(anchor_speaker_id, current_chunk, chunk_speakers, sample_diff_speaker_prob):
    is_speaking_val = 1 if np.random.uniform(0,1) <= sample_diff_speaker_prob else 0
    speakers_by_chunk = chunk_speakers[current_chunk]
    
    other_speaker_frames = [(spk, fr) for spk, fr, is_speaking in speakers_by_chunk 
                                if spk != anchor_speaker_id and is_speaking == is_speaking_val]
    if other_speaker_frames: #check if found in same chunk
        neg_face_id, neg_frame_id =  random.choice(other_speaker_frames)
        return current_chunk, neg_face_id, neg_frame_id
    else:
        remaining_chunk_ids = [ch for ch in chunk_speakers.keys() if ch != current_chunk]
        for chunk_id in remaining_chunk_ids:
            speakers_by_chunk = chunk_speakers[chunk_id]
            other_speaker_frames = [(spk, fr, is_speaking) for spk, fr,is_speaking in speakers_by_chunk 
                            if spk != anchor_speaker_id and is_speaking == is_speaking_val]
            if other_speaker_frames:
                neg_face_id, neg_frame_id = random.choice(other_speaker_frames)
                return chunk_id, neg_face_id, neg_frame_id
    return None


#INPUT: is_speaking.csv
#output: pairs.csv with 7 new columns:  PosChunkID, PosFrameID, NegChunkID, NegFrameID, NegSpeakerID, Video_Flag, Audio_Flag
def create_pairs(input_file_path):
    df = pd.read_csv(input_file_path, 
                names=['chunk_id', 'speaker_id', 'is_speaking', 'timestamp', 'frame_id'])
    
    speaker_frames, chunk_speakers = create_lookups(df)
    
    for anchor_speaker in speaker_frames: #iterate through speakers
        anchor_frames = speaker_frames[anchor_speaker] #get frames for current speaker
        #split anchor_frames into speaking/not speaking
        anchor_speaking_frames = [(ch, fr, is_speaking) for (ch, fr, is_speaking) in anchor_frames if is_speaking == 1]
        anchor_non_speaking_frames = [(ch, fr,is_speaking) for (ch, fr, is_speaking) in anchor_frames if is_speaking == 0]
        
        pair_info = {'chunk_id': [],
                     'speaker_id': [],
                     'is_speaking': [],
                     'frame_id':[],
                     'pos_chunk_id':[],
                     'pos_frame_id':[],
                     'neg_chunk_id':[],
                     'neg_speaker_id':[],
                     'neg_frame_id':[],
                     'video_flag':[],
                     'audio_flag':[]}
                     
        skipped_frames = [] #collect frames which are speaking but combined pair is not found
        
        for current_frame, current_chunk, _ in anchor_speaking_frames:
            pos_chunk, pos_frame = get_positive_pair(current_chunk, current_frame, anchor_speaking_frames) #get positive pair from speaking_frames only
            neg_chunk, neg_speaker, neg_frame = get_combined_negative_pair(anchor_speaker, current_chunk, chunk_speakers, AUDIO_SAMPLE_DIFFERENT_SPEAKER_PROB) #to be implemented - get negative pair for different face, speaking vs not speaking depends on probability
            if pos_chunk and neg_chunk:
                pair_info['chunk_id'].append(current_chunk)
                pair_info['speaker_id'].append(anchor_speaker)
                pair_info['is_speaking'].append(1) #always speaking
                pair_info['frame_id'].append(current_frame)
                pair_info['pos_chunk_id'].append(pos_chunk)
                pair_info['pos_frame_id'].append(pos_frame)
                pair_info['neg_chunk_id'].append(neg_chunk)
                pair_info['neg_speaker_id'].append(neg_speaker)
                pair_info['neg_frame_id'].append(neg_frame)
                pair_info['video_flag'].append(1) #works for both both video and audio pairs
                pair_info['audio_flag'].append(1)
                
            else:
                skipped_frames.append((current_frame, current_chunk, 1))
                
        for current_frame, current_frame, is_speaking in anchor_non_speaking_frames + skipped_frames: #visual only case
            pos_chunk, pos_frame = get_positive_pair(current_chunk, current_frame, anchor_frames) #get positive pair from any anchor frame
            neg_chunk, neg_speaker, neg_frame = get_visual_negative_pair(anchor_speaker, current_chunk, chunk_speakers) #get negative pair without speaking restriction
            if pos_chunk and neg_chunk:
                pair_info['chunk_id'].append(current_chunk)
                pair_info['speaker_id'].append(anchor_speaker)
                pair_info['is_speaking'].append(is_speaking) #is_speaking could be different
                pair_info['frame_id'].append(current_frame)
                pair_info['pos_chunk_id'].append(pos_chunk)
                pair_info['pos_frame_id'].append(pos_frame)
                pair_info['neg_chunk_id'].append(neg_chunk)
                pair_info['neg_speaker_id'].append(neg_speaker)
                pair_info['neg_frame_id'].append(neg_frame)
                pair_info['video_flag'].append(1) 
                pair_info['audio_flag'].append(0) #works for visual pairs only
                
        save_pair_info(pair_info)
        return pair_info

def save_pair_info(pair_info_dict, output_file_path):
    pairs_df = pd.DataFrame(pair_info_dict)
    pairs_df.to_csv(output_file_path, index=False)
    


def main():
    for video in videos:
        input_file_path = os.path.join(video, 'is_speaking.csv')
        output_file_path =  os.path.join(video, 'pairs.csv')
        pair_info_dict = create_pairs(input_file_path)
        save_pair_info(pair_info_dict, output_file_path)
    


   

