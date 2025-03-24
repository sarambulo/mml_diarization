from pairs.utils import *
from collections import defaultdict
import numpy as np
import pandas as pd
import random


def create_lookups(df):
    speaker_frames = defaultdict(list)  #speaker -> list of (chunk_id, frame_id) tuples
    chunk_speakers = defaultdict(list)  #chunk id -> list of (speaker_id, frame_id) tuples  
    
    for _, row in df.iterrows():
        spk = row['speaker_id']
        ch = row['chunk_id']
        fr = row['frame_id']
        is_speaking = row['is_speaking']
        speaker_frames[spk].append((ch, fr))
        chunk_speakers[ch].append((spk, fr))
    
    return speaker_frames, chunk_speakers

def get_visual_pairs(anchor_speaker_id, anchor_frames, chunk_speakers): #input: list of chunk_id, frame_id tuples from above
    positive_pairs = []
    negative_pairs = []
    for current_chunk, current_frame in anchor_frames: 
        pos_chunk, pos_frame = get_positive_pair(current_chunk, current_frame, anchor_frames)
        neg_chunk, neg_speaker, neg_frame = get_negative_pair(anchor_speaker_id, current_chunk, chunk_speakers)
    pass

def get_positive_pair(current_chunk, current_frame, anchor_frames):
    same_chunk_frames = [(ch, fr) for ch, fr in anchor_frames 
                               if ch == current_chunk and fr != current_frame]
    if same_chunk_frames:
        pos_chunk_id, pos_frame_id = random.choice(same_chunk_frames)
    else:
        diff_chunk_frames = [(ch, fr) for ch, fr in anchor_frames if ch != current_chunk]
        pos_chunk_id, pos_frame_id = random.choice(diff_chunk_frames)
        
    return pos_chunk_id, pos_frame_id
        
def get_negative_pair(anchor_speaker_id, current_chunk, chunk_speakers):
    speakers_by_chunk = chunk_speakers[current_chunk] #list of (speaker, frame) tuples
    other_speaker_frames = [(spk, fr) for spk, fr in speakers_by_chunk 
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
                
    

def create_pairs(chunks, input_file='is_speaking.csv'):
    df = pd.read_csv(input_file, 
                names=['chunk_id', 'speaker_id', 'is_speaking', 'timestamp', 'frame_id'],
                usecols=['speaker_id', 'is_speaking', 'frame_id', 'chunk_id'])
        
    speaker_frames, chunk_speakers = create_lookups(df)
    
    for speaker in speaker_frames:
        anchor_frames = speaker_frames[speaker] #get frames for current speaker
        get_visual_pairs(speaker, anchor_frames, chunk_speakers)


##FUTURE STATE for create pairs 
#INPUT: is_speaking.csv
#output: pairs.csv with 7 new columns:  PosChunkID, PosFrameID, NegChunkID, NegFrameID, NegSpeakerID, Video_Flag, Audio_Flag

def new_create_pairs(input_file='is_speaking.csv'):
    df = pd.read_csv(input_file, 
                names=['chunk_id', 'speaker_id', 'is_speaking', 'timestamp', 'frame_id'],
                usecols=['speaker_id', 'is_speaking', 'frame_id', 'chunk_id'])
        
    speaker_frames, chunk_speakers = create_lookups(df) #add is_speaking to create_lookups() function
    
    for speaker in speaker_frames: #iterate through speakers
        anchor_frames = speaker_frames[speaker] #get frames for current speaker
        #split anchor_frames into speaking/not speaking based on create_lookups update
        #for frame in is_speaking_frames:
            get_combined_pair() #to be implemented
        #for frame in non_speaking_frames:
            get_visual_pairs(speaker, anchor_frames, chunk_speakers) #mostly implemented above



   

