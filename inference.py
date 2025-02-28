import pandas as pd

def create_rttm_file(video_id, speaker_ids, labels, timestamps, output_path):
   """
   Create RTTM file from model predictions
   video_id: Vieo ID 
   labels: list of binary predictions (0/1) for active speaker detection
   timestamps: list of (start_time, end_time) for each prediction
   """
   # Expand speakers and labels into multiple rows
   # Set timestamps as index and melt
   # Sort by speakers then timestamps
   # Generate ids for intervals using cumsum
   # Drop rows were the person is not speaking label = 0
   # Start is the min(timestamp) in an interval
   # End is the min(timestamp) in an interval
   with open(output_path, 'w') as f:
      for video_id, speaker_ids, labels, timestamps in zip(video_id, speaker_ids, labels, timestamps):
         data = pd.DataFrame(columns=speaker_ids[0], data=labels)
         data['Timestamp'] = timestamps
         data = data.melt(id_vars=['Timestamp'], var_name='Speaker ID', value_name='Speaking')
         data = data.sort_values(['Speaker ID', 'Timestamp'])
         data['Interval Flag'] = (data['Speaking'] == 1) & (data.groupby('Speaker ID')['Speaking'].shift(1, fill_value=0) == 0)
         data['Interval ID'] = data.groupby('Speaker ID')['Interval Flag'].cumsum()
         data = data[data['Speaking'] == 1]
         data = data.groupby(['Speaker ID', 'Interval ID']).agg(**{
            'start': ('Timestamp', lambda x: x.min()),
            'end': ('Timestamp', lambda x: x.max()),
         })
         data['duration'] = data['end'] - data['start']

         speaker_ids = data['Speaker ID'].astype(int)
         start = data['start'].astype(float)
         duration = data['duration'].astype(float)

         for i in range(len(speaker_ids)):
            f.write(f"SPEAKER {int(video_id):05d} 0 {start[i]:.6f} {duration[i]:.6f} <NA> <NA> {speaker_ids[i]} <NA> <NA>\n")
   return