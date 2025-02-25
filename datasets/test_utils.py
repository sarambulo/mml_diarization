from .utils import parse_rttm
from pathlib import Path
import torch
import numpy as np

class TestParseRTTM():
   def test_ids(self):
      data_path = Path('data')
      rttm_files = ['few_train', 'few_val', 'many_val']
      for file in rttm_files:
         file_path = data_path / f'{file}.rttm'
         rttm = parse_rttm(file_path)
         # rttm is a dictionary with file ids as keys
         assert isinstance(rttm, dict)
         # each entry has a tuple with the timestamps and the speaker ids
         for file_id in rttm:
            data = rttm[file_id]
            assert len(data) == 2
            assert isinstance(data[0], np.ndarray)
            assert isinstance(data[1], np.ndarray)

