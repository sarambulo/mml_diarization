import torch
import torch.nn as nn
import torchaudio.transforms as AT
from torch.utils.data import Dataset
from ast_models import ASTModel

class AudioASTEncoder(nn.Module):
    def __init__(self, embedding_dims=768):
        super().__init__()
        self.ast_model = ASTModel(
            label_dim=527,  
            fstride=10, tstride=10,
            input_fdim=128, input_tdim=1024,
            imagenet_pretrain=True, audioset_pretrain=True,
            model_size='base384'
        )
        self.embedding_dims = embedding_dims

    def forward(self, x):
        x = (x + 4.26) / (4.57 * 2)  # normalization
        x = x.unsqueeze(1).transpose(2, 3)

        v = self.ast_model.v
        B = x.shape[0]
        x = v.patch_embed(x)
        cls_tokens = v.cls_token.expand(B, -1, -1)
        dist_token = v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + v.pos_embed
        x = v.pos_drop(x)
        for blk in v.blocks:
            x = blk(x)
        x = v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2  # [B, 768] — the embedding

        return torch.nn.functional.normalize(x, dim=-1)  


class AudioTripletDatasetWithLabels(Dataset):
    def __init__(self, msd_dataset, augment=True):
        self.msd_dataset = msd_dataset
        self.augment = augment
        self.mel_transform = AT.MelSpectrogram(
            sample_rate=16000,
            n_mels=128,
            n_fft=400,
            hop_length=160,
            win_length=400,
            pad_mode="reflect"
        )

    def __len__(self):
        return len(self.msd_dataset)

    def pad_or_crop(self, mel, target_length=1024):
        current_length = mel.shape[-1]
        if current_length < target_length:
            pad_amt = target_length - current_length
            mel = torch.nn.functional.pad(mel, (0, pad_amt))
        else:
            mel = mel[:, :target_length]
        return mel

    def process_audio(self, audio_segment):
        if audio_segment is None:
            return None
        if audio_segment.dim() == 1:
            audio_segment = audio_segment.unsqueeze(0)
        elif audio_segment.shape[-1] == 2:
            audio_segment = audio_segment.mean(dim=-1, keepdim=True)
        mel = self.mel_transform(audio_segment)
        mel = self.pad_or_crop(mel[0])
        return mel.T

    def __getitem__(self, index):
        try:
            _, audio_triplet, is_speaking = self.msd_dataset[index]
            anchor_audio, positive_audio, negative_audio = audio_triplet
            anchor_mel = self.process_audio(anchor_audio)
            positive_mel = self.process_audio(positive_audio)
            negative_mel = self.process_audio(negative_audio)
            if any(x is None for x in [anchor_mel, positive_mel, negative_mel]):
                return None
            label = torch.tensor(float(is_speaking), dtype=torch.float32)
            return anchor_mel, positive_mel, negative_mel, label
        except:
            return None
