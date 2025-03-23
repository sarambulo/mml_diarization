from pydub import AudioSegment
import os


def convert_mp4_to_wav(input_path, output_file_path):
    for file in os.listdir(input_path):
        # Load the MP4 file
        audio = AudioSegment.from_file(os.path.join(input_path, file), format="mp4")

        # Export as WAV
        audio.export(output_file_path + file[:-4] + ".wav", format="wav")


if __name__ == "__main__":
    input_mp4 = "data/many_val"
    output_wav = "data/many_val_audio"
    convert_mp4_to_wav(input_mp4, output_wav)
