import os


def create_numbered_file(dir, base_name, extension):
    counter = 0
    while True:
        if counter == 0:
            file_name = f"{base_name}.{extension}"
        else:
            file_name = f"{base_name}_{counter}.{extension}"

        if not os.path.exists(os.path.join(dir, file_name)):
            return file_name
        counter += 1


def get_all_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def save_triplet_metadata_to_csv(csv_path, speaker_id, status, path):
    pass


def save_triplet(tup):
    pass


def read_spectrogram(path):
    pass


def read_speaker_file(path):
    pass


def crawl_chunks_dir(dir):
    pass
