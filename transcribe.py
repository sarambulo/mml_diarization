import boto3
import json
import os
import requests
from dotenv import load_dotenv
from utils.data import load_rttm_by_video, load_rttm_by_video_from_folder

load_dotenv()


def audio_to_rttm_aws_transcribe(
    audio_file_path, output_rttm_path, region_name="us-east-2"
):
    """
    Transcribes an audio file using Amazon Transcribe with speaker diarization enabled,
    automating S3 upload and cleanup.

    Args:
        audio_file_path (str): Path to the local audio file.
        output_rttm_path (str): Path to save the generated RTTM file.
        region_name (str): AWS region for Amazon Transcribe.
    """
    session = boto3.session.Session(
        aws_access_key_id=os.getenv("AWS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET"),
    )
    # Initialize AWS clients
    s3_client = session.client("s3", region_name=region_name)
    transcribe_client = session.client("transcribe", region_name=region_name)

    # Create a temporary S3 bucket name and job name
    bucket_name = "temporary-transcribe-bucket"
    videoId = os.path.splitext(os.path.basename(audio_file_path))[0]
    job_name = videoId + "-job-2"

    output_rttm_path = os.path.join(rttm_output, videoId + ".rttm")

    # Ensure the bucket exists (create if necessary)
    try:
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": region_name},
        )
        print(f"{videoId}: Created temporary S3 bucket: {bucket_name}")
    except Exception as e:
        if "BucketAlreadyOwnedByYou" not in str(e):
            print(f"Error creating bucket: {e}")
            return

    # Upload the audio file to S3
    s3_key = os.path.basename(audio_file_path)
    s3_client.upload_file(audio_file_path, bucket_name, s3_key)
    s3_uri = f"s3://{bucket_name}/{s3_key}"

    # Start transcription job with diarization enabled
    try:
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": s3_uri},
            MediaFormat="wav",  # Update based on your audio format
            LanguageCode="en-US",
            Settings={
                "ShowSpeakerLabels": True,
                "MaxSpeakerLabels": 4,  # Adjust based on expected number of speakers
            },
        )
        print(f"{videoId}: Started transcription job: {job_name}")
    except Exception as e:
        print(f"{videoId}: Error starting transcription job: {e}")
        return

    # Wait for the transcription job to complete
    while True:
        status = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        if status["TranscriptionJob"]["TranscriptionJobStatus"] in [
            "COMPLETED",
            "FAILED",
        ]:
            break

    if status["TranscriptionJob"]["TranscriptionJobStatus"] == "FAILED":
        print(
            f"{videoId}: Transcription job failed - {status['TranscriptionJob']['FailureReason']}"
        )
        return

    # Get the transcription result
    transcript_uri = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]

    try:
        # Step 1: Download the JSON transcript file
        response = requests.get(transcript_uri)
        response.raise_for_status()  # Raise an error if the download fails
        transcript_data = response.json()
        print(f"{videoId}: Transcript file downloaded successfully.")

        # Step 2: Extract speaker labels and timestamps
        speaker_labels = (
            transcript_data.get("results", {})
            .get("speaker_labels", {})
            .get("segments", [])
        )

        if not speaker_labels:
            print("{videoId}: No speaker labels found in the transcript.")
            return

        # Step 3: Generate RTTM file
        with open(output_rttm_path, "w") as rttm_file:
            for segment in speaker_labels:
                start_time = float(segment["start_time"])
                duration = float(segment["end_time"]) - start_time
                speaker_label = segment["speaker_label"]

                # Write RTTM line
                rttm_line = f"SPEAKER {videoId} 1 {start_time:.2f} {duration:.2f} <NA> <NA> {speaker_label} <NA> <NA>\n"
                rttm_file.write(rttm_line)

        print(f"{videoId}: RTTM file saved at {output_rttm_path}")

    except requests.exceptions.RequestException as e:
        print(f"{videoId}: Error downloading transcript: {e}")
    except json.JSONDecodeError as e:
        print(f"{videoId}: Error parsing JSON: {e}")
    except Exception as e:
        print(f"{videoId}: An unexpected error occurred: {e}")


# Example usage
audio_file = "./few_val_audio/00017.wav"
rttm_output = "aws_transcribe_rttms"

audio_dir = "./few_val_audio"

aws = load_rttm_by_video_from_folder(rttm_output)
msdwild = load_rttm_by_video("data/few.val.rttm")
missing = msdwild.keys() - aws.keys()
for video in missing:
    print(video)
    # audio_file = os.path.join(audio_dir, video + ".wav")
    # audio_to_rttm_aws_transcribe(audio_file, rttm_output)
