"""
Mock abstraction layer for Google Cloud Storage.
In a real GCP environment with workload identity, these calls
interact with a gs:// bucket without needing service account JSON keys.
"""

# import os
# from google.cloud import storage

def download_model_from_gcs(bucket_name: str, source_blob_name: str, destination_file_name: str):
    """
    Downloads a blob from the bucket.
    Example: download_model("genai-models", "fashion-lora/adapter_model.safetensors", "models/fashion-lora/adapter_model.safetensors")
    """
    print(f"[GCP STUB] Downloading gs://{bucket_name}/{source_blob_name} to {destination_file_name}")
    # client = storage.Client()
    # bucket = client.bucket(bucket_name)
    # blob = bucket.blob(source_blob_name)
    # blob.download_to_filename(destination_file_name)
    # print(f"Downloaded {source_blob_name}.")
    pass

def upload_image_to_gcs(bucket_name: str, source_file_name: str, destination_blob_name: str) -> str:
    """
    Uploads an image to the bucket and returns the generated GCS URI.
    """
    print(f"[GCP STUB] Uploading {source_file_name} to gs://{bucket_name}/{destination_blob_name}")
    # client = storage.Client()
    # bucket = client.bucket(bucket_name)
    # blob = bucket.blob(destination_blob_name)
    # blob.upload_from_filename(source_file_name)
    # return f"gs://{bucket_name}/{destination_blob_name}"
    return f"gs://{bucket_name}/mocked_uri"
