
import boto3
import os
from urllib.parse import urlparse

def parse_s3_url(url):
    parts = urlparse(url)
    bucket = parts.netloc
    key = parts.path.lstrip('/')
    return bucket, key

def download_s3_file(url, local_path=None):
    bucket, key = parse_s3_url(url)
    s3 = boto3.client('s3')
    if not local_path:
        local_path = os.path.join(os.getcwd(), os.path.basename(key))
    s3.download_file(bucket, key, local_path)
    print(f"Downloaded file from {url} to {local_path}")

if __name__ == "__main__":
    URL = 's3://zrive-ds-data/groceries/sampled-datasets/'
    PATH = '~/Zrive/zrive-ds/data/'
    print(urlparse(url=URL))
   #download_s3_file(, local_path=PATH)


