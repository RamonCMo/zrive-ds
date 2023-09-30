import boto3 
import os

BUCKET_NAME = 'zrive-ds-data'
prefix = 'groceries/sampled-datasets/' # helps to search particular folder in bucket
session = boto3.Session(profile_name='default')
s3 = session.resource('s3')

def download_folder(bucket_name, s3_folder, local_dir=None):
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        bucket.download_file(obj.key, target)

download_folder(BUCKET_NAME, prefix, 'data/')