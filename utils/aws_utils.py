import boto3
import sagemaker
import sagemaker.amazon.common as smac


# upload/download regular files (txt, csv, json, ecc..)
def write_to_s3(filename, bucket, key):
    with open(filename, "rb") as f:  # Read in binary mode
        return (
            boto3.Session().resource("s3").Bucket(bucket).Object(key).upload_fileobj(f)
        )


def download_from_s3(filename, bucket, key):
    with open(filename, "wb") as f:
        return (
            boto3.Session()
            .resource("s3")
            .Bucket(bucket)
            .Object(key)
            .download_fileobj(f)
        )


# upload/download ReordIO format
def write_recordio_file(filename, x, y=None):
    with open(filename, "wb") as f:
        smac.write_numpy_to_dense_tensor(f, x, y)


def read_recordio_file(filename, recordsToPrint=10):
    with open(filename, "rb") as f:
        record = smac.read_records(f)
        for i, r in enumerate(record):
            if i >= recordsToPrint:
                break
            print("record: {}".format(i))
            print(r)
