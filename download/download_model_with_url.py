import logging
import os
import requests
import shutil
import sys
import tarfile
import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))

def download_model_with_url(url):
    with timer('download model file with url'):
        URL = url
        res = requests.get(URL, stream = True)
        file_path = 'pytorch_model_file.pth'
        print('downloading model file ...')
        with open(file_path, 'wb') as fp:
            shutil.copyfileobj(res.raw, fp)

        # use when tar.gz
        # with tarfile.open(file_path) as tar:
        #     tar.extractall(.)
        # os.remove(file_path)

sample_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
# download_model_with_url(sample_url)
