# How to use kaggle command at Google Colaboratory Environment

# =======
# set up
# =======
!pip install -q kaggle
!mkdir -p ~/.kaggle
## !cp kaggle.json ~/.kaggle/
!cp 'drive/My Drive/kaggle/kaggle.json' ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json


# =============================
# download competition datasets
# =============================
!kaggle competitions download -c bengaliai-cv19


# =======================
# download usersdatasets
# =======================

# example
# https://www.kaggle.com/kaerunantoka/japanese-bert-pretrained-model
!kaggle datasets download -w kaerunantoka/japanese-bert-pretrained-model

# ====================
# upload user datasets
# ====================

!mkdir upload

!kaggle datasets init -p upload

!cp fuga.txt upload

```
%%writefile upload/dataset-metadata.json
{
 "title": "hogefuga", # The dataset title must be between 6 and 50 character
 "id": "kaerunantoka/hogefuga",
 "licenses": [
  {
   "name": "CC0-1.0"
  }
 ]
}
```

# finally
!kaggle datasets create -p upload

# I can access https://www.kaggle.com/kaerunantoka/hogefuga
