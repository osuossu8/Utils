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

# only this part, I worked at local mac pc...
# I need overwrite some file.

kaggle datasets init -p hoge
vi hoge/dataset-metadata.json

# before
{
 "title": "INSERT_TITLE_HERE",
 "id": "kaerunantoka/INSERT_SLUG_HERE",
 "licenses": [
  {
   "name": "CCO-1.0"
  }
 ]
}

# overwrite hoge/dataset-metadata.json

# after
{
 "title": "hogefuga", # The dataset title must be between 6 and 50 character
 "id": "kaerunantoka/hogefuga",
 "licenses": [
  {
   "name": "CCO-1.0"
  }
 ]
}

# finally
kaggle datasets create -p hoge

# I can access https://www.kaggle.com/kaerunantoka/hogefuga
