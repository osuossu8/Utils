
import pandas as pd
import glob
import os

from tqdm import tqdm

#preprocessing
dirlist = ["dokujo-tsushin","it-life-hack","kaden-channel","livedoor-homme",
           "movie-enter","peachy","smax","sports-watch","topic-news"]
df = pd.DataFrame(columns=["class","news"])
for i in tqdm(dirlist):
    path = "text/"+i+"/*.txt"
    files = glob.glob(path)
    files.pop()
    for j in tqdm(files):
        f = open(j)
        data = f.read() 
        f.close()
        t = pd.Series([i,"".join(data.split("\n")[3:])],index = df.columns)
        df  = df.append(t,ignore_index=True)

# !tar -xvf ldcc-20140209.tar.gz
# df.to_csv("drive/My Drive/ELMo_allenlp_tutorial/livedoor_news_text.csv", index=False)
# !bzip2 -d 'drive/My Drive/ELMo_allenlp_tutorial/tweets_open.csv.bz2'