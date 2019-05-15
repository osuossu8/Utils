
import pandas as pd
import random
from tqdm import tqdm
import itertools as itr

path = "path/to/file"
df = pd.read_csv(path)

desc = df.title_a

meishi_list = [
 'googl', 
 'game',
 'trump',
 'blockchain',
 'window', 
]

# リスト内の名詞を含む文を抽出する
l = desc.apply(lambda x : x.lower()).to_list()
l_in = [s for s in l if meishi_list[0] in s]

# リストを半分に分割する.
txt_list_a = l_in[:len(l_in)//2]
txt_list_b = l_in[len(l_in)//2:]

lines = []
for i, j in tqdm(zip(txt_list_a, txt_list_b)):
    lines.append((i, j, str(0)))

tmp_df = pd.DataFrame(lines)
tmp_df.columns = ["title_a", "title_b", "target"]

'''
df の 別な列同士の要素からなる列を作成する.
'''
tmp_in_list = [
    (tmp_df.title_a[0], tmp_df.title_a[1], str(1)),
    (tmp_df.title_a[2], tmp_df.title_b[3], str(1)),
]

tmp_df.loc[5] = tmp_in_list[0]
tmp_df.loc[6] = tmp_in_list[1]


'''
指定列の target を変更する.
'''
false_list = [2, 5]
tmp_df.loc[false_list, 'target'] = 0


'''
連続した列を削除する.
'''
tmp_df = tmp_df.drop([i for i in range(7, 10)])


'''
指定単語を含むか否か(個数で対応)
'''
tmp_df['is_google'] = tmp_df.title_a.apply(lambda x: sum(x.count(w) for w in ["google", "Google"])).to_list()
tmp_df['is_trump'] = tmp_df.title_a.apply(lambda x: sum(x.count(w) for w in ["trump", "Trump"])).to_list()