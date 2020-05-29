# !pip install mecab-python3 > /dev/null
# !pip install transformers > /dev/null

import pandas as pd
import numpy as np
import torch
import transformers

from transformers import BertJapaneseTokenizer
from tqdm import tqdm
tqdm.pandas()


class BertSequenceVectorizer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = 'bert-base-japanese-whole-word-masking'
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128
            

    def vectorize(self, sentence : str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)
        
        seq_out, pooled_out = self.bert_model(inputs_tensor, masks_tensor)

        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy() # 0番目は [CLS] token, 768 dim の文章特徴量
        else:
            return seq_out[0][0].detach().numpy()


def cos_sim_matrix(matrix):
    """
    item-feature 行列が与えられた際に
    item 間コサイン類似度行列を求める関数
    """
    d = matrix @ matrix.T  # item-vector 同士の内積を要素とする行列

    # コサイン類似度の分母に入れるための、各 item-vector の大きさの平方根
    norm = (matrix * matrix).sum(axis=1, keepdims=True) ** .5

    # それぞれの item の大きさの平方根で割っている（なんだかスマート！）
    return d / norm / norm.T


if __name__ == '__main__':

    sample_df = pd.DataFrame(['お腹が痛いので遅れます。',
                              '頭が痛いので遅れます。',
                              'おはようございます！',
                              'kaggle が好きなかえる',
                              '味噌汁が好きなワニ'
                              ], columns=['text'])


    BSV = BertSequenceVectorizer()
    sample_df['text_feature'] = sample_df['text'].progress_apply(lambda x: BSV.vectorize(x))
    print(sample_df.head())

    print(cos_sim_matrix(np.stack(sample_df.text_feature)))

    '''
    array([[0.99999976, 0.96970403, 0.7450729 , 0.6458974 , 0.6031469 ],
           [0.96970403, 1.0000002 , 0.7233708 , 0.6374834 , 0.59641033],
           [0.7450729 , 0.7233708 , 1.0000002 , 0.6782884 , 0.62431043],
           [0.6458974 , 0.6374834 , 0.67828834, 1.0000001 , 0.8846145 ],
           [0.603147  , 0.5964103 , 0.62431043, 0.8846146 , 1.0000002 ]],
          dtype=float32)
    '''
