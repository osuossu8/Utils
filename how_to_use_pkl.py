
import pickle

# 保存
with open('drive/My Drive/ELMo_allenlp_tutorial/x_train.pkl', mode='wb') as f:
    pickle.dump(x_train, f)

# ロード
with open('drive/My Drive/ELMo_allenlp_tutorial/x_train.pkl', mode='rb') as f:
    x_train = pickle.load(f)
    
x_train.head()