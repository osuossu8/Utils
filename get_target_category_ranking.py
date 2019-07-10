
import pandas as pd

# ソフトマックス関数を定義
def softmax(x):
    u = np.sum(np.exp(x))
    return np.exp(x)/u

def get_target_category_ranking(df, predict_col,softmax_col):
    """
    target 毎の softmax 値のランキングを取得する.
    
    Parameters
    ----------
    df : dataframe
    predict_col : string
        予測値のラベル
    softmax_col : string
    
    Returns
    -------
    df : dataframe
        rank カラムが追加された df
    """
    for i in df[predict_col].unique():
        replace_index = list(df[df[predict_col]==i].index)
        df.loc[replace_index, "rank"] = df[df[predict_col]==i][softmax_col].rank(ascending=False).values
        df["rank"] = df["rank"].astype(np.int8)
        
    return df

# df = get_target_category_ranking(df, 'predict', 'softmax')