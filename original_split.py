
from sklearn.model_selection import train_test_split

def original_split(df, target, test_size=0.2, seed=42):
    '''
    df : DataFrame
    target : String
    test_size : float
    seed : int
    '''
    df_pos = df[df[target]==1]
    df_neg = df[df[target]==0]
        
    df_neg_sample = df_neg.sample(n=len(df_pos), random_state=seed)

    X = pd.concat([df_pos, df_neg_sample])
    
    train, valid = train_test_split(X, test_size=test_size, random_state=seed)
    
    return train, valid
