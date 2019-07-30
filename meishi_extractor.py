#!pip install janome

from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.tokenfilter import *
from janome.charfilter import *

char_filters = [UnicodeNormalizeCharFilter(), RegexReplaceCharFilter(r"[IiⅠｉ?.*/~=()〝 <>:：《°!！!？（）-]+", "")]
tokenizer = Tokenizer()
token_filters = [POSKeepFilter(["名詞"]), POSStopFilter(["名詞,非自立", "名詞,数", "名詞,代名詞", "名詞,接尾", "名詞,サ変接続"]),LowerCaseFilter()]
analyzer = Analyzer(char_filters, tokenizer, token_filters)

def preprocess(df_text_col):
    df_text_col = df_text_col.astype(str).progress_apply(lambda x: " ".join([token.surface for token in analyzer.analyze(x)]))
    return df_text_col


# ===========
# How to use
# ===========

# preprocess(df["text"])