import re
import nltk
import pandas as pd

ps = nltk.stem.PorterStemmer()
lc = nltk.stem.lancaster.LancasterStemmer()
sb = nltk.stem.snowball.SnowballStemmer('english')

nltk.download('averaged_perceptron_tagger')


def clean_kakko_zenkaku(x):
    '''
    (), [], <>, とその中身、全角単語、url をクリーニングする.
    '''

    # x = x.str
    x = x.lower()
    
    ptn1 = r'\(.+?\)'
    ptn2 = r'\[.+?\]'
    ptn3 = r'\<.+?\>'
    url = r'(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+\$,%#]+)'
    zenkaku = r'[^\x01-\x7E]'
    
    x = re.sub(ptn1, " ", x)  
    x = re.sub(ptn2, " ", x)  
    x = re.sub(ptn3, " ", x)  
    x = re.sub(url, " ", x)
    x = re.sub(zenkaku, " ", x)
    return x


def clean_numbers(x):
    '''
    数字をクリーニングする.
    '''

    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def analyzer(text):
    stop_words = ['i', 'a', 'an', 'the', 'to', 'and', 'or', 'if', 'is', 'are', 'am', 'it', 'this', 'that', 'of', 'from', 'in', 'on']
    text = text.lower() # 小文字化
    text = text.replace('\n', '') # 改行削除
    text = text.replace('\t', '') # タブ削除
    puncts = r',.":)(-!?|;\'$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√。【】'
    for punct in puncts:
        text = text.replace(punct, f' {punct} ')
    # for bad_word in contraction_mapping:
    #     if bad_word in text:
    #         text = text.replace(bad_word, contraction_mapping[bad_word])
    text = text.split(' ') # スペースで区切る
    text = [sb.stem(t) for t in text]
    
    words = []
    for word in text:
        if (re.compile(r'^.*[0-9]+.*$').fullmatch(word) is not None): # 数字が含まれるものは分割
            for w in re.findall(r'(\d+|\D+)', word):
                words.append(w)
            continue
        if word in stop_words: # ストップワードに含まれるものは除外
            continue
        if len(word) < 2: #  1文字、0文字（空文字）は除外
            continue
        words.append(word)
        
    return " ".join(words)


def get_words(texts):
    sentences = texts.apply(lambda x: x.split()).values
    words = []
    for sentence in sentences:
        for word in sentence:
          words.append(word)
    return words


def get_meishi(word_list):
    tags = nltk.pos_tag(word_list)
    meishi_list = []
    for tag in tags:
        # 品詞タグに NN が含まれていたら、名詞リストに加える.
        if tag[1] in 'NN':
        meishi_list.append(tag[0])
  return meishi_list


def build_vocab_meishi(texts):
    # sentences = texts.apply(lambda x: x.split()).values
    sentences = texts
    vocab = {}
    for word in sentences:
        try:
            vocab[word] += 1
        except KeyError:
            vocab[word] = 1
    return vocab


def get_sorted_freq_meishi(dct):
    sorted_words = []
    for k, v in sorted(dct.items(), key=lambda x: -x[1]):
      # print(str(k) + ": " + str(v))
      sorted_words.append(str(k))
    return sorted_words


'''
# Usage

df = pd.read_csv(path)
desc = df.Description

desc = desc.apply(lambda x: clean_kakko_zenkaku(x))
desc = desc.apply(lambda x: clean_numbers(x))
desc = desc.apply(lambda x: analyzer(x))
  
words = get_words(desc)
words[:10]

meishi_list = get_meishi(words)
print(len(meishi_list))
meishi_list[:10]

vocab_m = build_vocab_meishi(meishi_list)

get_sorted_freq_meishi(vocab_m)[:10]

'''