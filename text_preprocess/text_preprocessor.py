import re
import MeCab
import mojimoji


class MecabTokenizer(object):
    def __init__(self):
        self.wakati = MeCab.Tagger('-Owakati')
        self.wakati.parse('')

    def tokenize(self, line):
        txt = self.wakati.parse(line)
        txt = txt.split()
        return txt
    
    def mecab_tokenizer(self, line):
        node = self.wakati.parseToNode(line)
        keywords = []
        while node:
            if node.feature.split(",")[0] == "名詞":
                keywords.append(node.surface)
            node = node.next
        return keywords  

    def mecab_tokenizer_proper_noun(self, line):
        node = self.wakati.parseToNode(line)
        keywords = []
        while node:
            if node.feature.split(",")[1] == "固有名詞":
                keywords.append(node.surface)
            node = node.next
        return keywords 

    def mecab_tokenizer_other(self, line, other_list=[]):
        node = self.wakati.parseToNode(line)
        keywords = []
        while node:
            if node.feature.split(",")[0] == "名詞":
                if node.surface in other_list:
                    keywords.append(node.surface)
            node = node.next
        return keywords


class TextPreprocessorJP(object):
    def __init__(self):
        self.puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
                       '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\n', '\xa0', '\t',
                       '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
                       '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
                       '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '（', '）', '～',
                       '➡', '％', '⇒', '▶', '「', '➄', '➆',  '➊', '➋', '➌', '➍', '⓪', '①', '②', '③', '④', '⑤', '⑰', '❶', '❷', '❸', '❺', '❻', '❼', '❽',  
                       '＝', '※', '㈱', '､', '△', '℮', 'ⅼ', '‐', '｣', '┝', '↳', '◉', '／', '＋', '○',
                       '【', '】', '✅', '☑', '➤', 'ﾞ', '↳', '〶', '☛', '｢', '⁺', '『', '≫',
                       ]

        self.html_tags = ['<p>', '</p>', '<table>', '</table>', '<tr>', '</tr>', '<ul>', '<ol>', '<dl>', '</ul>', '</ol>',
                          '</dl>', '<li>', '<dd>', '<dt>', '</li>', '</dd>', '</dt>', '<h1>', '</h1>',
                          '<br>', '<br/>', '<strong>', '</strong>', '<span>', '</span>', '<blockquote>', '</blockquote>',
                          '<pre>', '</pre>', '<div>', '</div>', '<h2>', '</h2>', '<h3>', '</h3>', '<h4>', '</h4>', '<h5>', '</h5>',
                          '<h6>', '</h6>', '<blck>', '<pr>', '<code>', '<th>', '</th>', '<td>', '</td>', '<em>', '</em>']

        self.empty_expressions = ['&lt;', '&gt;', '&amp;', '&nbsp;', 
                                   '&emsp;', '&ndash;', '&mdash;', '&ensp;'
                                   '&quot;', '&#39;']

        self.spaces = ['\u200b', '\u200e', '\u202a', '\u2009', '\u2028', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\u3000', '\x10', '\x7f', '\x9d', '\xad',
                       '\x97', '\x9c', '\x8b', '\x81', '\x80', '\x8c', '\x85', '\x92', '\x88', '\x8d', '\x80', '\x8e', '\x9a', '\x94', '\xa0', 
                       '\x8f', '\x82', '\x8a', '\x93', '\x90', '\x83', '\x96', '\x9b', '\x9e', '\x99', '\x87', '\x84', '\x9f',
                      ]

        self.numbers = ["0","1","2","3","4","5","6","7","8","9","０","１","２","３","４","５","６","７","８","９"]

    def _pre_preprocess(self, x):
        return str(x).lower() 

    def rm_spaces(self, x):
        for space in self.spaces:
                x = x.replace(space, ' ')
        return x

    def clean_html_tags(self, x, stop_words=[]):      
        for r in self.html_tags:
            x = x.replace(r, '')
        for r in self.empty_expressions:
            x = x.replace(r, ' ')
        for r in stop_words:
            x = x.replace(r, '')
        return x

    def replace_num(self, x, use_num=True):
        if use_num:
            x = re.sub('[0-9]{5,}', '#####', x)
            x = re.sub('[0-9]{4}', '####', x)
            x = re.sub('[0-9]{3}', '###', x)
            x = re.sub('[0-9]{2}', '##', x)
            for i in self.numbers:
                x = x.replace(str(i), '#')
        else:
            x = re.sub('[0-9]{5,}', '', x)
            x = re.sub('[0-9]{4}', '', x)
            x = re.sub('[0-9]{3}', '', x)
            x = re.sub('[0-9]{2}', '', x)    
            for i in self.numbers:
                x = x.replace(str(i), '')        
        return x

    def clean_puncts(self, x):
        for punct in self.puncts:
            # x = x.replace(punct, f' {punct} ')
            x = x.replace(punct, '')
        return x

    def clean_text_jp(self, x):
        x = x.replace('。', '')
        x = x.replace('、', '')
        x = x.replace('\n', '') # 改行削除
        x = x.replace('\t', '') # タブ削除
        x = x.replace('\r', '')
        x = re.sub(re.compile(r'[!-\/:-@[-`{-~]'), ' ', x) 
        x = re.sub(r'\[math\]', ' LaTex math ', x) # LaTex削除
        x = re.sub(r'\[\/math\]', ' LaTex math ', x) # LaTex削除
        x = re.sub(r'\\', ' LaTex ', x) # LaTex削除
        x = re.sub(' +', ' ', x)
        return x

    def han_to_zen(self, x):
        word = []
        for x_s in x.split():
            x_s = mojimoji.han_to_zen(x_s)
            word.append(x_s)  
        return " ".join(word)

    def preprocess(self, sentence):
        sentence = sentence.fillna(" ")
        sentence = sentence.progress_apply(lambda x: self._pre_preprocess(x))
        sentence = sentence.progress_apply(lambda x: self.rm_spaces(x))
        sentence = sentence.progress_apply(lambda x: self.clean_puncts(x))
        sentence = sentence.progress_apply(lambda x: self.replace_num(x, use_num=False))
        sentence = sentence.progress_apply(lambda x: self.clean_html_tags(x))
        sentence = sentence.progress_apply(lambda x: self.clean_text_jp(x))
        sentence = sentence.progress_apply(lambda x: self.han_to_zen(x))
        return sentence

# tok = MecabTokenizer()
# textpreprocesser = TextPreprocessorJP()
