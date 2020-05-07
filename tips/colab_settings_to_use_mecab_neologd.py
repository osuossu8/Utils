# mecab settings
!apt install mecab
!apt install libmecab-dev
!apt install mecab-ipadic-utf8
!apt install file
!pip install mecab-python3

# neologd settings
!git clone https://github.com/neologd/mecab-ipadic-neologd.git
%cd mecab-ipadic-neologd
!bin/install-mecab-ipadic-neologd -y

# not have to
!cat /etc/mecabrc

%%writefile /etc/mecabrc

;
; Configuration file of MeCab
;
; $Id: mecabrc.in,v 1.3 2006/05/29 15:36:08 taku-ku Exp $;
;
dicdir = /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd

; userdic = /home/foo/bar/user.dic

; output-format-type = wakati
; input-buffer-size = 8192

; node-format = %m\n
; bos-format = %S\n
; eos-format = EOS\n


import MeCab


class MecabTokenizer(object):
    def __init__(self, mode=None):
        self.wakati = MeCab.Tagger('-Owakati')
        if mode == 'neologd':
            self.dic_dir = '/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd'
            self.wakati = MeCab.Tagger(f"-d {self.dic_dir}")
        self.wakati.parse('')

    def tokenize(self, line):
        return self.wakati.parse(line).strip().split(" ")
    
    def mecab_tokenizer(self, line):
        node = self.wakati.parseToNode(line)
        keywords = []
        while node:
            if node.feature.split(",")[0] == "名詞":
                keywords.append(node.surface)
            node = node.next
        if len(keywords) == 0:
            keywords.append(" ")
        return keywords 


MT = MecabTokenizer(mode=None)
print(MT.mecab_tokenizer("農林水産業"))
# ['農林', '水産', '業']

MT = MecabTokenizer(mode='neologd')
print(MT.mecab_tokenizer("農林水産業"))
# ['農林水産業']

