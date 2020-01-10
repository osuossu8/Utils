import regex

class JapaneseCharactorExtractor(object):
    def __init__(self, moji):
        self.moji = moji
        if self.moji == 'kanji':
            self.rgx = r'\p{Han}+'
        elif self.moji == 'katakana':
            self.rgx = r'\p{Katakana}+'
        elif self.moji == 'hiragana':
            self.rgx = r'\p{Hiragana}+'
        else:
            raise Exception

    def _get_len_percentage(self, list_a, list_b):
        return (len(list_a)/len(list_b)) * 100

    def generate_features(self, sentence):
        moji_list = regex.findall(self.rgx, sentence)
        percentage_moji = get_len_percentage(moji_list, sentence)
        return moji_list, percentage_moji


sentence = "私かえるるる！夜更かしが得意でネトゲ好きな普通の高校生。"

knj_e = JapaneseCharactorExtractor("kanji")
ktkn_e = JapaneseCharactorExtractor("katakana")
hrgn_e = JapaneseCharactorExtractor("hiragana")

print(knj_e.generate_features(sentence))
print(ktkn_e.generate_features(sentence))
print(hrgn_e.generate_features(sentence))

"""
(['私', '夜更', '得意', '好', '普通', '高校生'], 21.428571428571427)
(['ネトゲ'], 3.571428571428571)
(['かえるるる', 'かしが', 'で', 'きな', 'の'], 17.857142857142858)
"""
