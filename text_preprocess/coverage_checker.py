import logging

class CoverageChecker:

    def __init__(self, df1, df2, df3, keywords):
        self.keywords = keywords

        self.arr1 = df1[df1.target==1].body_meishi.values
        self.unique_words1 = list(set(' '.join(self.arr1).split()))

        self.arr2 = df2[df2.target==1].body_meishi.values
        self.unique_words2 = list(set(' '.join(self.arr2).split()))

        self.arr3 = df3[df3.target==1].body_meishi.values
        self.unique_words3 = list(set(' '.join(self.arr3).split()))

        self.coverage1 = self.get_coverage(self.keywords, self.unique_words1)[0]
        self.word_diff1 = self.get_coverage(self.keywords, self.unique_words1)[1]

        self.coverage2 = self.get_coverage(self.keywords, self.unique_words2)[0]
        self.word_diff2 = self.get_coverage(self.keywords, self.unique_words2)[1]

        self.coverage3 = self.get_coverage(self.keywords, self.unique_words3)[0]
        self.word_diff3 = self.get_coverage(self.keywords, self.unique_words3)[1]

        logging.info(f'train keywords coverage : {self.coverage1}, train word_diff : {self.word_diff1}')
        logging.info(f'valid keywords coverage : {self.coverage2}, valid word_diff : {self.word_diff2}')
        logging.info(f'test keywords coverage : {self.coverage3}, test word_diff : {self.word_diff3}')
        logging.info(f'train valid coverage : {self.get_coverage(self.unique_words1, self.unique_words2)[0]}')
        logging.info(f'train test coverage : {self.get_coverage(self.unique_words1, self.unique_words3)[0]}')


    def get_coverage(self, w1, w2):
        l1 = len(w1)
        s1 = set(w1)
        s2 = set(w2)

        inter = list(s1 & s2)
        return (len(inter) / l1), list(set(w1)-set(inter))


# CC = CoverageChecker(train, valid, test, keywords)
