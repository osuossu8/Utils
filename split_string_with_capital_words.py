
import re

def split_string_with_capital_words(text):
      '''
      大文字で連結された単語を大文字で区切る.
      InTheMoney --> In The Money
      '''
      splitted = re.split('[A-Z]', text)[1:]
      uppers = [c for c in text if c.isupper()]
      length = len(uppers)
      
      words = []
      for i in range(length):
          s = uppers[i] + splitted[i]
          words.append(s)
          
      return " ".join(words)