'''
http://blog.mwsoft.jp/article/113600124.html

pandasでカラムサイズが一定でないcsv/tsvを読み込む

'''

import pandas as pd

df = pd.read_csv( 'unstable_column_size.tsv', sep='\t', header=False )
  #=> CParserError: Error tokenizing data. C error: Expected 3 fields in line 4, saw 8


# これをカラム名にする
col_names = [ 'c{0:02d}'.format(i) for i in range(10) ]
  #=> ['c00', 'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09']

# 読んでみる
df = pd.read_csv( 'foo.tsv', sep='\t', names=col_names )
  #=>       c00    c01     c02    c03     c04    c05     c06     c07  c08  c09
  #=> 0  105497    NaN     NaN    NaN     NaN    NaN     NaN     NaN  NaN  NaN
  #=> 1  101731  90359  107575    NaN     NaN    NaN     NaN     NaN  NaN  NaN
  #=> 2  105320  76175   96971  95604  109100  72563  105730  109194  NaN  NaN
  #=> 3   96971  95604     NaN    NaN     NaN    NaN     NaN     NaN  NaN  NaN