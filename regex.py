import re

# ==================
# url match pattern
# ==================

title_list = query_df.fillna("")[0].values.tolist()[:10000]

temp_title_list = []
urls = []

for line in title_list:
    match = re.search(r'(https?://[a-zA-Z0-9.-]*)', line)
    if match:
            #url = match.group(1)
            #line = line.replace(url, '')
          urls.append(line)
          continue
    temp_title_list.append(line)
    
temp_title_list[:15]


# ========================================
# specific words alignmenmt match pattern
# ========================================

txt = '桐生信金、芙蓉総合リースと業務提携'
txt = 'トヨタ、北京汽車と提携'
txt = '工機ホールディングス×助太刀 出資と戦略提携について'

match = re.search(r'(と(..)?提携)', txt)

if match:
    print(txt)


# ========================================
# remove urls and kakko
# ========================================

def remove_urls_and_kakko(text):
    '''
    * url や 括弧(内の語も含む) を取り除く
    * how to use
    df['text_column'] = df['text_column'].apply(remove_urls_and_kakko)
    '''

    text = re.sub(r'(https?://[a-zA-Z0-9.-]*)', r'', text)
    
    # 全角括弧
    text = re.sub(r'\（.+?\）', r'', text)
    
    # 【】
    text = re.sub(r'\【.+?\】', r'', text)
    return text