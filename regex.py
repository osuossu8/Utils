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
