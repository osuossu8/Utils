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