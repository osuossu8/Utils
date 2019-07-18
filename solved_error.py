
# ============
# Error 1
# ============

'''
# Raised Error
TypeError: sequence item 0: expected str instance, int found

# Situation
[文章, 文章, 文章, ・・・] を "".join([list]) でひとつの文章列に結合しようとした。

# Solution
int が混じっていると error になるので, string 化する map 関数を適用
'''

# Example

a = [1,2,3,4]

print("".join(map(str, a)))

# => 1234




