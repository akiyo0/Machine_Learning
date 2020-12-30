from collections import defaultdict

words = ['hello', 'world', 'nice', 'world']
counter = dict()

for word in words:
    #counter[word] += 1 会报KeyError, 累加前未初始化
    if word in counter:
        counter[word] += 1
    else:
        counter[word] = 1
print(counter)

# 或者
for word in words:
    counter.setdefault(word, 0)
    counter[word] += 1

# 亦或者
#当所访问的键不存在的时候, 可以实例化一个值作为默认值
counter = defaultdict(lambda: 0)
for word in words:
    counter[word] += 1

print(counter)