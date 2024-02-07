# 假设这是你的字典列表
dict_list = [
    {'chromo_id': 2, 'chromo_idx': 10, 'other_key': 'value1'},
    {'chromo_id': 1, 'chromo_idx': 11, 'other_key': 'value2'},
    {'chromo_id': 2, 'chromo_idx': 9, 'other_key': 'value3'},
    {'chromo_id': 1, 'chromo_idx': 12, 'other_key': 'value4'}
]

# 使用sorted函数和lambda表达式进行排序
sorted_list = sorted(dict_list, key=lambda x: (x['chromo_id'], x['chromo_idx']))

# 打印排序后的列表
for item in sorted_list:
    print(item)
