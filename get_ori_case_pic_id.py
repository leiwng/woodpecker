import os
import openpyxl
import requests

url = 'https://workstation.kemoshen.com/api/workstation/case_picture_info?case_id=A2404070001&page=1&limit=3000&smart_sorting=false'
response = requests.get(url)
data = response.json()
print(data['data'])

list1=[]
for i in data['data']:
    dict1={}
    dict1['original_pic']=i['original_picture_file']
    dict1['analysis_score']=i['analysis_score']
    dict1['case_id']=i['original_picture_file'].split('.')[0]
    dict1['sample_id']=i['original_picture_file'].split('.')[1]
    dict1['count_score']=i['count_score']
    dict1['num']=i['num_score']
    dict1['distribution']=i['distribution_score']
    dict1['correct']=i['correct_score']
    dict1['length']=i['length_score']
    dict1['straight']=i['straight_score']
    dict1['clarity']=i['clarity_score']
    dict1['overlap']=i['overlap_times']
    list1.append(dict1)
print(list1)#原图和分析打分列表