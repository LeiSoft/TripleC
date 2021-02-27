import json

datas = json.load(open("./datasets/D08-1001.json", 'r', encoding='utf-8'))

for i in datas['sections']:
    print(i)