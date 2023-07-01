#data preprocess

import os
import json
import logging
import numpy as np
import pandas as pd

def preprocess(dir):



    train_data = []
    with open(dir, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
        for json_object in data:
            text = json_object['text']
            label_entities = json_object.get('entities', None)
            sc = json_object['sc']
            con = json_object.get('con')


            if con=='0':
                type_name = ""
                type_name = ':無い;' # はない　ない　Null :;
                #  Null也作为Verbalizer的一个标签
                #  type_name = ':Null;'
                    
            elif con=='1':
                type_name = ""
                type_all = ""
                for object in label_entities:
                    # print(object)
                    # print(5)
                    # for name, span, type in object:
                    name = object.get('name')
                    type = object.get('type')
                    line = f"：{type}；{name}"
                    # line = f"{name}は{type}である。"
                    # print(line)
                    type_name += line
                    #在source里仅插入已有NER Label
                    # type1 = f"{type},"
                    # type_all += type1
                    # print(type_name)
                    # data_line_enti = [data_line]
                type_name = f"{type_name}"
            
            
            # prompt template
            # sc_all_label = '社会、文芸、学問、技術、自然'
            # ner_all_label = '人名、法人名、政治的組織名、そのほかの組織名、地名、施設名、イベント名'
            # line_sc_text = f"{text}類別:社会;:文芸;:学問;:技術;:自然;{text}:人名;:法人名;:政治的組織名;:そのほかの組織名;:地名;:施設名;:イベント名;:無い;固有表現抽出,,類別:{sc};固有表現抽出{type_name}"
            # line_sc_text = f"{text}類別:社会;:文芸;:学問;:技術;:自然;{text}固有表現抽出:人名;:法人名;:政治的組織名;:そのほかの組織名;:地名;:施設名;:イベント名;:無い;,,類別:{sc};固有表現抽出{type_name}"
            # line_sc_text = f"{text}:社会;:文芸;:学問;:技術;:自然;{text}:人名;:法人名;:政治的組織名;:そのほかの組織名;:地名;:施設名;:製品名;:イベント名;:無い;,,:{sc};固有表現抽出{type_name}"
            # line_sc_text = f"{text}:人名;:法人名;:政治的組織名;:そのほかの組織名;:地名;:施設名;:製品名;:イベント名;:無い;固有表現抽出,,固有表現抽出{type_name}"
            # line_sc_text = f"{text}<社会><文芸><学問><技術><自然>,,<{sc}>"
            
            # singlepromptword
            # line_sc_text = f"センテンス:{text},,類別:{sc};固有表現抽出{type_name}"#

            #simple
            # line_sc_text = f"{text},,:{sc};{type_name}"
            # print(line_sc_text)


            train_data.append(line_sc_text)
    return train_data


train_data = preprocess('ner-wikipedia-dataset-main/ner.json')
pd = pd.DataFrame(train_data)



pd.to_csv("translated_format_dataset/SCNM dataset.csv", index = None)
