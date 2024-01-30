import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split

data = pd.read_csv('ad_verbalization_oversample.csv').drop_duplicates('video_id')
data['path']=data['video_id'].replace({int(f.split('_')[0]):f for f in os.listdir('videos')})

def convert_data(id, video_path, instruction, answer):
    pre = "<Video>VideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideoVideo</Video>"
    return {
        "id": id,
        "video": video_path,
        "conversations": [
            {
                "from": "human",
                "value": instruction.replace(pre, '').replace("0 to 99", "00 to 99") + "\n<image>\n"
            },
            {
                "from": "gpt",
                "value": str(answer).zfill(2)
            }
        ]
    }
train = data.apply(lambda x: convert_data(x.video_id, x.path, x.complete_prompt, x.score), axis=1).tolist()
train, test = train_test_split(train, test_size=0.1)
json.dump(train, open('lambda_train.json', 'w'), indent=4)
test_q = [{
    'video_name': q['video'],
    'question': q['conversations'][0]['value'],
    'question_id': q['id'],
}
    for q in test
]

test_a = [{
    'answer': q['conversations'][1]['value'],
    'question_id': q['id'],
}
    for q in test
]

json.dump(test_q, open("test_q.json", 'w'), indent=4)
json.dump(test_a, open("test_a.json", 'w'), indent=4)