import pickle as pkl
import numpy as np
def get_file(exp_id,ifcontrast):
    data_path='../save/HER-'+str(exp_id)+'/model_outputs'
    if ifcontrast:
        preds=pkl.load(open(data_path+'/contrast_mnli_contrast/test_preds_contrast.pkl','rb'))
        target=pkl.load(open(data_path+'/contrast_mnli_contrast/test_targets_contrast.pkl','rb'))
    else:
        preds=pkl.load(open(data_path+'/contrast_mnli_original/test_preds.pkl','rb'))
        target=pkl.load(open(data_path+'/contrast_mnli_original/test_targets.pkl','rb'))
    return preds, target

import json
map_dict=json.load(open('../data/contrast_mnli/id_matching.json','r'))

def metrics(ori_preds,ori_target,con_preds,con_target):
    count=0
    total=len(ori_preds)
    for i in range(len(ori_preds)):
        map=map_dict[str(i)]
        flag=True
        if ori_preds[i]==ori_target[i]:
            for j in map:
                if  con_preds[j]!=con_target[j]:
                    flag=False
            if flag==True:
                count+=1
    return count/total
# 8175 8181 8185 8196 8200 8204 8213 8216 8219 
ori_ids=[4184 ,4190, 4193, 4186, 4191, 4194, 4189, 4192, 4195]

consistency=[]
for i in range(len(ori_ids)):
    ori_preds,ori_target=get_file(ori_ids[i],False)
    con_preds,con_target=get_file(ori_ids[i],True)
    print(ori_target[0],con_target[0])
    consistency.append(metrics(ori_preds,ori_target,con_preds,con_target))
print(np.mean(consistency))
mean1=np.mean(consistency[:3])
mean2=np.mean(consistency[3:6])
mean3=np.mean(consistency[6:])
print(np.std([mean1,mean2,mean3],ddof=1))


