import numpy as np
import math

def cond_ent_cal(x_list=[], y_list=[]):
'''条件熵计算'''
    x_nparr = np.array(x_list)
    y_nparr = np.array(y_list)
    x_num = np.unique(x_nparr)
    y_num = np.unique(y_nparr)
    cond_prob_list = []
    cond_ent_list = []
    prob_list = []
    for p in x_num:
        cond_prob_list = []
        cond_prob_1 = np.sum(y_nparr[np.where(x_nparr == p)])/np.where(x_nparr == p)[0].size
        cond_prob_list.append(cond_prob_1)
        cond_prob_list.append(1-cond_prob_1)
        cond_prob_nparr = np.array(cond_prob_list)
        cond_prob_nparr[cond_prob_nparr == 0] = 0.000001
#         print("cond_prob_nparr:", cond_prob_nparr)
        cond_ent_list.append(np.sum(-np.array(cond_prob_nparr)*np.log2(cond_prob_nparr)))
    return cond_ent_list
    
    
def pos_pro_cal(property_list=[]):
'''后验概率计算'''
    total = len(property_list)
    props = []
    pos_pro = []
    for p in property_list:
        if not props.__contains__(p):
            pos_pro.append(property_list.count(p)/total)
            props.append(p)
    return pos_pro
    
    
def ent_cal(sample_list = []):
'''熵计算'''
    prop_list_2 = []
    result_list = []
    p_prob_list_2 = []
    r_prob_list = []
    cond_ent_list_2 = []
    cond_ent_list = []
    prob_list = []
    for p, r in sample_list:
        prop_list_2.append(p)
        result_list.append(r)
    r_prob_list = pos_pro_cal(result_list)
    prop_list = np.array(prop_list_2).transpose()
    for props in prop_list:
        p_prob_list_2.append(pos_pro_cal(list(props)))
        cond_ent_list_2.append(cond_ent_cal(list(props), result_list))
#     print("p_prob_list_2:")
#     print(p_prob_list_2)
#     print("cond_ent_list_2:")
#     print(cond_ent_list_2)
    for i in range(len(p_prob_list_2)):
        cond_ent_list.append(np.sum(np.array(p_prob_list_2[i])*np.array(cond_ent_list_2[i])))
    r_num = np.unique(np.array(result_list))
    y_nparr = np.array(result_list)
    for r in r_num:
        prob_temp = np.where(y_nparr == r)[0].size/y_nparr.size
        prob_list.append(prob_temp)
    prob_nparr = np.array(prob_list)
    prob_nparr[prob_nparr==0.0] = 0.00001
#     print("prob_nparr:",prob_nparr)
    y_ent = np.sum(-prob_nparr*np.log2(prob_nparr))
    return cond_ent_list, y_ent
    

def tree_gen(sample=None, eps=0.2, tree=None):
'''ID3方法生成决策树'''
    if sample is None:
        sample = []
    if tree is None:
        tree = []
    cond_ent_list, ent = ent_cal(sample)
    cond_ent_arr = np.array(cond_ent_list)
    sample_nparr = np.array(sample)
    class_nparr = np.array(sample).transpose()[-1]
    x_nparr = np.array(list(sample_nparr[:len(sample), 0:1][0:len(sample),0]))
    min_ent_index = np.where(cond_ent_arr == np.min(cond_ent_arr))[0][0]
    min_ent_feature = x_nparr.transpose()[min_ent_index]
    min_ent_feature_uniq = np.unique(min_ent_feature)
    tree_temp = []
    for f in min_ent_feature_uniq:
        np.count_nonzero(class_nparr[(np.where(min_ent_feature == f))])/len(class_nparr)
        tree_temp.append((1 if np.count_nonzero(class_nparr[(np.where(min_ent_feature == f))])/len(class_nparr[(np.where(min_ent_feature == f))]) > 0.5 else 0, f))
    tree_temp.append(min_ent_index)
    tree.append(tree_temp)
    if np.min(cond_ent_arr) < eps:
        # 停止条件成立，返回特征栏位索引和特征对应的类
        return tree
    else:
        sample_temp = []
        sample_next = []
        for f in min_ent_feature_uniq:
            sample_temp.append(np.array(sample)[np.where(x_nparr.transpose()[min_ent_index] == f)].tolist())
            for _s in sample_temp[0]:
                sample_next.append((_s[0], _s[1]))
            tree_gen(sample_next,tree = tree)
    return tree
