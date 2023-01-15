import os
from utils.example import Example


def cal_distance(string1, string2):
    # ref: https://blog.csdn.net/qq_22583833/article/details/80197716
    if len(string1) > len(string2):
        string1, string2 = string2, string1
    str1_length = len(string1) + 1
    str2_length = len(string2) + 1
    distance_matrix = [list(range(str2_length)) for _ in range(str1_length)]
    for i in range(1, str1_length):
        for j in range(1, str2_length):
            deletion = distance_matrix[i - 1][j] + 1
            insertion = distance_matrix[i][j - 1] + 1
            substitution = distance_matrix[i - 1][j - 1]
            if string1[i - 1] != string2[j - 1]:
                substitution += 1
            distance_matrix[i][j] = min(insertion, deletion, substitution)
    return distance_matrix[str1_length - 1][str2_length - 1]


def correct(pred, label_vocab):
    new_pred = []
    for slot_value in pred:
        asv = slot_value.split('-')
        if len(asv) == 2:
            new_pred.append(slot_value)
        if len(asv) == 3:
            act, sl, val = asv
            choice = label_vocab.ontology["slots"][sl]
            if not isinstance(choice, list):
                with open(os.path.join('./data', choice), "r", encoding='utf-8') as f:
                    choice = f.read().splitlines()
            if val in choice:
                new_pred.append(slot_value)
            else:
                best_dis = 999999
                for c in choice:
                    dis = cal_distance(val, c)
                    if best_dis > dis:
                        best_dis = dis
                        new_val = c
                if best_dis * 2 < len(val):
                    new_pred.append('-'.join([act, sl, new_val]))
                else:   # 假如最佳匹配也相差过多，直接舍去这一项/保留原来的值
                    new_pred.append(slot_value)   # 保留原来的值
                    # pass    # 舍去这一项
    return new_pred


if __name__ == '__main__':
    Example.configuration('../data', train_path='../data\\train.json', word2vec_path='../word2vec-768.txt')
    pred_tuple = ['inform-操作-导航', 'inform-终点名称-哈尔滨医科大学附属']
    pred_tuple = correct(pred_tuple, Example.label_vocab)