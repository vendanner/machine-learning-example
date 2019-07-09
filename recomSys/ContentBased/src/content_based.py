#-*-coding:utf-8-*-

"""
基于内容的召回：用户画像
    给item 打标签
    根据用户消费的item，给用户打标签
    根据用户标签和item 标签，给用户推荐item
"""
import os
import ready

def get_user_profile(item_cate_ratio,file_path,score_thr = 4.0,cate_num=2):
    """
    根据用户消费的item，描绘用户画像
    本例无近期和长期兴趣画像
    :param item_cate_ratio:
    :param file_path: ratings
    :param score_thr: 用户喜爱的阈值
    :param cate_num: 用户选取几个cate
    :return:dict {user_id:[cate1,cate2]}
    """
    if not os.path.exists(file_path):
        print(file_path+" no exist")
        return {}

    first_flag = 0
    records = {}
    user_profile = {}
    with open(file_path) as f:
        for line in f.readlines():
            if first_flag == 0:
                first_flag += 1
                continue
            words = line.strip().split(',')
            if len(words) < 4:
                continue
            user_id,item_id,rating,timestamp = words[0],words[1],float(words[2]),float(words[3])
            if rating < score_thr:
                continue
            if item_id not in item_cate_ratio:
                continue

            time_rate = get_time_rate(timestamp)
            if user_id not in records:
                records[user_id] = {}
            for cate in item_cate_ratio[item_id]:
                if cate not in records[user_id]:
                    records[user_id][cate] = 0
                records[user_id][cate] += rating * time_rate * item_cate_ratio[item_id][cate]
    for user_id in records:
        user_profile[user_id] = [arr[0] for arr in sorted(records[user_id].items(),key=lambda item:item[1],reverse = True)[:cate_num]]

    return user_profile

def recom(cate_item,user_profile,user_id,topK = 10):
    """
    根据用户画像 cate，推对应cate 下的 item
    :param cate_item:
    :param user_id:
    :param topK: 每个cate 推荐的item数目
    :return: [item1,item2...]
    """
    if user_id not in user_profile:
        return []
    recom_result = []
    for cate in user_profile[user_id]:
        if cate not in cate_item:
            continue
        recom_result += cate_item[cate][:topK]

    return recom_result


def get_time_rate(timestamp):
    """
    时间戳比例，观看时间越早，打分权重占比越小
    :param timestamp: 观看时间戳
    :return:
    """
    fix_time_stamp = 1476086345
    total_sec = 24*60*60
    delta = (fix_time_stamp - timestamp)/total_sec/100
    return round(1/(1+delta), 3)

if __name__ == "__main__":
    print("begin")
    ave_score = ready.get_ave_score('../data/ratings.txt')
    # print(ave_score['1'])
    item_cate_ratio, cate_item =ready.get_item_cate(ave_score,'../data/movies.txt')
    # for cate in cate_item:
    #     print(cate)
    #     print(cate_item[cate])
    #     break
    user_id = '4'
    user_profile = get_user_profile(item_cate_ratio,'../data/ratings.txt')
    print(user_profile[user_id])
    result = recom(cate_item,user_profile,user_id)
    for item_id in result:
        print(item_id,item_cate_ratio[item_id])