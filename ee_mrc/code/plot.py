"""
Author: Xinrui Ma
Date: 2021-06-29 10:51:51
LastEditTime: 2021-06-29 19:26:51
Description: NLP homework, event extraction, data analysis
FilePath: \mrc\src\plot.py
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import json
from collections import defaultdict, Counter

from utils import read_by_lines

sns.set_theme()
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
sns.set(font='SimHei') 


def stat_cnt(data_path):
    lines = read_by_lines(data_path)
    class_cnt = defaultdict(int)
    event_type_cnt = defaultdict(int)
    role_cnt = defaultdict(int)

    for line in lines:
        data = json.loads(line.strip())
        for event in data["event_list"]:
            event_type_cnt[event["event_type"]] += 1
            class_cnt[event["class"]] += 1
            for argu in event["arguments"]:
                role_key = event["event_type"] + "-" + argu["role"]
                role_cnt[role_key] += 1

    # print("In {}, \n Class count: {}. \n".format(data_path, class_cnt))
    return class_cnt, event_type_cnt, role_cnt


def plot_count(data, title):
    df = pd.DataFrame.from_dict(data, orient='index',columns=["count"])
    _,ax = plt.subplots(figsize=(12,6))
    ax.barh(df.index,df["count"], height=0.5)
    ax.set_xlabel('数量')
    ax.set_title(title)
    for x,y in enumerate(df["count"]):
        ax.text(y,x,y,va='center',fontsize=14)
    path = "../output/" + title + ".png"
    plt.savefig(path,dpi=200)
    plt.show()

def plot_event_role():
    # a-事件大类  b-事件小类  c-论元角色
    a1, b1, c1 = stat_cnt("../input/train.json")
    a2, b2, c2 = stat_cnt("../input/test.json")
    plot_count(a1, '事件分布')

    cnt_evt_ = defaultdict(int)
    for k in b1:
        class_, evt_ = k.split('-')
        if class_ == "竞赛行为":
            cnt_evt_[evt_] = b1[k]
    plot_count(cnt_evt_, '竞赛行为事件子类型分布')

    cnt_role_ = defaultdict(int)
    for k in c1:
        class_, evt_, role_ = k.split('-')
        if class_+"-"+ evt_== "竞赛行为-胜负":
            cnt_role_[role_] = c1[k]
    plot_count(cnt_role_, '竞赛行为-胜负-论元角色分布')


def cnt_sent_len(data_path):
    lens_cnt = []
    lines = read_by_lines(data_path)
    for l in lines:
        l = json.loads(l)
        lens_cnt.append(len(l["text"]))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(lens_cnt, bins=20,density=False)
    for rect in ax.patches:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
        xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 
    plt.xlabel('长度')
    plt.ylabel('数量')
    plt.title('句子长度统计')
    path = "../output/" + "句子长度统计" + ".png"
    plt.savefig(path,dpi=200)
    plt.show()
    # print(Counter(lens_cnt))
    print("句子长度，最长：",max(lens_cnt),"最短：", min(lens_cnt))


def len_argu(path):
    lines = read_by_lines(path)
    argum_len = []
    for line in lines:
        data = json.loads(line.strip())
        for event in data["event_list"]:
            for argu in event["arguments"]:
                argum_len.append(len(argu["argument"]))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(argum_len, bins=20,density=False)
    for rect in ax.patches:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
        xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 
    plt.xlabel('长度')
    plt.ylabel('数量')
    plt.title('论元长度统计')
    path = "../output/" + "论元长度统计" + ".png"
    plt.savefig(path,dpi=200)
    plt.show()
    print("论元长度，最长：",max(argum_len),"最短：", min(argum_len))


def sent_event_cnt(data_path):
    lines = read_by_lines(data_path)
    cnt = []
    for line in lines:
        l = json.loads(line.strip())
        cnt.append(len(l["event_list"]))
    print("句子中事件数量统计：",Counter(cnt))
# Counter({1: 10400, 2: 1292, 3: 193, 4: 46, 5: 16, 6: 4, 8: 3, 7: 2, 15: 1, 11: 1})

def sent_role_cnt(data_path):
    lines = read_by_lines(data_path)
    cnt = []
    for line in lines:
        l = json.loads(line.strip())
        t = sum([len(e["arguments"]) for e in l["event_list"]])
        cnt.append(t)
    print("句子中角色数量统计：", Counter(cnt))
    countnt = Counter(cnt)
    s5, s10 = 0, 0
    for k in countnt:
        if 5 <= k <= 10:
            s5 += countnt[k]
        elif k > 10:
            s10 += countnt[k]
    print("其中，5~10", s5, ">10", s10)
# Counter({2: 4507, 1: 2863, 3: 2459, 4: 968, 0: 305, 5: 290, 6: 240, 7: 137, 8: 68, 9: 33, 10: 26, 11: 13, 12: 12, 15: 9, 16: 6, 14: 4, 13: 4, 18: 4, 20: 3, 23: 2, 19: 1, 60: 1, 17: 1, 44: 1, 32: 1})


# 角色重叠、共享
def count_more_role():
    data_path = "../input/test.json"
    lines = read_by_lines(data_path)
    more = 0
    for line in lines:
        l = json.loads(line.strip())
        role = defaultdict(int)
        for event in l["event_list"]:
            for argu in event["arguments"]:
                role[argu["argument"]] += 1
        for k in role:
            if role[k] > 1:
                more += 1
                break
    print("角色重叠、共享: ", more) # 1120


# 论元重叠
def count_overlap_argument():
    data_path = "../input/test.json"
    lines = read_by_lines(data_path)
    overlap = 0
    for line in lines:
        l = json.loads(line.strip())
        argu_set = set()
        for event in l["event_list"]:
            for argu in event["arguments"]:
                argu_set.add(argu["argument"])
        al = sorted(list(argu_set))
        for i in range(len(al)):
            for j in range(i + 1, len(al)):
                if al[i] in al[j] or al[j] in al[i]:
                    overlap += 1
                    print(line)
                    # print(al[i], al[j])
    print("论元重叠: ", overlap) # 862


def plot_pie_overlap():
    fig,axes=plt.subplots(1,2,figsize=(8,4))#创建画布
    labels=['有','无']
    x1=[1120, 10838]  # 角色重叠
    x2=[862, 11096]   # 论元重叠
    colors=['indianred','lightsteelblue']#每块对应的颜色
    explode=(0.05,0.05)#将每一块分割出来，值越大分割出的间隙越大
    axes[0].pie(x1,
            colors=colors,
            labels=labels,
            explode=explode,
            autopct='%.2f%%', #数值设置为保留固定小数位的百分数
            shadow=False,#无阴影设置
            startangle=90,#逆时针起始角度设置
            pctdistance=0.5,#数值距圆心半径背书距离
            labeldistance=0.8#图例距圆心半径倍距离
        )
    axes[0].axis('equal')#x,y轴刻度一致，保证饼图为圆形
    axes[0].legend(loc='best')
    axes[0].set_title('角色重叠有/无')

    axes[1].pie(x2,colors=colors,labels=labels,explode=explode,autopct='%.2f%%',shadow=False,startangle=90,pctdistance=0.5,labeldistance=0.8)
    axes[1].axis('equal')
    axes[1].set_title('论元重叠有/无')
    axes[1].legend(loc='best')
    fig.savefig('../output/角色重叠论元重叠.jpg',dpi=200)
    plt.show()


def main():
    plot_event_role()  # 事件类型，大类、小类、论元
    
    path = "../input/train.json"
    cnt_sent_len(path)  # 句子长度
    len_argu(path)

    sent_event_cnt(path)  # 句子中事件数量

    sent_role_cnt(path)  # 句子中角色数量

    count_more_role()
    count_overlap_argument()
    plot_pie_overlap()  # 角色重叠论元重叠


if __name__ == "__main__":
    # main()
    count_overlap_argument()

    
