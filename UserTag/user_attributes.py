import pandas as pd
from wordcloud import WordCloud
import json
import matplotlib.pyplot as plt

'''
事实标签（微博直接爬取）=用户粉丝量follower，转发reports，评论comments，点赞likes，博文内容text，博文关键词key_words，用户合并标签hobby
模型标签（自定义规则计算得到）=1影响力（用户粉丝量）2博文热度（转评赞总数）3话题专业度（博文内容长度）4话题情感（博文内容情感识别）5用户兴趣领域（源自用户合并标签）
预测标签（用于用户行为、偏好预测）=1话题带动力（模型标签1+2+4）2话题持续关注度（模型标签3+4+5）
根据预测标签可以画散点图（所有用户的预测标签）
    1高2高的属于重点用户，这些用户对AI生成视频很感兴趣，且可以带动更多人了解相关话题
    1高2低的属于营销用户，可以收买这些用户，利用他们的话题带动力，推广AI生成视频
    1低2高的属于忠诚用户，虽然不能带动其他用户，但本身可以给AI生成视频话题持续带来热度
    1低2低的属于获取用户，要通过各种手段增加他们对AI生成视频的兴趣
'''
#爬取数据的读取路径
user_path = 'E:\\#电子商务系统\\users.csv'
content_path = 'E:\\#电子商务系统\\emotion.csv'
#emotion_path = 'E:\\#电子商务系统\\emotion.csv'

#数据预处理 主要是把users和content对应起来放在一个df里 以及把事实标签‘用户擅长领域’提取出来
user_df = pd.read_csv(user_path)
content_df = pd.read_csv(content_path)

follower = []
for item in user_df.iloc[:,2]:
    if '万' in item:
        temp = item.replace('万','')
        follower.append(float(temp)*10000)
    elif '亿' in item:
        temp = item.replace('亿','')
        follower.append(float(temp)*100000000)
    else:
        follower.append(int(item))

hobby = []
for item in user_df.iloc[:,4]:
    hlis = []
    for x in item.split(','):
        if 'V指数' in x:
            hlis.append(x.split(' ')[1])
        elif '博主' in x:
            hlis.append(x.split('博主')[0].strip(' \'[]'))
    if len(hlis) == 0:
        hlis.append('普通用户')
    hobby.append(hlis)

user_name = list(user_df.iloc[:,1])
reports = []
comments = []
likes = []
comm_text = []
key_words_text = []
time = []
emo_color = ['black','red','purple','orange','blue','green']
emotion = []
emotion_color = []
emo_name = ['neutral','anger','fear','joy','sadness','surprise']
non_index = [1658,2922,5759,4395,4929,4946,6272,3177,5202,3817,3970,3420,2362]
flag = 0
for url in user_df.iloc[:,0]:
    con_index = content_df.index[content_df['user']==url].to_list()
    if len(con_index) != 0:
        reports.append(content_df.iloc[con_index[0],9])
        comments.append(content_df.iloc[con_index[0],10])
        likes.append(content_df.iloc[con_index[0],11])
        comm_text.append(content_df.iloc[con_index[0],6])
        key_words_text.append(content_df.iloc[con_index[0],0])
        time.append(content_df.iloc[con_index[0],3])
        emotion.append(content_df.iloc[con_index,14].values[0])
        emotion_color.append(emo_color[emo_name.index(content_df.iloc[con_index,13].values[0])])
    else:
        reports.append(content_df.iloc[flag,9])
        comments.append(content_df.iloc[flag,10])
        likes.append(content_df.iloc[flag,11])
        comm_text.append(content_df.iloc[flag,6])
        key_words_text.append(content_df.iloc[flag,0])
        time.append(content_df.iloc[flag,3])
        emotion.append(content_df.iloc[flag,14])
        emotion_color.append(emo_color[emo_name.index(content_df.iloc[flag,13])])
        flag += 1
    
# 通过传递字典创建DataFrame
factual_label_data = {'user_name':user_name, 'followers':follower,'reports':reports,\
                      'comments':comments,'likes':likes,'text':comm_text,'key_words':key_words_text,\
                      'time':time,'user_hobby':hobby,'emo_label':emotion_color,'emotion':emotion}
df = pd.DataFrame(factual_label_data)
#df.to_csv('E:\\#电子商务系统\\factual_label_data.csv',encoding='utf-8')
#print(df)

'''
user_hobby用户领域标签
输入=df 
对所有用户的兴趣领域进行频率统计
输出=所有用户兴趣领域的词云图分布，所有用户兴趣领域的字典hobby_dict
'''
def hobby_wordcloud(df):
    hobby_dict = {}
    for item in df.iloc[:,8]:
        for area in item:
            if area in hobby_dict:
                hobby_dict[area] += 1
            else:
                hobby_dict[area] = 1
    '''
    frequencies = {key:value for key, value in hobby_dict.items()}
    wc = WordCloud(background_color='white', width=1200, height=900, font_path='simhei.ttf').generate_from_frequencies(frequencies)
    plt.imshow(wc,interpolation='bilinear')
    plt.axis('off')'''
    #plt.savefig('E:\\#电子商务系统\\user_hobby_wordcloud.png',dpi=300)
    #with open('E:\\#电子商务系统\\user_hobby_dict.txt','w',encoding='utf-8') as f:
        #json.dump(hobby_dict,f)
    return hobby_dict

hobby_dict = hobby_wordcloud(df)
#print(hobby_dict)

'''
模型标签1用户影响力=事实标签粉丝量
df中的followers列（float）

模型标签2博文热度=事实标签转发+事实标签评论+事实标签点赞
df中的heat列(float)

模型标签3话题专业度=事实标签博文内容的长度
df中的professionalism列（float）

模型标签4话题情感=事实标签博文内容提取情感关键词，再给情绪赋予权重
df中的emo列（float）

模型标签5用户兴趣领域权重=用户事实标签用户兴趣领域在dict_hobby中的频率
df中的hobby_weight列（float）
'''
followers_lis = []
heat_lis = []
text_length_lis = []
emotion_lis = []
hobby_lis = []
for user_num in range(df.shape[0]):
    followers_lis.append(df.iloc[user_num,1])
    heat_lis.append(df.iloc[user_num,2]+df.iloc[user_num,3]+df.iloc[user_num,4])
    text_length_lis.append(len(df.iloc[user_num,5]))
    emotion_lis.append(df.iloc[user_num,10])
    hobby_lis.append(hobby_dict[df.iloc[user_num,8][0]])

df['influence'] = followers_lis
df['heat'] = heat_lis
df['professionalism'] = text_length_lis
df['emo'] = emotion_lis
df['hobby_weight'] = hobby_lis
#df.to_csv('E:\\#电子商务系统\\model_label_data.csv',encoding='utf-8')
#print(df)

'''
所有用于计算预测标签的模型标签应该先归一化，消除不同模型标签之间的量纲

每个用户的预测标签1和预测标签2都需要计算
01
预测标签1话题带动力=influence+heat+emo
预测标签2话题持续关注度=professionalism+hobby_weight+emo
02
预测标签1话题带动力=influence+heat+emo
预测标签2话题持续关注度=professionalism+hobby_weight
03
预测标签1话题带动力=influence+heat
预测标签2话题持续关注度=professionalism+hobby_weight+emo
04
预测标签1话题带动力=influence+heat+emo
预测标签2话题持续关注度=professionalism+hobby_weight+heat
05
预测标签1话题带动力=influence+heat+emo
预测标签2话题持续关注度=professionalism+hobby_weight+heat+emo
06
预测标签1话题带动力=influence+heat+emo+hobby_weight
预测标签2话题持续关注度=professionalism+hobby_weight+heat

根据预测标签1和预测标签2绘制散点图
'''
normalize = lambda x: (x - x.min()) / (x.max() - x.min())
df[['influence','heat','professionalism','emo','hobby_weight']] = df[['influence','heat','professionalism','emo','hobby_weight']].apply(normalize)

topic_drives = []
topic_continuous_attention = []
for user_num in range(df.shape[0]):
    topic_drives.append(df.iloc[user_num,11]+df.iloc[user_num,12]+df.iloc[user_num,14]+df.iloc[user_num,15])
    topic_continuous_attention.append(df.iloc[user_num,13]+df.iloc[user_num,15]+df.iloc[user_num,12])
df['topic_drives'] = topic_drives
df['topic_continuous_attention'] = topic_continuous_attention
#df.to_csv('E:\\#电子商务系统\\predict_label_data_normalize.csv',encoding='utf-8')

def plot_sca(df,emo_color,emo_name):
    #legend_lis = []
    for color in emo_color:
        dil_df = df[df['emo_label'] == color]
        #plt.scatter(dil_df.iloc[:,16],dil_df.iloc[:,17],c=color,alpha=0.5)
        plt.scatter(dil_df.iloc[:,16],dil_df.iloc[:,17],edgecolors=color,alpha=0.5,marker='o',facecolors='none')
        #legend_lis.append(pl)
    plt.title('user portrait')
    plt.xlabel('topic drives')
    plt.ylabel('topic continuous attention')
    plt.legend(emo_name)
    plt.savefig('E:\\#电子商务系统\\user_portrait06.png',dpi=300)

plot_sca(df,emo_color,emo_name)
