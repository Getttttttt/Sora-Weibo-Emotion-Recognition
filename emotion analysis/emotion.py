from transformers import pipeline
import pandas as pd

def readData(content_path):
    df = pd.read_csv(content_path, encoding='ANSI')
    return df

def split_string(text, chunk_size=200):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def emotion(df):
    blog_contents = df['博文内容'].tolist()
    classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
    emotion_label = []
    emotion_score = []
    for content in blog_contents:
        content = content.strip()
        print(len(content))
        if(len(content)>200):
            substrings = split_string(content)
            print(substrings)
            e = []
            for i in substrings:
                emo = classifier(i)
                e.append(emo[0])
            print(e)
            flag=0
            for i in range(len(e)):
                for j in range(len(e)):
                    if(e[i]["label"]!=e[j]["label"]):
                        flag=1
            if(flag==0):
                score = [x["score"] for x in e]
                label = e[0]["label"]
                avg_score = sum(score) / len(e)
                emo = {'label': e[0]["label"], 'score': avg_score}
            else:
                sorted_dict_list = sorted(e, key=lambda x: x['score'], reverse=True)
                emo = {'label':sorted_dict_list[0]["label"], 'score':sorted_dict_list[0]["score"]}
        else:            
            emo = classifier(content)
            emo = emo[0]
        emotion_label.append(emo["label"])
        emotion_score.append(emo["score"])
    df["情绪标签"] = emotion_label
    df["标签得分"] = emotion_score
    df.to_csv("./emotion.csv", index=False)

    
def test_model():
    classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
    output = classifier("这是一段对 OpenAI 的 Sora 团队成员的采访，没有对 CTO 采访 微博正文 的信息量大，不过也可以看看：- **Sora的运行原理**：Sora是一个视频生成模型，借鉴了DALL-E的扩散模型和GPT系列的技术，能够处理和生成不同长度、宽高比和分辨率的视频。它通过公开数据和OpenAI持有许可的数据进行训练，创新地处理不同尺寸的视频。- **Sora的优势和劣势**：- **优势**：处理照片级细节能力强，视频长度可以达到一分钟，处理光线、反射、近景和纹理等方面表现出色。- **劣势**：处理具体物理问题和细节（如手指数量错误）方面存在挑战，特别是涉及时间推移的相机轨迹或动作时。- **未来方向和反馈**：Sora团队寻求公众反馈，特别关注如何提供更详细的内容控制。团队也意识到，随着技术的发展，确保内容安全和防止误用变得尤为重要。- **对声音的探索**：尽管Sora当前主要专注于视频生成，团队认为将来加入声音会使视频更具沉浸感，但目前还在早期阶段。- **发布和社会影响**：Sora的发布旨在搜集公众反馈，未来将探索如何确保技术的安全使用。团队希望Sora能帮助降低从创意到成品的制作成本，同时意识到技术的社会影响，特别是在内容验证和确保不被误用方面的重要性。- **AI多媒体的未来**：Sora团队期待AI工具将如何促进创造全新类型的内容，而不仅仅是模仿现有内容。他们强调了学习用户如何使用这些工具进行创新的重要性，并对AI创造的新媒体体验持开放态度。 L宝玉xp的微博视频 收起d")
    print(output)


if __name__=='__main__':
    #test_model()
    content_path = './content_noblank.csv'
    df = readData(content_path)
    emotion(df)