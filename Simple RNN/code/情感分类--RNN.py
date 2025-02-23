# 读取数据
import pandas as pd
import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 加载数据
file_path = r"../data/Sentiment Analysis Dataset.csv"

# 提取前1000条数据进行提取并训练
data = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip", nrows=5000)
data.head()

# 提取评论信息
sentimentText = data["SentimentText"]
sentimentText.head()

# 获取真实的标签
labels = data["Sentiment"].to_numpy()
labels = torch.tensor(labels)

# 对文本进行编码、分词的操作
tokenizer = Tokenizer(num_words=5000, oov_token="<oov>")
tokenizer.fit_on_texts(sentimentText)  # 创建词汇表
print("词汇表的长度是:", len(tokenizer.word_index))

sequences = tokenizer.texts_to_sequences(sentimentText)
# print(sequences)  # 将文本数据转化为token序列化数据

sequence_length = [len(sequence) for sequence in sequences]
max_length = max(sequence_length)

# 将token数据转化成长度一致的序列便于处理
padded_sequences = torch.tensor(
    pad_sequences(sequences, maxlen=max_length), dtype=torch.long
)

# 对token化后的数据进行embedding操作
embed_size = 5
vocab_size = len(tokenizer.word_index)  # 获得词典长度，输入embedding层
embedding = nn.Embedding(vocab_size, embedding_dim=embed_size)  # 建立embedding层
input = embedding(padded_sequences)


# 定义多层RNN模型
class MultiLayerRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, layer_num, class_num):
        super(MultiLayerRNN, self).__init__()
        self.rnn = nn.RNN(embed_size, hidden_size, layer_num, batch_first=True)
        self.linear = nn.Linear(
            hidden_size, class_num
        )  # 由于使用的交叉熵损失自带softmax激活函数，故不再加激活曾

    def forward(self, inputs):
        """
        :param inputs: 输入的embedding结果
        :return: 返回最终的预测
        """
        output, hidden = self.rnn(inputs)
        final_hidden = hidden[-1]  # 获取最后一层RNN的最后一个时间步的隐藏层结果
        linear_result = self.linear(final_hidden)
        return linear_result


# 设置参数值，并进行前向传播
hidden_size = 6
layer_num = 3
class_num = 2

# 定义RNN实例
MyRNN = MultiLayerRNN(embed_size, hidden_size, layer_num, class_num)

# 进行反向传播并优化

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(MyRNN.parameters(), lr=0.01)  # 定义优化器

# 训练次数设置
epochs = 6
accuracy_list = []

for epoch in range(epochs):
    # 前向传播
    outputs = MyRNN(input)

    # 计算损失
    loss = criterion(outputs, labels)

    # 执行优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward(retain_graph=True)
    optimizer.step()

    # 计算准确率
    with torch.no_grad():  # 禁用梯度计算以节省内存
        predictions = torch.argmax(outputs, dim=1)  # 获取预测类别
        correct = (predictions == labels).sum().item()  # 正确预测的数量
        accuracy = correct / labels.size(0)  # 计算准确率
        accuracy_list.append(accuracy)

    # 打印损失和准确率
    print(
        f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%"
    )

# 设定模型保存规则
if max(accuracy_list) >= 0.71:
    torch.save(
        MyRNN.state_dict(),
        r"../model/my_rnn_model.pth",
    )
    print("RNN模型保存成功！")
else:
    print("此次的模型没有达到标准")
