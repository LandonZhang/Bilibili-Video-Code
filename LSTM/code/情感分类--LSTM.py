from typing import List  # noqa: UP035

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 导入分析数据
data = pd.read_csv(
    r"../data/Sentiment Analysis Dataset.csv",  # noqa: E501
    encoding="UTF-8",
    on_bad_lines="skip",
    nrows=1000,
)

# 提取评论信息
sentimentText = data["SentimentText"]

# 获取真实的标签
labels = torch.tensor(data["Sentiment"])

# 清理数据
full_data = sentimentText.to_list()
cleaned_data = []

for sentence in full_data:
    sentence = str(sentence)
    # 去除双引号与单引号
    sentence = sentence.replace("'", "")
    sentence = sentence.replace('"', "")
    # 去除空白
    sentence = sentence.strip()
    cleaned_data.append(sentence)


class advancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer, drop_out=0.5):
        super().__init__()
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layer,
            batch_first=True,
            dropout=drop_out if num_layer > 1 else 0,  # 作用于层与层之间的连接
        )

        # 批标准化层
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Dropout层 - 使得输出不过多依赖某一个参数
        self.dropout = nn.Dropout(p=0.3)

        # 分类层
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, origin_data):
        # 调用LSTM层
        result, (h_n, c_n) = self.lstm(
            origin_data
        )  # h_n的形状是: (num_layer, batch_size, hidden_size)
        last_hidden = h_n[-1]

        # 批归一化
        x = self.batch_norm(last_hidden)

        # Droupout层
        x = self.dropout(x)

        # 分类层
        x = self.classifier(x)

        return F.softmax(x, dim=1)


class SentimentAnalyzer:
    def __init__(
        self,
        input_size=1536,
        hidden_size=512,
        output_size=2,
        num_layers=3,
    ):
        """初始化情感分析器"""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.model = None
        self.client = OpenAI()

    def _init_model(self):
        """初始化LSTM模型"""
        self.model = advancedLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            num_layer=self.num_layers,
        )

    def vectorize_data(
        self, cleaned_data, dimension=1536, batch_size=100, status="training"
    ):
        """数据向量化处理"""
        embedded_text = []

        if status == "training":
            for i in range(len(cleaned_data) // batch_size):
                batch = cleaned_data[i * batch_size : (i + 1) * batch_size]
                try:
                    full_data = self.client.embeddings.create(
                        model="text-embedding-3-small",
                        input=batch,
                        dimensions=dimension,
                        encoding_format="float",
                    )

                    for count in range(batch_size):
                        embedded_text.append(
                            full_data.data[count].embedding
                        )  # (batch_size, input_size)

                except Exception as e:
                    print(f"处理第{i}批时出错:{str(e)}")
                    break
        elif status == "predicting":
            full_data = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=cleaned_data,
                dimensions=dimension,
                encoding_format="float",
            )
            for i in range(len(cleaned_data)):
                embedded_text.append(full_data.data[i].embedding)

        return torch.tensor(np.array(embedded_text), dtype=torch.float32)

    def train(
        self, input_data, labels, n_epochs=50, learning_rate=0.001, test_size=0.2
    ):
        """训练模型"""
        # 初始化模型
        if self.model is None:
            self._init_model()

        # 数据集分割
        X_train, X_test, y_train, y_test = train_test_split(
            input_data, labels, test_size=test_size, random_state=42
        )

        # 添加序列维度
        X_train = X_train.unsqueeze(1)  # (batch_size, seq_len(1), input_size)
        X_test = X_test.unsqueeze(1)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # 训练循环
        for epoch in tqdm(range(n_epochs), desc="Training"):
            self.model.train()

            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}")

                # 评估
                accuracy = self.evaluate(X_test, y_test)
                print(f"Test Accuracy: {accuracy:.2%}")

        return self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        """评估模型"""
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test)
            _, predicted = torch.max(
                test_outputs.data, 1
            )  # 第一个参数是：最大值； 第二个参数是：最大值的索引
            accuracy = (predicted == y_test).sum().item() / len(y_test)
        return accuracy

    def predict(self, text):
        """预测输入文本的情感"""
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型")
        # 添加数据类型检查
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, List[str]):
            raise ValueError("输入必须是字符串或字符串列表")

        # 获取文本嵌入
        embedding = self.vectorize_data(text, status="predicting")
        embedding = embedding.unsqueeze(1)

        self.model.eval()
        with torch.no_grad():
            output = self.model(embedding)
            _, predicted = torch.max(output.data, 1)

        result = [
            "积极消息" if single_result == 1 else "消极消息"
            for single_result in predicted
        ]
        return result[0] if isinstance(text, str) else result

    def save_model(self, path):
        """保存模型"""
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            print(f"模型已保存至: {path}")
        else:
            print("没有可保存的模型")

    def load_model(self, path):
        """加载模型"""
        self._init_model()
        self.model.load_state_dict(torch.load(path))
        print(f"模型已从 {path} 加载")


if __name__ == "__main__":
    # # 创建情感分析器实例
    # analyzer = SentimentAnalyzer()

    # # 数据预处理和向量化
    # vectors = analyzer.vectorize_data(cleaned_data)

    # # 训练模型
    # final_accuracy = analyzer.train(vectors, labels)
    # print(f"最终准确率: {final_accuracy:.2%}")

    # # 保存模型
    # analyzer.save_model(
    #     r"E:\课程资料合集\数据分析原理与技术(2)\Code\Week7 lab\情感分类\model\LSTM.pth"
    # )

    # 加载模型
    analyzer = SentimentAnalyzer()
    analyzer.load_model(r"../model/LSTM.pth")

    # 预测新文本
    # text = ["I love this movie!", "I hate that bitch", "Shit!"]
    text = "Fuck you!"
    sentiment = analyzer.predict(text)
    print(f"文本情感: {sentiment}")
