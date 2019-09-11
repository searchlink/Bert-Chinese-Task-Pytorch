# encoding: utf-8
'''基于bert的finetuning, 对句子对进行分类'''
import logging
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# 1. 准备原始数据
# 读取数据
df_train = pd.read_csv("/home/wangwei/pt_workdir/data/fake_news/train.csv")
# 简单数据清理，去除空白标题的 examples
empty_title = ((df_train['title2_zh'].isnull()) \
               | (df_train['title1_zh'].isnull()) \
               | (df_train['title2_zh'] == '') \
               | (df_train['title2_zh'] == '0'))
df_train = df_train[~empty_title]
df_train = df_train[~empty_title]

# 剔除过长的样本以避免 BERT 无法将整个输入序列放入GPU种
MAX_LENGTH = 30
df_train = df_train[~(df_train.title1_zh.apply(lambda x : len(x)) > MAX_LENGTH)]
df_train = df_train[~(df_train.title2_zh.apply(lambda x : len(x)) > MAX_LENGTH)]

# 只用 1% 训练数据看看 BERT 对少标注数据有多少帮助
SAMPLE_FRAC = 0.01
df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=9527)

# 选择需要训练学习的列
df_train = df_train.reset_index()
df_train = df_train.loc[:, ['title1_zh', 'title2_zh', 'label']]
df_train.columns = ['text_a', 'text_b', 'label']

# 将处理結果另存成 tsv 供 PyTorch 使用
df_train.to_csv("./train.tsv", sep="\t", index=False)

# 读取测试集
df_test = pd.read_csv("/home/wangwei/pt_workdir/data/fake_news/test.csv")
df_test = df_test.loc[:, ["title1_zh", "title2_zh", "id"]]
df_test.columns = ["text_a", "text_b", "Id"]
df_test.to_csv("./test.tsv", sep="\t", index=False)

# 2. 将原始文本转换成Bert相容的输入格式
class FakeNewsDateset(Dataset):
    def __init__(self, mode, tokenizer):
        # 读取数据前的初始化操作
        assert mode in ["train", "test"]   # 一般还要dev
        self.mode = mode
        self.df = pd.read_csv(mode + ".tsv", sep="\t")
        self.length = len(self.df)
        self.label_map = {'agreed': 0, 'disagreed': 1, 'unrelated': 2}
        self.tokenizer = tokenizer  # 将使用bert tokenizer

    # @pysnooper.snoop()
    def __getitem__(self, index):
        if self.mode == "train":
            text_a, text_b, label = self.df.iloc[index, :].values
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)
        else:
            text_a, text_b = self.df.iloc[index, :2].values
            label_tensor = None

        # 分字， 建立第一个句子的Bert tokens并加入[SEP]
        word_pieces = ["[CLS]"]
        token_a = self.tokenizer.tokenize(text_a)
        word_pieces += token_a + ["[SEP]"]
        len_a = len(word_pieces)

        # 分字，建立第二个句子的Bert tokens并加入[SEP]
        token_b = self.tokenizer.tokenize(text_b)
        word_pieces += token_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a

        # 将整个token序列转化为索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        # 区分句子，将第一句包含 [SEP] 的 token 位置设为 0，其他为 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, dtype=torch.long)

        return (tokens_tensor, segments_tensor, label_tensor)

    def __len__(self):
        return self.length

# 初始化一个专门读取训练样本的Dataset， 使用中文BERT断词
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
trainset = FakeNewsDateset("train", tokenizer=tokenizer)

# 有了dataset之后， 需要一个dataloader来传入mini_batch的数据
def create_mini_batch(samples):
    """
    :param samples: 传入的samples是一个list， 里面每个element都是定义的FakeNewsDateset中一个样本
    - tokens_tensor
    - segments_tensor
    - label_tensor
    会对前面两个 tensors 作 zero padding，并产生前面说明过的 masks_tensors
    :return:
    """
    tokens_tensors = [sample[0] for sample in samples]
    segments_tensors = [sample[1] for sample in samples]
    if samples[0][2] is None:
        label_ids = None
    else:
        label_ids = torch.stack([sample[2] for sample in samples])  # (batch_size, 1)

    # 开始padding
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, label_ids

BATCH_SIZE = 64
train_loader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

# 直接预测准确率
def get_predictions(model, train_loader, compute_acc=False):
    predictions = None  # 预测结果
    total = 0   # 样本数
    correct = 0 # 预测正确个数

    model.eval()  # 推论模式
    with torch.no_grad():
        for data in train_loader:
            # 将所有 tensors 移到 GPU 上
            if next(model.parameters()).is_cuda:
                data = [t.to(device) for t in data if t is not None]
            # # 別忘记前 3 个 tensors 分别为 tokens, segments 以及 masks
            tokens_tensors = data[0]
            segments_tensors = data[1]
            masks_tensors = data[2]
            outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors)
            logits = outputs[0]
            # 返回元组， 对应最大值，另外相应的下标index
            _, pred = torch.max(logits, 1)

            # 计算准确率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()

            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))

    if compute_acc:
        print("correct:", correct, "total:", total)
        acc = correct / total
        return predictions, acc
    return predictions

# 4. 训练该下游任务模型
# 定义模型
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=3).to(device)
# 随机初始化，进行模型预测
print("*"*50)
print(model.config)
print("*"*50)
_, acc = get_predictions(model, train_loader, compute_acc=True)
print("classification acc:", acc)

# 检查参数个数同时选择需要进行梯度更新的参数
def get_learnable_params(module):
    return [p for p in module.parameters() if p.requires_grad == True]
model_params = get_learnable_params(model)
clf_params = get_learnable_params(model.classifier)
print("整个分类模型的参数量：", sum(p.numel() for p in model_params))
print("线性分类器的参数量：", sum(p.numel() for p in clf_params))

print("开始训练：")
model.train()
optimizer = torch.optim.Adam(model_params, lr=1.0e-4)
# 训练模型
epochs = 6
for epoch in range(epochs):
    running_loss = 0.
    for data in train_loader:
        tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]

        # 前向传播
        outputs = model(input_ids=tokens_tensors, token_type_ids=segments_tensors, attention_mask=masks_tensors,
                     labels=labels)

        loss = outputs[0]
        optimizer.zero_grad()

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录当前 batch loss
        running_loss += loss.item()

        # 计算分类准确率
    _, acc = get_predictions(model, train_loader, compute_acc=True)
    print('[epoch %d] loss: %.3f, acc: %.3f' % (epoch + 1, running_loss, acc))

testset = FakeNewsDateset("test", tokenizer=tokenizer)
testloader = DataLoader(testset, batch_size=256, collate_fn=create_mini_batch)
