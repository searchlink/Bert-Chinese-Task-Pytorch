# encoding: utf-8
import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertPreTrainedModel, BertConfig, BertModel, BertForTokenClassification
from seqeval.metrics import classification_report
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

'''
包含两个步骤：
1. 数据处理
2. 构建模型
'''
########################################################################################################################
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file, quotechar=None):
        """Reads a BIO data."""
        with open(input_file) as f:
            lines = []
            words = []
            labels = []
            for line in f:
                line = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                    w = ' '.join([word for word in words if len(word) > 0])
                    l = ' '.join([label for label in labels if len(label) > 0])
                    lines.append([w, l])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
        print(lines[:10])
        return lines

class NerProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self,data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    # 表示从1开始对label进行index化
    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    # 每个example实例
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label.split(' ')
        tokens = []
        labels = []
        # 每个token
        for i, word in enumerate(textlist):
            # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            labels.append(label_1)
        # 序列截断
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]") # 句子开始设置CLS 标志
        segment_ids.append(0)   # 用来识别句子的界限。第一个句子为0
        label_ids.append(label_map["[CLS]"])   # 与tokens保持一致
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]") # 句尾添加[SEP]标志
        segment_ids.append(0)
        label_ids.append(label_map["[SEP]"])     # 与tokens保持一致
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)    # 将序列中的字(ntokens)转化为ID形式
        input_mask = [1] * len(input_ids)
        # 开始进行padding处理
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        # 打印部分样本数据信息
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids)
                      )
    return features
###################################################################################################################
class BertNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)  # 载入bert模型
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 简单的线性层
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # 初始化权重
        self.init_weights(self.classifier)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)

# 开始设定训练的各种参数和配置
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

processors = {"ner": NerProcessor}
processor = processors["ner"]()
label_list = processor.get_labels()
num_labels = len(label_list) + 1    # 因为是从0开始
label_map = {i : label for i, label in enumerate(label_list,1)}

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
config = BertConfig.from_pretrained("bert-base-chinese", num_labels=num_labels)
model = BertNer.from_pretrained("bert-base-chinese", config=config).to(device)

# 检查参数个数同时选择需要进行梯度更新的参数
def get_learnable_params(module):
    return [p for p in module.parameters() if p.requires_grad == True]
model_params = get_learnable_params(model)
clf_params = get_learnable_params(model.classifier)
print("整个分类模型的参数量：", sum(p.numel() for p in model_params))
print("线性分类器的参数量：", sum(p.numel() for p in clf_params))

max_seq_length = 128
train_batch_size = 64
num_train_epochs = 10

print("开始训练：")
train_examples = processor.get_train_examples("/home/wangwei/pt_workdir/bert_ner_task/data")
train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)

dev_examples = processor.get_dev_examples("/home/wangwei/pt_workdir/bert_ner_task/data")
dev_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)

dev_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
dev_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
dev_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
dev_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
dev_data = TensorDataset(dev_input_ids, dev_input_mask, dev_segment_ids, dev_label_ids)
dev_loader = DataLoader(dev_data, sampler=RandomSampler(dev_data), batch_size=train_batch_size)

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", train_batch_size)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=train_batch_size)
optimizer = torch.optim.Adam(model_params, lr=1.0e-4)

for epoch in range(num_train_epochs):
    running_loss = 0.
    for batch in train_loader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # 前向传播
        outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
        loss = outputs[0]

        optimizer.zero_grad()

        # 反向传播
        loss.backward()
        optimizer.step()

        # 记录当前 batch loss
        running_loss += loss.item()

    # 计算准确率
    model.eval()  # 推论模式
    with torch.no_grad():
        y_true = []
        y_pred = []
        for batch in dev_loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            logits = outputs[0]

            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                # 第i个句子 第j个字， 剔除第一个[cls]
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(label_map):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])

        report = classification_report(y_true, y_pred, digits=4)
        logger.info("\n%s", report)