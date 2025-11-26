from torch.utils.data import DataLoader,Dataset
import json
from transformers import AutoTokenizer
import os
import torch
from utils import Hypernum
class my_dataset(Dataset):
    def __init__(self,hypernum,mode):
        self.mode = mode
        self.texts,self.labels = self.read_data(os.path.join(hypernum.data_dir+hypernum.dataset_name+f"{self.mode}.txt"))
        self.tokenizer = AutoTokenizer.from_pretrained(hypernum.model_path)
        self.max_len = hypernum.max_len
        self.label_map_path = hypernum.label_map_path
        self.label2index,self.index2label = self.build_label_index(self.labels)
        
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, index):
        return self.texts[index],self.labels[index]
    def align_labels(self,text,label):##对其策略是如果是B开头的，后面拆分的都变成I开头的，只存在一个B,对于I开头的，分裂之后的也都是I,O同理I的逻辑
        tokenized_inputs = self.tokenizer(text, truncation=True, is_split_into_words=True)
        word_ids = tokenized_inputs.word_ids()
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None: #CLS和SEQ
                label_ids.append("O")
            else:
                current_label = label[word_idx]
                if word_idx != previous_word_idx:#第一个子词
                    label_ids.append(current_label)
                else:#后续的
                    if current_label.startswith("B-"):
                        label_ids.append("I" + current_label[1:])
                    else:
                        label_ids.append(current_label)
                previous_word_idx = word_idx
        return label_ids
    
    def my_collate_fn(self,batch):
        batch_labels = []
        texts = [text for text,label in batch]
        for text,label in batch:
            labels = self.align_labels(text,label)
            labels = [self.label2index.get(label,0) for label in labels ]##默认是第一个“O”
            if len(labels)<self.max_len:
                labels += [-100]*(self.max_len - len(labels))##使用-100做填充，在计算损失的时候会自动忽略
            else :
                labels = labels[:self.max_len]
            batch_labels.append(labels)
        encoding_out = self.tokenizer(texts,is_split_into_words = True,max_length = self.max_len,padding="max_length",truncation=True,return_tensors="pt")
        labels = torch.tensor(batch_labels,dtype=torch.int64)
        return {
            "input_ids":encoding_out["input_ids"],
            "attention_mask":encoding_out["attention_mask"],
            "labels":labels
        }
        
    def read_data(self,file):
        datas = open(file, "r", encoding="utf-8").read().strip().split("\n\n")
        texts,labels = [], []
        for data in datas:
            text1, label1 = [], []
            data1 = data.split("\n")
            for line in data1:
                text,label = line.split("\t")
                text1.append(text)
                label1.append(label)
            texts.append(text1)
            labels.append(label1)    
        return texts, labels
    def build_label_index(self,labels):##创建标签和id的映射关系json文件，为避免多次构建，如果文件存在，就不在构建
        path = self.label_map_path
        label2index = {}
        index2label = []
        if os.path.exists(path) and os.path.getsize(path)>0:
            with open(path,"r",encoding="utf-8") as f:
                data = json.load(f)
                label2index = data["label2index"]
                index2label = data["index2label"]

        else:
            for label in labels:
                for label_1 in label:
                    if label_1 not in label2index:
                        label2index[label_1] = len(label2index)
            index2label = list(label2index)
        with open(path,"w",encoding="utf-8") as f:
            json.dump({"label2index":label2index,"index2label":index2label},f)
        return label2index,index2label


def create_dataloader(hypernum):
    train_dataset = my_dataset(hypernum , "train")     
    dev_dataset =   my_dataset(hypernum , "dev") 
    test_dataset = my_dataset(hypernum , "test") 

    train_loader = DataLoader(train_dataset,hypernum.batch_size,shuffle=True,collate_fn=train_dataset.my_collate_fn)
    dev_loader = DataLoader(dev_dataset,hypernum.batch_size,shuffle=False,collate_fn=dev_dataset.my_collate_fn)
    test_loader = DataLoader(test_dataset,hypernum.batch_size,shuffle=False,collate_fn=test_dataset.my_collate_fn)

    return train_loader,dev_loader,test_loader,train_dataset.index2label

if __name__ =="__main__":
    hypernum = Hypernum.from_yaml("./config/weibo.yaml")
    trainloader,devloader,testloader,index2label = create_dataloader(hypernum)
    print(index2label)
    for batch in trainloader:
        print(batch["input_ids"])
        print(batch["attention_mask"])
        print(batch["labels"])
        break
    for batch in devloader:
        print(batch["input_ids"])
        print(batch["attention_mask"])
        print(batch["labels"])
        break
    for batch in testloader:
        print(batch["input_ids"])
        print(batch["attention_mask"])
        print(batch["labels"])
        break
    
