from dataprocess_ner import create_dataloader
from model_ner import ner_model
from tqdm import tqdm
import torch
from metric import F1_ner
import torch.nn as nn
from utils import Hypernum
import argparse
def parse_args():##终端传入config配置文件路径
    parser = argparse.ArgumentParser(description="Running NER Model")
    parser.add_argument("--config", type=str, required=True, help="yaml文件路径")
    return parser.parse_args()

def train(model,train_loader,optimizer,epoch,device):
    model.train()
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()
    pbar = tqdm(train_loader,desc=f"Training {epoch}",unit="epoch",ncols=120)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids,attention_mask)
        loss = loss_fn(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1))#[bsz,seq_len,hid]->[bsz*seq_len,hid]

        total_loss+=loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pbar.set_postfix(loss = loss.item())
    avg_loss = total_loss/len(train_loader)
    return avg_loss

def dev(model,dev_loader,epoch,device,index2label):
    model.eval()
    F1cal_ = F1_ner(index2label)
    pbar = tqdm(dev_loader,desc=f"Deving{epoch}",unit="epoch",ncols=120)
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids,attention_mask)
            logits = torch.argmax(logits,dim=-1)
            F1cal_.update(logits,labels,attention_mask)

            f1 = F1cal_.f1cal()
            pbar.set_postfix(f1 = f1)
    f1 = F1cal_.f1cal()
    pbar.set_postfix(f1 = f1)
    return f1


def test(model,test_loader,epoch,device,index2label):
    model.eval()
    F1cal_ = F1_ner(index2label)
    pbar = tqdm(test_loader,desc=f"Testing{epoch}",unit="epoch",ncols=120)
    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids,attention_mask)
            logits = torch.argmax(logits,dim=-1)
            F1cal_.update(logits,labels,attention_mask)

            f1 = F1cal_.f1cal()
            pbar.set_postfix(f1 = f1)
    f1 = F1cal_.f1cal()
    pbar.set_postfix(f1 = f1)
    return f1


if __name__=="__main__":
    args = parse_args()
    hypernum = Hypernum.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader,dev_loader,test_loader,index2label = create_dataloader(hypernum)
    model = ner_model(hypernum.model_path,len(index2label)).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr = hypernum.lr)
    best_f1 = 0
    best_state_model_dict = None
    for epoch in range(hypernum.epochs):
        train_loss = train(model,train_loader,optimizer,epoch,device)

        dev_f1 = dev(model,dev_loader,epoch,device,index2label)

        if dev_f1>best_f1:
            best_f1 = dev_f1
            best_state_model_dict = model.state_dict()
    if best_state_model_dict !=None:
        torch.save(best_state_model_dict,hypernum.save_pth)
        print("最优模型已经保存")
    model.load_state_dict(best_state_model_dict)
    test_f1 = test(model,test_loader,hypernum.epochs,device,index2label)

    print("训练结束")
    