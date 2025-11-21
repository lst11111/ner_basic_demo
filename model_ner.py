from transformers import BertModel

import torch.nn as nn
class ner_model(nn.Module):
    def __init__(self, model_path, num_labels):
        super(ner_model,self).__init__()
        self.bert_model = BertModel.from_pretrained(model_path)
        self.fc = nn.Linear(768,num_labels)

    def forward(self,input_ids,attention_mask):
        hidden = self.bert_model(input_ids,attention_mask)
        last_hidden = hidden.last_hidden_state
        logits = self.fc(last_hidden)
        return logits