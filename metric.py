from seqeval.metrics.sequence_labeling import get_entities
class F1cal():
    def __init__(self,index2label):
        self.tp =0
        self.pre_len = 0
        self.gold_len = 0
        self.index2label = index2label
    def update(self,pred,gold,mask):
        for i in range(len(mask)):
            length = mask[i].sum()##计算每个样本的有效长度
            ##截取到最大长度
            _pre= pred[i][:length].tolist()
            _label = gold[i][:length].tolist()
            

            prediction = [self.index2label[pre] for pre in _pre]
            reference = [self.index2label[label] for label in _label]

            predictions = get_entities(prediction)##获取预测的实体信息
            references = get_entities(reference)##获取真实的实体信息

            for pre in predictions:
                if pre in references:
                    self.tp+=1
            
            self.gold_len+=len(references)
            self.pre_len +=len(predictions)
    def f1(self):
        p = self.tp/(self.pre_len+1e-13)
        r = self.tp/(self.gold_len+1e-13)

        f1 = 2*p*r/(p+r+1e-13)
        return f1