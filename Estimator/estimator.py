import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers import BertPreTrainedModel
import numpy as np
import time
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import KMeans
import os
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
import random   
from sklearn import cluster
from torch.nn import functional as F 
import matplotlib.pyplot as plt 


class args:
    use_pretrain='SBERT'
    bert='distilbert'
    num_classes=82
    text='text' 
    label='topic' 
    objective='SCCL'
    augtype='explicit'
    temperature =1
    alpha=1.0                                         
    eta=10 
    lr=1e-05            
    lr_scale=1      
    max_length=32   
    batch_size=8     
    max_iter=1200   
    gpuid=[0]
    augmentation_1='text_aug_one'
    augmentation_2='text_aug_two'
    dataname='twitter_aug_train' 
    datapath=''
    print_freq=100
    seed=42
    aug_type='explicit' 

eps = 1e-8  

class KLDiv(nn.Module):    
    def forward(self, predict, target): #log(output+10^-8),target
        assert predict.ndimension()==2,'Input dimension must be 2'
        target = target.detach()  
        p1 = predict + eps
        t1 = target + eps
        logI = p1.log()
        logT = t1.log()
        TlogTdI = target * (logT - logI)
        kld = TlogTdI.sum(1)
        return kld

class KCL(nn.Module):
    def __init__(self):
        super(KCL,self).__init__()
        self.kld = KLDiv()

    def forward(self, prob1, prob2):  
        kld = self.kld(prob1, prob2)
        return kld.mean()
#input - clusters_prob    
def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9) 
    #weight-tensor (82,8) 
    #return tensor (82,8)
    return (weight.t() / torch.sum(weight, 1)).t()  




class PairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(PairConLoss, self).__init__()  
        self.temperature = temperature
        self.eps = 1e-08
  
    def forward(self, features_1, features_2):
        device = features_1.device  
        batch_size = features_1.shape[0]
        features= torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)  
        mask = ~mask    
          
        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(2*batch_size, -1)
        
        neg_mean = torch.mean(neg)
        pos_n = torch.mean(pos)
        Ng = neg.sum(dim=-1)
            
        loss_pos = (- torch.log(pos / (Ng+pos))).mean()
        
        return {"loss":loss_pos, "pos_mean":pos_n.detach().cpu().numpy(), "neg_mean":neg_mean.detach().cpu().numpy(), "pos":pos.detach().cpu().numpy(), "neg":neg.detach().cpu().numpy()}
            
 

class SCCLBert(nn.Module):
    def __init__(self, bert_model, tokenizer, cluster_centers=None, alpha=1.0):
        super(SCCLBert, self).__init__()
        
        self.tokenizer = tokenizer
        self.bert = bert_model
        self.emb_size = self.bert.config.hidden_size
        self.alpha = alpha
        
        # Instance-CL head
        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.Dropout(0.2,inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128))
        
        # Clustering head
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)
      
    
    def forward(self, input_ids, attention_mask, task_type="virtual"):
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
        
        elif task_type == "virtual":
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1) 
            
            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            return mean_output_1, mean_output_2
        
        elif task_type == "explicit":
            input_ids_1, input_ids_2, input_ids_3 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2, attention_mask_3 = torch.unbind(attention_mask, dim=1) 
            
            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            mean_output_3 = self.get_mean_embeddings(input_ids_3, attention_mask_3)
            return mean_output_1, mean_output_2, mean_output_3
        
        else:  
            raise Exception("TRANSFORMER ENCODING TYPE ERROR! OPTIONS: [EVALUATE, VIRTUAL, EXPLICIT]")
      
    #help function
    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output  
    

    def get_cluster_prob(self, embeddings): 
        #cluster_centers-tensor (82,768) 
        #for every element calculate distance on every center of cluster
        #norm_squared-tensor(8,82)  
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        #for every item in tensor 1/(1+item/alpha)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))

        power = float(self.alpha + 1) / 2
        #item^((alpha+1)/2)  
        numerator = numerator ** power
        #normalization
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def local_consistency(self, embd0, embd1, embd2, criterion):
        #get cluster prob for every aug
        p0 = self.get_cluster_prob(embd0)
        p1 = self.get_cluster_prob(embd1)
        p2 = self.get_cluster_prob(embd2)     
        
        lds1 = criterion(p1, p0)
        lds2 = criterion(p2, p0)
        return lds1+lds2
    #input-embd2,embd3
    def contrast_logits(self, embd1, embd2=None):   
        #normalization
        feat1 = F.normalize(self.contrast_head(embd1), dim=1) #to tensor (8,128)  
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else: 
            return feat1

 
#calculate metrics

cluster_nmi = normalized_mutual_info_score
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
  
    # ind = sklearn.utils.linear_assignment_.linear_assignment(w.max() - w)
    # row_ind, col_ind = linear_assignment(w.max() - w)
    row_ind, col_ind = hungarian(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / self.count

class Timer(object):
    """
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.interval = 0
        self.time = time.time()

    def value(self):
        return time.time() - self.time
        
    def tic(self):
        self.time = time.time()
        
    def toc(self):
        self.interval = time.time() - self.time
        self.time = time.time()
        return self.interval

class Confusion(object):
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index  
    """
    def __init__(self, k, normalized = False):
        super(Confusion, self).__init__()
        self.k = k                          
        self.conf = torch.LongTensor(k,k)
        self.normalized = normalized
        self.reset()                   

    def reset(self):
        self.conf.fill_(0)
        self.gt_n_cluster = None

    def cuda(self):  
        self.conf = self.conf.cuda()

    def add(self, output, target):
        output = output.squeeze()
        target = target.squeeze()               
        assert output.size(0) == target.size(0), \
                'number of targets and outputs do not match'
        if output.ndimension()>1: #it is the raw probabilities over classes
            assert output.size(1) == self.conf.size(0), \
                'number of outputs does not match size of confusion matrix'  
         
            _,pred = output.max(1) #find the predicted class
        else: #it is already the predicted class
            pred = output       
        indices = (target*self.conf.stride(0) + pred.squeeze_().type_as(target)).type_as(self.conf)
        ones = torch.ones(1).type_as(self.conf).expand(indices.size(0))
        self._conf_flat = self.conf.view(-1)
        self._conf_flat.index_add_(0, indices, ones)    

    def classIoU(self,ignore_last=False):
        confusion_tensor = self.conf
        if ignore_last:
            confusion_tensor = self.conf.narrow(0,0,self.k-1).narrow(1,0,self.k-1)
        union = confusion_tensor.sum(0).view(-1) + confusion_tensor.sum(1).view(-1) - confusion_tensor.diag().view(-1)
        acc = confusion_tensor.diag().float().view(-1).div(union.float()+1)
        return acc
        
    def recall(self,clsId):
        i = clsId
        TP = self.conf[i,i].sum().item()
        TPuFN = self.conf[i,:].sum().item()
        if TPuFN==0:
            return 0
        return float(TP)/TPuFN
        
    def precision(self,clsId):
        i = clsId
        TP = self.conf[i,i].sum().item()
        TPuFP = self.conf[:,i].sum().item()
        if TPuFP==0:
            return 0
        return float(TP)/TPuFP
        
    def f1score(self,clsId):
        r = self.recall(clsId)
        p = self.precision(clsId)
        print("classID:{}, precision:{:.4f}, recall:{:.4f}".format(clsId, p, r))
        if (p+r)==0:
            return 0
        return 2*float(p*r)/(p+r)
        
    def acc(self):
        TP = self.conf.diag().sum().item()
        total = self.conf.sum().item()
        if total==0:
            return 0
        return float(TP)/total     
        
    def optimal_assignment(self,gt_n_cluster=None,assign=None):
        if assign is None:
            mat = -self.conf.cpu().numpy() #hungaian finds the minimum cost
            r,assign = hungarian(mat)
        self.conf = self.conf[:,assign]
        self.gt_n_cluster = gt_n_cluster
        return assign
        
    def show(self,width=6,row_labels=None,column_labels=None):
        print("Confusion Matrix:")      
        conf = self.conf
        rows = self.gt_n_cluster or conf.size(0)
        cols = conf.size(1)
        if column_labels is not None:
            print(("%" + str(width) + "s") % '', end='')
            for c in column_labels:
                print(("%" + str(width) + "s") % c, end='')
            print('')
        for i in range(0,rows):
            if row_labels is not None:
                print(("%" + str(width) + "s|") % row_labels[i], end='')
            for j in range(0,cols):
                print(("%"+str(width)+".d")%conf[i,j],end='')
            print('')
        
    def conf2label(self):
        conf=self.conf
        gt_classes_count=conf.sum(1).squeeze()
        n_sample = gt_classes_count.sum().item()
        gt_label = torch.zeros(n_sample)
        pred_label = torch.zeros(n_sample)
        cur_idx = 0
        for c in range(conf.size(0)):
            if gt_classes_count[c]>0:
                gt_label[cur_idx:cur_idx+gt_classes_count[c]].fill_(c)
            for p in range(conf.size(1)):
                if conf[c][p]>0:
                    pred_label[cur_idx:cur_idx+conf[c][p]].fill_(p)
                cur_idx = cur_idx + conf[c][p];
        return gt_label,pred_label
    
    def clusterscores(self):
        target,pred = self.conf2label()
        NMI = normalized_mutual_info_score(target,pred)
        ARI = adjusted_rand_score(target,pred)
        AMI = adjusted_mutual_info_score(target,pred)
        return {'NMI':NMI,'ARI':ARI,'AMI':AMI}



 
#help functions 
def get_mean_embeddings(bert, input_ids, attention_mask):
        bert_output = bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    
def get_batch_token(tokenizer, text, max_length):
    token_feat = tokenizer.batch_encode_plus(
        text, 
        max_length=max_length, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )
    return token_feat


def get_kmeans_centers(bert, tokenizer, train_loader, num_classes, max_length):
    for i, batch in enumerate(train_loader):
        text, label = batch['text'], batch['label']  
        tokenized_features = get_batch_token(tokenizer, text, max_length)
        corpus_embeddings = get_mean_embeddings(bert, **tokenized_features)
        
        if i == 0:  
            all_labels = label
            all_embeddings = corpus_embeddings.detach().numpy()
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.detach().numpy()), axis=0)
    #all_embeddings-tensor (1319,768)
    #all_labels-tensor (1319)  
    
    confusion = Confusion(num_classes)

    #create kmeans model

    clustering_model = KMeans(n_clusters=num_classes)

    #fit on all_embeddings
    
    clustering_model.fit(all_embeddings)

    #get a kmeans labels(pred_labels)

    cluster_assignment = clustering_model.labels_

    #true labels is topic

    true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)    
    confusion.add(pred_labels, true_labels)  
    confusion.optimal_assignment(num_classes)

    #get a metrics ACC on eazy kmeans
    #return begin cluster centers
    return clustering_model.cluster_centers_ 


 
BERT_CLASS = {
    "distilbert": 'distilbert-base-uncased', 
}

SBERT_CLASS = {
    "distilbert": 'distilbert-base-nli-stsb-mean-tokens',
}


def get_optimizer(model, args):
    
    optimizer = torch.optim.Adam([
        {'params':model.bert.parameters()}, 
        {'params':model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale},
        {'params':model.cluster_centers, 'lr': args.lr*args.lr_scale}
    ], lr=args.lr)
    
    print(optimizer)
    return optimizer   
    

def get_bert(args):
    
    if args.use_pretrain == "SBERT":
        bert_model = get_sbert(args)
        tokenizer = bert_model[0].tokenizer
        model = bert_model[0].auto_model
        print("..... loading Sentence-BERT !!!")
    else:
        config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])
        model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
        tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])
        print("..... loading plain BERT !!!")
        
    return model, tokenizer


def get_sbert(args):
    sbert = SentenceTransformer(SBERT_CLASS[args.bert])
    return sbert

def set_global_random_seed(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)  
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True 

     
class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader,val_loader,test_loader, args,NMI2,ARI2,epoch_count):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.eta = self.args.eta
        self.aug_type = self.args.aug_type
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.contrast_loss = PairConLoss(temperature=self.args.temperature)
        self.gstep = 0   
        self.NMI2=NMI2
        self.ARI2=ARI2
        self.epoch_count=epoch_count 
    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=self.args.max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_transformer_input(self, batch):
        if len(batch) == 4:
            text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
            feat1 = self.get_batch_token(text1)
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)
                                                           
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1), feat3['attention_mask'].unsqueeze(1)], dim=1)
        elif len(batch) == 2:
            text = batch['text']
            feat1 = self.get_batch_token(text)
            feat2 = self.get_batch_token(text)
                                                                       
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
            
        return input_ids.cuda(), attention_mask.cuda()
        
                                      
    
    def train_step_explicit(self, input_ids, attention_mask):  
          
        self.optimizer.zero_grad()
        embd1, embd2, embd3 = self.model(input_ids, attention_mask, task_type="explicit")
        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd2, embd3) #feat1-tensor (8,128)
        losses = self.contrast_loss(feat1, feat2)  
        loss = self.eta * losses["loss"]

        # Clustering loss
        
        output = self.model.get_cluster_prob(embd1) #cluster_prob tensor (82,8)
        target = target_distribution(output).detach() #target distribution tensor (82,8)
        cluster_loss = self.cluster_loss((output+1e-08).log(), target)/output.shape[0] #KLDivloss-torch function
        loss += cluster_loss                                     
        losses["cluster_loss"] = cluster_loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),0.1)  
        self.optimizer.step()
        return losses
      
    
    def train(self):
        #print('\n={}/{}=Iterations/Batches'.format(self.args.max_iter, len(self.train_loader)))

        self.model.train()  
        for i in np.arange(self.args.max_iter+1): 
            try:
                batch = next(train_loader_iter)
            except:  
                train_loader_iter = iter(self.train_loader)
                batch = next(train_loader_iter)

            input_ids, attention_mask = self.prepare_transformer_input(batch)

            losses =self.train_step_explicit(input_ids, attention_mask)  

            if (self.args.print_freq>0) and ((i%self.args.print_freq==0) or (i==self.args.max_iter)):  
                self.evaluate_embedding(i)
                self.model.train()  

        return None     

    
    def evaluate_embedding(self, step):                                     
        dataloader =self.val_loader 
        self.model.eval()                        
        for i, batch in enumerate(dataloader):  
            with torch.no_grad():
                text, label = batch['text'], batch['label']     
                feat = self.get_batch_token(text)
                #get embeddings of batch  
                embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")
  
                model_prob = self.model.get_cluster_prob(embeddings)             
                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_prob = model_prob
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_prob = torch.cat((all_prob, model_prob), dim=0)    
            
        # Initialize confusion matrices
        confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes)
        #all_prob-tensor(1319,82)
        all_pred = all_prob.max(1)[1]
         
        #all_pred-tensor (1319) , all_pred getting by bert-model 

        confusion_model.add(all_pred, all_labels)        
        confusion_model.optimal_assignment(self.args.num_classes) 
        acc_model = confusion_model.acc()
        #acc-score by bert-model   

        self.NMI2.append(confusion_model.clusterscores()['NMI'])
        self.ARI2.append(confusion_model.clusterscores()['ARI'])  
        self.epoch_count.append(step)
        return None
    def predict(self):
        dataloader =self.test_loader
          
        
        self.model.eval()
        for i, batch in enumerate(dataloader):  
            with torch.no_grad(): 
                text, label = batch['text'], batch['label']    
                feat = self.get_batch_token(text)  
                embeddings = self.model(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")  
                model_prob = self.model.get_cluster_prob(embeddings)    
                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_prob = model_prob
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_prob = torch.cat((all_prob, model_prob), dim=0)
              
        confusion, confusion_model = Confusion(self.args.num_classes), Confusion(self.args.num_classes) 
        all_pred = all_prob.max(1)[1]
        confusion_model.add(all_pred, all_labels)
        confusion_model.optimal_assignment(self.args.num_classes)
        acc_model = confusion_model.acc()    
          
        return confusion_model.clusterscores() 
class Estimator:
    def __init__(self):
        self.param=1
    def fit(self,train_loader,val_loader,test_loader):
        torch.cuda.set_device(args.gpuid[0])
        bert, tokenizer = get_bert(args)
              
		    # initialize cluster centers  
        cluster_centers = get_kmeans_centers(bert, tokenizer, train_loader, args.num_classes, args.max_length) 
        model = SCCLBert(bert, tokenizer, cluster_centers=cluster_centers, alpha=args.alpha) 
        model = model.cuda()

		    # optimizer             
        optimizer = get_optimizer(model, args)     
        NMI2=[]
        ARI2=[]  
        epoch_count=[]                
        trainer = SCCLvTrainer(model, tokenizer, optimizer, train_loader,val_loader,test_loader, args,NMI2,ARI2,epoch_count)   
        trainer.train()
        print(trainer.predict())    
        plt.plot(trainer.epoch_count,trainer.NMI2)
        plt.plot(trainer.epoch_count,trainer.ARI2)                                      