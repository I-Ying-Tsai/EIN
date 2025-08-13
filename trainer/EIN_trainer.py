import torch.nn.functional as F
import numpy as np
from utils.earlystopping import EarlyStopping
from torch_geometric.loader import DataLoader
from utils.dataloader import *
import time
import os
from utils.word2vec import *
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from utils.logger import (
    get_logger, 
    get_log_dir,
)

import pandas as pd
import matplotlib.pyplot as plt


class EINTrainer(object):
    def __init__(self, datasets, model, optimizer, args, device):
        super(EINTrainer, self).__init__()
        self.model = model 
        self.optimizer = optimizer
        self.device = device
        self.args = args

        train_dataset, val_dataset, test_dataset = datasets

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        self.train_per_epoch = len(self.train_loader)

        # log
        args.log_dir = get_log_dir(args)
        if os.path.isdir(args.log_dir) == False and not args.debug:
            os.makedirs(args.log_dir, exist_ok=True)
        self.logger = get_logger(args.log_dir, name=args.log_dir, debug=args.debug)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        
        self.logger.info('Experiment log path in: {}'.format(args.log_dir))
        self.logger.info('Experiment configs are: {}'.format(args))

        ##record
        self.raw_alpha_history = []
        self.alpha_history = []
        self.raw_beta_history = []
        self.beta_history = []
        self.train_loss_history = [] 


    
    def train_epoch(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data.to(self.device)
            out_labels, U, S, D = self.model(data)
            
            p_loss = self.model.physics_loss(U, S, D, data.user_state)

            loss = F.nll_loss(out_labels, data.y) + p_loss
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

        train_epoch_loss = train_loss/self.train_per_epoch

        ##record
        self.raw_alpha_history.append(self.model.raw_alpha.item())
        self.raw_beta_history.append(self.model.raw_beta.item())
        self.alpha_history.append(self.model.alpha.item())
        self.beta_history.append(self.model.beta.item())

        self.logger.info('*******Traininig Epoch {}: averaged Loss : {:.6f}'.format(epoch, train_epoch_loss))

        return train_epoch_loss

    def validate_epoch(self, epoch):

        val_losses = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.val_loader):
                data.to(self.device)
                val_out, _, _, _ = self.model(data)
                val_loss  = F.nll_loss(val_out, data.y)
                val_losses.append(val_loss.item())

        val_loss = np.mean(val_losses)
        self.logger.info('*******Val Epoch {}: averaged Loss : {:.6f}'.format(epoch, val_loss))
        
        ##record
        self.train_loss_history.append(val_loss)

        return val_loss
    
    def test(self):
        # test
        y_true = []
        y_pred = []
        all_confidence_scores = []  # 儲存所有信心度
        original_files = []  # 儲存原始檔案名

        print("test_loader.dataset size:", len(self.test_loader.dataset))

        self.model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                data.to(self.device)
                test_out, _, _, _ = self.model(data)

                y_true += data.y.tolist()
                y_pred += test_out.max(1).indices.tolist()

                # 獲取信心度
                scores, preds, reviews = self.evaluate_confidence(data)
                all_confidence_scores.extend(scores.cpu().numpy().tolist())  # 儲存信心度

                # 提取原始檔案名，這裡需要確保正確提取
                original_files.extend(data.original_file)  # 將原始檔案名添加到列表中

                
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            


            acc = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            self.logger.info("Test Acc: {:.4f} | AUC: {:.4f} | F1 {:.4f}".format(acc, auc, f1))

            return all_confidence_scores, original_files  # 返回所有信心度和原始檔案名
  

    def plot_parameters(self):
        # 创建一个 DataFrame 来存储参数
        data = {
            'Epochs': range(len(self.raw_alpha_history)),
            'raw_alpha': self.raw_alpha_history,
            'sigmoid_alpha': self.alpha_history,
            'raw_beta': self.raw_beta_history,
            'sigmoid_beta': self.beta_history,
            'average_loss': self.train_loss_history,
        }
        
        df = pd.DataFrame(data)

        # 使用 args.dataset 生成文件名
        csv_filename = f'parameters_table_{self.args.dataset}.csv'
        img_filename = f'parameters_table_{self.args.dataset}.png'

        # 保存为 CSV 文件
        df.to_csv(csv_filename, index=False)

        # 创建表格并保存为图片
        fig, ax = plt.subplots(figsize=(10, 4))  # 设置图形大小
        ax.axis('tight')
        ax.axis('off')
        table_data = df.values
        column_labels = df.columns
        table = ax.table(cellText=table_data, colLabels=column_labels, cellLoc='center', loc='center')
        
        # 调整表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)  # 调整表格大小

        # 保存为图片
        plt.savefig(img_filename, bbox_inches='tight', dpi=300)
        plt.show()

        
    def train_process(self):

        start_time = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.n_epochs):
            
            train_epoch_loss = self.train_epoch(epoch)
            
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break

            # validation
            val_loss = self.validate_epoch(epoch)
            
            early_stopping(val_loss, self.model, epoch, self.best_path)
            
            if early_stopping.early_stop:
                self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                "Training stops.".format(self.args.patience))
                break
        
        training_time = time.time() - start_time
        self.logger.info("== Training finished.\n"
                    "Total training time: {:.2f} min\t"
                    "best loss: {:.4f}\t"
                    "best epoch: {}\t".format(
                        (training_time / 60), 
                        -early_stopping.best_score, 
                        early_stopping.best_epoch))

        #self.test()

        # 獲取信心度和原始檔案名
        confidence_scores, original_files = self.test()  # 更新為返回原始檔案名

        # 儲存信心度和原始檔案名
        self.save_confidence_scores(confidence_scores, original_files)

    def evaluate_confidence(self, data):
        self.model.eval()
        with torch.no_grad():
            data.to(self.device)
            outputs, _, _, _ = self.model(data)
            probabilities = F.softmax(outputs, dim=1)
            confidence_scores, predicted_classes = torch.max(probabilities, dim=1)

        return confidence_scores, predicted_classes, data  # 確保返回三個值

    def save_confidence_scores(self, confidence_scores, original_files, filename='confidence_scores.csv'):
        df = pd.DataFrame({'Original File': original_files, 'Confidence': confidence_scores})
        df.to_csv(filename, index=False)
        self.logger.info(f"Confidence scores saved to {filename}")