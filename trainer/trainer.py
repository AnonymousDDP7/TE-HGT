import tqdm
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import os 

class NotALightningTrainer(object):
    def __init__(self, args, callbacks):
        self.args = args

        self.epoch = 0
        self.global_step = 0

        self.callbacks = callbacks
        for callback in callbacks:
            callback.trainer = self

        self.should_stop = False

    def stop(self):
        self.should_stop = True

    def fit(self, model, train_dataloader, val_dataloader):

        optimizer = model.configure_optimizers()

        if not hasattr(model.model, "module"):
            # distributed data parallel??
            model.model = nn.DataParallel(model.model, device_ids=[0])
            model.model = model.model.cuda()

        self.model_hook = model.model
        f1_max = 0.0
        log_data = []
        path2save = "model.pth"
        for epoch in tqdm.tqdm(range(self.args.epochs)):

            if self.should_stop:
                break

            for callback in self.callbacks:
                callback.on_epoch_start()

            pbar = tqdm.tqdm(train_dataloader, total=len(train_dataloader))

            model.training_epoch_start(epoch)
            loss_train = 0.0
            for i, data in enumerate(pbar):
                self.global_step += 1
                optimizer.zero_grad()

                for callback in self.callbacks:
                    callback.on_batch_start()

                # data = data.cuda()
                for key in data.keys():
                    data[key] = data[key].cuda()

                loss = model.training_step(data, i)
                loss = loss / self.args.accumulation_steps
                loss_train += loss.item()
                loss.backward()
                if (i + 1) % self.args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                    optimizer.step()
                    for callback in self.callbacks:
                        callback.on_batch_end()

            loss_train = loss_train / len(train_dataloader)
            print(f"::: Epoch {epoch} | Train Loss: {loss_train:.4f}")
            model.training_epoch_end()
            self.epoch += 1
            model.model.train(False)
            with torch.no_grad():
                outputs = []
                pbar_eval = tqdm.tqdm(val_dataloader, total=len(val_dataloader))
                for i, data in enumerate(pbar_eval):
                    # data = data.cuda()
                    for key in data.keys():
                        data[key] = data[key].cuda()
                    out = model.validation_step(data, i)
                    outputs.append(out)
                    pbar_eval.set_description(f"Validating")

                f1_val, rec_val, pre_val, auc_val, acc_val = model.validation_epoch_end(outputs)
                if f1_val >= f1_max:
                    f1_max = f1_val
                    if os.path.exists(path2save):
                        os.remove(path2save)   

                    if "twitter" in self.args.dataset:
                        path2save = "./checkpoints/{}-{}-ws-{}-{}-{}-K-{}-date_threshold-{}/fold-{}-epoch={}_f1_{:.4f}_rec_{:.4f}_pre_{:.4f}_auc_{:.4f}_acc_{:.4f}.pth".format(
                            self.args.model,
                            self.args.dataset,
                            self.args.window_size,
                            self.args.image_embeddings_type,
                            self.args.text_embeddings_type,
                            self.args.K,
                            self.args.date_threshold,
                            self.args.fold,
                            self.epoch,
                            f1_val,
                            rec_val,
                            pre_val,
                            auc_val,
                            acc_val,
                        )
                    else:  # reddit
                        path2save = "./checkpoints/{}-{}-ws-{}-{}-{}-K-{}-date_threshold-{}/val-epoch={}_f1_{:.4f}_rec_{:.4f}_pre_{:.4f}_auc_{:.4f}_acc_{:.4f}.pth".format(
                                self.args.model,
                                self.args.dataset,
                                self.args.window_size,
                                self.args.image_embeddings_type,
                                self.args.text_embeddings_type,
                                self.args.K,
                                self.args.date_threshold,
                                self.epoch,
                                f1_val,
                                rec_val,
                                pre_val,
                                auc_val,
                                acc_val,
                            )
                    
                    torch.save(model.model.state_dict(), path2save)
                    print(f"::: New best model saved with F1: {f1_max:.4f}")
            
            log_data.append([epoch, loss_train, f1_val, rec_val, pre_val, auc_val, acc_val])

            # Save log
            log_data_frame = np.asarray(log_data)
            log_data_frame = pd.DataFrame(log_data_frame, columns=[['Epoch', 'Loss_train', 'F1_val', 'Recall_val', 'Precision_val', 'AUC_val', 'Acc_val']])
            if self.args.dataset == "twitter":
                log_data_frame.to_csv("./checkpoints/{}-{}-ws-{}-{}-{}-K-{}-date_threshold-{}/training-log-fold-{}.csv".format(
                                self.args.model,
                                self.args.dataset,
                                self.args.window_size,
                                self.args.image_embeddings_type,
                                self.args.text_embeddings_type,
                                self.args.K,
                                self.args.date_threshold,
                                self.args.fold), index=False)
            else:  # reddit
                log_data_frame.to_csv("./checkpoints/{}-{}-ws-{}-{}-{}-K-{}-date_threshold-{}/training-log.csv".format(
                                self.args.model,
                                self.args.dataset,
                                self.args.window_size,
                                self.args.image_embeddings_type,
                                self.args.text_embeddings_type,
                                self.args.K,
                                self.args.date_threshold), index=False)
                
            model.model.train(True)
