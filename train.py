import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from core.modules.models import Log1DNetv2, Log1DNetv3
from ignite.contrib.metrics.regression.r2_score import R2Score
from tqdm.auto import tqdm
from numpy import Inf

def criterion(model:Log1DNetv2|Log1DNetv3, dataloader:DataLoader, train:bool=False, optimizer:Adam=None):
    dtc_loss, dts_loss = MSELoss(), MSELoss()
    dtc_score, dts_score = R2Score(), R2Score()#.to('cuda:0')

    dtc_avg_loss = 0; dtc_avg_score = 0
    dts_avg_loss = 0; dts_avg_score = 0

    count = 0

    for input, output in iter(dataloader):
        predictions = model(input)
        #dtc
        dtc_loss_ = torch.sqrt(dtc_loss(predictions[:, 0], output[:, 0]))
        dtc_score.update([predictions[:, 0], output[:, 0]])
        dtc_score_ = dtc_score.compute()

        #dts
        dts_loss_ = torch.sqrt(dts_loss(predictions[:, 1], output[:, 1]))
        dts_score.update([predictions[:, 1], output[:, 1]])
        dts_score_ = dts_score.compute()

        if train:
            optimizer.zero_grad()
            dtc_loss_.backward(retain_graph=True)
            dts_loss_.backward()
            optimizer.step()

        dtc_avg_loss += dtc_loss_.item()
        dtc_avg_score += dtc_score_

        dts_avg_loss += dts_loss_.item()
        dts_avg_score += dts_score_

        count += 1

    return dtc_avg_loss/count, dtc_avg_score/count, dts_avg_loss/count, dts_avg_score/count

def main(model:Log1DNetv2, trainloader:DataLoader, testloader:DataLoader, EPOCHS:int):
    train_dtc_loss, train_dtc_score, train_dts_loss, train_dts_score = list(), list(), list(), list()
    val_dtc_loss, val_dtc_score, val_dts_loss, val_dts_score = list(), list(), list(), list()

    optimizer = Adam(model.parameters(), lr=1e-1)

    best_val_loss = Inf

    for epoch in tqdm(range(EPOCHS)):
        tn_dtc_avg_loss, tn_dtc_avg_score, tn_dts_avg_loss, tn_dts_avg_score = criterion(model, trainloader, train=True, optimizer=optimizer)
        tt_dtc_avg_loss, tt_dtc_avg_score, tt_dts_avg_loss, tt_dts_avg_score = criterion(model, testloader, train=False, optimizer=optimizer)
        #train
        train_dtc_loss.append(tn_dtc_avg_loss); train_dtc_score.append(tn_dtc_avg_score)
        train_dts_loss.append(tn_dts_avg_loss); train_dts_score.append(tn_dts_avg_score)
        #val
        val_dtc_loss.append(tt_dtc_avg_loss); val_dtc_score.append(tt_dtc_avg_score)
        val_dts_loss.append(tt_dts_avg_loss); val_dts_score.append(tt_dts_avg_score)

        if (tt_dtc_avg_loss+tt_dts_avg_loss) < best_val_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            print('Model Saved')

            best_val_loss = tt_dtc_avg_loss+tt_dts_avg_loss

        print(f'Epoch {epoch+1} Train: DTC Loss={tn_dtc_avg_loss:.4f} | DTC Score={tn_dtc_avg_score:.4f} | DTS Loss={tn_dts_avg_loss:.4f} | DTS Score={tn_dts_avg_score:.4f}')
        print(f'Epoch {epoch+1} Val: DTC Loss={tt_dtc_avg_loss:.4f} | DTC Score={tt_dtc_avg_score:.4f} | DTS Loss={tt_dts_avg_loss:.4f} | DTS Score={tt_dts_avg_score:.4f}')
        print()


    return train_dtc_loss, train_dtc_score, train_dts_loss, train_dts_score, val_dtc_loss, val_dtc_score, val_dts_loss, val_dts_score

model = Log1DNetv3()

print(model.cuda())

random_seed = 9898
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

train_dtc_loss, train_dtc_score, train_dts_loss, train_dts_score,\
 val_dtc_loss, val_dtc_score, val_dts_loss, val_dts_score = main(model, trainloader, valloader)

