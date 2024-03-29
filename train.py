import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from core import *
from ignite.contrib.metrics.regression.r2_score import R2Score
from tqdm.auto import tqdm
from numpy import Inf
from argparse import ArgumentParser
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

# for reproducibilty
random_seed = 9898
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)


#------------------------------------------------------------------------------------------------
# ArgumentParser

parser = ArgumentParser(description= 'Argument to required')

parser.add_argument('-e', '--num_epochs', default=100, metavar='EPOCH', type=int, help='number of epochs')
parser.add_argument('-bs', '--batch_size', default=64, type=int, help='number of batch or sample size')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, metavar='LR', help='learning rate')
parser.add_argument('-path', '--data_dir', default='./data', help='data directory', type=str)

args = parser.parse_args()

#---------------------------------------------------------------

# Dataloading
# well_df = loader(folder=args.data_dir)
# well1,well2,well3,well4,well5,well6,well7,well8 = well_df

# get all paths and alphabetically ordered
# paths = sorted(glob.glob(os.path.join("./", "*.LAS")))
paths = ['15_9-F-11A.LAS', '15_9-F-11T2.LAS', '15_9-F-1A.LAS', '15_9-F-1B.LAS',
         '15_9-F-4.LAS', '15_9-F-14.LAS', '15_9-F-5.LAS', '15_9-F-15D.LAS']
well_df = [0] * 8

for i in range(len(paths)):
    # read with lasio and convert to dataframe
    df = (lasio.read(os.path.join(args.data_dir, paths[i]))).df()

    well_df[i] = df.reset_index()

well1, well2, well3, well4, well5, well6, well7, well8 = well_df

well1 = process_train(well1, col, name='15_9-F-11A')
well2 = process_test(well2, col, name='15_9-F-11T2') #validation
well3 = process_train(well3, col, name='15_9-F-1A')
well4 = process_train(well4, col, name='15_9-F-1B')
well5 = process_test(well5, col, name='15_9-F-4') #validation
# well6 = process_test(well6, col, name='15_9-F-14') #blind test
well7 = process_test(well7, col, name='15_9-F-5') #blind
well8 = process_test(well8, col, name='15_9-F-15D') #blind
print(well1)


train = pd.concat((well1, well3, well4), axis='index').reset_index(drop=True)
test = pd.concat((well2, well5), axis='index').reset_index(drop=True)
test2 = pd.concat((well7, well8), axis='index').reset_index(drop=True)


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
# data_processing
wells = pd.concat((train, test), axis='index').reset_index(drop=True)

wells['GR'] = np.where(wells['GR'] <= 200., wells['GR'], 200.)
wells['RT'] = np.where(wells['RT'] <= 2000., wells['RT'], 2000.)
# wells['NPHI'] = np.where(wells['NPHI'] <= 0.5, wells['NPHI'], 0.5)

wells['RT'] = np.log10(wells['RT'])

features = ['GR', 'RT', 'RHOB', 'NPHI'] # only feature column names
target = ['DTC', 'DTS'] # only target column names
feature_target = np.append(features, target)

# normalize using power transform Yeo-Johnson method
scaler = PowerTransformer(method='yeo-johnson')

# ColumnTransformer
ct = ColumnTransformer([('transform', scaler, features)], remainder='passthrough')

# fit and transform
new_wells = ct.fit_transform(wells)

#convert to dataframe
wells_norm = pd.DataFrame(new_wells, columns=['GR', 'RT', 'RHOB', 'NPHI', 'DEPTH', 'WELL', 'DTC', 'DTS'])

#convert into correct type
x = wells_norm[features].astype(float)
y = wells_norm['WELL'].astype(str)
z = wells_norm['DEPTH'].astype(float)

#joining data together
wells_norm = pd.concat([z, y, x], axis=1)
wells_norm['DTC'] = wells.DTC
wells_norm['DTS'] = wells.DTS

#-------------------------------------------------------------------------------------------------
# data spliting

train_ = wells_norm[:train.shape[0]]
test_ = wells_norm[train.shape[0]:]

train_x_ = train_[train_.columns[2:6]]
train_y_ = train_[['DTC', 'DTS']]

train_x, val_x, train_y, val_y = train_test_split(train_x_, train_y_, shuffle=True,
                                                  test_size=0.3, random_state=2024)

test_x = test_[test_.columns[2:6]]
test_y = test_[['DTC', 'DTS']]

test_well2_x = test_x[:well2.shape[0]]
test_well2_y = test_y[:well2.shape[0]]

test_well5_x = test_x[well2.shape[0]:]
test_well5_y = test_y[well2.shape[0]:]

train_well1_x = train_x_[:well1.shape[0]]
train_well1_y = train_y_[:well1.shape[0]]

train_well3_x = train_x_[well1.shape[0]:(well1.shape[0]+well3.shape[0])]
train_well3_y = train_y_[well1.shape[0]:(well1.shape[0]+well3.shape[0])]

train_well4_x = train_x_[(well1.shape[0]+well3.shape[0]):]
train_well4_y = train_y_[(well1.shape[0]+well3.shape[0]):]

#----------------------------------------------------------------------------------------------
# creating pytorch dataaloader

trainloader = dataloader(train_x, train_y, bs=args.bs)
valloader = dataloader(val_x, val_y, bs=args.bs)
testloader = dataloader(test_x, test_y, bs=args.bs)

#-------------------------------------------------------------------------------------------------------
# Model Training and Validation



def main(model:Log1DNetv2, trainloader:DataLoader, testloader:DataLoader, EPOCHS:int=args.e):
    train_dtc_loss, train_dtc_score, train_dts_loss, train_dts_score = list(), list(), list(), list()
    val_dtc_loss, val_dtc_score, val_dts_loss, val_dts_score = list(), list(), list(), list()

    optimizer = Adam(model.parameters(), lr=args.lr)

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


if __name__ == '__main__':


    model = Log1DNetv3(batch_size=args.bs)

    print(model.cuda())



    train_dtc_loss, train_dtc_score, train_dts_loss, train_dts_score,\
    val_dtc_loss, val_dtc_score, val_dts_loss, val_dts_score = main(model, trainloader, valloader)

