import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import random
import scipy.io as scion
import datetime
import calendar

def seed_torch(seed=200):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def Makedate(numtrain,yyear):
    fr = datetime.datetime(year=yyear, month=1, day=1)
    ed = datetime.datetime(year=yyear + 1, month=1, day=1)
    fr = fr.strftime('%Y-%m-%d')
    ffr = np.datetime64(fr)
    ed = ed.strftime('%Y-%m-%d')
    EDate = np.arange(fr, ed, dtype='datetime64')
    if EDate.shape[0] == 366:
        EEDate = np.delete(EDate,-1)
    else:
        EEDate = EDate.copy()

    EDay = (EEDate - ffr).astype(int) + 1
    EDay = np.tile(EDay, (numtrain, 1))
    EDay = EDay.T
    EDay = EDay.reshape(EDay.shape[0] * EDay.shape[1], 1)
    EDay = np.squeeze(EDay)
    TrainDate = np.expand_dims(EDay, axis=1)
    return TrainDate


def MakeData(ttestindex,year,TType,n,num_station):

    if calendar.isleap(year):
        days = 366
    else:
        days = 365
    DataPath = 'Data/'
    fdd = 1 + days * (year - 2010)# 起始日
    edd = (days + 1) + days * (year - 2010) # 结束日
    Ta = scion.loadmat(DataPath+TType)
    Ta = np.array(Ta[TType])
    Ta = Ta[fdd:edd, :]
    SSelObs = Ta.copy()
    SSelObs = SSelObs.reshape(SSelObs.shape[0] * SSelObs.shape[1], 1)
    Seldata = np.delete(Ta, ttestindex, 1)
    Seldata = np.kron(Seldata, np.ones((num_station, 1)))

    disM = scion.loadmat(DataPath+'disM')
    SelDisM = np.array(disM['disM'])
    SelDisM = np.delete(SelDisM, ttestindex, 1)
    Dindex = np.argsort(SelDisM)
    Dindexx = Dindex.copy()
    Dindex = Dindexx[:, 1:(n+1)].copy()
    Dindex[ttestindex, 0:n] = Dindexx[ttestindex, 0:n].copy()
    DDindex = np.tile(Dindex, [days, 1])
    Dobs = DDindex.copy()
    Dobs = Dobs.astype(float)
    for i in np.arange(0, DDindex.shape[0], 1):
        Dobs[i, :] = Seldata[i, DDindex[i, :]]

    DLST = scion.loadmat(DataPath+'DLST')
    DLST = np.array(DLST['DLST'])
    DLST = DLST[fdd:edd, :]
    DLST = DLST.reshape(DLST.shape[0] * DLST.shape[1], 1)

    NLST = scion.loadmat(DataPath+'NLST')
    NLST = np.array(NLST['NLST'])
    NLST = NLST[fdd:edd, :]
    NLST = NLST.reshape(NLST.shape[0] * NLST.shape[1], 1)

    EVI = scion.loadmat(DataPath+'EVI')
    EVI = np.array(EVI['EVI'])
    EVI = EVI[fdd:edd, :]
    EVI = EVI.reshape(EVI.shape[0] * EVI.shape[1], 1)

    DEM = scion.loadmat(DataPath+'DEM')
    DEM = np.array(DEM['DEM'])
    DEMM = DEM[:,1:4]
    DEMM = np.tile(DEMM, [days,1])

    TrainDate = Makedate(num_station,year)
    target = SSelObs

    train = np.concatenate((DLST, NLST, DEMM, EVI,Dobs,TrainDate), axis=1)
    return train, target


def MakeAllData(year,TType,n,num_station):

    if calendar.isleap(year):
        days = 366
    else:
        days = 365
    DataPath = 'Data/'
    fdd = 1 + days * (year - 2010)# 起始日
    edd = (days + 1) + days * (year - 2010)    # 结束日

    Ta = scion.loadmat(DataPath+TType)
    Ta = np.array(Ta[TType])
    Ta = Ta[fdd:edd, :]
    SSelObs = Ta.copy()
    SSelObs = SSelObs.reshape(SSelObs.shape[0] * SSelObs.shape[1], 1)
    Seldata = np.kron(Ta, np.ones((num_station, 1)))


    disM = scion.loadmat(DataPath+'disM')
    SelDisM = np.array(disM['disM'])
    Dindex = np.argsort(SelDisM)
    Dindexx = Dindex.copy()
    Dindex = Dindexx[:, 1:(n+1)].copy()
    DDindex = np.tile(Dindex, [days, 1])
    Dobs = DDindex.copy()
    Dobs = Dobs.astype(float)
    for i in np.arange(0, DDindex.shape[0], 1):
        Dobs[i, :] = Seldata[i, DDindex[i, :]]

    DLST = scion.loadmat(DataPath + 'DLST')
    DLST = np.array(DLST['DLST'])
    DLST = DLST[fdd:edd, :]
    DLST = DLST.reshape(DLST.shape[0] * DLST.shape[1], 1)

    NLST = scion.loadmat(DataPath + 'NLST')
    NLST = np.array(NLST['NLST'])
    NLST = NLST[fdd:edd, :]
    NLST = NLST.reshape(NLST.shape[0] * NLST.shape[1], 1)

    EVI = scion.loadmat(DataPath + 'EVI')
    EVI = np.array(EVI['EVI'])
    EVI = EVI[fdd:edd, :]
    EVI = EVI.reshape(EVI.shape[0] * EVI.shape[1], 1)

    DEM = scion.loadmat(DataPath + 'DEM')
    DEM = np.array(DEM['DEM'])
    DEMM = DEM[:, 1:4]
    DEMM = np.tile(DEMM, [days, 1])

    TrainDate = Makedate(num_station, year)
    target = SSelObs

    train = np.concatenate((DLST, NLST, DEMM, EVI,Dobs,TrainDate), axis=1)
    return train, target


def LoadData(traindex, valindex, testindex,year,TType,n,dd,num_station):

    if calendar.isleap(year):
        days = 366
    else:
        days = 365
    Sel_data = days * num_station
    train, target = MakeData(np.concatenate((valindex,testindex)),year,TType,n,num_station)
    dtarget = pd.DataFrame(data=target)
    dtrain = pd.DataFrame(data=train)
    dsize = dtrain.shape[1]
    dtrain = pd.get_dummies(dtrain, columns=[dsize - 1], dtype=int)
    dtrain = dtrain.to_numpy().astype(np.float32)
    dtarget = dtarget.to_numpy().astype(np.float32)

    delt = np.arange(0, Sel_data, num_station)
    ttrainindex = (np.tile(traindex, (len(delt), 1)).T + delt).T.flatten().tolist()
    vvalindex = (np.tile(valindex, (len(delt), 1)).T + delt).T.flatten().tolist()

    OHD = days + n
    dsize = dtrain.shape[1] - OHD

    Scaler = StandardScaler(with_mean=True, with_std=True)
    Ptrain = np.concatenate((dtrain[:, :dsize], np.tile(target, [1, n])), axis=1)
    Scaler.fit(Ptrain)

    trainx = dtrain[ttrainindex, :]
    trainy = dtarget[ttrainindex, :]
    valx = dtrain[vvalindex, :]
    valy = dtarget[vvalindex, :]

    dsize = Ptrain.shape[1]
    trainx[:, :dsize] = Scaler.transform(trainx[:, :dsize])
    valx[:, :dsize] = Scaler.transform(valx[:, :dsize])

    trainxx = np.reshape(trainx, (-1, len(traindex), trainx.shape[1]))
    trainyy = np.reshape(trainy, (-1, len(traindex)))

    valxx = np.reshape(valx, (-1, len(valindex), valx.shape[1]))
    valyy = np.reshape(valy, (-1, len(valindex)))

    kk = 2 * dd + 1
    trainxx = np.concatenate((np.tile(trainxx[0, :, :], (dd, 1, 1)), trainxx), axis=0)
    trainxx = np.concatenate((trainxx, np.tile(trainxx[-1, :, :], (dd, 1, 1))), axis=0)
    valxx = np.concatenate((np.tile(valxx[0, :, :], (dd, 1, 1)), valxx), axis=0)
    valxx = np.concatenate((valxx, np.tile(valxx[-1, :, :], (dd, 1, 1))), axis=0)

    TRX, TRY = CreateLSTMDataset(trainxx, trainyy, k=kk)
    VAX, VAY = CreateLSTMDataset(valxx, valyy, k=kk)

    return TRX, TRY, VAX, VAY


def LoadDataa(traindex, testindex,year,TType,n,dd,num_station):

    if calendar.isleap(year):
        days = 366
    else:
        days = 365
    Sel_data = days * num_station
    train, target = MakeData(testindex, year, TType,n,num_station)
    dtarget = pd.DataFrame(data=target)
    dtrain = pd.DataFrame(data=train)
    dsize = dtrain.shape[1]
    dtrain = pd.get_dummies(dtrain, columns=[dsize - 1], dtype=int)
    dtrain = dtrain.to_numpy().astype(np.float32)
    dtarget = dtarget.to_numpy().astype(np.float32)

    delt = np.arange(0, Sel_data, num_station)
    ttrainindex = (np.tile(traindex, (len(delt), 1)).T + delt).T.flatten().tolist()
    ttestindex = (np.tile(testindex, (len(delt), 1)).T + delt).T.flatten().tolist()

    OHD = days + n
    dsize = dtrain.shape[1] - OHD

    Scaler = StandardScaler(with_mean=True, with_std=True)
    Ptrain = np.concatenate((dtrain[:, :dsize], np.tile(target, [1, n])), axis=1)
    Scaler.fit(Ptrain)

    trainx = dtrain[ttrainindex, :]
    trainy = dtarget[ttrainindex, :]

    testx = dtrain[ttestindex, :]
    testy = dtarget[ttestindex, :]

    dsize = Ptrain.shape[1]
    trainx[:, :dsize] = Scaler.transform(trainx[:, :dsize])
    testx[:, :dsize] = Scaler.transform(testx[:, :dsize])

    trainxx = np.reshape(trainx, (-1, len(traindex), trainx.shape[1]))
    trainyy = np.reshape(trainy, (-1, len(traindex)))

    testxx = np.reshape(testx, (-1, len(testindex), testx.shape[1]))
    testyy = np.reshape(testy, (-1, len(testindex)))

    kk = 2 * dd + 1
    trainxx = np.concatenate((np.tile(trainxx[0, :, :], (dd, 1, 1)), trainxx), axis=0)
    trainxx = np.concatenate((trainxx, np.tile(trainxx[-1, :, :], (dd, 1, 1))), axis=0)

    testxx = np.concatenate((np.tile(testxx[0, :, :], (dd, 1, 1)), testxx), axis=0)
    testxx = np.concatenate((testxx, np.tile(testxx[-1, :, :], (dd, 1, 1))), axis=0)

    TRX, TRY = CreateLSTMDataset(trainxx, trainyy, k=kk)
    TEX, TEY = CreateLSTMDataset(testxx, testyy, k=kk)
    return TRX, TRY, TEX, TEY



def LoadAllData(year,TType,n,dd,num_station):

    if calendar.isleap(year):
        days = 366
    else:
        days = 365
    Sel_data = days * num_station
    train, target = MakeAllData(year,TType,n,num_station)
    dtarget = pd.DataFrame(data=target)
    dtrain = pd.DataFrame(data=train)
    dsize = dtrain.shape[1]
    dtrain = pd.get_dummies(dtrain, columns=[dsize - 1], dtype=int)
    dtrain = dtrain.to_numpy().astype(np.float32)
    dtarget = dtarget.to_numpy().astype(np.float32)

    delt = np.arange(0, Sel_data, num_station)
    ttrainindex = (np.tile(np.arange(0, num_station, 1), (len(delt), 1)).T + delt).T.flatten().tolist()

    OHD = days + n
    dsize = dtrain.shape[1] - OHD

    Scaler = StandardScaler(with_mean=True, with_std=True)
    Ptrain = np.concatenate((dtrain[:, :dsize], np.tile(target, [1, n])), axis=1)
    Scaler.fit(Ptrain)

    trainx = dtrain[ttrainindex, :]
    trainy = dtarget[ttrainindex, :]

    dsize = Ptrain.shape[1]
    trainx[:, :dsize] = Scaler.transform(trainx[:, :dsize])

    trainxx = np.reshape(trainx, (-1, num_station, trainx.shape[1]))
    trainyy = np.reshape(trainy, (-1, num_station))

    kk = 2 * dd + 1
    trainxx = np.concatenate((np.tile(trainxx[0, :, :], (dd, 1, 1)), trainxx), axis=0)
    trainxx = np.concatenate((trainxx, np.tile(trainxx[-1, :, :], (dd, 1, 1))), axis=0)

    TRX, TRY = CreateLSTMDataset(trainxx, trainyy, k=kk)

    return TRX, TRY

def CreateLSTMData(tx,ty,k):
    x = []
    y = []
    L = len(ty)
    for i in range(L):
        xx = tx[i:i+k]
        yy = ty[i]
        x.append(xx)
        y.append(yy)
    return x,y


def CreateLSTMDataset(ttx, tty, k):
    TXData = []
    TYData = []
    for i in range(ttx.shape[1]):
       tx, ty = CreateLSTMData(ttx[:,i,:], tty[:,i], k)
       if (np.isnan(tx).sum() + np.isnan(ty).sum()) == 0:
           TXData.extend(tx)
           TYData.extend(ty)

    return TXData,TYData

def CreateLSTMPre(tx):
    x = []
    x.append(tx)
    return x

def CreateLSTMPreset(ttx):
    TXData = []
    for i in range(ttx.shape[1]):
       tx= CreateLSTMPre(ttx[:,i,:])
       TXData.extend(tx)
    return TXData

class LoadDataset(Dataset):

    def __init__(self,train,target):

        self.x_data = torch.from_numpy(np.array(train)).to(torch.float32)
        self.y_data = torch.from_numpy(np.array(target)).to(torch.float32).unsqueeze(1)
        self.len = len(target)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class LoadPreDataset(Dataset):

    def __init__(self,train):

        self.x_data = torch.from_numpy(np.array(train)).to(torch.float32)
        self.len = len(train)

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len

def GetDatasets(traindex, valindex, testindex,year,TType,n,dd,num_station):

    trainx, trainy, valx, valy = LoadData(traindex, valindex, testindex, year, TType, n, dd,num_station)
    dtrainSet = LoadDataset(trainx, trainy)
    train_dataset = DataLoader(dataset=dtrainSet,
                               batch_size=2048,
                               num_workers=4,
                               pin_memory=True,
                               prefetch_factor=4,
                               drop_last=True,
                               shuffle=True)

    dvalSet = LoadDataset(valx, valy)
    val_dataset = DataLoader(dataset=dvalSet,
                              batch_size=50000,
                              drop_last=False,
                              shuffle=False)

    return train_dataset, val_dataset


def GetDatasett(traindex,testindex,year,TType,n,dd,num_station):

    trainx, trainy, testx, testy = LoadDataa(traindex,testindex,year,TType,n,dd,num_station)
    dtrainSet = LoadDataset(trainx, trainy)
    train_dataset = DataLoader(dataset=dtrainSet,
                               batch_size=2048,
                               num_workers=4,
                               pin_memory=True,
                               prefetch_factor=4,
                               drop_last=True,
                               shuffle=True)


    dtestSet = LoadDataset(testx, testy)
    test_dataset = DataLoader(dataset=dtestSet,
                              batch_size=50000,
                              drop_last=False,
                              shuffle=False)

    return train_dataset, test_dataset


def GetAllDatasets(year,TType,n,dd,num_station):

    trainx, trainy = LoadAllData(year, TType, n, dd,num_station)
    dtrainSet = LoadDataset(trainx, trainy)
    train_dataset = DataLoader(dataset=dtrainSet,
                               batch_size=2048,
                               num_workers=4,
                               pin_memory=True,
                               prefetch_factor=4,
                               drop_last=True,
                               shuffle=True)
    return train_dataset

