import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split,KFold
from osgeo import gdal
from sklearn.preprocessing import StandardScaler
import scipy.io as scion
from utils.CLSTNtools import seed_torch,GetDatasets,GetDatasett,GetAllDatasets,Makedate,MakeAllData,CreateLSTMPreset,LoadPreDataset,DataLoader
import time
import calendar
import datetime
import sys
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ATLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, dd):
        super(ATLSTM, self).__init__()
        seed_torch()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.dd = dd
        self.lstm = nn.LSTM(input_size,hidden_size,num_layer,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)
    def forward(self, x, type):
        if type == 1:
            h0 = torch.randn(self.num_layer*2,x.size(0), self.hidden_size).to(device)
            c0 =  torch.randn(self.num_layer*2,x.size(0), self.hidden_size).to(device)
        else:
            h0 = torch.randn(self.num_layer * 2, x.size(0), self.hidden_size).half().to(device)
            c0 =  torch.randn(self.num_layer*2,x.size(0), self.hidden_size).half().to(device)
        out,_ = self.lstm(x,(h0,c0))
        return self.fc(out[:,self.dd,:])


def TrainModel(year,TType):

    if calendar.isleap(year):
        days = 366
    else:
        days = 365

    if TType == 'TMean':
        hsize = 60
        dd = 5
        n=4
        hlayer = 3
    else:  # TTMax  TTMin
        hsize = 50
        dd = 4
        n=4
        hlayer = 3

    num_station = 117

    seed_torch()
    traindexx, othindex = train_test_split(np.arange(0, num_station, 1),test_size=0.2,shuffle=True)
    kf = KFold(n_splits=5, shuffle=True)
    max_num_epochs = 70
    Valscore = np.zeros([5, max_num_epochs])
    # 5-fold cross-validation for training data
    for nf, (traindex, testindex) in enumerate(kf.split(traindexx)):
        ttraindex = traindexx[traindex]
        ttestindex = traindexx[testindex]
        seed_torch()
        ttrain_dataset, ttest_dataset = GetDatasets(ttraindex,ttestindex,othindex,year,TType,n,dd,num_station)
        netmodel = ATLSTM(375, hsize, hlayer, dd).to(device)
        loss_fun = nn.MSELoss()
        loss_fun2 = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(netmodel.parameters(), lr=0.002)
        LType = torch.tensor(1, dtype=torch.int32)
        LType = LType.to(device)
        for epoch in range(max_num_epochs):
            netmodel.train()
            for step, (x, y) in enumerate(ttrain_dataset):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                toutputs = netmodel(x, LType)
                tloss = loss_fun(y, toutputs)
                tloss.backward()
                optimizer.step()

            netmodel.eval()
            with torch.no_grad():
                totloss = 0
                ssize = 0
                for data in ttest_dataset:
                    x, y = data
                    ssize += len(y)
                    x, y = x.to(device), y.to(device)
                    voutputs = netmodel(x, LType)
                    vtloss = loss_fun2(y, voutputs)
                    totloss += vtloss.item()
                totloss = np.sqrt(totloss / ssize)
            print("Epoch=%d:the RMSE of the validation is %f" % (epoch, totloss))
            Valscore[nf, epoch] = totloss

    Epoch = np.sum(Valscore, axis=0)
    Estd = np.std(Valscore, axis=0)
    ep = np.argmin(Epoch)
    std = Estd[ep]
    print("Year=%d: the optimal RMSE of the cross-validation is %f" % (year, np.min(Epoch)/5))
    print("Year=%d: the optimal Std of the cross-validation is %f" % (year, std))
    print("Year=%d: the optimal epocah of the cross-validation is %d" % (year, ep + 1))
    seed_torch()
    torch.cuda.empty_cache()

    # Retrain the model using the entire training data
    train_dataset, test_dataset = GetDatasett(traindexx, othindex, year, TType,n,dd,num_station)
    netmodel1 = ATLSTM(375, hsize, hlayer, dd).to(device)
    loss_fun = nn.MSELoss()
    loss_fun2 = nn.MSELoss(reduction='sum')
    optimizer1 = torch.optim.Adam(netmodel1.parameters(), lr=0.002)
    LType = torch.tensor(1, dtype=torch.int32)
    LType = LType.to(device)
    for epoch in range(ep+1):
            netmodel1.train()
            for step, (x, y) in enumerate(train_dataset):
                x, y = x.to(device), y.to(device)
                optimizer1.zero_grad()
                toutputs = netmodel1(x, LType)
                tloss = loss_fun(y, toutputs)
                tloss.backward()
                optimizer1.step()

            netmodel1.eval()
            totloss = 0
            ssize = 0
            with torch.no_grad():
                for data in test_dataset:
                        x, y = data
                        ssize += len(y)
                        x, y = x.to(device), y.to(device)
                        voutputs = netmodel1(x, LType)
                        vtloss = loss_fun2(y, voutputs)
                        totloss += vtloss.item()
            valloss = np.sqrt(totloss / ssize)
            print("Epoch=%d:the RMSE of the validation is %f" % (epoch, valloss))
    print("Year=%d:the optimal epoch is %d" % (year, ep + 1))
    print("Year=%d:the RMSE is %f" % (year, valloss))

    # Evaluate the model using GTa from test stations
    netmodel1.eval()
    totloss = 0
    ssize = 0
    ycpu=[]
    ycpu=np.array(ycpu)
    pcpu=[]
    pcpu=np.array(pcpu)
    with torch.no_grad():
        for data in test_dataset:
                x, y = data
                ssize += len(y)
                x, y = x.to(device), y.to(device)
                voutputs = netmodel1(x, LType)
                vtloss = loss_fun2(y, voutputs)
                totloss += vtloss.item()
                ycpu = np.concatenate((ycpu,np.squeeze(y.cpu().numpy())),axis=0)
                pcpu = np.concatenate((pcpu,np.squeeze(voutputs.cpu().numpy())),axis=0)
    totloss = np.sqrt(totloss / ssize)
    print("Year=%d:the RMSE of the validation is %f" % (year, totloss))

    testnum = len(othindex)
    ycpu = np.transpose(ycpu.reshape(testnum, days))
    pcpu = np.transpose(pcpu.reshape(testnum, days))

    Yresult = pd.DataFrame(ycpu)
    Yresult.to_csv('Result/' + TType + '/' + str(year) + 'Obs.csv', index=True, header=othindex)
    Presult = pd.DataFrame(pcpu)
    Presult.to_csv('Result/' + TType + '/' + str(year) + 'Pre.csv', index=True, header=othindex)

    # Retained for the entire data
    ttrain_dataset = GetAllDatasets(year, TType, n, dd, num_station)
    seed_torch()
    netmodel2 = ATLSTM(375, hsize, hlayer, dd).to(device)
    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(netmodel2.parameters(), lr=0.002)
    LType = torch.tensor(1, dtype=torch.int32)
    LType = LType.to(device)
    for epoch in range(ep+1):
        netmodel2.train()
        for step, (x, y) in enumerate(ttrain_dataset):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            toutputs = netmodel2(x, LType)
            tloss = loss_fun(y, toutputs)
            tloss.backward()
            optimizer.step()

    modelpath = 'Model/' + TType + '/' + str(year) + '.pth'
    torch.save(netmodel2.state_dict(), modelpath)

# preparedata for mapping
def preparedata(batchsize,day,TType,Scaler,year):

    if TType == 'TMean':
        dd = 5
        kk=11
    else:  # TTMax  TTMin
        dd = 4
        kk=9

    DataPath = 'Data/'
    train = scion.loadmat('Data/UT-Mapping/Mmatrix.mat')
    tindex = np.array(train['Mmatrix'])
    data = scion.loadmat(DataPath+TType)
    obPre = np.array(data[TType])

    base = datetime.datetime.strptime('2010-01-01', "%Y-%m-%d")
    stryd = datetime.datetime.strptime(str(year) + '-01' + '-01', "%Y-%m-%d")-base
    stryd = stryd.days
    
    if (day < (dd + 1)):
        rept = dd + 1 - day
        strd = stryd + 1
        yst = 1
        endd = stryd + day + dd + 1
        yed = day + dd + 1
        index = 0

    elif (day > 365 - dd):
        rept = day - (365 - dd)
        strd = stryd + day - dd
        yst = day - dd
        endd = stryd + 366
        yed = 366
        index = -1
    else:
        rept = 0
        strd = stryd + day - dd
        yst = day - dd
        endd = stryd + day + dd + 1
        yed = day + dd + 1
        index = 0

    Seldata = obPre[strd:endd, :]

    if rept > 0:
        if (index < 0):
            repday = np.tile(Seldata[index], (rept, 1))
            Seldata = np.vstack((Seldata, repday))
        else:
            repday = np.tile(Seldata[index], (rept, 1))
            Seldata = np.vstack((repday, Seldata))


    Dobs = np.empty(shape=(tindex.shape[0] * kk, 4))
    for i in np.arange(0, 4):
        Dobs[:, i] = np.ravel(Seldata[:, tindex[:, i]], order='C')

    Lat = scion.loadmat('Data/LOC-Mapping/Lat.mat')
    Lat = np.array(Lat['Lat'])
    Lat = np.tile(Lat,(kk,1))

    Lon = scion.loadmat('Data/LOC-Mapping/Lon.mat')
    Lon = np.array(Lon['Lon'])
    Lon = np.tile(Lon,(kk,1))

    DEM = scion.loadmat('Data/LOC-Mapping/DEM.mat')
    DEM = np.array(DEM['DEM'])
    DEM = np.tile(DEM,(kk,1))

    for i in np.arange(yst,yed):

        DT = scion.loadmat('Data/LST-Mapping/LSTday'+ str(year)+str(i).zfill(3)+'.mat')
        DT = np.array(DT['LST'])
        if i == yst:
          DDT = DT
          numsel = DT.shape[0]
        else:
          DDT = np.concatenate([DDT,DT],axis=0)

        NT = scion.loadmat('Data/LST-Mapping/LSTnight'+ str(year)+str(i).zfill(3)+'.mat')
        NT = np.array(NT['LST'])
        if i == yst:
          NNT = NT
        else:
          NNT = np.concatenate([NNT,NT],axis=0)

        EVI = scion.loadmat('Data/EVI-Mapping/EVI' + str(year) + str(i).zfill(3) + '.mat')
        EVI = np.array(EVI['VI'])
        if i == yst:
            EEVI = EVI
        else:
            EEVI = np.concatenate([EEVI, EVI], axis=0)

    if rept > 0:
        if index < 0:
            repddt = np.tile(DDT[-numsel:], (rept, 1))
            repnnt = np.tile(NNT[-numsel:], (rept, 1))
            repevi = np.tile(EEVI[-numsel:], (rept, 1))
            DDT = np.concatenate((DDT, repddt), axis=0)
            NNT = np.concatenate((NNT, repnnt), axis=0)
            EEVI = np.concatenate((EEVI, repevi), axis=0)
        else:
            repddt = np.tile(DDT[:numsel], (rept, 1))
            repnnt = np.tile(NNT[:numsel], (rept, 1))
            repevi = np.tile(EEVI[:numsel], (rept, 1))
            DDT = np.concatenate((repddt, DDT), axis=0)
            NNT = np.concatenate((repnnt, NNT), axis=0)
            EEVI = np.concatenate((repevi, EEVI), axis=0)

    SPDate = np.concatenate([DDT,NNT,Lat,Lon,DEM,EEVI,Dobs] ,axis=1)

    SPDate[:,:10]=Scaler.transform(SPDate[:,:10])
    SPDate[np.isnan(SPDate)] = 0
    SPDate = np.reshape(SPDate,(-1,len(tindex),SPDate.shape[1]))
    SPDate = CreateLSTMPreset(SPDate)

    SPDate = LoadPreDataset(SPDate)
    SPDate = DataLoader(dataset=SPDate,
                               batch_size=batchsize,
                               drop_last=False,
                               num_workers=4,
                               pin_memory= True,
                               prefetch_factor=4,
                               shuffle=False)
    return SPDate,yst,yed,rept,index


def ModelPrediction(TType,year,dataset):

    if calendar.isleap(year):
        days = 366
    else:
        days = 365

    if TType == 'TMean':
        hsize = 60
        dd = 5
        n = 4
        hlayer = 3
        kk=11
    else:  # TTMax  TTMin
        hsize = 50
        dd = 4
        n = 4
        hlayer = 3
        kk=9

    num_station = 117

    seed_torch()
    batchsize = 250000
    Pdate = Makedate(1,year)
    Pdate = pd.DataFrame(data=Pdate)
    Pdate = pd.get_dummies(Pdate, columns=[0], dtype=int)
    Pdate = Pdate.to_numpy().astype(np.float32)
    netmodel = ATLSTM(375, hsize, hlayer, dd) # TTMean
    LType = torch.tensor(2, dtype=torch.int32)
    LType = LType.to(device)

    train, target = MakeAllData(year,TType,n,num_station)
    dtrain = pd.DataFrame(data=train)
    dsize = dtrain.shape[1]
    dtrain = pd.get_dummies(dtrain, columns=[dsize - 1], dtype=int)
    dtrain = dtrain.to_numpy().astype(np.float32)

    OHD = days + n
    dsize = dtrain.shape[1] - OHD

    Scaler = StandardScaler(with_mean=True, with_std=True)
    Ptrain = np.concatenate((dtrain[:, :dsize], np.tile(target, [1, n])), axis=1)
    Scaler.fit(Ptrain)

    for day in np.arange(6,8, 1):
        stime = time.time()
        Traindata,strd,endd,rept,index = preparedata(batchsize, day, TType,Scaler,year)
        SelDate = Pdate[(strd-1):(endd-1), :]
        if rept > 0:
            if (index < 0):
                repday = np.tile(SelDate[index], (rept, 1))
                SelDate = np.vstack((SelDate, repday))
            else:
                repday = np.tile(SelDate[index], (rept, 1))
                SelDate = np.vstack((repday, SelDate))
        SelDate1 = np.reshape(np.tile(SelDate, (batchsize, 1)), (-1, kk, SelDate.shape[1]))
        SelDate1 = torch.from_numpy(SelDate1)
        modelpath = 'Model/' + TType + '/' + str(year) + '.pth'
        if os.path.exists(modelpath):
            netmodel.load_state_dict(torch.load(modelpath))
        else:
            print("No file exist")
            sys.exit()
        netmodel.half()
        netmodel.to(device)
        netmodel.eval()
        with torch.no_grad():
            for step, x in enumerate(Traindata):
                if x.shape[0] == batchsize:
                    x = torch.cat((x, SelDate1), 2)
                    x = x.type(torch.HalfTensor).to(device)
                else:
                    SelDate2 = np.reshape(np.tile(SelDate, (x.shape[0], 1)), (-1, kk, SelDate.shape[1]))
                    SelDate2 = torch.from_numpy(SelDate2)
                    x = torch.cat((x, SelDate2), 2)
                    x = x.type(torch.HalfTensor).to(device)
                voutputs = netmodel(x,LType)
                voutputs = voutputs.float()
                if step == 0:
                    Predata = np.array(voutputs.to('cpu').tolist())
                else:
                    Predata = np.concatenate((Predata, np.array(voutputs.to('cpu').tolist())), axis=0)
        #Predata[Predata<-50]=None
        Predata = np.around(Predata,2)*100
        outpath = 'Mapping/'+ TType +'/' + TType + str(year) + str(day).zfill(3) + '.tif'
        write_geotif(outpath, dataset, Predata.astype(np.int16))
        etime = time.time()
        elapsed_time = etime - stime
        print(f"runtime is：{elapsed_time} seconds")

def write_geotif(data_path,dataset,data):

    driver = gdal.GetDriverByName('GTiff')
    geodata = driver.Create(data_path,dataset.RasterXSize,dataset.RasterYSize,1,gdal.GDT_Int16)
    geodata.SetGeoTransform(dataset.GetGeoTransform())
    geodata.SetProjection(dataset.GetProjection())
    imgdata = dataset.ReadAsArray()
    imgdata = np.array(imgdata)
    imgdata = imgdata.T
    imgdata[imgdata > 0] = np.squeeze(data)
    imgdata = imgdata.T
    geodata.GetRasterBand(1).WriteArray(imgdata)
    geodata = None

if __name__ == '__main__':

    for year in np.array([2010]):
        TrainModel(year, 'TMean')
        dataset = gdal.Open('Mapping/ExtLL.tif')
        ModelPrediction('TMean',2010,dataset)


# Year=2010: the optimal RMSE of the cross-validation is 0.853524
# Year=2010: the optimal Std of the cross-validation is 0.079336
# Year=2010: the optimal epocah of the cross-validation is 56
# Year=2010:the optimal epoch is 56
# Year=2010:the RMSE is 0.680707
# Year=2010:the RMSE of the validation is 0.681320