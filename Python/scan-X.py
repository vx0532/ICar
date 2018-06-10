import tushare as ts
import numpy as np
#import xgboost as xgb
import matplotlib.pyplot as plt
#import matplotlib.finance as mpf
import mpl_finance as mpf
import datetime,time,shutil,os,pickle,pdb

backDay=0
showHere=1
diffBars=1
reload=1

saveFolder='/home/austin/Documents/figures' #save stocks' pictures which were selected.
if reload:
    todayTime=datetime.datetime.today()
    today=todayTime.date()    
    tradeDay= (not ts.is_holiday(datetime.datetime.strftime(today,'%Y-%m-%d'))) and todayTime.hour*100+todayTime.minute>930
    try:
        tmp=open('scan_trade.pkl','rb')
        pklData=pickle.load(tmp)
        tmp.close()
        todayLast=pklData['today']
        dataCall=pklData['dataCall']
        stocks=pklData['stocks']
    except:
        todayLast=0
    if today!=todayLast:
        tmp=ts.get_stock_basics()
        stocks=tmp.index
        L=len(stocks)
        nDays=260+int(1.5*backDay) #number of natural days;
        dataCall=[]
        t1=time.time()
        for i in range(L):
            tmp=ts.get_k_data(code = stocks[i],start = datetime.datetime.strftime(today-datetime.timedelta(nDays),'%Y-%m-%d'))
            dataCall.append(tmp)
            ratioStocks=(i+1)/L
            t2=time.time()
            print(stocks[i]+':'+str(round(100*ratioStocks,2))+'%,and need more time:{} minutes'.format(round((t2-t1)*(1/ratioStocks-1)/60,2)))
        
        t2=datetime.datetime.now()
        print('Updating data has been completed now! and time lapses {} minutes.'.format(round((t2-todayTime).seconds/60,2)))
        pklData={'today':today,'dataCall':dataCall,'stocks':stocks}
        tmp=open('scan_trade.pkl','wb')
        pickle.dump(pklData,tmp)
        tmp.close()
    
    if tradeDay:
        dataNow=ts.get_today_all()
        dataNow=dataNow.set_index('code')
else:
    tmp=open('scan_trade.pkl','rb')
    pklData=pickle.load(tmp)
    tmp.close()
    today=pklData['today']
    dataCall=pklData['dataCall']
    stocks=pklData['stocks']
    tradeDay=0
stockPass=[]
stockSelect=[]
try:
    shutil.rmtree(saveFolder)
except:
    pass
os.mkdir(saveFolder)
for i in range(len(stocks)):
    if len(dataCall[i])<1:
        stockPass.append(stocks[i])
        continue
    if tradeDay:
        try:
            tmpNow=dataNow.loc[stocks[i]]
            opens=np.r_[dataCall[i].open, tmpNow.open]
            closes=np.r_[dataCall[i].close, tmpNow.trade]
            highs=np.r_[dataCall[i].high, tmpNow.high]
            lows=np.r_[dataCall[i].low, tmpNow.low]
            vols=np.r_[dataCall[i].volume, tmpNow.volume/100]
        except:
            print('cant download current data for '+stocks[i])
            continue
    else:
        opens=dataCall[i].open.values
        closes=dataCall[i].close.values
        highs=dataCall[i].high.values
        lows=dataCall[i].low.values
        vols=dataCall[i].volume.values
    Li=len(opens)-1
    dataCollect=[]
    for i2 in range(4,20):
        if Li-3*i2-6-backDay<0:
            break               
        pointZero=Li-6*i2
        if pointZero-backDay-1<0:
            pointZero=backDay+1
        if abs(max(highs[Li-3*i2-4-backDay:Li-3*i2+4-backDay])-max(highs[pointZero-backDay-1:Li+1-backDay]))<0.00000001 and \
        abs(min(lows[Li-3*i2-backDay-1:Li-i2-backDay])-min(lows[Li-2*i2-3-backDay:Li-2*i2+3-backDay]))<0.00000001 and \
        abs(max(highs[Li-2*i2-backDay-1:Li-backDay])-max(highs[Li-i2-2-backDay:Li-i2+2-backDay]))<0.00000001 and \
        (lows[Li-backDay]<min(lows[Li-i2-backDay-2:Li-backDay]) or \
        (highs[Li-backDay]<highs[Li-backDay-1] and lows[Li-1-backDay]<min(lows[Li-i2-backDay-2:Li-backDay]) )) and \
        min(highs[Li-3*i2-6-backDay:Li-backDay]-lows[Li-3*i2-6-backDay:Li-backDay])>0.0000001 :
            addDays=9*i2
            if Li-3*i2-addDays-backDay<0:
                addDays=Li-3*i2-backDay
            P1=np.argmax(highs[Li-3*i2-addDays-backDay:Li-2*i2-backDay])
            P2=np.argmin(lows[Li-3*i2-backDay:Li-2*i2+3-backDay])+addDays
            P3=np.argmax(highs[Li-2*i2-backDay:Li-i2+3-backDay])+i2+addDays
            P4=3*i2+addDays
            baseI=Li-3*i2-addDays-backDay
            down1=P2-P1
            up=P3-P2
            down2=P4-P3
            if i2>8:
                DBtmp=diffBars+1
            else:
                DBtmp=diffBars
            startTmp=max(0,Li-backDay-int(3.5*(P4-P1))-4)
            if max([down1,up,down2])-min([down1,up,down2])<=DBtmp:# \
#            and min(lows[Li-3*i2-4-backDay:Li+1-backDay])>min(lows[startTmp:Li-backDay+1])+0.45*(max(highs[startTmp:Li-backDay+1])-min(lows[startTmp:Li-backDay+1])):# and fig>0:
                periodBoll=20#i2*3  #################################
                upLine=[]
                middleLine=[]
                downLine=[]
                for i3 in range(baseI+periodBoll,Li+1):#periodBoll-1,Li+1-baseI
                    tmpMean=closes[i3-periodBoll+1:i3+1].mean()
                    tmpStd=closes[i3-periodBoll+1:i3+1].std()
                    upLine.append(tmpMean+2*tmpStd)
                    middleLine.append(tmpMean)
                    downLine.append(tmpMean-2*tmpStd)
                dataCollect.append([range(periodBoll,Li+1-baseI),[upLine,middleLine,downLine],[P1,P2,P3,P4],\
                                    [highs[baseI+P1],lows[baseI+P2],highs[baseI+P3],lows[baseI+P4]],\
                                    baseI])
    LdataCollect=len(dataCollect)
    if LdataCollect:
        fig=plt.figure(figsize=(10,6))
        ax=fig.add_axes([0.1,0.3,0.8,0.6])                     
        monitorTmp=[]
        monitorInd=[]
        baseRec=[]
        for i2 in range(LdataCollect):
            datai2=(np.array(dataCollect[i2][2])+dataCollect[i2][4]).tolist()
            if datai2 not in monitorTmp:
                monitorTmp.append(datai2)
                monitorInd.append(i2)
                baseRec.append(dataCollect[i2][4])
        baseImin=min(baseRec)
        for i2 in monitorInd:
            bollx=dataCollect[i2][0]
            bollLine=dataCollect[i2][1]
            ax.plot(bollx,bollLine[0],'r--',bollx,bollLine[1],'b--',bollx,bollLine[2],'r--')
            P1,P2,P3,P4=dataCollect[i2][2][0],dataCollect[i2][2][1],dataCollect[i2][2][2],dataCollect[i2][2][3]
#            price4=dataCollect[i2][3]
            baseI=dataCollect[i2][4]            
            baseIadd=baseI-baseImin
            ax.plot([baseIadd+P1,baseIadd+P2,baseIadd+P3,baseIadd+P4],[highs[baseI+P1],lows[baseI+P2],highs[baseI+P3],lows[baseI+P4]],color='k')
            upTmp=highs[baseI+P3]-lows[baseI+P2]
            ax.text(baseIadd+P1,highs[baseI+P1],round((highs[baseI+P1]-lows[baseI+P2])/upTmp,3))
            ax.text(baseIadd+P3,highs[baseI+P3],round((highs[baseI+P3]-lows[baseI+P4])/upTmp,3))
        
        candleData=np.column_stack([range(baseIadd+P4+backDay+1),opens[baseI-baseIadd:Li+1],highs[baseI-baseIadd:Li+1],lows[baseI-baseIadd:Li+1],closes[baseI-baseIadd:Li+1]])
        mpf.candlestick_ohlc(ax,candleData,width=0.5,colorup='r',colordown='g')
        bars=len(opens[baseI:Li+1])
        ax.set_xticks(range(0,bars,3))
        ax.grid()
        plt.title(stocks[i])
        ax1=fig.add_axes([0.1,0.1,0.8,0.2])
        vol=vols[baseI-baseIadd:Li+1]
        ax1.bar(range(len(vol)),vol)
        ax1.set_xticks(range(0,bars,3))
        ax1.grid()
        plt.savefig(saveFolder+'/'+stocks[i])
        if not showHere:
            plt.clf()
        stockSelect.append(stocks[i])
                
print('Stocks which are past: '+','.join(stockPass))
print('Stocks which are select: ',end='')
print(stockSelect)

