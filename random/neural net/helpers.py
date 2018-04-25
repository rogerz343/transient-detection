from time import gmtime, strftime
import numpy as np
import matplotlib.pyplot as plt
def perfMeasures(p1, n, lbl, nm = strftime("%H_%M_%S", gmtime())):
  thresholds = np.linspace(np.min(p1),np.max(p1),n)
  y1 = lbl == 1
  yz = np.logical_not(y1)
  
  
  FPRL = []
  TPRL = []
  RL, PL = [], []
  dic = {}
  #TN, FN, FP, TP, AM, GM, F1
  RL, PL = [], []
  dic['TN'] = []
  dic['FN'] = []
  dic['FP'] = []
  dic['TP'] = []
  dic['AM'] = []
  dic['GM'] = []
  dic['F1'] = []
  for th in thresholds:
    h1 = p1 >= th
    hz = np.logical_not(h1)

    TN = float(np.sum(np.logical_and(hz,yz)))
    FN = float(np.sum(np.logical_and(hz,y1)))
    FP = float(np.sum(np.logical_and(h1,yz)))
    TP = float(np.sum(np.logical_and(h1,y1)))


    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    Precision = TP/(TP+FP)

    AM = (TPR+TNR)/2
    GM = np.sqrt(TPR*TNR)

    F1 = 2 * TPR * Precision / (TPR + Precision)

    FPR = 1 - TNR

    FPRL.append(FPR)
    TPRL.append(TPR)
    RL.append(TPR)
    PL.append(Precision)
    dic['TN'].append(TN)
    dic['FN'].append(FN)
    dic['FP'].append(FP)
    dic['TP'].append(TP)
    dic['AM'].append(AM)
    dic['GM'].append(GM)
    dic['F1'].append(F1)
  Tar = np.array(TPRL)
  Far = np.array(FPRL)
  print(TPRL)
  print(FPRL)
  plt.figure()
  plt.scatter(Far,1-Tar)
  plt.xlim(0.0, 0.1)
  plt.ylim(0, 0.05)
  plt.title("ROC")
  plt.savefig("out/ROC"+nm+".png")
  AUCROC = -np.trapz(Tar, x=Far)

  Tar = np.array(PL)
  Far = np.array(RL)
  print("PL")
  print(PL)
  print("RL")
  print(RL)
  plt.figure()
  plt.scatter(Far,Tar)
  #plt.xlim(0.0, 0.1)
  #plt.ylim(0, 0.05)
  plt.title("Precision")
  plt.savefig("out/Precision"+nm+".png")
  AUCPR = -np.trapz(Tar, x=Far)
  dic['Precision']=PL
  dic['Recall']=RL
  dic['TPR']=TPRL
  dic['FPR']=FPRL
  dic['AUCPR']=AUCPR
  dic['AUCROC']=AUCROC
  return dic


