import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch
import copy
import math
from sklearn.metrics import roc_curve
from scipy.optimize import curve_fit
from torchsummary import summary

from sklearn import tree

#from sklearn.externals.six import StringIO
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

plt.rcParams.update({'font.size': 10})


LR = 0.0003
Gamma_Scheduler=0.999945
Nepoch = 1500
split_train_valid_test = 0.8
N_batch_size = 20
filename = 'NN.dat'

class DataSet:
    def __init__(self):
        fyr = ['TGFa pg/ml','IFNa2 pg/ml','IFNg pg/ml','IL-10 pg/ml','IL-12P40 pg/ml','IL-12P70 pg/ml','IL-13 pg/ml','IL-15 pg/ml',	'IL-17A pg/ml',	'IL-1RA pg/ml',	'IL-1a pg/ml',	'IL-9 pg/ml', 'IL-1b pg/ml',	'IL-2 pg/ml',	'IL-3 pg/ml',	'IL-4 pg/ml',	'IL-5 pg/ml',	'IL-6 pg/ml',	'IL-7 pg/ml', 'IL-8 pg/ml', 'TNFa pg/ml', 'TNFb pg/ml']


def plot_roc_curve(fper, tper):
    fper, tper = fper[0], tper[0]
    #for el in range(len(fper)):
    plt.plot(fper, tper)
    maxindex = 0
    max_ = 0
    averx,avery = 0,0
    for i in range(6,len(fper)):
    	if fper[i] < 0.35: continue
    	if tper[i]/fper[i] > max_: 
    		maxindex = i
    		max_ = tper[i]/fper[i]
    		print(max_)
    print(' -------------------------------------------- the best perfoance at ',maxindex,1-fper[maxindex],tper[maxindex])
    
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('Rate of incorrect recognition of health people')
    plt.ylabel('Rate of correct recognition of patients')
    #plt.title('Receiver Operating Characteristic Curve')
    plt.plot([fper[maxindex]],[tper[maxindex]],marker='*', markersize=15)
    return (fper[maxindex],tper[maxindex])


def cross_valid(len,train_ratio = 0.8):
    np.random.seed(136)
    indices = np.random.choice(len, len, replace=False)
    nvalid = math.ceil((1. - train_ratio)*len)
    i, ind_h = 0,0
    res = []
    while ind_h < len - nvalid/2:
        ind_l = nvalid*i
        ind_h = ind_l + nvalid
        if ind_h > len: ind_h = len
        #print(ind_l,ind_h)
        valid = indices[ind_l:ind_h]
        train = set(indices) - set(valid)
        valid = np.array(list(valid))
        train = np.array(list(train))
        valid = np.random.choice(valid,size=valid.shape[-1], replace=False)
        train = np.random.choice(train, size=train.shape[-1], replace=False)
        res.append((train,valid))
        i+=1
    return res

def dataset_to_torch(dataset, train_indices, valid_indices):
    # Convert features and labels to numpy arrays.'
    target = 'res'

    labels = dataset[target].to_numpy()
    feature_dataset = dataset.drop([target], axis=1)
    data = feature_dataset.to_numpy()
    # Separate training and test sets
    train_features = data[train_indices]
    train_labels = labels[train_indices]
    valid_features = data[valid_indices]
    valid_labels = labels[valid_indices]

    input_tensor = torch.from_numpy(train_features).type(torch.FloatTensor)
    label_tensor = torch.from_numpy(train_labels).type(torch.LongTensor)
    valid_input_tensor = torch.from_numpy(valid_features).type(torch.FloatTensor)
    valid_label_tensor = torch.from_numpy(valid_labels).type(torch.LongTensor)

    return input_tensor, label_tensor, valid_input_tensor, valid_label_tensor


def MSE(x,y):
    return (x-y)**2 # +torch.abs(torch.abs(x)-1)

def BLE(x,y):
    # x - predicted
    # y - label
    res = -1/2*(y*torch.log(x) + (1-y)*torch.log(1-x))
    return res




class NNModel(nn.Module):
    '''Neural network model used in the classification'''
    def __init__(self,len, logist = True):
        super().__init__()
        self.linear2 = nn.Linear(len, int(len))
        self.linear3 = nn.Linear(int(len), int(len/2))
        self.linear4 = nn.Linear(int(len/2), int(len/3))
        self.lastlayer = nn.Linear(int(len/3), 1)
        self.activation = nn.ReLU()
        self.activation1 = nn.Sigmoid()
        self.logist = logist 
        self.linear_logist = nn.Linear(len, 1)

    def forward(self, x):
        if self.logist:
            btv = self.activation1(self.linear_logist(x))
            return btv
        out2 = self.activation(self.linear2(x))
        out3 = self.activation(self.linear3(out2))
        out4 = self.activation(self.linear4(out3))
        out5 = self.lastlayer(out4)
        out6 = self.activation1(out5)
        return out6
        
        

def low_correlation(dataset):
    colum = list(set(dataset.columns) - set(['res']))
    colum = list(dataset.columns)
    colum.remove("res")
    res = []
    names = []
    for col in colum:
        corr = np.corrcoef(dataset[col], dataset['res'])[0,1]
        names.append(col.replace("pg/ml","").replace("/BB",""))
        res.append(corr)
    print(names,res)
    return res,names

def overlap(dataset,thres):
    colum  = list(set(dataset.columns) - set(['res']))
    res = []
    for col in colum:
        mean1 = dataset[dataset['res']==0][col].mean()
        mean2 = dataset[dataset['res']==1][col].mean()
        std1 = dataset[dataset['res']==0][col].std()
        std2 = dataset[dataset['res']==1][col].std()
        corr = np.abs(mean1-mean2)/(std1+std2)
        #print(corr)
        if corr < thres: res.append(col)
    return res

def preprocess_nn(df,number):
    categorical = [df.columns[0]]
    target = 'res'
    print("================= categorical ", categorical)
    continuous  = list(set(df.columns) - set(categorical) - set([target]))

    #for col in categorical:
    #    df = df.drop(col, axis=1)

    fyr = ['TGFa pg/ml', 'IFNa2 pg/ml', 'IFNg pg/ml', 'IL-10 pg/ml', 'IL-12P40 pg/ml', 'IL-12P70 pg/ml', 'IL-13 pg/ml',
           'IL-15 pg/ml', 'IL-17A pg/ml', 'IL-1RA pg/ml', 'IL-1a pg/ml', 'IL-9 pg/ml', 'IL-1b pg/ml', 'IL-2 pg/ml',
           'IL-3 pg/ml', 'IL-4 pg/ml', 'IL-5 pg/ml', 'IL-6 pg/ml', 'IL-7 pg/ml', 'IL-8 pg/ml', 'TNFa pg/ml',
           'TNFb pg/ml']
    print(df.columns)
    rest_for_df = list(set(df.columns) - set(fyr) - set([target]))
    rest_for_df = ['Пол','BDNF, ng/ml','PAI-1(total), ng/ml', 'RANTES pg/ml', 'VEGF pg/ml ', 'Fractalkine pg/ml'] 
    for col in rest_for_df:
        df = df.drop(col, axis=1)
    file = open(filename, 'a+')
    file.write(f'{df.columns[number]} ')
    file.close()
    if number!= -1: df = df.drop(df.columns[number], axis=1)
    df = df.astype('float')
    return df



def uniform(df):
    target = 'res'
    df = df.astype('float')
    for col in list(set(df.columns) - set([target])):
        mean = df[col].mean()
        std = df[col].std()
        df[col] = 1 - 2 / (1 + np.exp(-(df[col] - mean) / std))
    return df




def read_df(file):
    df = pd.read_csv(file);
    df.replace(",", ".", regex=True, inplace=True);
    df.replace(">", "", regex=True, inplace=True);
    df.replace("↑", "", regex=True, inplace=True);
    df.replace(". ", "", regex=True, inplace=True);
    df.replace(" ", 0, regex=True, inplace=True);

    return df

def describe(df):
    print(len(df),len(df.columns))

def process_df(number):
    df_schizo = read_df("data_2022.csv")
    print('df_schizo.shape after reading from file = ',df_schizo.shape)
    df_control = read_df("data_2022_control.csv")
    print('df_schizo.columns = ',df_schizo.columns)
    res = [1]*df_schizo.shape[0]
    df_schizo['res'] = res
    res = [0]*df_control.shape[0]
    df_control['res'] = res
    extra = set(df_schizo.columns) - set(df_control.columns)
    print('extra = ', extra)
    for el in extra: df_schizo = df_schizo.drop(el,axis=1)

    extra = set(df_control.columns) - set(df_schizo.columns)
    print('extra = ', extra)
    for el in extra: df_control = df_control.drop(el, axis=1)

    print(df_schizo.shape,df_control.shape)
    print('df_schizo.columns = ', df_schizo.columns)
    print('=============================================')

    print('--------------- INITIAL DATA ANALYSIS ----------------')
    describe(df_control)
    describe(df_schizo)

   # df_control_null = df_control.isnull().any(axis=1)
   # df_schizo_null = df_schizo.isnull().any(axis=1)
   # describe(df_control[df_control_null])
   # describe(df_schizo[df_schizo_null])
    print('=============================================')
  #  df_control = df_control[df_control_null]
  #  df_schizo = df_schizo[df_schizo_null]
    print('combine patients and control')
    df = df_schizo._append(df_control, ignore_index=True)

    df = preprocess_nn(df,number)

    df_null = df[df.isnull().any(axis=1)]
    print('df_null.shape = ', df_null.shape)
    for el in df.columns:
        df = df.loc[df[el].notnull()]
    print('df.shape = ',df.shape)
    print('df.columns = ',df.columns)
    print('===================')
    #df.fillna(0,inplace=True)

    return df



def draw_correlation(df):
    # high_overl = overlap(df, 0.1)
    # print("high_overl = ", high_overl)
    #  df = df.drop(high_overl, axis=1)

    set1 = set(df.columns)
    set2 = set(df.select_dtypes(include=['number']).columns)
    # print(df[df.columns[15]].values)

    print(df.shape)

    correlation1 = df.loc[df['res'] == 1].corr()
    correlation0 = df.loc[df['res'] == 0].corr()
    plt.figure(figsize=(13, 13), dpi=80)
    plt.subplot(121)

    plt.title("Матрица корреляций dataset, отобранного для обучения нейронной сети", pad=20)
    sns.heatmap(correlation0, square=True, annot=True)
    plt.subplot(122)
    sns.heatmap(correlation1, square=True, annot=True)
    plt.show()


def perform(name_save, input_tensor, label_tensor, logist = False):

    # high_overl = overlap(df, 0.1)
    # print("high_overl = ", high_overl)
    #  df = df.drop(high_overl, axis=1)

    #set1 = set(df.columns)
    #set2 = set(df.select_dtypes(include=['number']).columns)
    #print(df[df.columns[15]].values)


    #print(df.shape)

    #dataset = dataset_to_torch(df, split_train_valid_test)

    model1 = NNModel(input_tensor.shape[1],logist = logist)

    criterion1 = torch.nn.MSELoss()  #  torch.nn.BCELoss() # BLE # MSE
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=LR, weight_decay=0.01)
    # Работа SGD в данном случае эквивалента обычному GD, так как функция ошибок и градиенты считаются для каждой строки.
    scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, gamma=Gamma_Scheduler)


    num_of_parameters = sum(map(torch.numel, model1.parameters()))
    list_par_before = []
    for p in model1.parameters():
        list_par_before.append(copy.copy(p.data.cpu().detach().numpy()))

    cost, cost0, cost1, val_array = np.array([]), np.array([]), np.array([]), np.array([])
    aver_v = 0
    variant = True
    valid = 0
    stop_v = np.inf
    weight = []
    optimizer1.zero_grad()
    lrs = []
    try:
        stream_hist = open("results/" + name_save + ".dat",'w')
    except:
        os.mkdir("results/")
        stream_hist = open("results/" + name_save + ".dat", 'w')
    loss_v = 0
    # train_set, valid_sate
    aver_loss = 0.25
    break_now = False
    for epoch in range(Nepoch):
        if break_now: break
        len_inp_tens = len(input_tensor)
        for t in np.random.choice(len_inp_tens,len_inp_tens,replace=False):
            label = label_tensor[t]
            label = label.to(torch.float32)
            if label.item() == 1:
                if variant == False: continue
            else:
                if variant == True: continue
            elem = copy.copy(list(model1.parameters())[1].cpu().detach().numpy())
            weight.append(elem)
            output = model1(input_tensor[t])
            output = output.to(torch.float32)
            loss = criterion1(output[0], label)
            cost = np.append(cost, loss.item())

            if label.item() == -1:
                cost0 = np.append(cost0, loss.item())
            else:
                cost1 = np.append(cost1, loss.item())
            lrs.append(optimizer1.param_groups[0]["lr"])
            loss.backward()
            if aver_v == N_batch_size:
                optimizer1.step()
                optimizer1.zero_grad()
                scheduler1.step()
                loss_v += loss.item()
                stream_hist.write(f'{loss_v/(aver_v+1)} {label.item()} {output[0]} {optimizer1.param_groups[0]["lr"]}\n')
                print(f'epoch = {epoch}', format(t / input_tensor.shape[0], '.2f'),
                      'leraning rate = ', format(optimizer1.param_groups[0]["lr"], '.6f'),
                      'loss = ', format(loss.item(), '.6f'),
                      'NN_output = ', format(output.item(), '.6f'),
                      'target = ', label.item(), "  aver_loss = ", aver_loss)
                aver_loss = 0.95*aver_loss + loss_v/(aver_v+1)/20.
                if aver_loss < 0.05:
                    break_now = True
                    break
                loss_v = 0
                aver_v = 0

            else:
                aver_v += 1
                loss_v+=loss.item()

            if label.item() == 1:
                variant = False
            else:
                variant = True

    torch.save(model1.state_dict(),"results/" + name_save + ".pth")
            
    #plt.show()
    #plt.plot(cost, linewidth=1.3, label='total loss')
    #plt.show()

    return 0


def make(df, logist):
    dset = cross_valid(len(df), split_train_valid_test)
    i = 0
    for el in dset:
        set_ = dataset_to_torch(df, el[0], el[1])
        print(set_)
        perform(str(i), set_[0], set_[1], logist)
        i+=1

def make_grad():
        dset = cross_valid(len(df), split_train_valid_test)
        model1 = NNModel(len(df.columns) - 1)
        i = 0
        grad = [0]*22
        for el in dset:
            set_ = dataset_to_torch(df, el[0], el[1])
            model1.load_state_dict(torch.load(f"results/{i}.pth"))
            model_input = set_[2]
            for j in range(len(grad)):
                s2 = model1(model_input)
                model_input[:,j]+= 0.5
                grad[j] += torch.sum(model1(model_input) - s2).item()
                model_input[:, j]-= 0.5
            i+=1
        grad = [np.abs(el) for el in grad]
        plt.xlabel('Признак')
        plt.ylabel('Чувствительность сети к признаку')
        plt.bar(list(range(len(grad))),grad)
        plt.show()
        print(grad)



def validate(df,logist):
    file = open(filename, 'a+')
    dset = cross_valid(len(df), split_train_valid_test)
    model1 = NNModel(len(df.columns) - 1,logist)
    i = 0
    min_health, max_shcizo = [],[]
    vect0,vect1 = [],[]
    for el in dset:
        set_ = dataset_to_torch(df, el[0], el[1])
        #print(set_)
        model1.load_state_dict(torch.load(f"results/{i}.pth"))
        #model1.to
        #matr = model1.parameters()
        #for matr_el in matr:
        #    print(matr_el.shape)
        #    plt.imshow(matr_el.detach().numpy())
        #    plt.colorbar()
        #    plt.show()
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print("Device:", device)    
        #model1.to(device)
        #print(next(model1.parameters()).device)
        #model1 = model1.to('cuda:0')
        #summary(model1, (1, set_[2].shape[1]))

        s1 = set_[1]
        print('======================')
        print('total number of training dataset = ',len(s1))
        print('number of health = ', len([el for el in s1 if el ==0]))
        print('number of schizo = ', len([el for el in s1 if el == 1]))
        print('======================')

        s1 = set_[3]
        model_input = set_[2]
        s2 = model1(model_input)

        label = s1
        label = label.to(torch.float32)
        output = s2
        output = output.to(torch.float32)
        #los = criterion1(output[0], label).item()

        print('======================')
        print('total number of validation dataset = ',len(s1))
        print('number of health = ', len([el for el in s1 if el ==0]))
        print('number of schizo = ', len([el for el in s1 if el == 1]))
        print('======================')
        #plt.hist([el.item() for el in s2[s1 == 0]], histtype='bar', bins=100, label='health')
        #plt.hist([el.item() for el in s2[s1 == 1]], histtype='step', bins=100, label='schizo')
        [vect0.append(el.item()) for el in s2[s1 == 0]]
        [vect1.append(el.item()) for el in s2[s1 == 1]]
        plt.legend()
        plt.xlabel('NN output')
        #plt.show()
        s2 = s2.cpu().detach().numpy()
        fper, tper, thresholds = roc_curve(s1, s2)
        mk = plot_roc_curve([fper], [tper])
        min_health.append(1-mk[0])
        max_shcizo.append(mk[1])
        i+=1

    plt.show()
    plt.hist(vect0, histtype='bar', bins=50, label='health')
    plt.hist(vect1, histtype='step', bins=50, label='patients')
    min_health = np.array(min_health)
    max_shcizo = np.array(max_shcizo)
    print('accuracy_control = ','{:.2f}'.format(min_health.mean()), "+/-", '{:.2f}'.format(min_health.std()))
    print('accuracy_patients = ', '{:.2f}'.format(max_shcizo.mean()), "+/-", '{:.2f}'.format(max_shcizo.std()))
    file.write(f'{min_health} {max_shcizo} \n')
    file.close()
    plt.xlabel('model output')
    plt.ylabel('#test events')
    plt.legend()
    plt.show()



def func(x, a, b, c, d):
    res = a + b*x # + c*x**2 + d*x**3
    return res



def draw_history():
    df = process_df(-1)
    dset = cross_valid(len(df), split_train_valid_test)

    for el in range(len(dset)):
        print('=====================================')
        print(f'results/{el}.dat')
        df = pd.read_table(f'results/{el}.dat',delimiter=' ')
        print(len(df))
        #df = df.loc[df[df.columns[1]]==0]
        df[df.columns[0]].plot(style='.', xlabel = 'training batch', ylabel = 'MSE loss',markersize=4)
        print('len(df) = ',len(df))
        xdata = df.index[-400:]
        ydata = df[df.columns[0]][-400:].values
        print('hello0')
        popt, pcov = curve_fit(func, xdata, ydata)
        print('hello')
        print('dddd',*popt)
        plt.plot(xdata, func(xdata, *popt), 'r-', label = 'аппроксимация',color = 'black', linewidth = 4)
    plt.show()

def plot_data(df):
    sns.relplot(
        data=df,
        x = df.columns[13], y = df.columns[6], #col=df.columns[2]#,
        hue='res'#, style=df.columns[4], size=df.columns[5]
    )
    plt.show()




def make_NN(train = True, logist = False, number=-1):


    df = process_df(number)

    #plt.xscale('log')
    numer = 9
    min_ = df[df.columns[numer]].min()
    max_ = df[df.columns[numer]].max()
    plt.hist(df[df.columns[numer]].loc[df['res'] == 1],bins= 40, range = (min_,max_),density=True,label='shizo')
    plt.hist(df[df.columns[numer]].loc[df['res'] == 0],bins= 40, range = (min_,max_),density=True,fc=(0.3, 0.3, 0.3, 0.36),label='health')
    plt.xlabel(df.columns[numer])
    plt.legend()
    plt.show()

    #
    correlation_w_res = low_correlation(df)
    plt.bar(correlation_w_res[1],correlation_w_res[0])
    plt.xticks(rotation=45)
    plt.show()
    #
    plot_data(df)
    #
    df = uniform(df)
    #
    #print(df.columns)
    dset = cross_valid(len(df), split_train_valid_test)
    print("============================ split ====================== ")
    #for el in dset:
    #    print(len(el[0]), len(el[1]))
    #    print("============================ end split ====================== ")

    #draw_correlation(df)
    
    if train: make(df,logist)
    draw_history()
    validate(df,logist)
    
    #print(cross_valid(10,0.8))
    
    #make_grad()


def make_DTrees(number):
    #file = open(filename, 'a+')

    df = process_df(number)
    df = uniform(df)
    dset = cross_valid(len(df), split_train_valid_test)
    #file.write(f'{df.columns[number]} ')
    i = 0
    feature_cols = ['res']
    
    accur_health_, accur_schizo_ = 0,0
    accur_health, accur_schizo = [],[]

    for el in dset:
        set_ = dataset_to_torch(df, el[0], el[1])
        n_h = len(set_[1][set_[1]==0])
        n_sc = len(set_[1][set_[1] == 1])
        wei = n_sc/n_h
        weight = {0: 1, 1: wei}

        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth = 8, class_weight = weight) #int(len(df.columns)/2), class_weight = weight)
        clf = RandomForestClassifier(random_state=5, n_jobs=-1, n_estimators=3, class_weight = weight);
        print('n_h = ',n_h)
        print('n_sc = ',n_sc)
        weight = {0: wei*2, 1: 1}
        clf = svm.SVC(C = 2, kernel = 'poly', class_weight=weight, max_iter = -1, degree=3)
        clf = KNeighborsClassifier(n_neighbors=2)
        #pca = PCA(n_components=len(df.columns)-1-10)  # https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/
        X_train = set_[0]
        #X_test = pca.transform(set_[2])
        #print(pca.explained_variance_ratio_)
        #print(pca)
        #clf = clf.fit(set_[0], set_[1])
        clf = clf.fit(set_[0], set_[1])
        
        set2_0 = set_[2][set_[3] == 0]
        set3_0 = set_[3][set_[3] == 0]
        set2_1 = set_[2][set_[3] == 1]
        set3_1 = set_[3][set_[3] == 1]
        accur_health_ = clf.score(set2_0, set3_0)
        accur_schizo_ = clf.score(set2_1, set3_1)

        print('training score = ','%.3f'%clf.score(X_train, set_[1]),'  test score health = ','%.3f'%accur_health_,'  test score schizo = ','%.3f'%accur_schizo_)
        accur_health.append(accur_health_)
        accur_schizo.append(accur_schizo_)

        print(classification_report(set_[3], clf.predict(set_[2])))
        print(confusion_matrix(set_[3], clf.predict(set_[2])))
        print('===========================')
        dot_data = StringIO()
        #export_graphviz(clf, out_file=dot_data,  
        #                filled=True, rounded=True,feature_names = fyr,
        #        special_characters=True, class_names=['0','1'])
        #graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        #graph.write_png('Tree.png')
        #Image(graph.create_png())
        i+=1

    print('accuracy_health = ', round(np.mean(accur_health),3),'+/-',round(np.std(accur_health),3))
    print('accuracy_schizo = ', round(np.mean(accur_schizo),3),'+/-',round(np.std(accur_schizo),3))
    #file.write(f' {accur_health} {accur_schizo} \n')
    bin = 0
    plt.hist(df[df.columns[bin]].loc[df['res'] == 1],bins= 40, range = (-0.001,0.05),label='shizo')
    plt.hist(df[df.columns[bin]].loc[df['res'] == 0],bins= 40, range = (-0.001,0.05),fc=(0.3, 0.3, 0.3, 0.36),label='health')
    plt.xlabel(df.columns[bin])
    plt.yscale('log')
    plt.legend()
    #plt.show()
    #file.close()


np.random.seed(136)        
torch.manual_seed(136)
torch.cuda.manual_seed(136)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def plot_file(name_f):
    df = pd.read_csv(name_f,index_col=0,delimiter=' ')
    print(df)
    df.iloc[:,:-1].plot(kind="bar", figsize = (2, 4))
    plt.show()

def analyze_database():
    df = process_df(number=-1)

    print("Больных женщин ", len(df[df["res"]==1][df["Пол"]=="F"]))
    print("Больных мужчин ", len(df[df["res"]==1][df["Пол"]=="M"]))
    print("Здоровых женщин ", len(df[df["res"]==0][df["Пол"]=="F"]))
    print("Здоровых мужчин ", len(df[df["res"]==0][df["Пол"]=="M"]))

     
    df1= df[df["res"]==1]["Возраст"]
    print("Возраст больных mean std min max ", df1.mean(), df1.std(), df1.min(), df1.max())
    df1 = df[df["res"]==0]["Возраст"]
    print("Возраст здоровых mean std min max ", df1.mean(), df1.std(), df1.min(), df1.max())

if __name__=='__main__':
    #for i in range(-1,0):
    #    make_DTrees(i)
    make_DTrees(-1)
    #for i in range(-1, 42):
    #    make_NN(train = True, logist = False, number=i)
    #plot_file('NN.dat')
    #make_NN(train=True, logist=False, number=-1)
    #draw_history()
    #analyze_database()