import matplotlib.pyplot as plt
import random
import torch
import pandas as pd

num_data = 20000
#---------------------一次関数の組み合わせ-------------------
a = [random.random()/5 - 0.1 for i in range(num_data)] #y=ax+bの前半のa -0.1<a<0.1
changet = torch.tensor([40*random.random() + 30 for i in range(num_data)]) #途中で傾きを変えるのでそのタイミングをランダムに決定する
changet = torch.floor(changet)#整数に丸める
b = [random.random()/5 - 0.1 for i in range(num_data)] #y=ax+bの後半のa -0.1<a<0.1
x = torch.arange(0, 120, 1)

y = torch.zeros(num_data, len(x))
for i in range(num_data):
    t_sca = changet[i].item()#下の式で使えるようにスカラーに変更
    y[i,:] = (a[i]*x + 25)*(x<t_sca) + (b[i]*x +  (a[i]-b[i])*t_sca +25)*(x>=t_sca)
    #y[i,:] = (a[i]*x + 25 + 10*a[i])*(x<t_sca) + (b[i]*x +  (a[i]-b[i])*t_sca +25+10*a[i])*(x>=t_sca)

l_test = torch.zeros(10,len(x))
t_sca = changet[1000].item()
l_test[0,:] = (0.02*x + 25)*(x<43) + (-0.08*x +  (0.02+0.08)*43 +25)*(x>=43)
t_sca = changet[1001].item()
l_test[1,:] = (-0.087131*x + 25)*(x<61) + (0.0584079*x +  (-0.087131-0.0584079)*61 +25)*(x>=61)
t_sca = changet[1002].item()
l_test[2,:] = (-0.038239*x + 25)*(x<55) + (0.00575258*x +  (-0.038239-0.00575258)*55 +25)*(x>=55)
for i in range(7):
    t_sca = changet[i].item()#下の式で使えるようにスカラーに変更
    l_test[i+3,:] = (a[1000+i]*x + 25)*(x<t_sca) + (b[1000+i]*x +  (a[1000+i]-b[1000+i])*t_sca +25)*(x>=t_sca)
#-------------------ただの一次関数----------------------
# a = [random.random()/10 - 0.05 for i in range(num_data)] #y=ax+bの前半のa -0.1<a<0.1
# b = [random.random() - 0.5 for i in range(num_data)]
# x = torch.arange(0, 151, 1)

# y = torch.zeros(num_data, len(x))
# for i in range(num_data):
#     y[i,:] = a[i]*x + 25 + b[i]

#-------------------最初一定途中で傾きあり------------------
# a = [random.random()/10 - 0.05 for i in range(num_data)] #y=ax+bの前半のa -0.1<a<0.1
# changet = torch.tensor([100*random.random() + 20 for i in range(num_data)]) 
# changet = torch.floor(changet)#整数に丸める
# b = [random.random()/10 - 0.05 for i in range(num_data)] #y=ax+bの後半のa -0.1<a<0.1
# x = torch.arange(0, 150, 1)

# y = torch.zeros(num_data, len(x))
# for i in range(num_data):
#     t_sca = changet[i].item()#下の式で使えるようにスカラーに変更
#     y[i,:] = (25+b[i])*(x<t_sca) + (a[i]*x - a[i]*t_sca+25)*(x>=t_sca)
    #y[i,:] = (a[i]*x + 25 + 10*a[i])*(x<t_sca) + (b[i]*x +  (a[i]-b[i])*t_sca +25+10*a[i])*(x>=t_sca)

#--------------------sin関数----------------------------
# a = [random.random()/10 - 0.05 for i in range(num_data)] #y=ax+bの前半のa -0.1<a<0.1

# x = torch.arange(0, 150, 1)
# y = torch.zeros(num_data, len(x))
# for i in range(num_data):
#     y[i,:] = 3*torch.sin(x/20) + 25

# --------------------ただの一次関数--------------------------
# a = [random.random()*10 + 20 for i in range(num_data)] #y=ax+bの前半のa -0.1<a<0.1
# changet = torch.tensor([40*random.random() + 30 for i in range(num_data)]) #途中で傾きを変えるのでそのタイミングをランダムに決定する
# changet = torch.floor(changet)#整数に丸める
# x = torch.arange(0, 151, 1)
# y = torch.zeros(num_data, len(x))

# for i in range(num_data):
#     t_sca = changet[i].item()#下の式で使えるようにスカラーに変更
#     y[i,:] = (25)*(x<t_sca) + (a[i])*(x>=t_sca)

# -------------プロット----------------------------------
plt.rcParams["figure.figsize"] = 8,6
plt.rcParams["figure.subplot.left"] = 0.14  # 余白
plt.rcParams["figure.subplot.bottom"] = 0.14# 余白
plt.rcParams["figure.subplot.right"] =0.90  # 余白
plt.rcParams["figure.subplot.top"] = 0.90   # 余白
plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 18 # 全体のフォントサイズが変更されます。
# plt.rcParams['xtick.labelsize'] = 18 # 軸だけ変更されます。
# plt.rcParams['ytick.labelsize'] = 18 # 軸だけ変更されます
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid
plt.rcParams["legend.fancybox"] = False # 凡例の丸角をとる
plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
# plt.rcParams["legend.labelspacing"] = 5. # 垂直（縦）方向の距離の各凡例の距離
# plt.rcParams["legend.handletextpad"] = 3. # 凡例の線と文字の距離の長さ
plt.rcParams["legend.markerscale"] = 2 # 点がある場合のmarker scale
plt.rcParams["legend.borderaxespad"] = 0.5 # 凡例の端とグラフの端を合わせる

fig = plt.figure()
for i in range(10):
    plt.plot(x, y[i,:])
plt.xlabel('time[s]')
plt.ylabel('velocity[m/s]')
# plt.show()

# # save
# fig.savefig('test_5.pdf', bbox_inches="tight", pad_inches=0.05)

# print("finish")

# -------------git公開用データ保存----------------------------------
# # y を NumPy に変換して DataFrame に
# df = pd.DataFrame(y.numpy())

# # 列名を "t0", "t1", ..., "t99" に設定（任意）
# df.columns = [f"t{i}" for i in range(y.shape[1])]

# # CSV として保存
# df.to_csv("leadin_vehicle.csv", index=False)