import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import matplotlib.pyplot as plt
from tr_v1 import y , l_test
import DNN_func as df
import pickle
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#以下適宜変更する
horizon = 100
num_test = 10


# パラメータの読み込み
with open('DNN_result/9/control_data','rb') as control_data:
	data_dict =pickle.load(control_data)
state_size = data_dict['state_size']
control_size = data_dict['control_size']
control_bound = data_dict['control_bound']
learning_rate = data_dict['learning_rate']
y = data_dict['lead_test_data']
x0 = data_dict['x0']
num_vehicles = data_dict['num_vehicles']

class DNN_Controller(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear1 = nn.Linear(state_size, 64)
		self.linear2 = nn.Linear(64,128)
		self.linear4 = nn.Linear(128,128)
		self.linear5 = nn.Linear(128,64)
		self.linear3 = nn.Linear(64, control_size)
	def forward(self, x):
		x = nn.functional.relu(self.linear1(x))
		x = nn.functional.relu(self.linear2(x))
		x = nn.functional.relu(self.linear4(x))
		x = nn.functional.relu(self.linear4(x))
		x = nn.functional.relu(self.linear4(x))
		x = nn.functional.relu(self.linear4(x))
		x = nn.functional.relu(self.linear5(x))
		x = self.linear3(x)
		x = control_bound * torch.tanh(x)
		return x



# # ------------------------------------------------------------
# モデル
controller = DNN_Controller().to(device)
# オプティマイザ
optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)

# モデルを読み込む。
checkpoint = torch.load("DNN_result/9/controller_save.tar") # 読み込みたいtarファイル名を入力
controller.load_state_dict(checkpoint["controller_state_dict"])

# # ----------ここで計算----------------
M = [1000,1200,1400,1600,1800]

m = torch.zeros(num_vehicles)
for i in range(num_vehicles):
	random_int = random.randint(0, 4)
	m[i] = M[0]
with torch.no_grad():
	_, x_all = df.test_controller(controller,num_test,num_vehicles,l_test,horizon,x0,m)
t = np.linspace(0, horizon-1,horizon)#サンプリングタイム変更したときはここも変更
# # -----------------------------------

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
plt.rcParams["legend.fontsize"] = 10 # 凡例のフォントサイズ
plt.rcParams["legend.loc"] = "upper right"         # 凡例の位置、"best"でいい感じのところ
# plt.rcParams["legend.labelspacing"] = 5. # 垂直（縦）方向の距離の各凡例の距離
# plt.rcParams["legend.handletextpad"] = 3. # 凡例の線と文字の距離の長さ
plt.rcParams["legend.markerscale"] = 2 # 点がある場合のmarker scale
plt.rcParams["legend.borderaxespad"] = 0 # 凡例の端とグラフの端を合わせる
plt.rcParams['pdf.fonttype'] = 42

fig_v, ax_v = plt.subplots(3,1)
for i in range(3):
	ax_v[i].set_xlim((0, horizon))
	result = x_all[i,:,:]
	for k in range(num_vehicles+1):
		ax_v[i].plot(t, result[k, :], label=f"v_{k}")
	ax_v[i].set_xlabel(r"$time$")
	ax_v[i].set_ylabel(r"$velocity$")
	ax_v[i].legend()
plt.savefig("temporaly_data_preservation/v")

fig_e, ax_e = plt.subplots(3,1)
error = np.zeros((num_vehicles+1,horizon))
sum_error = np.zeros((num_vehicles+1,10))
dr = np.zeros((num_vehicles+1,horizon))
for i in range(10):
	result = x_all[i,:,:]
	for ti in range(100):
		for k in range(1,num_vehicles+1):
			dr[k,ti] = df.ref_vth(result[k-1,ti],result[k,ti])
	
	if i < 3:
		ax_e[i].set_xlim((0, horizon+1))
		for k in range(1,num_vehicles+1):
			error[k] = result[k+num_vehicles]- dr[k]
			ax_e[i].plot(t, error[k, :], label=f"e_{k}")
			sum_error[k,i] = np.sum(np.abs(error[k,:]))
		ax_e[i].set_xlabel(r"$time$")
		ax_e[i].set_ylabel(r"$error$")
		ax_e[i].legend()
	
	else:
		for k in range(1,num_vehicles+1):
			error[k] = result[k+num_vehicles]- dr[k]
			sum_error[k,i] = np.sum(np.abs(error[k,:]))
plt.savefig("temporaly_data_preservation/e")

fig_d, ax_d = plt.subplots(3,1)
for i in range(3):
	ax_d[i].set_xlim((0, horizon))
	result = x_all[i,:,:]
	for k in range(1,num_vehicles+1):
		ax_d[i].plot(t, result[k+num_vehicles, :], label=f"d_{k}")
	ax_d[i].set_xlabel(r"$time$")
	ax_d[i].set_ylabel(r"$distance$")
	ax_d[i].legend()
plt.savefig("temporaly_data_preservation/d")
plt.rcParams["xtick.major.width"] = 1.0     # x軸主目盛り線の線幅


ave_sum_err = np.zeros(num_vehicles+1)
for k in range(1,num_vehicles+1):
	ave_sum_err[k] =  np.mean(sum_error[k])
	print(k,"台目の誤差の総和の平均:",ave_sum_err[k] )
car = np.arange(0, num_vehicles+1, 1)
fig_loss, ax_loss = plt.subplots(1)            
ax_loss.plot(car[1:num_vehicles+1], ave_sum_err[1:num_vehicles+1])
plt.xlabel(r"number of car")
plt.ylabel(r"average of sum error")
plt.savefig("temporaly_data_preservation/ave_e")

#LSTMで制御したものも一緒にプロット----------------
# plt.rcParams["legend.fontsize"] = 18 # 凡例のフォントサイズ
# with open('DNN_result/9/lstmcontrol_data','rb') as control_data:
# 	data_dict =pickle.load(control_data)
# LSTM_ave_sum_err = data_dict['LSTM_ave_sum_err']
# ave_sum_err = np.zeros(num_vehicles+1)
# for k in range(1,num_vehicles+1):
# 	ave_sum_err[k] =  np.mean(sum_error[k])
# 	print(k,"台目の誤差の総和の平均:",ave_sum_err[k] )
# car = np.arange(0, num_vehicles+1, 1)
# fig_loss, ax_loss = plt.subplots(1)            
# ax_loss.plot(car[1:num_vehicles+1], LSTM_ave_sum_err[1:num_vehicles+1],label="LSTM")
# ax_loss.plot(car[1:num_vehicles+1], ave_sum_err[1:num_vehicles+1],label="DNN")
# ax_loss.legend()
# plt.xlabel(r"number of car")
# plt.ylabel(r"average of sum error")
# plt.savefig("temporaly_data_preservation/ave_e")


plt.rcParams["legend.fontsize"] = 9 # 凡例のフォントサイズ
fig_all, ax_all = plt.subplots(3,1)
for i in range(3):
	ax_all[i].set_xlim((0, horizon))

result = x_all[0,:,:]
for k in range(num_vehicles+1):
	ax_all[0].plot(t, result[k, :], label=f"v_{k}")
ax_all[0].set_ylabel('velocity[m/s]')
ax_all[0].legend()

plt.rcParams["legend.fontsize"] = 10 # 凡例のフォントサイズ
error = np.zeros((num_vehicles+1,horizon))
dr = np.zeros((num_vehicles+1,horizon))

for ti in range(100):
	for k in range(1,num_vehicles+1):
		dr[k,ti] = df.ref_vth(result[k-1,ti],result[k,ti])
for k in range(1,num_vehicles+1):
	error[k] = result[k+num_vehicles]- dr[k]
	ax_all[1].plot(t, error[k, :], label=f"e_{k}")
	sum_error[k,i] = np.sum(np.abs(error[k,:]))
ax_all[1].set_ylabel('error[m]')
ax_all[1].legend()
	

for k in range(1,num_vehicles+1):
	ax_all[2].plot(t, result[k+num_vehicles, :], label=f"d_{k}")
ax_all[2].set_xlabel('time[s]')
ax_all[2].set_ylabel('distance[m]')
ax_all[2].legend()
plt.savefig("temporaly_data_preservation/all")
plt.rcParams["xtick.major.width"] = 1.0     # x軸主目盛り線の線幅


#-----------------------------------------------
# plt.show()
# # save
# fig_v.savefig('DNN_6_v.pdf', bbox_inches="tight", pad_inches=0.05)
# fig_e.savefig('DNN_6_e.pdf', bbox_inches="tight", pad_inches=0.05)
# fig_d.savefig('DNN_6_d.pdf', bbox_inches="tight", pad_inches=0.05)
fig_loss.savefig('DNN_loss.pdf', bbox_inches="tight", pad_inches=0.05)
fig_all.savefig('DNN_all.pdf', bbox_inches="tight", pad_inches=0.05)