import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import matplotlib.pyplot as plt
from leading import y
import func
import test_controller as tc
import pickle
import time 
import random
import copy
start_time = time.time()
# パラメータ 制御対象のダイナミクス
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 制御関連パラメータ
horizon = 120						# 制御ホライズン
f_state_size = 7					# 状態次元
f_control_size = 1					# 制御入力次元
f_control_bound = 3					# 各制御入力変数の大きさの上界
coeff_loss = 3						# lossの係数

# NNハイパーパラメータ
f_hidden = 256						# 隠れ状態h,cの次元数（STLの時は32が適当でした）
f_layers = 10						# LSTMの層数
f_learning_rate = 0.0005			# 学習率
epoch = 60							# エポック数
iteration = 100						# イテレーション
batch_size = 4						# バッチサイズ
q = iteration*batch_size			#コードを短くするためだけに作った
qq = int(0.8*q)						# 後方車両のデータを多く含ませるために作った変数

loss_ave_all = []					# テスト時のloss平均格納用
car_number = 4						# 全車両台数はcar_number+3台
# d = [random.random()*10 - 5 for i in range(q*num_vehicles)]	#初期位置用外乱
M = [1000,1200,1400,1600,1800]
#---------------確認用----------------------
x_aaa = torch.zeros(0,7,horizon)
#-----------2台分のコントローラをロードして使う-------------
num_test = 10
num_vehicles = 2

# パラメータの読み込み
with open('result_main/4/control_data','rb') as control_data:
	data_dict =pickle.load(control_data)
state_size = data_dict['state_size']
control_size = data_dict['control_size']
control_bound = data_dict['control_bound']
hidden_size = data_dict['hidden_size']
num_layers = data_dict['num_layers']
learning_rate = data_dict['learning_rate']

class LSTM_Controller(nn.Module):
	def __init__(self, f_state_size, f_hidden, f_layers, f_control_size, control_bound):
		super().__init__()
		self.f_lstm = nn.LSTM(f_state_size, f_hidden, f_layers, batch_first=True)
		self.f_fc = nn.Linear(f_hidden, f_control_size)
		self.control_bound = control_bound
			
	def forward(self, x, h, c):
		x = x.permute(*torch.arange(x.ndim - 1, -1, -1))
		out, (f_hn, f_cn) = self.f_lstm(x,(h,c)) # 現在の状態と隠れ状態をlstmに入力し，制御入力のもととなるoutおよび次の時刻の隠れ状態を計算
		out = self.f_fc(out)#out.size() = ([hidden_size]) 1*hidden_sizeの配列になっている
		out = self.control_bound * torch.tanh(out) # 制御入力の大きさをcontrol_bound以下に限定する（tanhの出力は-1～1）
		
		return out, f_hn, f_cn

# モデル
controller = LSTM_Controller(state_size, hidden_size, num_layers, control_size, control_bound).to(device)
# モデルの読み込み
checkpoint = torch.load("result_main/4/controller_save.tar") # 読み込みたいtarファイル名を入力
controller.load_state_dict(checkpoint["controller_state_dict"])

# ----------------------------訓練データ作成---------------------------
d0 = func.ref_vth(25,25) 
lead_x0 = torch.tensor([[25],[25],[d0],[d0]])
# 1台分のデータを入力して2,3台目のデータを作成
lead_init = torch.zeros(q,horizon)
for i in range (q):
    lead_init[i,:] = y[i,:]
m = torch.zeros(2)
for i in range(2):
	random_int = random.randint(0, 4)
	m[i] = M[random_int]
with torch.no_grad():
	_, x_012 = tc.test_controller(controller,q,2,lead_init,horizon,lead_x0,hidden_size,num_layers,m)
x_train = x_012

f_controller = LSTM_Controller(f_state_size, f_hidden, f_layers, f_control_size, control_bound).to(device)
optimizer = torch.optim.Adam(f_controller.parameters(), lr=f_learning_rate)

#----------------------------followモデル読み込み----------------------------------------------
checkpoint = torch.load("result_main/f72/controller_save.tar") # 読み込みたいtarファイル名を入力
f_controller.load_state_dict(checkpoint["f_controller_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# k=0の時のためのleadtest
lead_test = torch.zeros(num_test,5,horizon)
for i in range (num_test):
	random_number = random.randint(0, q-1)
	lead_test[i,:,:] = x_train[random_number ,:,:]
#----------------------------メインループ------------------------------------------------
for k in range(car_number):
	print("全部で",k+4,"台")
	for i in range(epoch):
		print("epoch:", i+1)
		next_point = 0
		rand_indx = torch.randperm(q)
		for ite in range(iteration):
			loss = 0
			for n in range(batch_size):
				lead_car_temp = x_train[rand_indx[next_point+n],:,:]
				xf0 = torch.tensor([[25],[d0]])
				# 初期状態作成 
				x = torch.zeros(7,1)
				x[0:3,0] = lead_car_temp[0:3,0]	#v
				x[ 3, 0] = xf0[0,0]					#vf
				x[4:6,0] = lead_car_temp[3:5,0]	#d
				x[6:7,0] = xf0[1,0]					#df

				f_xt = x
				f_hn = torch.zeros(f_layers, f_hidden) # 隠れ状態初期化
				f_cn = torch.zeros(f_layers, f_hidden)
				 
				random_int = random.randint(0, 4)
				m = M[random_int]
				
				for t in range(horizon-1):
					u, f_hn, f_cn = f_controller(f_xt, f_hn, f_cn) # lstmコントローラで制御入力計算
					
					f_xt = func.f_dynamics(x[:,-1], u.T,m) # ダイナミクスに制御入力を印加して状態を更新
					f_xt[0:3] = lead_car_temp[0:3,t+1]	#v
					f_xt[4:6] = lead_car_temp[3:5,t+1]	#d
					x = torch.hstack((x,f_xt.unsqueeze(1)))
					f_xt = f_xt.reshape(-1,1)
				loss = func.f_sigma(x)
				
			next_point = next_point + batch_size

			# パラメータ更新
			f_controller.zero_grad() # zero_gradで勾配を初期化（これをしないと以前計算した勾配に足されていきます）
			optimizer.zero_grad()
			loss.backward()        # back propagationで各パラメータの勾配を計算
			optimizer.step()       # 上で計算した勾配を用いてパラメータ更新（adam）
				
			# 以下test_controller関数による制御性能評価＆結果表示（evaluate_each毎に実施）
			if(i == 10 or i==20 or i==30 or i == 40 or i==50 or i==60 or i == 70 or i==80 or i==90 or i==epoch-1):
				if (ite == iteration-1):#試行回数がevaluate_eachと同じになったら評価する
					f_m = torch.zeros(num_test)
					for j in range(num_test):
						random_int = random.randint(0, 4)
						f_m[j] = M[random_int]
					with torch.no_grad(): #更新せずにmodelを利用する場合はwith torch.no_grad()
						print("iterarion:", ite)
						
						loss_test, _ = tc.f_test_controller(f_controller,num_test,lead_test,horizon,xf0,f_hidden,f_layers,f_m)
						loss_test = torch.tensor(loss_test, dtype=torch.float64)
						loss_test_ave = torch.mean(loss_test)

						print("averaged loss = ", loss_test_ave)            
			
						loss_ave_all.append(loss_test_ave)
		if(i == int(epoch/3 or 2*epoch/3)):
			f_m = torch.zeros(q)
			for i in range(q):
				random_int = random.randint(0, 4)
				f_m[i] = M[random_int]
			x_input = copy.deepcopy(x_012)
			x_all_temp = x_012
			for i in range(k+1):
				with torch.no_grad():
					_, x_out = tc.f_test_controller(f_controller,q,x_input,horizon,xf0,f_hidden,f_layers,f_m)
				
				x_input[:,0:3,:] = x_out[:,1:4,:]	#v後ろ3つ
				x_input[:,3:5,:] = x_out[:,5:7,:]	#d後ろ2つ
				x_all_temp = torch.vstack((x_all_temp,x_input))
			#=-------------------5割後方車両---------------------------------
			x_train_temp = x_all_temp[0:q*(k+2):2*(k+2),:,:]
			x_train = torch.vstack((x_train_temp,x_input[0:q:2,:,:]))

	#----controllerから最後方の車両データを作成して次の訓練データにする----
	#----1台目から最後方までのデータは得られない---
	
	f_m = torch.zeros(q)
	for i in range(q):
		random_int = random.randint(0, 4)
		f_m[i] = M[random_int]
	x_input = copy.deepcopy(x_012)
	x_all_temp = x_012
	for i in range(k+1):
		with torch.no_grad():
			_, x_out = tc.f_test_controller(f_controller,q,x_input,horizon,xf0,f_hidden,f_layers,f_m)
		
		x_input[:,0:3,:] = x_out[:,1:4,:]	#v後ろ3つ
		x_input[:,3:5,:] = x_out[:,5:7,:]	#d後ろ2つ
		x_all_temp = torch.vstack((x_all_temp,x_input))
	#=-------------------5割後方車両---------------------------------
	x_train_temp = x_all_temp[0:q*(k+2):2*(k+2),:,:]
	x_train = torch.vstack((x_train_temp,x_input[0:q:2,:,:]))

	int_q = int(q/num_test)
	lead_test = x_all_temp[0:q*(k+2):int_q*(k+2),:,:] # 全種類のテストデータ作成 num_test,5,horizon

# DNNでプロット重ねるために変数保存
preserve_cntrl = {"x_aaa":x_aaa}
with open('temporaly_fdata_preservation/x_aaa','wb') as control_data:
	pickle.dump(preserve_cntrl,control_data)
	

print("close the graph")  
end_time = time.time()
elapsed_time = (end_time - start_time)/60
print(f"elapsed time: {elapsed_time:.2f}minutes") 
fig, ax = plt.subplots(1, figsize=(5,5))            
x_eva = range(len(loss_ave_all))    	#何回目のevaluate_eachか
y_eva = loss_ave_all#[5::]            	#その時のloss(最初のlossは大きすぎるからカットしている)
ax.plot(x_eva,y_eva)                	#だんだん減っていってくれると学習が進んでいるという話になる
plt.xlabel("Epoch ")
plt.ylabel("Averaged Loss")
plt.ylim([0, 10000])
plt.show()

#メインで使ったデータの保存
preserve_main = {"input":y,"x_eva":x_eva,"y_eva":y_eva}
with open('temporaly_fdata_preservation/main_data','wb') as main_data:
	pickle.dump(preserve_main,main_data)
# controllerを使うときのためのパラメータ保存 load_controllerで利用
preserve_cntrl = {"f_state_size":f_state_size,
				"f_hidden":f_hidden,
				"f_layers":f_layers,
				"f_control_size":f_control_size,
				"f_control_bound":f_control_bound,
				"f_learning_rate":f_learning_rate,
				"xf0":xf0}
with open('temporaly_fdata_preservation/control_data','wb') as f_control_data:
	pickle.dump(preserve_cntrl,f_control_data)
# 追従確認のためのデータ保存
with torch.no_grad():
	_, f_x_all = tc.f_test_controller(f_controller,num_test ,lead_test,horizon,xf0,f_hidden,f_layers,f_m)
	result = f_x_all[1,:,:]
preserve_data ={"f_x_all":f_x_all,"horizon":horizon}
with open('temporaly_fdata_preservation/validation', 'wb') as validation:
	pickle.dump(preserve_data, validation)

t = np.linspace(0,horizon-1,horizon)#サンプリングタイム変更したときはここも変更

# ここのプロットは後方4台の動きを出すだけ
fig_v, ax_v = plt.subplots(3,1)
for i in range(3):
	ax_v[i].set_xlim((0, horizon+1))
	result = f_x_all[i,:,:]
	for k in range(4):
		ax_v[i].plot(t, result[k, :], label=f"v_{k}")
	ax_v[i].set_xlabel("t")
	ax_v[i].set_ylabel("v")
	ax_v[i].legend()
plt.savefig("temporaly_fdata_preservation/v")

fig_e, ax_e = plt.subplots(3,1)
error = np.zeros((4+1,horizon))
sum_error = np.zeros((4+1,10))
dr = np.zeros((4+1,horizon))
for i in range(10):
	result = f_x_all[i,:,:]
	for ti in range(100):
		for k in range(1,3+1):
			dr[k,ti] = func.ref_vth(result[k-1,ti],result[k,ti])

	if i < 3:
		ax_e[i].set_xlim((0, horizon+1))
		for k in range(1,3+1):
			error[k] = result[k+3]-dr[k]
			ax_e[i].plot(t, error[k, :], label=f"e_{k}")
			sum_error[k,i] = np.sum(np.abs(error[k,:]))
		ax_e[i].set_xlabel("t")
		ax_e[i].set_ylabel("e")
		ax_e[i].legend()

	else:
		for k in range(1,3+1):
			error[k] = result[k+3]-dr[k]
			sum_error[k,i] = np.sum(np.abs(error[k,:]))

plt.savefig("temporaly_fdata_preservation/e")

ave_sum_err = np.zeros(4+1)
for k in range(1,4):
	ave_sum_err[k] =  np.mean(sum_error[k])
	print(k,"台目の誤差の総和の平均:",ave_sum_err[k] )
car = np.arange(0, 4, 1)
fig_loss, ax_loss = plt.subplots(1, figsize=(5,5))            
ax_loss.plot(car[1:4], ave_sum_err[1:4])
plt.xlabel('x')
plt.ylabel('average of sum error')
plt.show()


torch.save(
	{
		"f_controller_state_dict": f_controller.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
	},
	"temporaly_fdata_preservation/controller_save.tar",
)