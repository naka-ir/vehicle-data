import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import matplotlib.pyplot as plt
from leading import y
import DNN_func as df
import pickle
import time 
import random
start_time = time.time()
# パラメータ 制御対象のダイナミクス
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 制御関連パラメータ
horizon = 100							# 制御ホライズン
num_vehicles = 6						# 後続車両の数
state_size = 1 + 2*num_vehicles			# 状態次元
control_size = num_vehicles				# 制御入力次元
control_bound = 3						# 各制御入力変数の大きさの上界
coeff_loss = 3							# lossの係数

# NNハイパーパラメータ
learning_rate = 0.0005					# 学習率
epoch = 100  							# エポック数
iteration = 50							# イテレーション
batch_size = 8							# バッチサイズ
evaluate_each = 60						# iterationがevaluate_eachごとにモデルをtest_controller関数で評価

num_test = 10							# テスト時に評価する初期状態数
loss_ave_all = []						# テスト時のloss平均格納用
lead_car = y
M = [1000,1200,1400,1600,1800]
# ----------------------------入力データ---------------------------
## 入力データをあらかじめ準備
lead_test = torch.zeros(num_test,horizon)
for i in range (num_test):
		lead_test[i,:] = lead_car[100 + i,:]
# -------------------------------------------------------
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
# モデル
controller = DNN_Controller().to(device)
# オプティマイザ
optimizer = torch.optim.Adam(controller.parameters(), lr=learning_rate)

#----------------------------モデル読み込み----------------------------------------------
# checkpoint = torch.load("result/52/controller_save.tar") # 読み込みたいtarファイル名を入力
# controller.load_state_dict(checkpoint["controller_state_dict"])
# optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#----------------------------メインループ------------------------------------------------
q = iteration*batch_size#コードを短くするためだけに作った
for i in range(epoch):
	print("epoch:", i+1)
	next_point = 0
	for ite in range(iteration):
		loss = 0
		for n in range(batch_size):
			n = next_point + n #事前に用意したデータを上から順番に使う
			lead_car_tempo = lead_car[n, :]#先頭車両のite番目の時系列データを抽出
			d0 = df.ref_vth(25,25) 
			x0 = torch.tensor([[25],[25],[d0],[d0]])
			x0 = torch.tensor([[25],[25],[25],[25],[25],[25],[d0],[d0],[d0],[d0],[d0],[d0]])
			x = torch.vstack((lead_car_tempo[0],x0))#t=0の時の全車両の初期状態
			xt = x

			m = torch.zeros(num_vehicles)
			for i in range(num_vehicles):
				random_int = random.randint(0, 4)
				m[i] = M[random_int]
			
			for t in range(horizon-1): #t 0~horizon-2	
				u = controller(torch.t(xt)) # コントローラで制御入力計算
				
				xt = df.dynamics(x[:,-1] , u.T, num_vehicles,m) # ダイナミクスに制御入力を印加して状態を更新
				xt[0] = lead_car_tempo[t+1]
				x = torch.hstack((x,xt.unsqueeze(1)))
				xt = xt.reshape(-1,1) 
			calcu = df.sigma(num_vehicles,x)
			loss = loss + calcu
			
		next_point = next_point + batch_size

		# パラメータ更新
		controller.zero_grad() # zero_gradで勾配を初期化（これをしないと以前計算した勾配に足されていきます）
		optimizer.zero_grad()
		loss.backward()        # back propagationで各パラメータの勾配を計算
		optimizer.step()       # 上で計算した勾配を用いてパラメータ更新（adam）
			
		# 以下test_controller関数による制御性能評価＆結果表示（evaluate_each毎に実施）
		
		if (ite % evaluate_each == 0):#試行回数がevaluate_eachと同じになったら評価する
			
			with torch.no_grad(): #更新せずにmodelを利用する場合はwith torch.no_grad()
				print("iterarion:", ite)
				
				loss_test, _ = df.test_controller(controller,num_test,num_vehicles,lead_test,horizon,x0,m)
				loss_test = torch.tensor(loss_test, dtype=torch.float64)
				loss_test_ave = torch.mean(loss_test)

				print("averaged loss = ", loss_test_ave)            
	
				loss_ave_all.append(loss_test_ave)
				
	
print("close the graph")  
end_time = time.time()
elapsed_time = (end_time - start_time)/60
print(f"elapsed time: {elapsed_time:.2f}minutes") 
fig, ax = plt.subplots(1, figsize=(5,5))            
x_eva = range(len(loss_ave_all)-5)    	#何回目のevaluate_eachか
y_eva = loss_ave_all[5::]            	#その時のloss(最初のlossは大きすぎるからカットしている)
ax.plot(x_eva,y_eva)                	#だんだん減っていってくれると学習が進んでいるという話になる
plt.xlabel("Epoch (* {})".format(evaluate_each))
plt.ylabel("Averaged Loss")
plt.ylim([0, 10000])
plt.show()

#メインで使ったデータの保存
preserve_main = {"input":y,"x_eva":x_eva,"y_eva":y_eva}
with open('temporaly_data_preservation/main_data','wb') as main_data:
	pickle.dump(preserve_main,main_data)
# controllerを使うときのためのパラメータ保存 load_controllerで利用
preserve_cntrl = {"state_size":state_size,
				"control_size":control_size,
				"control_bound":control_bound,
				"learning_rate":learning_rate,
				"lead_test_data":y,
				"x0":x0,
				"num_vehicles":num_vehicles}
with open('temporaly_data_preservation/control_data','wb') as control_data:
	pickle.dump(preserve_cntrl,control_data)
# 追従確認のためのデータ保存
with torch.no_grad():
	_, x_all = df.test_controller(controller,num_test,num_vehicles,lead_test,horizon,x0,m)
	result = x_all[1,:,:]
preserve_data ={"x_all":x_all,"horizon":horizon}
with open('temporaly_data_preservation/validation', 'wb') as validation:
	pickle.dump(preserve_data, validation)

t = np.linspace(0,horizon-1,horizon)#サンプリングタイム変更したときはここも変更


fig_v, ax_v = plt.subplots(3,1)
for i in range(3):
	ax_v[i].set_xlim((0, horizon))
	result = x_all[i,:,:]
	for k in range(num_vehicles+1):
		ax_v[i].plot(t, result[k, :], label=f"v_{k}")
	ax_v[i].set_xlabel("t")
	ax_v[i].set_ylabel("v")
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
		ax_e[i].set_xlabel("t")
		ax_e[i].set_ylabel("e")
		ax_e[i].legend()
	
	else:
		for k in range(1,num_vehicles+1):
			error[k] = result[k+num_vehicles]- dr[k]
			sum_error[k,i] = np.sum(np.abs(error[k,:]))
plt.savefig("temporaly_data_preservation/e")

ave_sum_err = np.zeros(num_vehicles+1)
for k in range(1,num_vehicles+1):
	ave_sum_err[k] =  np.mean(sum_error[k])
	print(k,"台目の誤差の総和の平均:",ave_sum_err[k] )
car = np.arange(0, num_vehicles+1, 1)
fig_loss, ax_loss = plt.subplots(1, figsize=(5,5))            
ax_loss.plot(car[1:num_vehicles+1], ave_sum_err[1:num_vehicles+1])
plt.xlabel('x')
plt.ylabel('average of sum error')
plt.show()

torch.save(
	{
		"controller_state_dict": controller.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
	},
	"temporaly_data_preservation/controller_save.tar",
)
