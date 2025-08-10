import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import matplotlib.pyplot as plt
from leading import y ,l_test
import func
import test_controller as tc
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
hidden_size = 128						# 隠れ状態h,cの次元数（STLの時は32が適当でした）
num_layers = 6							# LSTMの層数
learning_rate = 0.0005					# 学習率
epoch = 70  							# エポック数
iteration = 100							# イテレーション
batch_size = 8							# バッチサイズ
evaluate_each = 60						# iterationがevaluate_eachごとにモデルをtest_controller関数で評価

num_test = 10							# テスト時に評価する初期状態数
loss_ave_all = []						# テスト時のloss平均格納用
lead_car = y
#d = [random.random()*10 - 5 for i in range(iteration*batch_size*num_vehicles)]	#初期位置用外乱研究には結局使わなかった
M = [1000,1200,1400,1600,1800]
# ----------------------------入力データ---------------------------
## 入力データをあらかじめ準備
lead_test = torch.zeros(num_test,horizon)
for i in range (num_test):
        lead_test[i,:] = lead_car[100 + i,:]
# -------------------------------------------------------

# controller
class LSTM_Controller(nn.Module):
	def __init__(self, state_size, hidden_size, num_layers, control_size, control_bound):
		super().__init__()
		self.lstm = nn.LSTM(state_size, hidden_size, num_layers, batch_first=True)
		self.fc = nn.Linear(hidden_size, control_size)
		self.control_bound = control_bound
			
	def forward(self, x, h, c):
		x = x.permute(*torch.arange(x.ndim - 1, -1, -1))
		out, (hn, cn) = self.lstm(x,(h,c)) # 現在の状態と隠れ状態をlstmに入力し，制御入力のもととなるoutおよび次の時刻の隠れ状態を計算
		out = self.fc(out)#out.size() = ([hidden_size]) 1*hidden_sizeの配列になっている
		out = self.control_bound * torch.tanh(out) # 制御入力の大きさをcontrol_bound以下に限定する（tanhの出力は-1～1）
		
		return out, hn, cn
# モデル
controller = LSTM_Controller(state_size, hidden_size, num_layers, control_size, control_bound).to(device)
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
			lead_car_temp = lead_car[n, :]#先頭車両のite番目の時系列データを抽出
			d0 = func.ref_vth(25,25) 
			x0 = torch.tensor([[25],[25],[d0],[d0]])
			x0 = torch.tensor([[25],[25],[25],[25],[25],[25],[d0],[d0],[d0],[d0],[d0],[d0]])
			x = torch.vstack((lead_car_temp[0],x0))#t=0の時の全車両の初期状態
			xt = x
			hn = torch.zeros(num_layers, hidden_size) # 隠れ状態初期化
			cn = torch.zeros(num_layers, hidden_size)

			m = torch.zeros(num_vehicles)
			for i in range(num_vehicles):
				random_int = random.randint(0, 4)
				m[i] = M[random_int]
			
			for t in range(horizon-1): #t 0~horizon-2	
				u, hn, cn = controller(xt, hn, cn) # lstmコントローラで制御入力計算
				
				xt = func.dynamics(x[:,-1] , u.T, num_vehicles,m) # ダイナミクスに制御入力を印加して状態を更新
				xt[0] = lead_car_temp[t+1]
				x = torch.hstack((x,xt.unsqueeze(1)))
				xt = xt.reshape(-1,1) 
			calcu = func.sigma(num_vehicles,x)
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
				
				loss_test, _ = tc.test_controller(controller,num_test,num_vehicles,lead_test,horizon,x0,hidden_size,num_layers,m)
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
				"hidden_size":hidden_size,
				"num_layers":num_layers,
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
	_, x_all = tc.test_controller(controller,num_test,num_vehicles,l_test,horizon,x0,hidden_size,num_layers,m)
	result = x_all[1,:,:]
preserve_data ={"x_all":x_all,"horizon":horizon}
with open('temporaly_data_preservation/validation', 'wb') as validation:
	pickle.dump(preserve_data, validation)

t = np.linspace(0,horizon-1,horizon)#サンプリングタイム変更したときはここも変更


plt.rcParams["figure.figsize"] = 8,6
plt.rcParams["figure.subplot.left"] = 0.14  # 余白
plt.rcParams["figure.subplot.bottom"] = 0.14# 余白
plt.rcParams["figure.subplot.right"] =0.90  # 余白
plt.rcParams["figure.subplot.top"] = 0.90   # 余白
plt.rcParams['font.family'] = 'Times New Roman' # font familyの設定
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 18 # 全体のフォントサイズが変更
# plt.rcParams['xtick.labelsize'] = 18 # 軸だけ変更
# plt.rcParams['ytick.labelsize'] = 18 # 軸だけ変更
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in 
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid
plt.rcParams["legend.fancybox"] = False # 凡例の丸角をとる
plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
plt.rcParams["legend.handlelength"] = 1 # 凡例の線の長さを調節
plt.rcParams["legend.fontsize"] = 10 # 凡例のフォントサイズ
# plt.rcParams["legend.labelspacing"] = 5. # 垂直（縦）方向の距離の各凡例の距離
# plt.rcParams["legend.handletextpad"] = 3. # 凡例の線と文字の距離の長さ
plt.rcParams["legend.markerscale"] = 2 # 点がある場合のmarker scale
plt.rcParams["legend.borderaxespad"] = 0 # 凡例の端とグラフの端を合わせる
plt.rcParams["legend.loc"] = 'upper right' # 凡例の位置を右上にする
plt.rcParams['pdf.fonttype'] = 42
fig_v, ax_v = plt.subplots(3,1)
for i in range(3):
	ax_v[i].set_xlim((0, horizon))
	result = x_all[i,:,:]
	for k in range(num_vehicles+1):
		ax_v[i].plot(t, result[k, :], label=f"v_{k}")
	ax_v[i].set_xlabel("time[s]")
	ax_v[i].set_ylabel("velocity[m/s]")
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
			dr[k,ti] = func.ref_vth(result[k-1,ti],result[k,ti])
	
	if i < 3:
		ax_e[i].set_xlim((0, horizon+1))
		for k in range(1,num_vehicles+1):
			error[k] = result[k+num_vehicles]- dr[k]
			ax_e[i].plot(t, error[k, :], label=f"e_{k}")
			sum_error[k,i] = np.sum(np.abs(error[k,:]))
		ax_e[i].set_xlabel("time[s]")
		ax_e[i].set_ylabel("error[m]")
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
plt.xlabel('number of car')
plt.ylabel('average of sum error')
plt.show()

torch.save(
	{
		"controller_state_dict": controller.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
	},
	"temporaly_data_preservation/controller_save.tar",
)