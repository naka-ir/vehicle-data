import torch
#loss計算

def silu(x):
    return x * torch.sigmoid(x)
def sigma(num_vehicles,x):
	result = 0
	loss1 = 0 #目標車間距離との二乗和誤差
	loss2 = 0 #ストリングスタビリティ目指す
	D = torch.zeros(num_vehicles,100)
	for t in range(100):
		for i in range(1, num_vehicles +1):
			loss1 =  loss1 + (x[num_vehicles + i,t] - ref_vth(x[i-1,t],x[i,t]))**2
			D[i-1,t] = x[num_vehicles+i,t] - ref_vth(x[i-1,t],x[i,t])-(x[num_vehicles+i,t-1] - ref_vth(x[i-1,t-1],x[i,t-1]))
	
	for t in range(100):	
		for j in range(1,num_vehicles):
			if (D[j-1,t])**2 < 1:
				loss2 = loss2 + silu((D[j,t])**2-1)
			else:
				loss2 = loss2 + silu((D[j,t])**2-(D[j-1,t])**2)


	result = loss1 + loss2
	return result

# ダイナミクス x[0]先頭車の速度 x[2k-1]後続車両の速度 x[2k]車間距離
para1 = 50
para2 = 2
para3 = 0.1

def dynamics(x, u, num_vehicles,m):#状態，入力，全部でk台   
	x_next = torch.zeros(2*num_vehicles+1)
	for j in range(1,num_vehicles+1):
		x_next[j] = x[j] + u[j-1] - (para1 + para2*x[j]+ para3*x[j]**2)/m[j-1]
		x_next[num_vehicles+j] = x[num_vehicles+j] + x[j-1] -x[j]
	return x_next

def ref_vth(vl,vf): #vthによる目標車間距離の決定
	h_temp = 0.1 - 0.2*(vl - vf)
	if h_temp > 1:
		h = 1
	elif h_temp < 0:
		h = 0
	else:
		h = h_temp
	dr = 3 + h*vf
	return dr

def test_controller(controller,num_test,num_vehicles,lead_test,horizon,x0,m):
	
	loss_test = []
	x_all = torch.empty((num_test,num_vehicles*2+1,horizon))
	
	for i in range(num_test):
		
		lead_test_temp = lead_test[i,:]            #先頭車両のi番目の時系列データを抽出
		x = torch.vstack((lead_test_temp[0],x0))   #t=0の全車両の初期状態
		xt = x
		X = x.clone()                               #Xにこの状態を保存する
		
		loss = 0
		for t in range(horizon-1): #t 0~horizon-2	
			u = controller(torch.t(xt)) # コントローラで制御入力計算

			xt = dynamics(x[:,-1], u.T, num_vehicles,m) # ダイナミクスに制御入力を印加して状態を更新
			xt[0] = lead_test_temp[t+1]
			xt_2d = xt.reshape(-1,1)                    #hstackに合うようにuを2次元にする
			X = torch.hstack((X,xt_2d))					#N+1行 horizon列のデータができる
			x = torch.hstack((x,xt.unsqueeze(1)))
			xt = xt.reshape(-1,1)
		calcu = sigma(num_vehicles,x)
		loss = loss + calcu
			
		loss_test.append(loss)
		#i,:,:にi回目のテスト結果を代入
		# x_all[i,:,:] = np.expand_dims(np.expand_dims(X, axis=0), axis=0)
		x_all[i,:,:] = torch.unsqueeze(torch.unsqueeze(X, 0), 0)

	
	return loss_test, x_all
