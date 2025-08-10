import torch
import random

def silu(x):
    return x * torch.sigmoid(x)
def sigma(num_vehicles,x):
	num_cols = x.size(dim=1)
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


	# if num_cols < 10:
	# 	for t in range(num_cols):
	# 		for i in range(1, num_vehicles +1):
	# 			loss1 =  loss1 + 0.1*(t+1)*(x[num_vehicles + i,t] - goal*x[i,t])**2
	# 		for j in range(1,num_vehicles):
	# 			if (x[num_vehicles*2+j,t])**2 < 1:
	# 				loss2 = loss2 + silu((t+1)*((x[num_vehicles*2+j+1,t])**2-1))
	# 			else:
	# 				loss2 = loss2 + silu((t+1)*((x[num_vehicles*2+j+1,t])**2-(x[num_vehicles*2+j,t])**2))

	# else:
	# 	for t in range(num_cols-10 ,num_cols):
	# 		for i in range(1, num_vehicles +1):
	# 			loss1 =  loss1 + 0.1*(t+1)*(x[num_vehicles + i,t] - goal*x[i,t])**2
	# 		for j in range(1,num_vehicles):
	# 			if (x[num_vehicles*2+j,t])**2 < 1:
	# 				loss2 = loss2 + silu((t+1)*((x[num_vehicles*2+j+1,t])**2-1))
	# 			else:
	# 				loss2 = loss2 + silu((t+1)*((x[num_vehicles*2+j+1,t])**2-(x[num_vehicles*2+j,t])**2))

	result = loss1 + 0.1*loss2
	return result

# ダイナミクス x[0]先頭車の速度 x[2k-1]後続車両の速度 x[2k]車間距離
#ダイナミクスパラメータ
para1 = 50
para2 = 2
para3 = 0.1

def dynamics(x, u, num_vehicles,m):#状態，入力，全部でk台   
	x_next = torch.zeros(2*num_vehicles+1)
	for j in range(1,num_vehicles+1):
		x_next[j] = x[j] + u[j-1] - (para1 + para2*x[j]+ para3*x[j]**2)/m[j-1]
		x_next[num_vehicles+j] = x[num_vehicles+j] + x[j-1] -x[j]
	return x_next

def f_sigma(x):
	result = 0
	loss1 = 0 
	loss2 = 0
	D = torch.zeros(2,100)
	for t in range(100):
		loss1 =  loss1 + (x[6,t] - ref_vth(x[2,t],x[3,t]))**2 
		for i in range(2):
			D[i,t] = x[5+i,t] - ref_vth(x[1+i,t],x[2+i,t])-(x[5+i,t-1] - ref_vth(x[1+i,t-1],x[2+i,t-1]))
	
	for t in range(100):
		if (D[0,t])**2 < 1:
			loss2 = loss2 + silu((D[1,t])**2-1)
		else:
			loss2 = loss2 + silu((D[1,t])**2-(D[0,t])**2)

	result = loss1 + loss2
	return result

def f_dynamics(x, u ,m):#状態，入力，全部でk台   
	x_next = torch.zeros(2)
	x_result = torch.zeros(7)
	x_next[0] = x[3] + u - (para1 + para2*x[3]+ para3*x[3]**2)/m	#v
	x_next[1] = x[6] + x[2] - x[3]									#d
	x_result[0:3] = x[0:3]			#v
	x_result[3] = x_next[0]			#vf
	x_result[4:6] = x[3:5]			#d
	x_result[6:7] = x_next[1]	 	#df
	return x_result

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

def m(m):
	M = [1000,1200,1400,1600,1800]
	for i in range(m.shape[0]):
		random_int = random.randint(0,4)
		m[i] = M[random_int]
	return m