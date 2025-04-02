import numpy as np
import matplotlib.pyplot as plt
import time
import RTSsmoother, LAE, LSR
from scipy.linalg import qr
from scipy.linalg import qr, cholesky
from sklearn.linear_model import LinearRegression
from scipy.linalg import block_diag




## 速度\加速度Q值分開調整
def AKF(dt, Pos):
    A = np.array([[1, dt, 0.5*dt**2],
                  [0, 1, dt],
                  [0, 0, 1 ]])
    B = np.array([[0.5*dt**2],
                  [dt],
                  [1]])
    # u = AccCmd
    C = np.array([[1, 0, 0]]) 
    
############################# Q #############################
    # Q = np.array([[0.01, 0, 0],
    #               [0, 1, 0],
    #               [0, 0, 100]]) 
    # Q = np.array([[0.01, 0, 0],
    #               [0, 0.01, 0],
    #               [0, 0, 0.01]]) 
    # Q = np.array([[0.1, 0, 0],
    #               [0, 0.01, 0],
    #               [0, 0, 0.001]])
    # Q = np.array([[0.01, 0.01, 0.01],
    #               [0.01, 0.01, 0.01],
    #               [0.01, 0.01, 0.01]])
    # Q = np.array([[1e-4, 1e-4, 1e-4],
    #               [1e-4, 1e-4, 1e-4],
    #               [1e-4, 1e-4, 1e-4]])
    # Q = np.array([[1e-4, 0.01, 0.01],
    #               [0.01, 1e-4, 0.01],
    #               [0.01, 0.01, 1e-4]])
    Q = np.array([[1e-4, 0, 0],
                  [0, 1e-4, 0],
                  [0, 0, 1e-4]])
    # Q = np.array([[1e-2, 0, 0],
    #               [0, 1e-2, 0],
    #               [0, 0, 1e-2]])
    # Q = np.array([[1e-6, 0, 0],
    #               [0, 1e-6, 0],
    # #               [0, 0, 1e-6]])
    # Q = np.array([[1e-8, 0, 0],
    #               [0, 1e-8, 0],
    #               [0, 0, 1e-8]])
############################# R #############################
    R = 0.1
############################# P #############################
    # P = np.array([[1e-4, 1e-4, 1e-4],
    #                 [1e-4, 1e-4, 1e-4],
    #                 [1e-4, 1e-4, 1e-4]])
    P = np.array([[1e-2, 1e-2, 1e-2],
                    [1e-2, 1e-2, 1e-2],
                    [1e-2, 1e-2, 1e-2]])
    P = np.array([[1e-5, 1e-5, 1e-5],
                    [1e-5, 1e-5, 1e-5],
                    [1e-5, 1e-5, 1e-5]])
    # P = np.array([[1e-6, 1e-6, 1e-6],
    #                 [1e-6, 1e-6, 1e-6],
    #                 [1e-6, 1e-6, 1e-6]])
    # P = np.array([[1e-8, 1e-8, 1e-8],
    #                 [1e-8, 1e-8, 1e-8],
    #                 [1e-8, 1e-8, 1e-8]])
    P = np.array([[1e-10, 1e-10, 1e-10],
                    [1e-10, 1e-10, 1e-10],
                    [1e-10, 1e-10, 1e-10]])
    P = np.array([[1e-12, 1e-12, 1e-12],
                    [1e-12, 1e-12, 1e-12],
                    [1e-12, 1e-12, 1e-12]])
    # P = np.array([[10, 10, 10],
    #                 [10, 10, 10],
    #                 [10, 10, 10]])

    Wt = 0
    # pose = np.zeros(len(Pos))
    # vele = np.zeros(len(Pos))
    # acce = np.zeros(len(Pos))
    pose = []
    vele = []
    acce = []
    xm = np.zeros((3, 1))  
    x = np.zeros((3, 1))  
    Pm = P
    u_p_values = []
    u_v_values = []
    u_a_values = []
    y_values = []
    delta_x_values = []
    Q_acc = []
    Q_vel = []
    Q_pos = []
    Q_0err = []
    Q_save = []
    v_data = [0]
    a_data = [0, 0]
    v_cov_data = []
    a_cov_data = []
    ek_2_data = []

    x_k_update_data = []
    k_y_data = []
    x_tel = []
    x_true_data = [] 
    x_true_data_noise = []
    x_k_predict_data = []

    P_k_update_data = []
    KCP_data = []

    v1_values = []
    v2_values = []
    a1_values = []
    a2_values = []

    vel_values = []
    vel1_values = []
    vel2_values = []
    acc_values = []
    acc1_values = []
    acc2_values = []

    vele_data = [0]
    acce_data = [0]

    LAE_acce_data = []
    LAE_vele_data = []

    Pos_data = []
    P_list = []
    Q_list = []

    vel_gls_data = []
    acc_gls_data = []

    y_est = np.zeros((3, 1))  

#---------------------------EKF---------------------------#
    def f(x, u):
        """非線性狀態轉移模型"""
        # 這裡用的是等加速度模型
        return np.array([x[0] + x[1] * dt + 0.5 * x[2] * dt**2,
                        x[1] + x[2] * dt,
                        x[2]])

    def h(x):
        """非線性觀測模型，這裡假設我們只觀測位置"""
        return np.array([x[0]])

    def F(x, u):
        """狀態轉移模型的雅可比矩陣"""
        return np.array([[1, dt, 0.5 * dt**2],
                         [0, 1, dt],
                         [0, 0, 1]])

    def H(x):
        """觀測模型的雅可比矩陣"""
        return np.array([[1, 0, 0]])

    star_time = time.time()
    print("---------------------------------AKF2_new_try--------------------------------")
    for i in range(len(Pos)): # m = measurement;p = predict len(pos)
        print("------------------------------------------------------------------------------------")
        Pos_data.append(Pos[i])
        # # 用LAE、LSF求初始P矩陣
        # if i < 20:
        #     LAE_acce = LAE.LAE(dt, Pos[0:20])
        #     LAE_vele = np.cumsum(LAE_acce) * dt
        #     # LAE_acce_data.append(LAE_acce)
        #     # LAE_vele_data.append(LAE_vele)
        # x_LAE = np.concatenate((Pos[0:20], LAE_acce.reshape(-1,1), LAE_acce.reshape(-1,1)), axis=1)
        # P_LAE = np.cov(x_LAE.T)
        # if i == 20:
        #     Pm = P_LAE
        #---------------------------KF---------------------------#
        Pp = np.dot(np.dot(A, Pm), A.T) + Q
        print("Q =", Q)
        print("Pp =", Pp)
        xp = np.dot(A, xm) + Wt
        Km = np.dot(Pp, C.T) / (np.dot(np.dot(C, Pp), C.T) + R)
        print("Km =", Km)
        dk = (Pos[i] - np.dot(C, xp)) # dk = zk - hk*xk
        # if i>2:
        #     y = np.array([[Pos[i] - xp[0]],
        #                 [u_v2 - xp[1]],
        #                 [u_a2 - xp[2]]])
        # else:
        y = (Pos[i] - np.dot(C, xp))
        k_y = Km @ y
        xm = xp + np.dot(Km, y) # =x_hat
        ek = (Pos[i] - np.dot(C, xm)) # ek = zk - hk*xk+1
        Pm = np.dot((np.eye(3) - np.dot(Km, C)), Pp)
        KCP = np.dot((np.dot(Km, C)), Pp)
        #---------------------------EKF---------------------------#
        # print("i =", i)
        # Pp = np.dot(np.dot(F(xm, 0), Pm), F(xm, 0).T) + Q  # 預測協方差
        # print("Q =", Q)
        # print("Pp =", Pp)
        # xp = f(xm, 0)  # 預測狀態
        # Km = np.dot(Pp, H(xm).T) / (np.dot(np.dot(H(xm), Pp), H(xm).T) + R)  # 卡爾曼增益
        # y = (Pos[i] - h(xp))  # 創新
        # k_y = Km @ y
        # xm = xp + np.dot(Km, y)
        # Pm = np.dot((np.eye(3) - np.dot(Km, H(xm))), Pp)  # 更新協方差
        # KCP = np.dot((np.dot(Km, C)), Pp)

    # #--------------------------------------------數據儲存--------------------------------------------#
        # x_data_all
        x_k_update_data.append(xm.flatten())
        k_y_data.append(k_y.flatten())
        x_true_data.append(xm[0].flatten())
        x_tel = np.array(x_true_data) - np.array(x_k_update_data)
        # x_tel.append(xm.flatten())
        x_true_data_noise = np.append(x_true_data_noise, Pos[i])
        x_true_data_noise = np.expand_dims(x_true_data_noise, axis=1)
        # print("x_true_data_noise.shape =", x_true_data_noise.shape)
        x_k_predict_data.append(xp.flatten())
        # P_data_all
        P_k_update_data.append(Pm.flatten())
        KCP_data.append(KCP.flatten())

    # #--------------------------------------------週期性重置--------------------------------------------#
        # if i % 500 == 0:  # 每 1000 次迭代重新調整 P
        #     # Pm = Pm * 1e-4
        #     # print("Pm =", Pm)
        #     Q = Q * 1e-4
        #     print("Q =", Q)
                                        ####################
##########################################  Q值自適應過程  ###############################################
                                        ####################
##########################################################################################################
################                               位置求Q                               #####################
##########################################################################################################
        ## 求M
        u_p = (Pos[i]- xm[0]) # 實際值-卡爾曼估測值
        # print("u_p=", u_p)
        u_p_values.append(u_p)
        n = i + 1
        N = 5

        u_sqr = [val**2 for val in u_p_values[:n]]
        # M = sum(u_sqr[:n])/n
        M = sum(u_sqr[:i])/(i+1)
        # M = sum(u_sqr[n-N:n])/N
        ## 求G_telda
        Y = np.dot(np.dot(C, Pp), C.T) + R
        # print("Y =", Y)
        G_tel = Km[0]**2 * Y
        ## 求G_hat
        delta_x = xm[0] - xp[0] # x - x-
        delta_x_values.append(delta_x)
        m = i + 1
        delta_x_values_sqr = [val**2 for val in delta_x_values[:m]]
        # G_hat = sum(delta_x_values_sqr[:m]) / m
        # G_hat = sum(delta_x_values_sqr[:i]) / (i+1)
        G_hat = sum(delta_x_values_sqr[m - N:m]) / N
        ## 求S
        G = G_hat/G_tel
        # print("G_hat=", G_hat)
        # print("G_tel=", G_tel)
        # print("G0=", G)
        a = 1e-1
        # a = 0.5
        a = 1 
        S = np.maximum(a, G_hat/G_tel)
        # S = G_hat/G_tel
        # if G>1:
        #     count0 = count0 + 1
        ## 求Q_hat
        Q_hat = S * M 
        Q[0, 0] = Q_hat
        Q_pos.append(Q[0, 0])
        # print("--------------------------------------------------")
        
##########################################################################################################
################                               速度求Q                               #####################
##########################################################################################################
        ## 求M
        # if i == 0:
        if i < 2:
            # if i == 0:
            #     vel_pred_ols= 0
            # u_v = (1/1)* Pos[i] - xm[1] + 1e-3
            if i == 0:
                x_mean = 0
                v_mean = 0
                cov_xtrue_vtrue = 1
                cov_xtrue_xtrue = 1
            v2 = (cov_xtrue_vtrue / cov_xtrue_xtrue)
            # v2 = 2
            # u_v = v2 * (Pos[i] - x_mean) + v_mean - xm[1] 
            u_v = 1 * (Pos[i] - x_mean) + v_mean - xm[1] 
            # v2 = (Pm[0][1] / Pm[0][0])
            # u_v = u_p *0.1
        else:
            v1 = (Km[1] / Km[0])
            if v1 == 0 or v1 == float('inf'):
                v1 = 1
            v1_values.append(v1)
            s = 5
            # M = sum(u_sqr[:n])/n
            v1 = sum(v1_values[:i])/(i+1)
            # v1 = sum(v1_values[n-N:n])/N
            # x_mean = sum(Pos[n-N:n])/N
            x_mean = sum(Pos_data)/(i+1)
            print("x_mean =", x_mean)
            print("v_mean =", v_mean)
            # v_mean = sum(xm[:i])/i
            # u_v1 = v1 * (Pos[i] - x_mean) + v_mean
            u_v1 = vel_pred_gls_P 
            # u_v1 = v1 * Pos[i]
            # u_v1 = v1 * Pos[i]**2 + v1 * Pos[i]
            print("u_v1 =", u_v1)
            # v2 = (Pm[0][1] / Pm[0][0])
            v2 = (cov_xtrue_vtrue / cov_xtrue_xtrue)
            print("v2 =", v2)
            # v2 = ((cov_data[0][1] - Pm[0][1]) / (cov_xtrue_xtrue))
            if v2 == 0 or v2 == float('inf'):
                v2 = 1
            v2_values.append(v2)
            # M = sum(u_sqr[:n])/n
            # if i == 0:
            #     v2 = sum(v2_values[:i])/(1)
            v2 = sum(v2_values[:i])/(i+1)
            # v2 = sum(v2_values[n-N:n])/N
            print("Pos[i] =", Pos[i])
            u_v2 = v2 * (Pos[i] - x_mean) + v_mean
            # u_v2 = v2 * Pos[i]
            # u_v2 = v2 * Pos[i]**2 + v2 * Pos[i]
            print("u_v2 =", u_v2)
            alpha = 1.0 # 越小越平滑
            # alpha = v1 / (v1 + v2)
            u_v = (1-alpha) * u_v1 + alpha * u_v2 - xm[1] # OLS fusion
            # u_v = (1-alpha) * vel_pred_gls_P + alpha * vel_pred_gls_Q - xm[1] # OLS/GLS fusion
            # u_v = u_v2 - xm[1]
            # u_v = vel_pred_gls_P - xm[1]
            # print("vel_pred =", vel_pred)
            print("u_v =", u_v)

        ##-----------------------------儲存模擬速度----------------------------------#
        ## 用Q求速度
            vel1 = u_v1
            vel1_values.append(vel1)
        ## 用P求速度
            # vel2 = vel_pred
            vel2 = u_v2
            vel2_values.append(vel2)
        ##--------------------------------------------------------------------------#

        u_v_values.append(u_v)
        n = i + 1
        # N = 5
        u_sqr = [val**2 for val in u_v_values[:n]]
        # u_sqr = u_sqr * 1e3
        # M = sum(u_sqr[:n])/n
        M = sum(u_sqr[:i])/(i+1)
        # M = sum(u_sqr[n-N:n])/N
        M = M * 1e0
        print("M =", M)
        ## 求G_tel
        Y = np.dot(np.dot(C, Pp), C.T) + R
        Y = Pm[1][1] + R
        G_tel = Km[1]**2 * Y
        ## 求G_hat
        delta_x = xm[1] - xp[1] # x - x-
        delta_x_values.append(delta_x)
        m = i + 1
        delta_x_values_sqr = [val**2 for val in delta_x_values[:m]]
        # G_hat = sum(delta_x_values_sqr[:m]) / m
        # G_hat = sum(delta_x_values_sqr[:i]) / (i+1)
        G_hat = sum(delta_x_values_sqr[m - N:m]) / N
        ## 求S
        G = G_hat/G_tel
        b = 1e-2
        # b = 0.7
        b = 1
        S = np.maximum(b, G_hat/G_tel)
        # S = G_hat/G_tel
        print("G_hat=", G_hat)
        print("G_tel=", G_tel)
        print("G1=", G)
        # if S > b:
        #     count1 = count1 + 1
        ## 求Q_hat
        Q_hat = S * M *1e-2 #*1e1 #*1e-2
        print("Q_hat =", Q_hat)
        Q[1, 1] = Q_hat
        
        # if np.abs(u_v_values[i]) > np.abs(u_v_values[i-1]):
        #     Q[1, 1] = Q_hat * 0.1
        # else:    
        #     Q[1, 1] = Q_hat * 1
        Q_vel.append(Q[1, 1])

        # print("--------------------------------------------------")

##########################################################################################################
################                              加速度求Q                               #####################
##########################################################################################################
        ## 求M
        u_p = (Pos[i] - xm[0]) # y(k)-x_hat(k)  實際值-卡爾曼估測值
        # if i == 0:
        if i < 2:
            # if i == 0:
            #     acc_pred_ols= 0
            # u_a = (1/1)* Pos[i] - xm[2] + 1e-3
            if i ==0 :
                x_mean = 0
                a_mean = 0
                cov_xtrue_atrue = 1
                cov_xtrue_xtrue = 1
            a2 = (cov_xtrue_atrue / cov_xtrue_xtrue)
            # a2 = 2
            # u_a = a2 * (Pos[i] - x_mean) + a_mean - xm[2]
            u_a = 1 * (Pos[i] - x_mean) + a_mean - xm[2]
            # a2 = (Pm[0][2] / Pm[0][0])
            # u_a = u_p *0.1
            
            # u_a = acc_pred_ols - xm[2]
            # a2 = (Pm[0][2] / Pm[0][0])
            # a2 = (cov_xtrue_atrue / cov_xtrue_xtrue)
            # print("a2 =", a2)
            # # a2 = ((cov_data[0][2] - Pm[0][2]) / (cov_xtrue_xtrue))
            # if a2 == 0 or a2 == float('inf'):
            #     a2 = 1
            # a2_values.append(a2)
            # # N = 10
            # # M = sum(u_sqr[:n])/n
            # a2 = sum(a2_values[:i])/(i+1)
            # u_a = a2 * u_p
        else:
        # if i >= 0:
            # x_mean = 0
            # a_mean = 0
            # cov_xtrue_atrue = 1
            # cov_xtrue_xtrue = 1

            a1 = (Km[2] / Km[0])
            if a1 == 0 or a1 == float('inf'):
                a1 = 1
            a1_values.append(a1)
            # N = 5
            # s = 10
            # M = sum(u_sqr[:n])/n
            # a1 = sum(a1_values[:i])/(i+1)
            a1 = sum(a1_values[n-N:n])/N
            # u_a1 = a1 * (Pos[i] - x_mean) + a_mean
            u_a1 = acc_pred_gls_P  # GLS
            # u_a1 = a1 * Pos[i]
            # u_a1 = a1 * Pos[i]**2 + a1 * Pos[i]
            print("u_a1 =", u_a1)
            a2 = (Pm[0][2] / Pm[0][0])
            a2 = (cov_xtrue_atrue / cov_xtrue_xtrue)
            print("a2 =", a2)
            # a2 = ((cov_data[0][2] - Pm[0][2]) / (cov_xtrue_xtrue))
            if a2 == 0 or a2 == float('inf'):
                a2 = 1
            a2_values.append(a2)
            # N = 10
            M = sum(u_sqr[:n])/n
            # if i == 0:
            #     a2 = sum(a2_values[:i])/(1)
            a2 = sum(a2_values[:i])/(i+1)
            # a2 = sum(a2_values[n-N:n])/N
            u_a2 = a2 * (Pos[i] - x_mean) + a_mean
            # u_a2 = a2 * Pos[i]
            # u_a2 = a2 * Pos[i]**2 + a2 * Pos[i]
            print("u_a2 =", u_a2)   
            alpha = 1.0 #越小越平滑 # u_v:0 ，u_a:0.1
            # alpha = a1 / (a1 + a2)
            u_a = (1-alpha) * u_a1 + alpha * u_a2 - xm[2] # OLS fusion
            # u_a = (1-alpha) * acc_pred_gls_P + alpha * acc_pred_gls_Q - xm[2] # OLS/GLS fusion

            # u_a = u_a2 - xm[2]
            # u_a = acc_pred_gls_P - xm[2] # GLS
            # print("acc_pred =", acc_pred)
            print("u_a =", u_a)
        
        ##-----------------------------儲存模擬速度----------------------------------#
        ## 用Q求加加速度
            acc1 = u_a1
            acc1_values.append(acc1)
        ## 用P求加速度
            # acc2 = acc_pred
            acc2= u_a2
            acc2_values.append(acc2)
        ##--------------------------------------------------------------------------#
        
        u_a_values.append(u_a)
        n = i + 1
        # N = 5
        u_sqr = [val**2 for val in u_a_values[:n]]
        # M = sum(u_sqr[:n])/n
        M = sum(u_sqr[:i])/(i+1)
        # M = sum(u_sqr[n-N:n])/N

        M = M * 1e0 *1 # M越小越平滑，越大追越準
        ## 求G_telda
        Y = np.dot(np.dot(C, Pp), C.T) + R
        Y = Pm[2][2] + R
        G_tel = Km[2]**2 * Y
        ## 求G_hat
        delta_x = xm[2] - xp[2] # x - x-
        # print("delta_x =", delta_x)
        delta_x_values.append(delta_x)
        m = i + 1
        delta_x_values_sqr = [val**2 for val in delta_x_values[:m]]
        # G_hat = sum(delta_x_values_sqr[:m]) / m
        # G_hat = sum(delta_x_values_sqr[:i]) / (i+1)
        G_hat = sum(delta_x_values_sqr[m - N:m]) / N
        ## 求S
        # print("G_hat=", G_hat)
        # print("G_tel=", G_tel)
        G = G_hat/G_tel
        c = 1e-4
        # c = 1e6
        # c = 0.5
        c = 1
        S = np.maximum(c, G_hat/G_tel) 

        # S = G_hat/G_tel
        # print("G2=", G)
        # print("-----------------")
        # if S > c:
        #     count2 = count2 + 1

        ## 求Q_hat
        Q_hat = S * M *1e-2 #*1e3 #*1e-1/1e-2
        Q[2, 2] = Q_hat

        # if np.abs(u_a_values[i]) > np.abs(u_a_values[i-1]):
        #     Q[2, 2] = Q_hat * 0.01
        # else:    
        #     Q[2, 2] = Q_hat * 10
        Q_acc.append(Q[2, 2])
        Q_save.append(Q.flatten())

        # pose[i] = xm[0]
        # vele[i] = xm[1]
        # acce[i] = xm[2]
        pose.append(xm[0])
        vele.append(xm[1])  
        acce.append(xm[2])

##########################################################################################################
################                    估測的p、v、a covariances計算                     #####################
##########################################################################################################
#================================================ 1.OLS基本線性回歸方法 ================================================#
        # data = np.vstack((pose, vele, acce))
        data = np.concatenate((np.array(pose).reshape(1, -1), np.array(vele).reshape(1, -1), np.array(acce).reshape(1, -1)), axis=0)
        # data = np.concatenate((np.array(pose[m - N:m]).reshape(1, -1), np.array(vele[m - N:m]).reshape(1, -1), np.array(acce[m - N:m]).reshape(1, -1)), axis=0)
        # print("data =", data)
        # print("data.shape =", data.shape)
        cov_data = np.cov(data) 
        # print("cov_data =", cov_data)
        cov_true_data = cov_data + Pm
        # print("cov_true_data =", cov_true_data)
        cov_xtrue_xtrue = cov_true_data[0, 0] 
        cov_xtrue_vtrue = cov_true_data[0, 1] 
        cov_xtrue_atrue = cov_true_data[0, 2] 
        cov_vtrue_atrue = cov_true_data[1, 2]
        # # cov_xtrue_xtrue = np.cov(x)
        # print("cov_x_x =", cov_xtrue_xtrue)
        #-------------------------- LSR ----------------------------#
        T = len(pose)
        X = np.hstack([np.ones((T, 1)), np.array(pose).reshape(-1, 1)])  # 自變數 X（共用）
        # # ----------- Y: 速度與加速度堆疊成 [Y_vel, Y_acc]（兩個 T x 1）-----------
        Y_all = np.hstack([np.array(vele).reshape(-1, 1), np.array(acce).reshape(-1, 1)])  # shape (T, 2)
        # vel_pred_data_ols, acc_pred_data_ols, vel_pred_ols, acc_pred_ols = LSR.ols_regression(X, Y_all)

#================================================ 2. WLS加權最小平方法回歸 ================================================#
        # # 接下來做 WLS 回歸：由位置估計來回歸速度與加速度
        # # 形式：vel = b0 + b1 * pos ，用 P 中的 Var(pos) 當作權重
        # X = np.vstack([np.ones_like(xm[0]), xm[0]]).T  # 加上常數項
        # W_vel = 1 / Pm[0, 1]  # 假設位置誤差協方差 P[0,0]
        # W = np.eye(len(X)) * W_vel

        # # WLS 回歸：速度
        # beta_wls_vel = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ xm[1]

        # X = np.vstack([np.ones_like(xm[0]), xm[1]]).T  # 加上常數項
        # W_vel = 1 / Pm[0, 2]  # 假設位置誤差協方差 P[0,0]
        # W = np.eye(len(X)) * W_vel

        # # WLS 回歸：加速度
        # beta_wls_acc = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ xm[2]

        # # 回歸預測結果
        # vel_pred_wls_P = X @ beta_wls_vel
        # acc_pred_wls_P = X @ beta_wls_acc

#================================================ 3. GLS廣義最小平方法回歸 ================================================#
        #-------------------------- LSR ----------------------------#
        T = len(pose)
        # T = N

        X = np.hstack([np.ones((T, 1)), np.array(pose).reshape(-1, 1)])  # 自變數 X（共用）
        # # ----------- Y: 速度與加速度堆疊成 [Y_vel, Y_acc]（兩個 T x 1）-----------
        Y_all = np.hstack([np.array(vele).reshape(-1, 1), np.array(acce).reshape(-1, 1)])  # shape (T, 2)
        P_list.append(Pm.copy())
        Q_list.append(Q.copy())
        vel_pred_data_gls_P, acc_pred_data_gls_P, vel_pred_gls_P, acc_pred_gls_P = LSR.gls_regression(X[m - N:m], Y_all[m - N:m], P_list[m - N:m])
        vel_pred_data_gls_Q, acc_pred_data_gls_Q, vel_pred_gls_Q, acc_pred_gls_Q = LSR.gls_regression(X[m - N:m], Y_all[m - N:m], Q_list[m - N:m])

        # 儲存回歸的速度加速度資料
        vel_gls_data.append(vel_pred_gls_P)
        acc_gls_data.append(acc_pred_gls_P)
       
        # print("vel_pred =", vel_pred)
        # print("acc_pred =", acc_pred)
#=========================================================================================================================#

        vele_data.append(vele[i]) 
        # print("vele_data =", vele_data)
        # if i == 0
        #     v_mean = sum(vele_data[:i]) / 1
        # else:
        v_mean = sum(vele_data) / (i+1)
        # v_mean = sum(vele_data[n-N:n])/N
        print("v_mean =", v_mean)

        acce_data.append(acce[i]) 
        # if i == 0:
        #     a_mean = sum(acce_data[:i]) / 1
        # else:
        a_mean = sum(acce_data) / (i+1)
        # a_mean = sum(acce_data[n-N:n])/N
        print("a_mean =", a_mean)
    #------------------------------------------------資料儲存---------------------------------------------#
        x_data_all = np.concatenate((x_k_update_data, k_y_data, x_tel, x_true_data, x_true_data_noise, x_k_predict_data), axis=1)# me
        P_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)# me
    end_time = time.time()
    vel_gls_data = np.array(vel_gls_data).reshape(-1, 1)
    acc_gls_data = np.array(acc_gls_data).reshape(-1, 1)
    #------------------------------------------------RTS平滑器---------------------------------------------#
    x_RTS, P_RTS, K_RTS, Pp_RTS = RTSsmoother.rts_smoother(i, x_k_update_data, P_k_update_data, A, Q_save)
    # print("count0 =", count0)
    # print("count1 =", count1)
    # print("count2 =", count2)
    print("執行時間：", end_time - star_time)
    print("平均時間：", (end_time - star_time) / len(pose))
    print('-----------------------------------------------------------------')

    return pose, vele, acce, Q_pos, Q_acc, Q_vel, u_p_values, u_v_values, u_a_values, Q_save, x_data_all, P_data_all, vel1_values, vel2_values, acc1_values, acc2_values, x_RTS, vel_gls_data, acc_gls_data

