import numpy as np
import matplotlib.pyplot as plt
import time
import RTSsmoother
from scipy.linalg import qr
from scipy.linalg import qr, cholesky


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
    Q = np.array([[1e-2, 0, 0],
                  [0, 1e-2, 0],
                  [0, 0, 1e-2]])
    Q = np.array([[1e-1, 0, 0],
                  [0, 1e-1, 0],
                  [0, 0, 1e-1]])
    # Q = np.array([[1e-6, 0, 0],
    #               [0, 1e-6, 0],
    #               [0, 0, 1e-6]])
    # Q = np.array([[1e-8, 0, 0],
    #               [0, 1e-8, 0],
    #               [0, 0, 1e-8]])
############################# R #############################
    R = 0.001
############################# P #############################
    # P = np.array([[1e-4, 1e-4, 1e-4],
    #                 [1e-4, 1e-4, 1e-4],
    #                 [1e-4, 1e-4, 1e-4]])
    P = np.array([[1e-2, 1e-2, 1e-2],
                    [1e-2, 1e-2, 1e-2],
                    [1e-2, 1e-2, 1e-2]])
    # P = np.array([[1e-5, 1e-5, 1e-5],
    #                 [1e-5, 1e-5, 1e-5],
    #                 [1e-5, 1e-5, 1e-5]])
    # P = np.array([[1e-6, 1e-6, 1e-6],
    #                 [1e-6, 1e-6, 1e-6],
    #                 [1e-6, 1e-6, 1e-6]])
    # P = np.array([[1e-8, 1e-8, 1e-8],
    #                 [1e-8, 1e-8, 1e-8],
    #                 [1e-8, 1e-8, 1e-8]])
    P = np.array([[1e-10, 1e-10, 1e-10],
                    [1e-10, 1e-10, 1e-10],
                    [1e-10, 1e-10, 1e-10]])

    Wt = 0
    pose = np.zeros(len(Pos))
    vele = np.zeros(len(Pos))
    acce = np.zeros(len(Pos))
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

    count0 = 0
    count1 = 0
    count2 = 0

    v1_values = []
    v2_values = []
    a1_values = []
    a2_values = []

    vel1_values = []
    vel2_values = []
    acc1_values = []
    acc2_values = []

    vele_data = []
    acce_data = []
    Q_old1 = []
    Q_old2 = []
    v_mean = 0
    a_mean = 0
    y_save = []
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
    for i in range(len(Pos)): # m = measurement;p = predict len(pos)
        #---------------------------KF---------------------------#
        Pp = np.dot(np.dot(A, Pm), A.T) + Q
        # print("Q =", Q)
        # print("Pp =", Pp)
        xp = np.dot(A, xm) + Wt
        Km = np.dot(Pp, C.T) / (np.dot(np.dot(C, Pp), C.T) + R)
        # print("Km =", Km)
        dk = (Pos[i] - np.dot(C, xp)) # dk = zk - hk*xk
        y = (Pos[i] - np.dot(C, xp))
        k_y = Km @ y
        xm = xp + np.dot(Km, y) # =x_hat
        ek = (Pos[i] - np.dot(C, xm)) # ek = zk - hk*xk+1
        Pm = np.dot((np.eye(3) - np.dot(Km, C)), Pp)
        KCP = np.dot((np.dot(Km, C)), Pp)
        # #---------------------------EKF---------------------------#
        # Pp = np.dot(np.dot(F(xm, 0), Pm), F(xm, 0).T) + Q  # 預測協方差
        # print("Q =", Q)
        # print("Pp =", Pp)
        # xp = f(xm, 0)  # 預測狀態
        # Km = np.dot(Pp, H(xm).T) / (np.dot(np.dot(H(xm), Pp), H(xm).T) + R)  # 卡爾曼增益
        # print("Km =", Km)
        # y = (Pos[i] - h(xp))  # 創新
        # k_y = Km @ y
        # xm = xp + np.dot(Km, y)
        # ek = (Pos[i] - np.dot(C, xm)) # ek = zk - hk*xk+1
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
        # del_Q = (np.dot(Km, y)) @ (np.dot(Km, y).T)
        y_save.append(y)
        y_2 = [val**2 for val in y_save[:i]]
        Ck = sum(y_2[:i])/(i+1)
        del_Q = Ck * np.dot(Km, Km.T)

#--------------------------------------------R值自適應--------------------------------------------#
        # alpha = 0.2
        # # R = alpha * R + (1 - alpha) * (ek * ek + np.dot(np.dot(C, Pp), C.T))
        # R = alpha * R + (1 - alpha) * (Ck + np.dot(np.dot(C, Pp), C.T))

##########################################################################################################
################                               位置求Q                               #####################
##########################################################################################################
        ## 求M
        u_p = (Pos[i]- xm[0]) # 實際值-卡爾曼估測值
        # print("u_p=", u_p)
        u_p_values.append(u_p)
        n = i + 1
        N = 10
        u_sqr = [val**2 for val in u_p_values[:n]]
        # M = sum(u_sqr[:n])/n
        # M = sum(u_sqr[:i])/(i+1)
        M = sum(u_sqr[n-N:n])/N
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
        # print("G_hat0=", G_hat)
        # print("G_tel0=", G_tel)
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
        # u_p = (Pos[i] - xm[0]) # 實際值-卡爾曼估測值
        if i == 0:
            u_v = (1/1)* Pos[i] - xm[1] + 1e-3
            # u_v = 1 
            # u_v = (Km[1] / Km[0]) * Pos[i]
        else:
        # if i >= 0:
            v1 = (Km[1] / Km[0])
            if v1 == 0 or v1 == float('inf'):
                v1 = 1
            v1_values.append(v1)
            s = 10
            # M = sum(u_sqr[:n])/n
            v1 = sum(v1_values[:i])/(i+1)
            # v1 = sum(v1_values[n-N:n])/N
            x_mean = sum(Pos[:i])/(i+1)
            # v_mean = sum(xm[:i])/i
            u_v1 = v1 * (Pos[i] - x_mean) + v_mean
            # u_v1 = v1 * u_p
            # u_v1 = (del_Q[0][1] / del_Q[0][0]) * (Pos[i] - x_mean) + v_mean
            # u_v1 = v1 * Pos[i]
            # u_v1 = v1 * (Pos[i] - x_mean)**2 + v1 * (Pos[i] - x_mean) + v_mean
            # print("(del_Q[0][1] / del_Q[0][0])*(Pos[i] - x_mean) + v_mean =", (del_Q[0][1] / del_Q[0][0])*(Pos[i] - x_mean) + v_mean)
            # print("u_v1 =", u_v1)
            v2 = (Pm[0][1] / Pm[0][0])
            if v2 == 0 or v2 == float('inf'):
                v2 = 1
            v2_values.append(v2)
            # M = sum(u_sqr[:n])/n
            v2 = sum(v2_values[:i])/(i+1)
            # v2 = sum(v2_values[n-N:n])/N
            u_v2 = v2 * (Pos[i] - x_mean) + v_mean
            # u_v2 = v2 * u_p
            # u_v2 = v2 * (Pos[i] - x_mean)**2 + v2 * (Pos[i] - x_mean) + v_mean
            # print("u_v2 =", u_v2)
            alpha =1 # 越小越平滑
            # alpha = v1 / (v1 + v2)
            u_v = (1-alpha) * u_v1 + alpha * u_v2 - xm[1]
            # u_v = (1-alpha) * u_v1 + alpha * u_v2 
            # u_v = u_v2
            # print("u_v =", u_v)

        ##-----------------------------儲存模擬速度----------------------------------#
        ## 用Q求速度
            vel1 = u_v1
            vel1_values.append(vel1)
        ## 用P求速度
            vel2= u_v2
            vel2_values.append(vel2)
        ##--------------------------------------------------------------------------#
        u_v_values.append(u_v)
        n = i + 1
        # N = 5
        u_sqr = [val**2 for val in u_v_values[:n]]
        # u_sqr = u_sqr * 1e3
        # M = sum(u_sqr[:n])/n
        # M = sum(u_sqr[:i])/(i+1)
        M = sum(u_sqr[n-N:n])/N
        M = M * 1e0*1
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
        b = 1e3
        b = 1
        S = np.maximum(b, G_hat/G_tel)
        # S = G_hat/G_tel
        # print("G_hat1=", G_hat)
        # print("G_tel1=", G_tel)
        # print("G1=", G)
        # if S > b:
        #     count1 = count1 + 1
        ## 求Q_hat
        Q_hat = S * M
        Q[1, 1] = Q_hat
        
        if np.abs(u_v_values[i]) > np.abs(u_v_values[i-1]):
            Q[1, 1] = Q_vel[-1]
        # if np.abs(u_v_values[i]) < 0:
        #     Q[1, 1] = Q_vel[-1]
        # else:    
        #     Q[1, 1] = Q_old1[-1]
        Q_vel.append(Q[1, 1])

        # print("--------------------------------------------------")

##########################################################################################################
################                              加速度求Q                               #####################
##########################################################################################################
        ## 求M
        # u_p = (Pos[i] - xm[0]) # y(k)-x_hat(k)  實際值-卡爾曼估測值
        if i == 0:
            u_a = (1/1)* Pos[i] - xm[2] + 1e-3
            # u_a = 1
            # u_a = (Km[2] / Km[0]) * Pos[i]
        else:
        # if i >= 0:
            a1 = (Km[2] / Km[0])
            if a1 == 0 or a1 == float('inf'):
                a1 = 1
            a1_values.append(a1)
            # N = 5
            # s = 10
            # M = sum(u_sqr[:n])/n
            a1 = sum(a1_values[:i])/(i+1)
            # a1 = sum(a1_values[n-N:n])/N
            u_a1 = a1 * (Pos[i] - x_mean) + a_mean
            # u_a1 = a1 * u_p
            # u_a1 = (del_Q[0][2] / del_Q[0][0]) * (Pos[i] - x_mean) + a_mean
            # u_a1 = a1 * Pos[i]
            # u_a1 = a1 * (Pos[i] - x_mean)**2 + a1 * (Pos[i] - x_mean) + a_mean
            # print("u_a1 =", u_a1)
            a2 = (Pm[0][2] / Pm[0][0])
            if a2 == 0 or a2 == float('inf'):
                a2 = 1
            a2_values.append(a2)
            # N = 10
            M = sum(u_sqr[:n])/n
            a2 = sum(a2_values[:i])/(i+1)
            # a2 = sum(a2_values[n-N:n])/N
            u_a2 = a2 * (Pos[i] - x_mean) + a_mean
            # u_a2 = a2 * u_p
            # u_a2 = a2 * (Pos[i] - x_mean)**2 + a2 * (Pos[i] - x_mean) + a_mean
            # print("u_a2 =", u_a2)   
            alpha = 1 #越小越平滑 # u_v:0 ，u_a:0.1
            # alpha = a1 / (a1 + a2)
            u_a = (1-alpha) * u_a1 + alpha * u_a2 - xm[2]
            # u_a = (1-alpha) * u_a1 + alpha * u_a2 
            # u_a = u_a2
            # print("u_a =", u_a)
        
        ##-----------------------------儲存模擬速度----------------------------------#
        ## 用Q求加加速度
            acc1 = u_a1
            acc1_values.append(acc1)
        ## 用P求加速度
            acc2= u_a2
            acc2_values.append(acc2)
        ##--------------------------------------------------------------------------#
        u_a_values.append(u_a)
        n = i + 1
        # N = 5
        u_sqr = [val**2 for val in u_a_values[:n]]
        # M = sum(u_sqr[:n])/n
        # M = sum(u_sqr[:i])/(i+1)
        M = sum(u_sqr[n-N:n])/N
        M = M * 1e0 *4 # M越小越平滑，越大追越準
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
        # print("G_hat2=", G_hat)
        # print("G_tel2=", G_tel)
        G = G_hat/G_tel
        c = 1e-4
        c = 1e6
        c = 0.5
        c = 1
        S = np.maximum(c, G_hat/G_tel)
        
        # S = G_hat/G_tel
        # print("G2=", G)
        # print("-----------------")
        # if S > c:
        #     count2 = count2 + 1

        ## 求Q_hat
        Q_hat = S * M 
        Q[2, 2] = Q_hat

        # if np.abs(u_a_values[i]) > np.abs(u_a_values[i-1]):
        #     Q[2, 2] = Q_acc[-1]
        # else:    
        #     Q[2, 2] = Q_old2p[-1]
        Q_acc.append(Q[2, 2])

        # if Q[0, 0] < 1e-6:  # 設定 Q 的最小值
        #     Q[0, 0] = 1e-6
        # if Q[1, 1] < 1e-11:  # 設定 Q 的最小值
        #     Q[1, 1] = 1e-9
        # if Q[2, 2] < 1e-8:  # 設定 Q 的最小值
        #     Q[2, 2] = 1e-6
        Q_save.append(Q.flatten())

        pose[i] = xm[0]
        vele[i] = xm[1]
        acce[i] = xm[2]


        # print(i)
        # s = 5
        vele_data.append(vele[i]) 
        # print("vele_data =", vele_data)
        # if i == 0
        #     v_mean = sum(vele_data[:i]) / 1
        # else:
        v_mean = sum(vele_data[:i]) / (i+1)

        acce_data.append(acce[i]) 
        # if i == 0:
        #     a_mean = sum(acce_data[:i]) / 1
        # else:
        a_mean = sum(acce_data[:i]) / (i+1)
        # print("x_k_update_data =", x_k_update_data[-1])
    #------------------------------------------------資料儲存---------------------------------------------#
        x_data_all = np.concatenate((x_k_update_data, k_y_data, x_tel, x_true_data, x_true_data_noise, x_k_predict_data), axis=1)# me
        P_data_all = np.concatenate((P_k_update_data, KCP_data), axis=1)# me

        # #------------------------------------------------RTS平滑器---------------------------------------------#
        # if i >= 3:
        #     x_k_update_data_RTS = x_k_update_data[i-3:i]
        #     P_k_update_data_RTS = P_k_update_data[i-3:i]
        #     Q_save_RTS = Q_save[i-3:i]
        #     xm, Pm, Km_RTS, Pp_RTS = RTSsmoother.rts_smoother(i, x_k_update_data_RTS, P_k_update_data_RTS, A, Q_save_RTS)
        #     print("xm =", xm[1])
        #     print("Pm =", Pm[1])
        #     print("Km =", Km)
        #     xm = xm[1].reshape(3, 1)
        #     Pm = np.array(Pm)[1].reshape(3, 3)
        #     # Km = np.array(Km)[-1].reshape(3, 1)
    end_time = time.time()
    #------------------------------------------------RTS平滑器---------------------------------------------#
    x_RTS, P_RTS, K_RTS, Pp_RTS = RTSsmoother.rts_smoother(i, x_k_update_data, P_k_update_data, A, Q_save)
    # print("count0 =", count0)
    # print("count1 =", count1)
    # print("count2 =", count2)
    print("執行時間：", end_time - star_time)
    print("平均時間：", (end_time - star_time) / len(Pos))

    return pose, vele, acce, Q_pos, Q_acc, Q_vel, u_p_values, u_v_values, u_a_values, Q_save, x_data_all, P_data_all, vel1_values, vel2_values, acc1_values, acc2_values, x_RTS

