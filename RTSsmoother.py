import numpy as np

# 參考網址:https://blog.csdn.net/qq_38410730/article/details/131236286
# github:https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/14-Adaptive-Filtering.ipynb
def rts_smoother(i, Xs_data, Ps_data, A, Q_save):
    # print("Xs.shape =", Xs_data.shape)
    # n, dim_x, _ = Xs.shape

    # smoother gain
    K = np.zeros((3, 1))
    # x, P, Pp = Xs_data.copy(), Ps_data.copy(), Ps_data.copy
    x = Xs_data
    # print("Xs_data =", Xs_data[-1])
    P = Ps_data
    Q_save = Q_save
    
    x_1 = 0
    P_1 = 0

    x_1_data = []
    P_1_data = []
    K_data = []
    Pp_data = []
    Q = np.array([[9.238203052156316038e-08, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 1.570573423589525795e-05, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 4.756964538875590315e-02]])

    for k in range(len(Xs_data)-2,-1,-1):

        Pp = A @ P[k].reshape(3, 3) @ A.T + Q_save[k].reshape(3, 3) # predicted covariance
        # Pp = A @ P[k].reshape(3, 3) @ A.T + Q # predicted covariance
        K = P[k].reshape(3, 3) @ A.T @ np.linalg.inv(Pp)
        x_1 = x[k].reshape(3, 1) + K @ (x[k+1].reshape(3, 1) - (A @ x[k].reshape(3, 1)))  
        x[k] = x_1 
        P_1 = P_1 + K @ (P[k+1].reshape(3, 3) - Pp) @ K.T
        # P_1 = P[k].reshape(3, 3) + K @ (P[k+1].reshape(3, 3) - Pp) @ K.T
        P[k] = P_1
        
        x_1_data.append(x_1.flatten())
        P_1_data.append(P_1.flatten())
        K_data.append(K.flatten())
        Pp_data.append(Pp.flatten())


#------------------------------僅一筆-----------------------------------#
    # K = np.zeros((3, 1))
    # x_1 = Xs_data[i-1].reshape(3, 1)
    # x = Xs_data[i].reshape(3, 1)
    # P_1 = Ps_data[i-1].reshape(3, 3)
    # P = Ps_data[i].reshape(3, 3)

    # Pp = A @ P @ A.T + Q # predicted covariance
    # K  = P @ A.T @ np.linalg.inv(Pp)
    # x += K @ (x_1 - (A @ x))     
    # P += K @ (P_1 - Pp) @ K.T
    
    return x_1_data, P_1_data, K_data, Pp_data
