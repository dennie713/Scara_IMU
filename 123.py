import numpy as np
import matplotlib.pyplot as plt
import zero_phase_filter, AKF_modify, AKF_modify2, AKF, LAE

if __name__ == "__main__":
    dt1 = 0.01     
    start_size1 = 0 # 1700
    data_size1 = 890
    t1 = np.arange(0, (data_size1 - start_size1)*dt1, dt1)
    total_time1 = (data_size1 - start_size1) / 1000 #1.5

    # IMU實際收回二軸加速度資料
    path = 'data/IMU_data8.txt'
    data = np.genfromtxt(path, delimiter=' ')
    data = np.array(data)
    i = data[start_size1:data_size1, 0] 
    time = data[start_size1:data_size1, 1]
    gyro_x = data[start_size1:data_size1, 2] # rad/s
    gyro_y = data[start_size1:data_size1, 3] 
    gyro_z = data[start_size1:data_size1, 4]
    acc_x = data[start_size1:data_size1, 5] # G
    acc_y = data[start_size1:data_size1, 6] 
    acc_z = data[start_size1:data_size1, 7]  

    # IMU資料上新的時間軸
    time2 = time - time[0]
    # print('time2 =', time2)
#-------------------------------------------------------------------#
    dt2 = 0.001      
    start_size2 = 0 # 1700
    data_size2 = 500
    t2 = np.arange(0, (data_size2 - start_size2)*dt2, dt2)
    total_time2 = (data_size2 - start_size2) / 1000 #1.5

    # 馬達ENCODER收回位置資料
    path = 'data/output7_first_ax3.txt'
    data = np.genfromtxt(path, delimiter='\t')
    # data = np.array(data).reshape(-1, 12)
    # print('data =', data)
    # cmd
    poscmd_1 = data[start_size2:data_size2, 0] 
    poscmd_2 = data[start_size2:data_size2, 1] 
    velcmd_1 = data[start_size2:data_size2, 2] 
    velcmd_2 = data[start_size2:data_size2, 3] 
    acccmd_1 = data[start_size2:data_size2, 4] 
    acccmd_2 = data[start_size2:data_size2, 5] 
    # encoder
    pos_1 = data[start_size2:data_size2, 6] 
    pos_2 = data[start_size2:data_size2, 7] 
    vel_1 = data[start_size2:data_size2, 8] 
    vel_2 = data[start_size2:data_size2, 9] 
    tor_1 = data[start_size2:data_size2, 10] 
    tor_2 = data[start_size2:data_size2, 11] 
#-------------------------------------------------------------------#

    # 劃出IMU加速度資料綠波前後
    # 用zero_phase_filter濾波:單位G
    filtered_acc_x = zero_phase_filter.zero_phase_filter(3, 10, acc_x) # 50/12
    filtered_acc_y = zero_phase_filter.zero_phase_filter(3, 10, acc_y)
    # x方向
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.title("Acc_x", loc="center")
    plt.plot(time2, -acc_x, label="acc_x", linewidth=1)
    plt.plot(time2, -filtered_acc_x, label="filtered_acc_x", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("acc(G)")
    plt.subplot(2, 1, 2)
    plt.plot(time2, -filtered_acc_x, label="filtered_acc_x", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("acc(G)")
    # y方向
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.title("Acc_y", loc="center")
    plt.plot(time2, -acc_y, label="acc_y", linewidth=1)
    plt.plot(time2, -filtered_acc_y, label="filtered_acc_y", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("acc(G)")
    plt.subplot(2, 1, 2)
    plt.plot(time2, -filtered_acc_y, label="filtered_acc_y", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("acc(G)")

    # # 對IMU加速度資料積分得速度:單位G
    # filtered_vel_x = np.cumsum(filtered_acc_x) * dt1
    # filtered_vel_y = np.cumsum(filtered_acc_y) * dt1
    # # x方向
    # plt.figure(figsize=(8, 4))
    # plt.subplot(2, 1, 1)
    # plt.title("X_axis", loc="center")
    # plt.plot(time, -acc_x, label="acc_x", linewidth=1)
    # plt.plot(time, -filtered_acc_x, label="filtered_acc_x", linewidth=1)
    # plt.legend(loc='upper right')
    # plt.xlabel("t")
    # plt.ylabel("acc(G)")
    # plt.subplot(2, 1, 2)
    # plt.plot(time, -filtered_vel_x, label="filtered_vel_x", linewidth=1)
    # plt.legend(loc='upper right')
    # plt.xlabel("t")
    # plt.ylabel("vel(G)")
    # # y方向
    # plt.figure(figsize=(8, 4))
    # plt.subplot(2, 1, 1)
    # plt.title("Y_axis", loc="center")
    # plt.plot(time, -acc_y, label="acc_y", linewidth=1)
    # plt.plot(time, -filtered_acc_y, label="filtered_acc_y", linewidth=1)
    # plt.legend(loc='upper right')
    # plt.xlabel("t")
    # plt.ylabel("acc(G)")
    # plt.subplot(2, 1, 2)
    # plt.plot(time, filtered_vel_y, label="filtered_vel_y", linewidth=1)
    # plt.legend(loc='upper right')
    # plt.xlabel("t")
    # plt.ylabel("vel(G)")
#-------------------------------------------------------------------#

    # 將第一軸的end-effectore'關節空間中的速度'轉換成'卡式空間中的速度'
    l1 = 225 # mm
    l2 = 0 # mm
    # 加速度G轉換成rad/s^2
    # r = l1
    # end-effector的半徑位置
    # x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
    # y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
    # r = np.sqrt(x**2 + y**2)
    # r = 這裡的關節角度是用馬達ENCODER收回的資料嗎? 還是要用IMU收回的資料?但是IMU收回的資料是G，不是rad/s^2
    r = l1

    # filtered_acc_x = filtered_acc_x * 9.81 #/ (r / 1000) # m/s^2 to rad/s^2
    # filtered_acc_y = filtered_acc_y * 9.81 #/ (r / 1000)
    # print('filtered_acc_x =', filtered_acc_x)
    # print('filtered_acc_y =', filtered_acc_y)
    # 計算速度rad/s
    for i in range(len(filtered_acc_x)):
        if i == 0:
            filtered_vel_x = np.cumsum(filtered_acc_x) * (time[i] - 0)
            filtered_vel_y = np.cumsum(filtered_acc_y) * (time[i] - 0)
        else:
            filtered_vel_x = np.cumsum(filtered_acc_x) * (time[i] - time[i-1])
            filtered_vel_y = np.cumsum(filtered_acc_y) * (time[i] - time[i-1])

    # filtered_vel_x = np.cumsum(filtered_acc_x) * dt1
    # filtered_vel_y = np.cumsum(filtered_acc_y) * dt1

    q1_dot_data = []
    q2_dot_data = []

    for i in range(len(filtered_vel_x)):
        # q1, q2
        # q2 = np.arccos(((filtered_vel_x[i]**2 + filtered_vel_y[i]**2) - l1**2 - l2**2) / (2*l1*l2))
        q2=0
        # print('q2 =', q2)
        # alpha = l1 + l2*np.cos(q2)
        # beta = l2*np.sin(q2)
        # A = -beta * filtered_vel_x[i] + alpha * filtered_vel_y[i]
        # B = alpha * filtered_vel_x[i] + beta * filtered_vel_y[i]
        # print('A, B =',A, B)
        # # q1
        q1 = np.arctan2(filtered_vel_y[i], filtered_vel_x[i])
        # if (B >= 0):
        #     q1 = np.arctan(A/B)
        # elif (B < 0) and (A >= 0):
        #     q1 = np.arctan(A/B) + np.pi
        # else:
        #     q1 = np.arctan(A/B) - np.pi
        # print('q1 =', q1)
        # Jacobian matrix
        # J_q = np.array([[-l1*np.sin(q1) - l2*np.sin(q1+q2), - l2*np.sin(q1+q2)],
        #                 [-l1*np.cos(q1) + l2*np.cos(q1+q2), l2*np.cos(q1+q2)]])
        J_q = np.array([[-l1*np.sin(q1) , 0],
                        [-l1*np.cos(q1) , 0]])
        # print('J_q =', J_q)
        # print("inv J_q = ", np.linalg.pinv(J_q))
        # q_dot
        q_dot = np.dot(np.linalg.pinv(J_q), np.array([filtered_vel_x[i], filtered_vel_y[i]]))
        q1_dot = q_dot[0]
        q2_dot = q_dot[1]
        # print('q_dot =', q_dot)
        # print('q1_dot =', q1_dot)
        # print('q2_dot =', q2_dot)
        q1_dot_data.append(q1_dot)
        q2_dot_data.append(q2_dot)

    #======================================畫圖==========================================#
    # plt.figure(figsize=(8, 4))
    # plt.subplot(2, 1, 1)
    # plt.title("q1_dot & q2_dot", loc="center")
    # plt.plot(t1, q1_dot_data, label="q1_dot", linewidth=1)
    # plt.legend(loc='upper right')
    # plt.xlabel("t")
    # plt.ylabel("vel(rad/s)")
    # plt.subplot(2, 1, 2)
    # plt.plot(t1, q2_dot_data, label="q2_dot", linewidth=1)
    # plt.legend(loc='upper right')
    # plt.xlabel("t")
    # plt.ylabel("vel(rad/s)")

#=================================================馬達命令=================================================#
    # pos vel acc cmd
    plt.figure(figsize=(8, 6))
    plt.subplot(3, 1, 1)
    plt.plot(data[:, 0] , "black", label="Pos CMD", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("pos")
    plt.legend(loc='upper right')
    plt.title("Command data", loc="center")
    plt.subplot(3, 1, 2)
    plt.plot(data[:, 2], "blue", label="Vel CMD", linewidth=1)
    plt.xlabel("t")
    plt.ylabel("vel")
    plt.legend(loc='upper right')
    plt.subplot(3, 1, 3)
    plt.plot(data[:, 4], "red", label="Acc CMD", linewidth=1)  
    plt.xlabel("t")
    plt.ylabel("acc")
    plt.legend(loc='upper right')
    plt.tight_layout()

#=================================================馬達速度估測=================================================#
    # clip = 6000
    # pos_1_clip = pos_1[:clip]
    pose, vele, acce, Q_pos, Q_acc, Q_vel, u_p_values, u_v_values, u_a_values, Q_save, x_data_all, P_data_all, vel1_values, vel2_values, acc1_values, acc2_values, x_RTS = AKF_modify.AKF(dt2, pos_1)
    # pose, vele, acce, Q_pos, Q_acc, Q_vel, u_p_values, u_v_values, u_a_values, Q_save, x_data_all, P_data_all, vel1_values, vel2_values, acc1_values, acc2_values, x_RTS = AKF.AKF(dt2, pos_1)
    acc_LAE = LAE.LAE(dt2, pos_1)
    # IMU收到的角速度及角加速度
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.title("IMU Data - Vel & Acc", loc="center")
    plt.plot(time2, -gyro_x, label="gyro_x", linewidth=1)
    plt.plot(time2, -gyro_y, label="gyro_y", linewidth=1)
    plt.plot(time2, -gyro_z, label="gyro_z", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("vel(rad/s)")
    plt.subplot(2, 1, 2)
    # plt.title("IMU Data Acc", loc="center")
    plt.plot(time2, -filtered_acc_x/(l1/1000), label="acc_x", linewidth=1)
    plt.plot(time2, -filtered_acc_y/(l1/1000), label="acc_y", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("acc(rad/s^2)")

    # 先把IMU的資料對齊到馬達的命令
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.title("Cmd Data & IMU Data", loc="center")
    # plt.plot(t2, vel_1, label="vel", linewidth=1)
    plt.plot(time2[315:773]-time2[315], -gyro_z[315:773], label="IMU vel", linewidth=1)
    plt.plot(t2, velcmd_1, label="velcmd", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("vel(rad/s)")
    plt.subplot(2, 1, 2)
    plt.plot(time2[315:773]-time2[315], (-acc_y/(l1/1000))[315:773], label="IMU acc", linewidth=1)
    # plt.plot(time2[299:]-time2[299], (-filtered_acc_y/(l1/1000))[299:], label="IMU acc", linewidth=1)
    plt.plot(t2, acccmd_1, label="acccmd", linewidth=1) 
    plt.plot(t2, acc_LAE, label="acc_LAE", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("acc(rad/s^2)")

    # 馬達的速度估測
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.title("AKF Est Data", loc="center")
    plt.plot(t2[200:], vele[200:], label="vel", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("vel(rad/s)")
    plt.subplot(2, 1, 2)
    plt.plot(t2[200:], acce[200:], label="acc", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("acc(rad/s^2)")

    # IMU與馬達資料比較
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.title("Scara Encoder & IMU Data ", loc="center")
    # plt.plot(t2, vel_1, label="vel", linewidth=1)
    plt.plot(time2[315:773]-time2[315], -gyro_z[315:773], label="IMU vel", linewidth=1)
    # plt.plot(t2, velcmd_1, label="velcmd", linewidth=1)
    plt.plot(t2, vele, label="vel", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("vel(rad/s)")
    plt.subplot(2, 1, 2)
    plt.plot(time2[315:773]-time2[315], (-acc_y/(l1/1000))[315:773], label="IMU acc", linewidth=1)
    # plt.plot(time2[299:]-time2[299], (-filtered_acc_y/(l1/1000))[299:], label="IMU acc", linewidth=1)
    # plt.plot(t2, acccmd_1, label="acccmd", linewidth=1) 
    plt.plot(t2, vele, label="vel", linewidth=1)
    plt.plot(t2, acc_LAE, label="acc_LAE", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("acc(rad/s^2)")

    # IMU與馬達資料比較-取穩定後數據比較
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.title("Scara Encoder & IMU Data after stable", loc="center")
    # plt.plot(t2, vel_1, label="vel", linewidth=1)
    plt.plot(time2[315:773]-time2[315], -gyro_z[315:773], label="IMU vel", linewidth=1)
    # plt.plot(t2, velcmd_1, label="velcmd", linewidth=1)
    plt.plot(t2[200:], vele[200:], label="vel", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("vel(rad/s)")
    plt.subplot(2, 1, 2)
    plt.plot(time2[315:773]-time2[315], (-acc_y/(l1/1000))[315:773], label="IMU acc", linewidth=1)
    # plt.plot(time2[299:]-time2[299], (-filtered_acc_y/(l1/1000))[299:], label="IMU acc", linewidth=1)
    # plt.plot(t2, acccmd_1, label="acccmd", linewidth=1) 
    plt.plot(t2[200:], vele[200:], label="vel", linewidth=1)
    plt.plot(t2[200:], acc_LAE[200:], label="acc_LAE", linewidth=1)
    plt.legend(loc='upper right')
    plt.xlabel("t")
    plt.ylabel("acc(rad/s^2)")
    
    plt.show()



