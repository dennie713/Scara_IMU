import numpy as np
import matplotlib.pyplot as plt

# print("===================================== Calculating Error =====================================")
def RMSE(y, y_true):
    """Calculate the Root Mean Square Error between true and predicted values."""
    return np.sqrt(np.mean((y - y_true) ** 2))

def MAE(y, y_true):
    """Calculate the Mean Absolute Error between true and predicted values."""
    return np.mean(np.abs(y - y_true))

def calError(true_vel, true_acc, vele, acce, LSF_vele, LAE_acce, CFD_vele, CFD_acce, LSF28_vele, LSF28_acce):
    
    # 計算RMSE -> RMSE_pos = np.sqrt(np.mean((pose - true_pos)**2))
    vele = np.array(vele).flatten()
    acce = np.array(acce).flatten()
    # ================================= AKF ================================
    AKF_RMSE_vele = RMSE(vele, true_vel)
    AKF_RMSE_acce = RMSE(acce, true_acc)
    # ================================= LSF ================================
    LSF_RMSE_vele = RMSE(LSF_vele, true_vel)
    # ================================= LSF28 ================================
    LSF28_RMSE_vele = RMSE(LSF28_vele, true_vel)
    LSF28_RMSE_acce = RMSE(LSF28_acce, true_acc)
    # ================================= LAE ================================
    LAE_RMSE_acce = RMSE(LAE_acce, true_acc)
    # ================================= CFD ================================
    CFD_RMSE_vele = RMSE(CFD_vele, true_vel)
    CFD_RMSE_acce = RMSE(CFD_acce, true_acc)

    # AKF_RMSE_vele = AKF_RMSE_vele.reshape(-1, 1)
    # AKF_RMSE_acce = AKF_RMSE_acce.reshape(-1, 1)

    # print results
    print("--------------- Error Result ---------------")
    print("------------------- RMSE -------------------")
    print("AKF RMSE vel :", AKF_RMSE_vele)
    print("AKF RMSE acc :", AKF_RMSE_acce)
    print("LSF14 RMSE vel :", LSF_RMSE_vele)
    print("LSF28 RMSE vel :", LSF28_RMSE_vele)
    print("LSF28 RMSE acc :", LSF28_RMSE_acce)
    print("LAE RMSE acc :", LAE_RMSE_acce)
    print("--------------------------------------------")

    # 計算MAE -> MAE_pos = np.mean(np.abs(pose - true_pos))
    # ================================= AKF ================================
    AKF_MAE_vele = MAE(vele, true_vel)
    AKF_MAE_acce = MAE(acce, true_acc)
    # AKF_MAE_vele = AKF_MAE_vele.reshape(-1, 1)
    # AKF_MAE_acce = AKF_MAE_acce.reshape(-1, 1)
    # ================================= LSF ================================
    LSF_MAE_vele = MAE(LSF_vele, true_vel)
    # ================================= LSF28 ================================
    LSF28_MAE_vele = MAE(LSF28_vele, true_vel)
    LSF28_MAE_acce = MAE(LSF28_acce, true_acc)
    # ================================= LAE ================================
    LAE_MAE_acce = MAE(LAE_acce, true_acc)
    # ================================= CFD ================================
    CFD_MAE_vele = MAE(CFD_vele, true_vel)
    CFD_MAE_acce = MAE(CFD_acce, true_acc)
    # print results
    print("------------------- MAE -------------------")
    print("AKF MAE vel :", AKF_MAE_vele)
    print("AKF MAE acc :", AKF_MAE_acce)
    print("LSF14 MAE vel :", LSF_MAE_vele)
    print("LSF28 MAE vel :", LSF28_MAE_vele)
    print("LSF28 MAE acc :", LSF28_MAE_acce)
    print("LAE MAE acc :", LAE_MAE_acce)
    print("CFD RMSE vel :", CFD_MAE_vele)
    print("CFD RMSE acc :", CFD_MAE_acce)
    print("--------------------------------------------")

    # plot RMSE
    plt.figure(figsize=(10, 7))
    plt.suptitle("RMSE and MAE Result", fontsize=20, fontweight='bold')
    plt.subplot(2, 2, 1)
    plt.title("RMSE Result vel")
    x = ['AKF vel', 'LSF14 vel', 'LSF28 vel']
    h = [AKF_RMSE_vele, LSF_RMSE_vele, LSF28_RMSE_vele]
    c = ['skyblue', 'orange', 'green']
    bars = plt.bar(x, h, color=c, width=0.1)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.xlabel('Method')
    plt.ylabel('rad/s')
    # plt.legend()

    plt.subplot(2, 2, 2)
    x = np.arange(2)
    width = 0.4
    plt.title("RMSE Result acc")
    x = ['AKF acc', 'LAE acc', 'LSF28 acc']
    h = [AKF_RMSE_acce, LAE_RMSE_acce, LSF28_RMSE_acce]
    c = ['skyblue', 'orange', 'green']
    bars = plt.bar(x, h, color=c, width=0.1)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.xlabel('Method')
    plt.ylabel('rad/s^2')
    # plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("MAE Result vel")
    x = ['AKF vel', 'LSF14 vel', 'LSF28 vel']
    h = [AKF_MAE_vele, LSF_MAE_vele, LSF28_MAE_vele]
    c = ['skyblue', 'orange', 'green']
    bars = plt.bar(x, h, color=c, width=0.1)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.xlabel('Method')
    plt.ylabel('rad/s')
    # plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.title("MAE Result acc")
    x = ['AKF acc', 'LAE acc', 'LSF28 acc']
    h = [AKF_MAE_acce, LAE_MAE_acce, LSF28_MAE_acce]
    c = ['skyblue', 'orange', 'green']
    bars = plt.bar(x, h, color=c, width=0.1)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.xlabel('Method')
    plt.ylabel('rad/s^2')
    # plt.legend()
    plt.tight_layout()

# ===================================== Calculating Error -after stabilization =====================================
def calError2(true_vel, true_acc, vele, acce, LSF_vele, LAE_acce, CFD_vele, CFD_acce, LSF28_vele, LSF28_acce):
    
    # 計算RMSE -> RMSE_pos = np.sqrt(np.mean((pose - true_pos)**2))
    vele = np.array(vele).flatten()
    acce = np.array(acce).flatten()
    # ================================= AKF ================================
    AKF_RMSE_vele = RMSE(vele, true_vel)
    AKF_RMSE_acce = RMSE(acce, true_acc)
    # ================================= LSF ================================
    LSF_RMSE_vele = RMSE(LSF_vele, true_vel)
    # ================================= LSF28 ================================
    LSF28_RMSE_vele = RMSE(LSF28_vele, true_vel)
    LSF28_RMSE_acce = RMSE(LSF28_acce, true_vel)
    # ================================= LAE ================================
    LAE_RMSE_acce = RMSE(LAE_acce, true_acc)
    # ================================= CFD ================================
    CFD_RMSE_vele = RMSE(CFD_vele, true_vel)
    CFD_RMSE_acce = RMSE(CFD_acce, true_acc)

    # AKF_RMSE_vele = AKF_RMSE_vele.reshape(-1, 1)
    # AKF_RMSE_acce = AKF_RMSE_acce.reshape(-1, 1)

    # print results
    print("--------------- Error Result -after stabilization ---------------")
    print("------------------- RMSE -------------------")
    print("AKF RMSE vel :", AKF_RMSE_vele)
    print("AKF RMSE acc :", AKF_RMSE_acce)
    print("LSF14 RMSE vel :", LSF_RMSE_vele)
    print("LSF28 RMSE vel :", LSF28_RMSE_vele)
    print("LSF28 RMSE acc :", LSF28_RMSE_acce)
    print("LAE RMSE acc :", LAE_RMSE_acce)
    print("--------------------------------------------")

    # 計算MAE -> MAE_pos = np.mean(np.abs(pose - true_pos))
    # ================================= AKF ================================
    AKF_MAE_vele = MAE(vele, true_vel)
    AKF_MAE_acce = MAE(acce, true_acc)
    # AKF_MAE_vele = AKF_MAE_vele.reshape(-1, 1)
    # AKF_MAE_acce = AKF_MAE_acce.reshape(-1, 1)
    # ================================= LSF ================================
    LSF_MAE_vele = MAE(LSF_vele, true_vel)
    # ================================= LSF28 ================================
    LSF28_MAE_vele = MAE(LSF28_vele, true_vel)
    LSF28_MAE_acce = MAE(LSF28_acce, true_vel)
    # ================================= LAE ================================
    LAE_MAE_acce = MAE(LAE_acce, true_acc)
    # ================================= CFD ================================
    CFD_MAE_vele = MAE(CFD_vele, true_vel)
    CFD_MAE_acce = MAE(CFD_acce, true_acc)
    # print results
    print("------------------- MAE -------------------")
    print("AKF MAE vel :", AKF_MAE_vele)
    print("AKF MAE acc :", AKF_MAE_acce)
    print("LSF14 MAE vel :", LSF_MAE_vele)
    print("LSF28 MAE vel :", LSF28_MAE_vele)
    print("LSF28 MAE acc :", LSF28_MAE_acce)
    print("LAE MAE acc :", LAE_MAE_acce)
    print("CFD RMSE vel :", CFD_MAE_vele)
    print("CFD RMSE acc :", CFD_MAE_acce)
    print("--------------------------------------------")

    # plot RMSE
    plt.figure(figsize=(10, 7))
    plt.suptitle("RMSE and MAE Result -after stabilization", fontsize=20, fontweight='bold')
    plt.subplot(2, 2, 1)
    plt.title("RMSE Result vel")
    x = ['AKF vel', 'LSF14 vel', 'LSF28 vel']
    h = [AKF_RMSE_vele, LSF_RMSE_vele, LSF28_RMSE_vele]
    c = ['skyblue', 'orange', 'green']
    bars = plt.bar(x, h, color=c, width=0.1)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.xlabel('Method')
    plt.ylabel('rad/s')
    # plt.legend()

    plt.subplot(2, 2, 2)
    x = np.arange(2)
    width = 0.4
    plt.title("RMSE Result acc")
    x = ['AKF acc', 'LAE acc', 'LSF28 acc']
    h = [AKF_RMSE_acce, LAE_RMSE_acce, LSF28_RMSE_acce]
    c = ['skyblue', 'orange', 'green']
    bars = plt.bar(x, h, color=c, width=0.1)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.xlabel('Method')
    plt.ylabel('rad/s^2')
    # plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("MAE Result vel")
    x = ['AKF vel', 'LSF14 vel', 'LSF28 vel']
    h = [AKF_MAE_vele, LSF_MAE_vele, LSF28_MAE_vele]
    c = ['skyblue', 'orange', 'green']
    bars = plt.bar(x, h, color=c, width=0.1)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.xlabel('Method')
    plt.ylabel('rad/s')
    # plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.title("MAE Result acc")
    x = ['AKF acc', 'LAE acc', 'LSF28 acc']
    h = [AKF_MAE_acce, LAE_MAE_acce, LSF28_MAE_acce]
    c = ['skyblue', 'orange', 'green']
    bars = plt.bar(x, h, color=c, width=0.1)
    for bar in bars:
        yval = bar.get_height()  # 取得 bar 的高度（數值）
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.xticks(x, x)
    plt.xlabel('Method')
    plt.ylabel('rad/s^2')
    # plt.legend()
    plt.tight_layout()
