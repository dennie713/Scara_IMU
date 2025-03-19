import numpy as np
from scipy.linalg import block_diag

# 初期回歸，使用 OLS（無卡爾曼濾波器的 P、Q）
def ols_regression(X, Y_all):
    # beta = np.linalg.lstsq(X, Y_all, rcond=None)[0]

    beta_ols_vel = np.linalg.lstsq(X, Y_all[:, [0]])[0]  # 速度回歸
    beta_ols_acc = np.linalg.lstsq(X, Y_all[:, [1]])[0]  # 加速度回歸

    vel_pred_data = X @ beta_ols_vel
    acc_pred_data = X @ beta_ols_acc

    
    vel_pred = vel_pred_data[-1]
    acc_pred = acc_pred_data[-1]

    return vel_pred_data, acc_pred_data, vel_pred, acc_pred

# 假設卡爾曼濾波器已經運行一段時間並且 P 收斂了
# 我們使用 GLS 進行回歸，並考慮 P 和 Q
def gls_regression(X, Y_all, P_list):
    # 建立共變異矩陣（包含 P 和 Q）
    Sigma_vel = block_diag(*[np.array([[Pm[1, 1]]]) for Pm in P_list])  # 只使用速度的共變異
    Sigma_acc = block_diag(*[np.array([[Pm[2, 2]]]) for Pm in P_list])  # 只使用加速度的共變異

    # 計算Sigma_inv
    Sigma_inv_vel = np.linalg.pinv(Sigma_vel)
    Sigma_inv_acc = np.linalg.pinv(Sigma_acc)

    lambda_reg = 1e-6
    # GLS 解速度
    beta_gls_vel = np.linalg.solve(X.T @ Sigma_inv_vel @ X + lambda_reg * np.eye(X.shape[1]), X.T @ Sigma_inv_vel @ Y_all[:, [0]])
    # GLS 解加速度
    beta_gls_acc = np.linalg.solve(X.T @ Sigma_inv_acc @ X + lambda_reg * np.eye(X.shape[1]), X.T @ Sigma_inv_acc @ Y_all[:, [1]])

    vel_pred_data = X @ beta_gls_vel
    acc_pred_data = X @ beta_gls_acc

    vel_pred = vel_pred_data[-1]
    acc_pred = acc_pred_data[-1]

    return vel_pred_data, acc_pred_data, vel_pred, acc_pred
