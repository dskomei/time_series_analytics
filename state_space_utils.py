import numpy as np
import scipy
import matplotlib.pyplot as plt


# トレンド推定モデルの行列の設定
def set_trend_matrixes(F, H, G, n_dim_trend):
    
    G[0, 0] = 1
    H[0, 0] = 1
    if n_dim_trend == 1:
        F[0, 0] = 1
    elif n_dim_trend == 2:
        F[0, 0] = 2
        F[0, 1] = -1
        F[1, 0] = 1
    elif n_dim_trend == 3:
        F[0, 0] = 3
        F[0, 1] = -3
        F[0, 2] = 1
        F[1, 0] = 1
        F[2, 1] = 1
        
    index_state, index_obj = n_dim_trend, n_dim_trend
    
    return F, H, G, index_state, index_obj


# 季節周期モデルの行列の設定
def set_series_matrixes(F, H, G, n_dim_series, index_state, index_obj):
    
    if n_dim_series > 0:
        
        G[index_state, 1] = 1
        H[0, index_obj] = 1
        
        for i in range(n_dim_series - 1):
            F[index_state, index_state + i] = -1
            
        for i in range(n_dim_series - 2):
            F[index_state + i + 1, index_state + i] = 1
            
        index_state = index_state + n_dim_series - 1
        index_obj = index_obj + n_dim_series - 1

    return F, H, G, index_state, index_obj


# ARモデルの行列を設定
def set_ar_matrixes(F, H, G, n_dim_ar, index_state, index_obj):
        
    if n_dim_ar > 0:
        
        G[index_state, 2] = 1
        H[0, index_obj] = 1
        
        for i in range(n_dim_ar):
            F[index_state, index_state + i] = 0
            
        for i in range(n_dim_ar - 1):
            F[index_state + i + 1, index_state + i] = 1
            
    return F, H, G, index_state, index_obj


# 状態空間モデルの構築
def set_state_space_model_matrixes(
    n_dim_trend, n_dim_obs=1, n_dim_series=0, n_dim_ar=0, Q_sigma2=10
):
    
    if n_dim_series > 0 or n_dim_ar > 0:
        n_dim_state = n_dim_trend + n_dim_series + n_dim_ar - 1
    else:
        n_dim_state = n_dim_trend
    n_dim_Q = (n_dim_trend != 0) + (n_dim_series != 0) + (n_dim_ar != 0)
    
    G = np.zeros((n_dim_state, n_dim_Q))
    F = np.zeros((n_dim_state, n_dim_state))
    H = np.zeros((n_dim_obs, n_dim_state))
    Q = np.eye(n_dim_Q) * Q_sigma2
    
    F, H, G, index_state, index_obj = set_trend_matrixes(
        F=F, H=H, G=G,
        n_dim_trend=n_dim_trend
    )

    F, H, G, index_state, index_obj = set_series_matrixes(
        F=F, H=H, G=G,
        n_dim_series=n_dim_series,
        index_state=index_state,
        index_obj=index_obj
    )
    
    F, H, G, index_state, index_obj = set_ar_matrixes(
        F=F, H=H, G=G,
        n_dim_ar=n_dim_ar,
        index_state=index_state,
        index_obj=index_obj
    )
            
    Q = G.dot(Q).dot(G.T)
    
    return F, H, Q, n_dim_state


# 状態空間モデルの予測をグラフ化
def plot_state_space_model_pred(
    kf, y, n_train, credible_interval=True, img_file_path=None
):
    
    train_data, test_data = y[:n_train], y[n_train:]
    
    state_means, state_covs = kf.smooth(train_data)
    ovsevation_means_predicted = np.dot(state_means, kf.observation_matrices.T)
    ovsevation_covs_predicted = kf.observation_matrices.dot(
        np.abs(state_covs)
    ).transpose(1, 0, 2).dot(kf.observation_matrices.T)

    lowers, uppers = scipy.stats.norm.interval(
        0.95,
        ovsevation_means_predicted.flatten(),
        scale=np.sqrt(ovsevation_covs_predicted.flatten())
    )

    current_state = state_means[-1]
    current_cov = state_covs[-1]

    pred_means = np.array([])
    tmp_lowers = []
    tmp_uppers = []
    for i in range(len(test_data)):

        current_state, current_cov = kf.filter_update(
            current_state, current_cov, observation=None
        )
        pred_mean = current_state.dot(kf.observation_matrices.T)
        pred_cov = kf.observation_matrices.dot(
            np.abs(current_cov)
        ).dot(kf.observation_matrices.T)

        pred_means = np.r_[pred_means, pred_mean]

        lower, upper = scipy.stats.norm.interval(
            0.95, pred_mean, scale=np.sqrt(pred_cov)
        )
        tmp_lowers.append(lower)
        tmp_uppers.append(upper)

    lowers = np.hstack([lowers, np.array(tmp_lowers).flatten()])
    uppers = np.hstack([uppers, np.array(tmp_uppers).flatten()])

    plt.figure(figsize=(8, 6))
    plt.plot(y, label="observation")
    plt.plot(
        np.hstack([
            ovsevation_means_predicted.flatten(),
            pred_means.flatten()
        ]),
        '--', label="forecast"
    )
    if credible_interval:
        plt.fill_between(range(len(y)), uppers, lowers, alpha=0.5, label="credible interval")
    plt.legend()
    plt.tight_layout()

    if img_file_path:
        plt.savefig(img_file_path)


# 状態空間モデルの分解した状態をグラフ化
def plot_state_space_model_process(
    kf, y, n_train, n_dim_trend, n_dim_series=0,
    n_dim_ar=0, img_file_path=None
):
    
    train_data, test_data = y[:n_train], y[n_train:]

    state_means, state_covs = kf.smooth(train_data)
    
    index_start = 0
    index_end = n_dim_trend
    smoothed_means_trend = np.dot(
        state_means[:, index_start:index_end],
        kf.observation_matrices[:, index_start:index_end].T
    )
    index_start = index_end
    
    if n_dim_series > 0:
        index_end = index_start + n_dim_series - 1
        smoothed_means_series = np.dot(
            state_means[:, index_start:index_end],
            kf.observation_matrices[:, index_start:index_end].T
        )
        index_start = index_end
        
    if n_dim_ar > 0:
        index_end = index_start + n_dim_ar
        smoothed_means_ar = np.dot(
            state_means[:, index_start:index_end],
            kf.observation_matrices[:, index_start:index_end].T
        )

    pred_means_trend = []
    if n_dim_series > 0:
        pred_means_series = []
    if n_dim_ar > 0:
        pred_means_ar = []

    current_state = state_means[-1]
    current_cov = state_covs[-1]
    for i in range(len(test_data)):
        current_state, current_cov = kf.filter_update(
            current_state,
            current_cov,
            observation=None
        )
        
        index_start = 0
        index_end = n_dim_trend
        pred_means_trend.append(
            kf.observation_matrices[:, index_start:index_end].dot(current_state[index_start:index_end])
        )
        index_start = index_end
        
        if n_dim_series > 0:
            index_end = index_start + n_dim_series - 1
            pred_means_series.append(
                kf.observation_matrices[:, index_start:index_end].dot(current_state[index_start:index_end])
            )
            index_start = index_end
            
        if n_dim_ar > 0:
            index_end = index_start + n_dim_ar
            pred_means_ar.append(
                kf.observation_matrices[:, index_start:index_end].dot(current_state[index_start:index_end])
            )
            index_start = index_end
            
    plt.figure(figsize=(8, 6))
    plt.plot(y, label='observation')
    plt.plot(np.hstack([smoothed_means_trend.flatten(), np.array(pred_means_trend).flatten()]), '--', label='trend')
    if n_dim_series > 0:
        plt.plot(np.hstack([smoothed_means_series.flatten(), np.array(pred_means_series).flatten()]), ':', label='series')
    if n_dim_ar > 0:
        plt.plot(np.hstack([smoothed_means_ar.flatten(), np.array(pred_means_ar).flatten()]), '+-', label='ar')
    plt.legend()
    plt.tight_layout()
    
    if img_file_path:
        plt.savefig(img_file_path)


# 引数の遷移行列を代入し、LogLikelihoodを返す
def minimize_likelihood_ar_matrix(
    matrix, kf, train_data, index_ar_target, index_series_end
):
    
    kf.transition_matrices[index_series_end, index_series_end + index_ar_target] = matrix
    kf.smooth(train_data)
    
    return -kf.loglikelihood(train_data)


# 引数の共分散値を代入し、LogLikelihoodを返す
def minimize_likelihood_q(value, kf, train_data, index_target):
    
    kf.transition_covariance[index_target, index_target] = value
    kf.smooth(train_data)
    
    return -kf.loglikelihood(train_data)


# ARの係数
def ar_coef(vector, n_dim_ar):
    
    ar_hat = np.zeros(n_dim_ar)
    
    if n_dim_ar == 1:
        ar_hat = vector
    else:
        am = np.zeros(n_dim_ar)
        
        for i in range(n_dim_ar):
            ar_hat[i] = vector[i]
            am[i] = vector[i]
            if i > 0:
                for j in range(i - 1):
                    ar_hat[j] = am[j] - vector[i] * am[i - j]
                    
                if i < n_dim_ar - 1:
                    for j in range(i - 1):
                        am[j] = ar_hat[j]
                        
    return ar_hat
