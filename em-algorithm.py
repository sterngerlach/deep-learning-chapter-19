
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

def calc_nmat(X, mu, sigma):
    N, D = np.shape(X)
    K, _ = np.shape(mu)
    
    diff = np.reshape(X, (N, 1, D)) - np.reshape(mu, (1, K, D))
    beta = np.linalg.inv(sigma)
    exp = np.einsum("nkj,nkj->nk", np.einsum("nki,kij->nkj", diff, beta), diff)
    result = np.exp(-0.5 * exp) / np.sqrt(np.linalg.det(sigma))
    result *= (2 * np.pi) ** (D / 2)
    
    return result

def e_step(X, pi, mu, sigma):
    N, D = np.shape(X)
    
    nmat = calc_nmat(X, mu, sigma)
    numerator = pi * nmat
    gamma = numerator / np.reshape(np.sum(numerator, axis=1), (N, 1))
    
    return gamma

def m_step(X, mu, gamma):
    N, D = np.shape(X)
    _, K = np.shape(gamma)
    
    diff = np.reshape(X, (N, 1, D)) - np.reshape(mu, (1, K, D))
    nk = np.sum(gamma, axis=0)
    pi = nk / N
    mu = np.dot(gamma.T, X) / np.reshape(nk, (K, 1))
    sigma = np.einsum("nki,nkj->kij", np.einsum("nk,nki->nki", gamma, diff), diff)
    sigma /= np.reshape(nk, (K, 1, 1))
    
    return (pi, mu, sigma)

def calc_prob(X, pi, mu, sigma):
    prob = np.dot(calc_nmat(X, mu, sigma), pi)
    return prob

def calc_log_likelihood(X, pi, mu, sigma):
    log_likelihood = np.sum(np.log(calc_prob(X, pi, mu, sigma)))
    return log_likelihood

def plot(X, Y, pi, mu, sigma):
    # 確率分布の等高線の描画
    nx = 200
    ny = 200
    
    x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), nx)
    y = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), ny)
    
    xx, yy = np.meshgrid(x, y)
    x0 = np.stack([xx.flatten(), yy.flatten()]).T
    zz = calc_prob(x0, pi, mu, sigma).reshape(nx, ny)
    
    plt.cla()
    plt.contour(xx, yy, zz, animated=True)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y)
    plt.show()

def main():
    # データ数, 各データの次元, クラスタ数の設定
    N = 200
    D = 2
    K = 3
    
    # データの生成
    X, Y = make_blobs(n_samples=N,
                      n_features=D,
                      centers=[[1.0, 1.0], [3.5, 6.0], [6.0, 1.0]],
                      cluster_std=[0.8, 1.0, 0.8])
    
    # パラメータの初期化
    pi = np.ones(K) / K
    mu = X[np.random.choice(N, size=K, replace=False)]
    sigma = np.tile(np.diag(np.var(X, axis=0)), (K, 1, 1))
    
    # 対数尤度
    log_likelihood = -np.inf
    
    # 対数尤度の差分の閾値
    epsilon = 1e-05
    
    # EMアルゴリズムの実行
    while True:
        gamma = e_step(X, pi, mu, sigma)
        (pi, mu, sigma) = m_step(X, mu, gamma)
        
        # 収束条件の判定
        log_likelihood_prev = log_likelihood
        log_likelihood = calc_log_likelihood(X, pi, mu, sigma)
        
        # 対数尤度の変化量が閾値未満であれば, EMアルゴリズムを終了
        if log_likelihood - log_likelihood_prev < epsilon:
            break
        
        print("log_likelihood: {}".format(log_likelihood))
    
    plot(X, Y, pi, mu, sigma)

if __name__ == "__main__":
    main()
    