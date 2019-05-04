
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

from scipy.special import digamma, gamma

def e_step(N, D, K, X, alpha, beta, m, W, nu):
    # alpha: (K, 1), beta: (K, 1), m: (D, K), W: (D, D, K), nu: (K, 1)
    # X[:, :, None]: (N, D, 1), d: (N, D, K)
    d = X[:, :, None] - m
    
    # (10.67)式の計算
    gauss = np.exp(-0.5 * D / beta
                   - 0.5 * nu * np.sum(np.einsum("ijk,njk->nik", W, d) * d, axis=1))
    # (10.66)式の計算
    pi = np.exp(digamma(alpha) - digamma(alpha.sum()))
    # (10.65)式の計算
    lamb = np.exp(digamma((nu + 1 - np.arange(D)[:, None]) / 2.0).sum(axis=0) +
                  D * np.log(2) + np.linalg.slogdet(W.T)[1])
    
    r = pi * np.sqrt(lamb) * gauss
    r /= np.sum(r, axis=-1, keepdims=True)
    r[np.isnan(r)] = 1.0 / K
    
    return r

def m_step(X, r, alpha0, beta0, m0, W0, nu0, nk):
    # (10.51)式の計算
    nk = r.sum(axis=0)
    # (10.52)式の計算
    xk = X.T.dot(r) / nk
    # (10.53)式の計算
    d = X[:, :, None] - xk
    sk = np.einsum("nik,njk->ijk", d, r[:, None, :] * d) / nk
    
    # (10.57)式の計算
    alpha = alpha0 + nk
    # (10.60)式の計算
    beta = beta0 + nk
    # (10.61)式の計算
    m = (beta0 * m0[:, None] + nk * xk) / beta
    # (10.62)式の計算
    d = xk - m0[:, None]
    W = np.linalg.inv(np.linalg.inv(W0) +
                      (nk * sk).T +
                      (beta0 * nk * np.einsum("ik,jk->ijk", d, d) / (beta0 + nk)).T).T
    # (10.63)式の計算
    nu = nu0 + nk
    
    return alpha, beta, m, W, nu, nk

def calc_prob(N, D, K, X, m, W, nu, nk, alpha0):
    covs = nu * W
    precisions = np.linalg.inv(covs.T).T
    
    d = X[:, :, None] - m
    exponents = np.sum(np.einsum("nik,ijk->njk", d, precisions) * d, axis=1)
    gauss = np.exp(-0.5 * exponents) \
            / np.sqrt(np.linalg.det(covs.T).T * (2 * np.pi) ** D)
    gauss *= (alpha0 + nk) / (K * alpha0 + N)
    
    return np.sum(gauss, axis=-1)

def classify(N, D, K, X, alpha, beta, m, W, nu):
    return np.argmax(e_step(N, D, K, X, alpha, beta, m, W, nu), 1)

def student_t(D, X, beta, m, W, nu):
    nu = nu + 1 - D
    L = nu * beta * W / (1.0 + beta)
    d = X[:, :, None] - m
    maha_sq = np.sum(np.einsum("nik,ijk->njk", d, L) * d, axis=1)
    
    result = gamma(0.5 * (nu + D)) \
             * np.sqrt(np.linalg.det(L.T)) \
             * (1.0 + maha_sq / nu) ** (-0.5 * (nu + D)) \
             / (gamma(0.5 * nu) * (nu * np.pi) ** (0.5 * D))
    
    return result

def predict_dist(D, X, alpha, beta, m, W, nu):
    return (alpha * student_t(D, X, beta, m, W, nu)).sum(axis=-1) / alpha.sum()

def plot(N, D, K, X, Y, alpha, beta, m, W, nu):
    # 確率分布の等高線の描画
    nx = 200
    ny = 200
    
    x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), nx)
    y = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), ny)
    
    xx, yy = np.meshgrid(x, y)
    x0 = np.array([xx, yy]).reshape(2, -1).transpose()
    zz = predict_dist(D, x0, alpha, beta, m, W, nu).reshape(nx, ny)
    labels = classify(N, D, K, X, alpha, beta, m, W, nu)
    
    plt.cla()
    plt.contour(xx, yy, zz)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=labels)
    plt.show()

def main():
    N = 200
    D = 2
    K = 10
    alpha0 = 0.01
    
    # データの生成
    X, Y = make_blobs(n_samples=N,
                      n_features=D,
                      centers=[[1.0, 1.0], [3.5, 6.0], [6.0, 1.0]],
                      cluster_std=[0.8, 1.0, 0.8])
    
    # 事前分布のパラメータの設定
    alpha0 = np.ones(K) * alpha0
    m0 = np.zeros(D)
    W0 = np.eye(D)
    nu0 = D
    beta0 = 1.0
    
    # 確率分布のパラメータを初期化
    nk = N / K + np.zeros(K)
    alpha = alpha0 + nk
    beta = beta0 + nk
    m = X[np.random.choice(N, size=K, replace=False)].T
    W = np.tile(W0, (K, 1, 1)).T
    nu = nu0 + nk
    
    while True:
        params = np.hstack([ar.flatten() for ar in (alpha, beta, m, W, nu)])
        
        r = e_step(N, D, K, X, alpha, beta, m, W, nu)
        alpha, beta, m, W, nu, nk = m_step(X, r, alpha0, beta0, m0, W0, nu0, nk)
        
        params_new = np.hstack([ar.flatten() for ar in (alpha, beta, m, W, nu)])
        
        if np.allclose(params, params_new):
            break
    
    plot(N, D, K, X, Y, alpha, beta, m, W, nu)
    
if __name__ == "__main__":
    main()
    