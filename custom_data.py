import numpy as np



def generate_linear_data(n_samples=500, noise=0.0):
    n_noise = int(noise*n_samples)
    noise_idx = np.random.choice(n_samples, n_noise, replace=False)
    mean0 = np.array([-2, -2])
    mean1 = np.array([2, 2])

    cov = np.array([
        [1, 0],
        [0, 1]
    ])

    X0 = np.random.multivariate_normal(
        mean0,
        cov,
        n_samples // 2
    )

    X1 = np.random.multivariate_normal(
        mean1,
        cov,
        n_samples // 2
    )

    y0 = np.zeros(n_samples // 2)
    y1 = np.ones(n_samples // 2)

    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    y[noise_idx] = 1-y[noise_idx]

    return X, y

def generate_circle_data(n_samples=500, radius=2, noise=0.0):
    n_noise = int(noise*n_samples)
    noise_idx = np.random.choice(n_samples, n_noise, replace=False)

    X = np.random.uniform(-4, 4, (n_samples, 2))
    y = (X[:, 0]**2 + X[:, 1]**2 > radius**2).astype(int)
    y[noise_idx] = 1-y[noise_idx]
    return X, y