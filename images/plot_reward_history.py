plt.clf()

def load_data(fn):
    out = []
    with open(fn, 'r') as fp:
        for line in fp:
            n = line.strip().split(',')
            n = list(map(float, n))
            n = np.mean(n)
            out.append(n)
    return np.array(out)

fedavg = load_data('fedavg.csv')
fedprox = load_data('fedprox.csv')
fedtrpo_global = load_data('fedtrpo-2021-11-18-10-57.csv')
fedtrpo_tv_15 = load_data('fedtrpo-tv-5e-4-1.5e0.csv')
fedtrpo_tv_20 = load_data('fedtrpo-tv-5e-4-2e0.csv')
fedtrpo_mahalanobis = load_data('fedtrpo-mahalanobis-5e-4-1e0.csv')

offset = 0
low = np.min([fedavg[offset], fedprox[offset], fedtrpo_global[offset], fedtrpo_mahalanobis[offset]])
# fedavg[:offset] = low
fedavg = fedavg[offset:]
# fedprox[:offset] = low
fedprox = fedprox[offset:]
# fedtrpo_global[:offset] = low
fedtrpo_global = fedtrpo_global[offset:]
# fedtrpo_tv[:offset] = low
fedtrpo_tv_15 = fedtrpo_tv_15[offset:]
fedtrpo_tv_20 = fedtrpo_tv_20[offset:]
# fedtrpo_mahalanobis[:offset] = low
fedtrpo_mahalanobis = fedtrpo_mahalanobis[offset:]
h1, = plt.plot(range(len(fedavg)), fedavg)
h2, = plt.plot(range(len(fedprox)), fedprox)
# h3, = plt.plot(range(len(fedtrpo_global)), fedtrpo_global)
# h4, = plt.plot(range(len(fedtrpo_local)), fedtrpo_local)
h5, = plt.plot(range(len(fedtrpo_tv_15)), fedtrpo_tv_15)
h7, = plt.plot(range(len(fedtrpo_tv_20)), fedtrpo_tv_20)
h6, = plt.plot(range(len(fedtrpo_mahalanobis)), fedtrpo_mahalanobis)
plt.legend(handles=[h1, h2, h5, h7, h6], labels=['FedAvg', 'FedProx', 'FedTRPO-tv-15', 'FedTRPO-tv-20', 'FedTRPO-mahalanobis'], loc='best')
