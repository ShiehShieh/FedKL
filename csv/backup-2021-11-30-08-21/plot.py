# coding: utf-8
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

def advantage(a, b):
    out = 0.0
    l = min(len(a), len(b))
    for i in range(l):
        out += a[i] - b[i]
    return out

def run(start, end, group1, group2, group3, group4, group5):
    # fedavg = load_data('fedavg.csv')
    # fedprox = load_data('fedprox.csv')
    # fedtrpo_global = load_data('fedtrpo-2021-11-18-10-57.csv')
    # fedtrpo_local = load_data('fedtrpo-2021-11-18-11-00.csv')
    # fedtrpo_tv_15 = load_data('fedtrpo-tv-5e-4-1.5e0.csv')
    # fedtrpo_tv_20 = load_data('fedtrpo-tv-5e-4-2e0.csv')
    # fedtrpo_mahalanobis = load_data('fedtrpo-mahalanobis-5e-4-1e0.csv')

    # fedavg_nonlinear_14 = load_data('fedavg-nonlinear-1e-4.csv')
    # fedprox_nonlinear_14 = load_data('fedprox-nonlinear-1e-4-1e-2.csv')
    # fedtrpo_nonlinear_14_20 = load_data('fedtrpo-nonlinear-tv-1e-4-2e0.csv')
    # fedtrpo_nonlinear_14_10 = load_data('fedtrpo-nonlinear-tv-1e-4-1e0.csv')
    # fedtrpo_nonlinear_14_51 = load_data('fedtrpo-nonlinear-tv-1e-4-5e-1.csv')

    # fedavg_nonlinear_55 = load_data('fedavg-nonlinear-5e-5.csv')
    # fedprox_nonlinear_55 = load_data('fedprox-nonlinear-5e-5-1e-2.csv')
    # fedtrpo_nonlinear_55 = load_data('fedtrpo-nonlinear-tv-5e-5-2e0.csv')

    # low = np.min([fedavg[offset], fedprox[offset], fedtrpo_global[offset], fedtrpo_mahalanobis[offset]])
    # fedavg[:offset] = low
    # fedavg = fedavg[offset:]
    # fedprox[:offset] = low
    # fedprox = fedprox[offset:]
    # fedtrpo_global[:offset] = low
    # fedtrpo_global = fedtrpo_global[offset:]
    # fedtrpo_local = fedtrpo_local[offset:]
    # fedtrpo_tv[:offset] = low
    # fedtrpo_tv_15 = fedtrpo_tv_15[offset:]
    # fedtrpo_tv_20 = fedtrpo_tv_20[offset:]
    # fedtrpo_mahalanobis[:offset] = low
    # fedtrpo_mahalanobis = fedtrpo_mahalanobis[offset:]

    # fedavg_nonlinear_14 = fedavg_nonlinear_14[start:end]
    # fedprox_nonlinear_14 = fedprox_nonlinear_14[start:end]
    # fedtrpo_nonlinear_14_20 = fedtrpo_nonlinear_14_20[start:end]
    # fedtrpo_nonlinear_14_10 = fedtrpo_nonlinear_14_10[start:end]
    # fedtrpo_nonlinear_14_51 = fedtrpo_nonlinear_14_51[start:end]

    # fedavg_nonlinear_55 = fedavg_nonlinear_55[start:end]
    # fedprox_nonlinear_55 = fedprox_nonlinear_55[start:end]
    # fedtrpo_nonlinear_55 = fedtrpo_nonlinear_55[start:end]

    if group1:
        h1, = plt.plot(range(len(fedavg)), fedavg)
        h2, = plt.plot(range(len(fedprox)), fedprox)
        # h3, = plt.plot(range(len(fedtrpo_global)), fedtrpo_global)
        # h4, = plt.plot(range(len(fedtrpo_local)), fedtrpo_local)
        h5, = plt.plot(range(len(fedtrpo_tv_15)), fedtrpo_tv_15)
        h7, = plt.plot(range(len(fedtrpo_tv_20)), fedtrpo_tv_20)
        h6, = plt.plot(range(len(fedtrpo_mahalanobis)), fedtrpo_mahalanobis)
        plt.legend(handles=[h1, h2, h5, h7, h6], labels=['FedAvg', 'FedProx', 'FedTRPO-tv-15', 'FedTRPO-tv-20', 'FedTRPO-mahalanobis'], loc='best')

    if group2:
        h8, = plt.plot(range(len(fedavg_nonlinear_14)), fedavg_nonlinear_14)
        h9, = plt.plot(range(len(fedprox_nonlinear_14)), fedprox_nonlinear_14)
        h10, = plt.plot(range(len(fedtrpo_nonlinear_14)), fedtrpo_nonlinear_14)

        h11, = plt.plot(range(len(fedavg_nonlinear_55)), fedavg_nonlinear_55)
        h12, = plt.plot(range(len(fedprox_nonlinear_55)), fedprox_nonlinear_55)
        h13, = plt.plot(range(len(fedtrpo_nonlinear_55)), fedtrpo_nonlinear_55)

        plt.legend(handles=[h8, h9, h10, h11, h12, h13], labels=['FedAvg-nonlinear-1e-4', 'FedProx-nonlinear-1e-4', 'FedTRPO-nonlinear-1e-4', 'FedAvg-nonlinear-5e-5', 'FedProx-nonlinear-5e-5', 'FedTRPO-nonlinear-5e-5'], loc='best')

    if group3:
        h14, = plt.plot(range(len(fedavg_nonlinear_14)), fedavg_nonlinear_14)                                                                       
        h15, = plt.plot(range(len(fedprox_nonlinear_14)), fedprox_nonlinear_14)                                                                     
        h16, = plt.plot(range(len(fedtrpo_nonlinear_14_20)), fedtrpo_nonlinear_14_20)
        h17, = plt.plot(range(len(fedtrpo_nonlinear_14_10)), fedtrpo_nonlinear_14_10)
        h18, = plt.plot(range(len(fedtrpo_nonlinear_14_51)), fedtrpo_nonlinear_14_51)

        plt.legend(handles=[h14, h15, h16, h17, h18], labels=['FedAvg-nonlinear-1e-4', 'FedProx-nonlinear-1e-4', 'FedTRPO-nonlinear-1e-4-2e0', 'FedTRPO-nonlinear-1e-4-1e0', 'FedTRPO-nonlinear-1e-4-5e-1'], loc='best')

    if group4:
        fedavg_nonlinear_14 = load_data('fedavg-nonlinear-1e-4.csv')
        fedprox_nonlinear_14 = load_data('fedprox-nonlinear-1e-4-1e-1.csv')
        fedtrpo_nonlinear_14_20 = load_data('fedtrpo-nonlinear-tv-1e-4-2e0.csv')
        fedtrpo_nonlinear_14_10 = load_data('fedtrpo-nonlinear-tv-1e-4-1e0.csv')
        fedtrpo_nonlinear_14_51 = load_data('fedtrpo-nonlinear-tv-1e-4-5e-1.csv')

        fedavg_nonlinear_14 = fedavg_nonlinear_14[start:end]
        fedprox_nonlinear_14 = fedprox_nonlinear_14[start:end]
        fedtrpo_nonlinear_14_20 = fedtrpo_nonlinear_14_20[start:end]
        fedtrpo_nonlinear_14_10 = fedtrpo_nonlinear_14_10[start:end]
        fedtrpo_nonlinear_14_51 = fedtrpo_nonlinear_14_51[start:end]

        h14, = plt.plot(range(len(fedavg_nonlinear_14)), fedavg_nonlinear_14)
        h15, = plt.plot(range(len(fedprox_nonlinear_14)), fedprox_nonlinear_14)
        h16, = plt.plot(range(len(fedtrpo_nonlinear_14_20)), fedtrpo_nonlinear_14_20)
        h17, = plt.plot(range(len(fedtrpo_nonlinear_14_10)), fedtrpo_nonlinear_14_10)
        h18, = plt.plot(range(len(fedtrpo_nonlinear_14_51)), fedtrpo_nonlinear_14_51)
        plt.legend(handles=[h14, h15, h16, h17, h18], labels=['FedAvg-nonlinear-1e-4', 'FedProx-nonlinear-1e-4-1e-1', 'FedTRPO-nonlinear-1e-4-2e0', 'FedTRPO-nonlinear-1e-4-1e0', 'FedTRPO-nonlinear-1e-4-5e-1'], loc='best')

    def work(fn, label, line_style='-'):
        fed = load_data(fn)
        fed = fed[start:end]
        h, = plt.plot(range(len(fed)), fed, line_style)
        return h, label

    if group5:
       pairs = [
        ['fedavg-nonlinear-1e-4.csv', 'FedAvg-nonlinear-1e-4', '-'],
        ['fedavg-nonlinear-1e-4-iid.csv', 'FedAvg-nonlinear-1e-4-iid', '-'],
        ['fedavg-nonlinear-1e-4-iid-ec.csv', 'FedAvg-nonlinear-1e-4-iid-ec', '-'],
        ['fedavg-nonlinear-1e-4-ec.csv', 'FedAvg-nonlinear-1e-4-ec', '--'],
        ['fedprox-nonlinear-1e-4-1e-1.csv', 'FedProx-nonlinear-1e-4-1e-1', '-'],
        ['fedprox-nonlinear-1e-4-1e0.csv', 'FedProx-nonlinear-1e-4-1e0', '-'],
        ['fedprox-nonlinear-1e-4-1e0-ec.csv', 'FedProx-nonlinear-1e-4-1e0-ec', '--'],
        ['fedprox-nonlinear-1e-4-1e0-iid.csv', 'FedProx-nonlinear-1e-4-1e0-iid', '-'],
        ['fedprox-nonlinear-1e-4-1e0-iid-ec.csv', 'FedProx-nonlinear-1e-4-1e0-iid-ec', '-'],
        ['fedtrpo-nonlinear-tv-1e-4-4e-1.csv', 'FedTRPO-nonlinear-1e-4-4e-1', '-'],
        ['fedtrpo-nonlinear-tv-1e-4-5e-1.csv', 'FedTRPO-nonlinear-1e-4-5e-1', '-'],
        ['fedtrpo-nonlinear-tv-1e-4-6e-1.csv', 'FedTRPO-nonlinear-1e-4-6e-1', '-'],
        ['fedtrpo-nonlinear-tv-1e-4-5e-1-iid.csv', 'FedTRPO-nonlinear-1e-4-5e-1-iid', '-'],
        ['fedtrpo-nonlinear-tv-1e-4-5e-1-iid-ec.csv', 'FedTRPO-nonlinear-1e-4-5e-1-iid-ec', '-'],
        ['fedtrpo-nonlinear-tv-1e-4-4e-1-ec.csv', 'FedTRPO-nonlinear-1e-4-4e-1-ec', '--'],
        ['fedtrpo-nonlinear-tv-1e-4-5e-1-ec.csv', 'FedTRPO-nonlinear-1e-4-5e-1-ec', '--'],
        ['fedtrpo-nonlinear-tv-1e-4-6e-1-ec.csv', 'FedTRPO-nonlinear-1e-4-6e-1-ec', '--'],
       ]
       handles, labels = [], []
       for pair in pairs:
           if 'ec' in pair[1] or 'iid' in pair[1]:
               pass
           if 'iid-ec' not in pair[1]:
               continue
           h, label = work(pair[0], pair[1], pair[2])
           handles.append(h)
           labels.append(label)
       plt.legend(handles=handles, labels=labels, loc='best')
       plt.ylim(-100, 0.0)

run(0, 200, False, False, False, False, True)