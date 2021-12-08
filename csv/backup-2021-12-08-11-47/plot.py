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

def run(start, end, flag):
    def work(fn, label, color='black', line_style='-'):
        fed = load_data(fn)
        fed = fed[start:end]
        h, = plt.plot(range(len(fed)), fed, color=color, linestyle=line_style)
        return h, label

    if True:
       pairs1 = [
        ['fedavg-nonlinear-1e-4-iid-ec.csv', 'FedAvg-nonlinear-1e-4-iid-ec', 'grey', '-'],
        ['fedprox-nonlinear-1e-4-1e0-iid-ec.csv', 'FedProx-nonlinear-1e-4-1e0-iid-ec', 'orangered', '-'],
        # ['fedtrpo-nonlinear-tv-1e-4-2e-1-iid-ec.csv', 'FedTRPO-nonlinear-1e-4-2e-1-iid-ec', '-'],
        # ['fedtrpo-nonlinear-tv-1e-4-3e-1-iid-ec.csv', 'FedTRPO-nonlinear-1e-4-3e-1-iid-ec', '-'],
        # ['fedtrpo-nonlinear-tv-1e-4-4e-1-iid-ec.csv', 'FedTRPO-nonlinear-1e-4-4e-1-iid-ec', '-'],
        ['fedtrpo-nonlinear-tv-1e-4-5e-1-iid-ec.csv', 'FedTRPO-nonlinear-1e-4-5e-1-iid-ec', 'dodgerblue', '-'],
        #
        ['fedavg-nonlinear-reacherv2-5iter-1e-4-dynamics-ec.csv', 'FedAvg-nonlinear-reacherv2-5iter-1e-4-dynamics-ec', 'grey', '--'],
        ['fedprox-nonlinear-reacherv2-5iter-1e-4-1e0-dynamics-ec.csv', 'FedProx-nonlinear-reacherv2-5iter-1e-4-1e0-dynamics-ec', 'orangered', '--'],
        ['fedtrpo-nonlinear-reacherv2-5iter-tv-1e-4-5e-1-dynamics-ec.csv', 'FedTRPO-nonlinear-reacherv2-5iter-1e-4-5e-1-dynamics-ec', 'dodgerblue', '--'],
        #
        ['fedavg-nonlinear-reacherv2-5iter-1e-4-initstate-ec.csv', 'FedAvg-nonlinear-reacherv2-5iter-1e-4-initstate-ec', 'grey', '-.'],
        ['fedprox-nonlinear-reacherv2-5iter-1e-4-1e0-initstate-ec.csv', 'FedProx-nonlinear-reacherv2-5iter-1e-4-1e0-initstate-ec', 'orangered', '-.'],
        ['fedtrpo-nonlinear-reacherv2-5iter-tv-1e-4-5e-1-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-5iter-1e-4-5e-1-initstate-ec', 'dodgerblue', '-.'],
        #
        ['fedavg-nonlinear-reacherv2-10iter-1e-4-initstate-ec.csv', 'FedAvg-nonlinear-reacherv2-10iter-1e-4-initstate-ec', 'grey', '-'],
        ['fedprox-nonlinear-reacherv2-10iter-1e-4-1e0-initstate-ec.csv', 'FedProx-nonlinear-reacherv2-10iter-1e-4-1e0-initstate-ec', 'orangered', '-'],
        ['fedtrpo-nonlinear-reacherv2-10iter-tv-1e-4-5e-1-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-10iter-1e-4-5e-1-initstate-ec', 'dodgerblue', '-'],
        #
        ['fedavg-nonlinear-reacherv2-10iter-1e-4-dynamics-ec.csv', 'FedAvg-nonlinear-reacherv2-10iter-1e-4-dynamics-ec', 'grey', '-'],
        ['fedprox-nonlinear-reacherv2-10iter-1e-4-1e0-dynamics-ec.csv', 'FedProx-nonlinear-reacherv2-10iter-1e-4-1e0-dynamics-ec', 'orangered', '-'],
        # ['fedtrpo-nonlinear-reacherv2-10iter-tv-1e-4-5e-1-dynamics-ec.csv', 'FedTRPO-nonlinear-reacherv2-10iter-1e-4-5e-1-dynamics-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-reacherv2-10iter-tv-1e-4-8e-1-dynamics-ec.csv', 'FedTRPO-nonlinear-reacherv2-10iter-1e-4-8e-1-dynamics-ec', 'dodgerblue', '-'],
        ['fedtrpo-nonlinear-reacherv2-10iter-tv-1e-4-1e0-dynamics-ec.csv', 'FedTRPO-nonlinear-reacherv2-10iter-1e-4-1e0-dynamics-ec', 'dodgerblue', '-'],
       ]
       pairs2 = [
        ['fedavg-nonlinear-figureeightv1-5iter-1e-4-iid-ec.csv', 'FedAvg-nonlinear-figureeightv1-5iter-1e-4-iid-ec', 'grey', '-'],
        ['fedprox-nonlinear-figureeightv1-5iter-1e-4-1e0-iid-ec.csv', 'FedProx-nonlinear-figureeightv1-5iter-1e-4-1e0-iid-ec', 'orangered', '-'],
        ['fedtrpo-nonlinear-figureeightv1-5iter-tv-1e-4-3e-2-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv1-5iter-1e-4-3e-2-iid-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-figureeightv1-5iter-tv-1e-4-5e-2-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv1-5iter-1e-4-5e-2-iid-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-figureeightv1-5iter-tv-1e-4-1e-1-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv1-5iter-1e-4-1e-1-iid-ec', 'turquoise', '-'],
       ]
       pairs = pairs2
       handles, labels = [], []
       for pair in pairs:
           if '10iter' not in pair[0] or 'dynamics' not in pair[0]:
               # continue
               pass
           h, label = work(pair[0], pair[1], pair[2], pair[3])
           handles.append(h)
           labels.append(label)
       plt.legend(handles=handles, labels=labels, loc='best')
       plt.ylim(-100.0, 0.0)
       plt.ylim(0.0, 600.0)

run(0, 500, True)
