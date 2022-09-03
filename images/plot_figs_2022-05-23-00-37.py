plt.clf()

def load_data(fn, proc_func):
    if not os.path.isfile(fn):
        return np.array([0.0])

    out = []
    with open(fn, 'r') as fp:
        for line in fp:
            n = line.strip().split(',')
            n = list(map(float, n))
            n = proc_func(n)
            out.append(n)
    return np.array(out)

def advantage(a, b):
    out = 0.0
    l = min(len(a), len(b))
    for i in range(l):
        out += a[i] - b[i]
    return out

def run(start, end, flag):
    def work(fn, label, color='black', line_style='-', ax=plt, proc_func=np.mean, marker=''):
        fed = load_data(fn, proc_func)
        fed = fed[start:end]
        h, = ax.plot(range(len(fed)), fed, color=color, linestyle=line_style, marker=marker, markersize=10)
        return h, label, fed

    def works(fns, label, color='black', line_style='-', ax=plt, proc_func=np.mean, marker=''):
        datum = []
        for fn in fns:
            data = load_data(fn, proc_func)
            datum.append(data)
            print(fn, ': ', np.var(data[20:]))
        l = np.min([len(d) for d in datum])
        print(l)
        datum = [d[:l] for d in datum]
        datum = np.mean(datum, axis=0)
        datum = datum[start:end]
        h, = ax.plot(range(len(datum)), datum, color=color, linestyle=line_style, marker=marker, markersize=10)
        return h, label, datum

    if True:
       pairs1 = [
        # ['fedavg-nonlinear-reacherv2-10iter-1e-4-initstate-ec.csv', 'FedAvg-nonlinear-reacherv2-10iter-1e-4-initstate-ec', 'grey', '-'],
        # ['fedavg-nonlinear-reacherv2-10iter-1e-4-1e-3-initstate-ec.csv', 'FedAvg-nonlinear-reacherv2-10iter-1e-4-1e-3-initstate-ec', 'black', '-'],
        # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-5e-3-initstate-ec.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-3-5e-3-initstate-ec', 'black', '-'],
        # ['fedavg-nonlinear-reacherv2-10iter-1e-3-1e-2-initstate-ec.csv', 'FedAvg-nonlinear-reacherv2-10iter-1e-3-1e-2-initstate-ec', 'black', '-'],
        # ['fedavg-nonlinear-reacherv2-6c-100iter-1e-3-1e-3-initstate-ec.csv', 'FedAvg-nonlinear-reacherv2-6c-100iter-1e-3-1e-3-initstate-ec', 'black', '-.'],
        # ['fedavg-nonlinear-reacherv2-6c-100iter-1e-3-5e-3-initstate-ec.csv', 'FedAvg-nonlinear-reacherv2-6c-100iter-1e-3-5e-3-initstate-ec', 'black', '-'],
        ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-1e-2-initstate-ec.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-3-1e-2-initstate-ec', 'black', '-'],
        # ['fedprox-nonlinear-reacherv2-10iter-1e-4-1e0-initstate-ec.csv', 'FedProx-nonlinear-reacherv2-10iter-1e-4-1e0-initstate-ec', 'orangered', '-'],
        # ['fedtrpo-nonlinear-reacherv2-10iter-tv-1e-4-5e-1-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-10iter-1e-4-5e-1-initstate-ec', 'dodgerblue', '-'],
        #
        # ['fedavg-nonlinear-reacherv2-10iter-1e-4-dynamics-ec.csv', 'FedAvg-nonlinear-reacherv2-10iter-1e-4-dynamics-ec', 'grey', '-'],
        # ['fedprox-nonlinear-reacherv2-10iter-1e-4-1e0-dynamics-ec.csv', 'FedProx-nonlinear-reacherv2-10iter-1e-4-1e0-dynamics-ec', 'orangered', '-'],
        # ['fedprox-nonlinear-reacherv2-10iter-1e-3-1e-2-1e-1-initstate-ec.csv', 'FedProx-nonlinear-reacherv2-10iter-1e-3-1e-2-1e-1-initstate-ec', 'orangered', '-'],
        ['fedprox-nonlinear-reacherv2-6c-10iter-1e-3-1e-2-1e-1-initstate-ec.csv', 'FedProx-nonlinear-reacherv2-6c-10iter-1e-3-1e-2-1e-1-initstate-ec', 'orangered', '-'],
        # ['fedprox-nonlinear-reacherv2-6c-10iter-1e-3-1e-2-5e-1-initstate-ec.csv', 'FedProx-nonlinear-reacherv2-6c-10iter-1e-3-1e-2-5e-1-initstate-ec', 'orangered', '--'],
        # ['fedtrpo-nonlinear-reacherv2-10iter-tv-1e-4-5e-1-dynamics-ec.csv', 'FedTRPO-nonlinear-reacherv2-10iter-1e-4-5e-1-dynamics-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-reacherv2-10iter-tv-1e-4-8e-1-dynamics-ec.csv', 'FedTRPO-nonlinear-reacherv2-10iter-1e-4-8e-1-dynamics-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-reacherv2-10iter-tv-1e-4-1e-3--1e0-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-10iter-1e-4-1e-3-1e0-initstate-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-reacherv2-10iter-sqrt_kl-1e-3-1e-2--3e-1-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-10iter-sqrt_kl-1e-3-1e-2-3e-1-initstate-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-reacherv2-10iter-sqrt_kl-1e-3-1e-2-1e-1-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-10iter-sqrt_kl-1e-3-1e-2-1e-1-initstate-ec', 'dodgerblue', '-.'],
        # ['fedtrpo-nonlinear-reacherv2-10iter-sqrt_kl-1e-3-1e-2-2e-1-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-10iter-sqrt_kl-1e-3-1e-2-2e-1-initstate-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-1e-2-2e-1-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-1e-2-2e-1-initstate-ec', 'dodgerblue', '--'],
        ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-1e-2-2e-1-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-2e-2-2e-1-initstate-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-5e-3-3e-1-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-5e-3-3e-1-initstate-ec', 'dodgerblue', '-.'],
        # ['fedtrpo-nonlinear-reacherv2-6c-50iter-sqrt_kl-1e-3-5e-3-6e-1-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-6c-50iter-sqrt_kl-1e-3-5e-3-6e-1-initstate-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-reacherv2-6c-100iter-sqrt_kl-1e-3-5e-3-1e0-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-6c-100iter-sqrt_kl-1e-3-5e-3-1e0-initstate-ec', 'dodgerblue', '--'],
        # ['fedtrpo-nonlinear-reacherv2-6c-100iter-sqrt_kl-1e-3-5e-3-1.5e0-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-6c-100iter-sqrt_kl-1e-3-5e-3-1.5e0-initstate-ec', 'dodgerblue', '--'],
        # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-5e-3-2e-1-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-5e-3-2e-1-initstate-ec', 'dodgerblue', '--'],
        # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-5e-3-2e-1-disable_kl-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-5e-3-2e-1-disable_kl-initstate-ec', 'dodgerblue', '--'],
       ]
       pairs2 = [
        # ['fedavg-nonlinear-figureeightv2-1iter-1e-2-1e-1-iid-ec.csv', 'FedAvg-nonlinear-figureeightv2-1iter-1e-2-1e-1-iid-ec', 'grey', '-.'],
        # ['fedavg-nonlinear-figureeightv2-1iter-1e-2-6e-2-iid-ec.csv', 'FedAvg-nonlinear-figureeightv2-1iter-1e-2-6e-2-iid-ec', 'grey', '-.'],
        # ['fedavg-nonlinear-figureeightv2-1iter-1e-2-5e-2-iid-ec.csv', 'FedAvg-nonlinear-figureeightv2-1iter-1e-2-5e-2-iid-ec', 'grey', '-'],
        ['fedavg-nonlinear-figureeightv2-50iter-1e-2-1e-3-iid-ec-seed0.csv', 'FedAvg-nonlinear-figureeightv2-50iter-1e-2-1e-3-iid-ec-seed0', 'grey', 'dotted'],
        ['fedavg-nonlinear-figureeightv2-50iter-1e-2-2e-3-iid-ec-seed0.csv', 'FedAvg-nonlinear-figureeightv2-50iter-1e-2-2e-3-iid-ec-seed0', 'grey', '-'],
        # ['fedavg-nonlinear-figureeightv2-50iter-1e-2-2e-3-iid-ec-seed10.csv', 'FedAvg-nonlinear-figureeightv2-50iter-1e-2-2e-3-iid-ec-seed10', 'grey', '-.'],
        ['fedavg-nonlinear-figureeightv2-50iter-1e-2-3e-3-iid-ec.csv', 'FedAvg-nonlinear-figureeightv2-50iter-1e-2-3e-3-iid-ec', 'grey', '-'],
        ['fedavg-nonlinear-figureeightv2-50iter-1e-2-5e-3-iid-ec.csv', 'FedAvg-nonlinear-figureeightv2-50iter-1e-2-5e-3-iid-ec', 'grey', '--'],
        # ['fedavg-nonlinear-figureeightv2-5iter-1e-4-iid-ec.csv', 'FedAvg-nonlinear-figureeightv2-5iter-1e-4-iid-ec', 'grey', '-'],
        # ['fedavg-nonlinear-figureeightv2-10iter-1e-4-iid-ec.csv', 'FedAvg-nonlinear-figureeightv2-10iter-1e-4-iid-ec', 'grey', '-.'],
        # ['fedavg-nonlinear-figureeightv2-50iter-1e-4-iid-ec.csv', 'FedAvg-nonlinear-figureeightv2-50iter-1e-4-iid-ec', 'black', '-.'],
        # ['fedavg-nonlinear-figureeightv2-50iter-1e-4-5e-5-iid-ec.csv', 'FedAvg-nonlinear-figureeightv2-50iter-1e-4-5e-5-iid-ec', 'grey', '-.'],
        # ['fedprox-nonlinear-figureeightv2-5iter-1e-4-1e-1-iid-ec.csv', 'FedProx-nonlinear-figureeightv2-5iter-1e-4-1e-1-iid-ec', 'orangered', '-.'],
        # ['fedprox-nonlinear-figureeightv2-5iter-1e-4-1e0-iid-ec.csv', 'FedProx-nonlinear-figureeightv2-5iter-1e-4-1e0-iid-ec', 'orangered', '-'],
        # ['fedprox-nonlinear-figureeightv2-10iter-1e-4-1e0-iid-ec.csv', 'FedProx-nonlinear-figureeightv2-10iter-1e-4-1e0-iid-ec', 'orangered', '-.'],
        # ['fedprox-nonlinear-figureeightv2-50iter-1e-4-1e0-iid-ec.csv', 'FedProx-nonlinear-figureeightv2-50iter-1e-4-1e0-iid-ec', 'orangered', '-.'],
        # ['fedprox-nonlinear-figureeightv2-50iter-1e-4-5e-5-1e-1-iid-ec.csv', 'FedProx-nonlinear-figureeightv2-50iter-1e-4-5e-5-1e-1-iid-ec', 'orangered', '-.'],
        # ['fedprox-nonlinear-figureeightv2-50iter-1e-4-1e0-iid-ec.csv', 'FedProx-nonlinear-figureeightv2-50iter-1e-4-1e0-iid-ec', 'orangered', '-.'],
        ['fedprox-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-3e-3-1e-2-iid-ec.csv', 'FedProx-nonlinear-figureeightv2-50iter-1e-2-3e-3-1e-2-iid-ec', 'orangered', '-'],
        # ['fedprox-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-1e-2-iid-ec.csv', 'FedProx-nonlinear-figureeightv2-50iter-1e-2-2e-3-1e-2-iid-ec', 'orangered', '-'],
        ['fedprox-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-1e-2-iid-ec-seed0.csv', 'FedProx-nonlinear-figureeightv2-50iter-1e-2-2e-3-1e-2-iid-ec-seed0', 'orangered', '-'],
        # ['fedprox-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-1e-2-iid-ec-seed10.csv', 'FedProx-nonlinear-figureeightv2-50iter-1e-2-2e-3-1e-2-iid-ec-seed10', 'orangered', '-.'],
        # ['fedtrpo-nonlinear-figureeightv2-1iter-sqrt_kl-1e-2-1e-1-2e-1-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-1iter-1e-2-1e-1-2e-1-iid-ec', 'dodgerblue', '-.'],
        # ['fedtrpo-nonlinear-figureeightv2-1iter-sqrt_kl-1e-2-1e-1-1e-1-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-1iter-1e-2-1e-1-1e-1-iid-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-figureeightv2-1iter-sqrt_kl-1e-2-5e-2-2e-1-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-1iter-1e-2-5e-2-2e-1-iid-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-figureeightv2-1iter-sqrt_kl-1e-2-5e-2-1e-1-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-1iter-1e-2-5e-2-1e-1-iid-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-figureeightv2-1iter-sqrt_kl-1e-2-5e-2-5e-2-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-1iter-1e-2-5e-2-5e-2-iid-ec', 'dodgerblue', '-.'],
        # ['fedtrpo-nonlinear-figureeightv2-1iter-sqrt_kl-1e-2-5e-2-6e-2-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-1iter-1e-2-5e-2-6e-2-iid-ec', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-figureeightv2-1iter-sqrt_kl-1e-2-5e-2-6e-2-disable_kl-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-1iter-1e-2-5e-2-6e-2-disable_kl-iid-ec', 'black', '-'],
        # ['fedtrpo-nonlinear-figureeightv2-1iter-sqrt_kl-1e-2-5e-2-8e-2-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-1iter-1e-2-5e-2-8e-2-iid-ec', 'dodgerblue', '--'],
        # ['fedtrpo-nonlinear-figureeightv2-1iter-sqrt_kl-1e-2-5e-2-6e-2-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-1iter-1e-2-5e-2-6e-2-iid-ec', 'dodgerblue', '--'],
        # ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-5e-3-4e-1-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-5e-3-4e-1-iid-ec', 'dodgerblue', '-.'],
        ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-1e-3-4e-1-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-1e-3-4e-1-iid-ec', 'dodgerblue', 'dotted'],
        ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-2e-3-6e-1-iid-ec', 'dodgerblue', '-.'],
        # ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-5e-1-iid-ec-seed10.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-2e-3-5e-1-iid-ec-seed10', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-1e-4-fixed_sigma-iid-ec-seed0.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-2e-3-6e-1-1e-4-fixed_sigma-iid-ec-seed0', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-1e-6-fixed_sigma-iid-ec-seed0.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-2e-3-6e-1-1e-6-fixed_sigma-iid-ec-seed0', 'dodgerblue', '--'],
        ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed0.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-2e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed0', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-iid-ec-seed10.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-2e-3-6e-1-iid-ec-seed10', 'dodgerblue', '-.'],
        # ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-7e-1-iid-ec-seed10.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-2e-3-7e-1-iid-ec-seed10', 'dodgerblue', '--'],
        # ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-7e-1-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-2e-3-7e-1-iid-ec', 'dodgerblue', '-.'],
        ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-3e-3-6e-1-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-3e-3-6e-1-iid-ec', 'dodgerblue', '-'],
        ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-5e-3-6e-1-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-5e-3-6e-1-iid-ec', 'black', '-'],
        ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-5e-3-7e-1-iid-ec.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-5e-3-7e-1-iid-ec', 'dodgerblue', '--'],

        ['fedavg-nonlinear-figureeightv2-50iter-1e-2-2e-3-iid-ec-seed1.csv', 'FedAvg-nonlinear-figureeightv2-50iter-1e-2-2e-3-iid-ec-seed1', 'grey', '-'],
        ['fedprox-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-1e-2-iid-ec-seed1.csv', 'FedProx-nonlinear-figureeightv2-50iter-1e-2-2e-3-1e-2-iid-ec-seed1', 'orangered', '-'],
        ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed1.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-2e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed1', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-3e-3-fixed_sigma-iid-ec-seed1.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-2e-3-6e-1-3e-3-fixed_sigma-iid-ec-seed1', 'dodgerblue', '--'],
        # ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-iid-ec-seed1.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-2e-3-6e-1-iid-ec-seed1', 'dodgerblue', '--'],

        ['fedavg-nonlinear-figureeightv2-50iter-1e-2-2e-3-iid-ec-seed2.csv', 'FedAvg-nonlinear-figureeightv2-50iter-1e-2-2e-3-iid-ec-seed2', 'grey', '-'],
        ['fedprox-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-1e-2-iid-ec-seed2.csv', 'FedProx-nonlinear-figureeightv2-50iter-1e-2-2e-3-1e-2-iid-ec-seed2', 'orangered', '-'],
        ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed2.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-2e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed2', 'dodgerblue', '-'],
        # ['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-3e-3-fixed_sigma-iid-ec-seed2.csv', 'FedTRPO-nonlinear-figureeightv2-sqrt_kl-50iter-1e-2-2e-3-6e-1-3e-3-fixed_sigma-iid-ec-seed2', 'dodgerblue', '--'],
       ]
       pairs3 = [
           ['fedavg-nonlinear-reacherv2-6c-20iter-1e-3-1e-3-initstate-ec.csv', 'FedAvg-nonlinear-reacherv2-6c-20iter-1e-3-1e-3-initstate-ec', 'black', '--'],
           ['fedavg-nonlinear-reacherv2-6c-20iter-1e-3-2e-3-initstate-ec.csv', 'FedAvg-nonlinear-reacherv2-6c-20iter-1e-3-5e-3-initstate-ec', 'black', '-'],
           ['fedprox-nonlinear-reacherv2-6c-20iter-1e-3-2e-3-1e-2-initstate-ec.csv', 'FedProx-nonlinear-reacherv2-6c-20iter-1e-3-5e-3-1e-2-initstate-ec', 'orangered', '-'],
           ['fedtrpo-nonlinear-reacherv2-6c-20iter-sqrt_kl-1e-3-2e-3-6e-1-1e-3-fixed_sigma-initstate-ec.csv', 'FedTRPO-nonlinear-reacherv2-6c-20iter-sqrt_kl-1e-3-2e-3-6e-1-1e-3-fixed_sigma-initstate-ec', 'dodgerblue' , '-'],
       ]
       pairs4 = [
           # [['fedavg-nonlinear-figureeightv2-50iter-1e-2-1e-3-iid-ec-seed0.csv', 'fedavg-nonlinear-figureeightv2-50iter-1e-2-1e-3-iid-ec-seed0.csv', 'fedavg-nonlinear-figureeightv2-50iter-1e-2-1e-3-iid-ec-seed0.csv'], 'FedAvg-nonlinear-figureeightv2-50iter-1e-2-1e-3-iid-ec', 'black', '--'],
           [['fedavg-nonlinear-figureeightv2-50iter-1e-2-2e-3-iid-ec-seed0.csv', 'fedavg-nonlinear-figureeightv2-50iter-1e-2-2e-3-iid-ec-seed1.csv', 'fedavg-nonlinear-figureeightv2-50iter-1e-2-2e-3-iid-ec-seed2.csv', 'fedavg-nonlinear-figureeightv2-50iter-1e-2-2e-3-iid-ec-seed3.csv'], 'FedAvg-nonlinear-figureeightv2-50iter-1e-2-2e-3-iid-ec', 'grey', '--'],
           [['fedprox-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-1e-2-iid-ec-seed0.csv', 'fedprox-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-1e-2-iid-ec-seed1.csv', 'fedprox-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-1e-2-iid-ec-seed2.csv', 'fedprox-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-1e-2-iid-ec-seed3.csv'], 'FedProx-nonlinear-figureeightv2-50iter-1e-2-2e-3-1e-2-iid-ec', 'orangered', '--'],
           [['fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed0.csv', 'fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed1.csv', 'fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed2.csv', 'fedtrpo-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed3.csv'], 'FedTRPO-nonlinear-figureeightv2-50iter-sqrt_kl-1e-2-2e-3-6e-1-1e-3-fixed_sigma-iid-ec', 'dodgerblue', '--'],
       ]
       pairs5 = [
           # [['fedavg-nonlinear-figureeightv2-50iter-1e-2-1e-3-iid-ec-seed0.csv', 'fedavg-nonlinear-figureeightv2-50iter-1e-2-1e-3-iid-ec-seed0.csv', 'fedavg-nonlinear-figureeightv2-50iter-1e-2-1e-3-iid-ec-seed0.csv'], 'FedAvg-nonlinear-figureeightv2-50iter-1e-2-1e-3-iid-ec', 'black', '--'],
           # [['fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec-seed0.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec-seed1.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec-seed2.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec-seed4.csv'], 'FedAvg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec', 'grey', '--'],
           # [['fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1e-2-iid-ec-seed0.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1e-2-iid-ec-seed1.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1e-2-iid-ec-seed2.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1e-2-iid-ec-seed4.csv'], 'FedProx-nonlinear-figureeightv1-50iter-1e-2-1e-3-1e-2-iid-ec', 'orangered', '--'],
           # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed1.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed2.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-6e-1-1e-3-fixed_sigma-iid-ec-seed4.csv'], 'FedTRPO-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-6e-1-1e-3-fixed_sigma-iid-ec', 'dodgerblue', '--'],
           # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-6e-1-1e-2-fixed_sigma-iid-ec-seed0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-6e-1-1e-2-fixed_sigma-iid-ec-seed1.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-6e-1-1e-2-fixed_sigma-iid-ec-seed2.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-6e-1-1e-2-fixed_sigma-iid-ec-seed4.csv'], 'FedTRPO-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-6e-1-1e-2-fixed_sigma-iid-ec', 'dodgerblue', '-'],
          # [['fedavg-nonlinear-figureeightv1-1iter-1e-2-1e-3-iid-ec-wseed0.csv', 'fedavg-nonlinear-figureeightv1-1iter-1e-2-1e-3-iid-ec-wseed1.csv', 'fedavg-nonlinear-figureeightv1-1iter-1e-2-1e-3-iid-ec-wseed2.csv', 'fedavg-nonlinear-figureeightv1-1iter-1e-2-1e-3-iid-ec-seed3.csv'], 'FedAvg-nonlinear-figureeightv1-1iter-1e-2-1e-3-iid-ec', 'black', 'dotted'],
          # [['fedavg-nonlinear-figureeightv1-50iter-1e-2-3e-3-iid-ec-wseed0.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-3e-3-iid-ec-wseed1.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-3e-3-iid-ec-wseed2.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-3e-3-iid-ec-seed4.csv'], 'FedAvg-nonlinear-figureeightv1-50iter-1e-2-3e-3-iid-ec', 'grey', '-'],
          # [['fedavg-nonlinear-figureeightv1-1iter-1e-2-5e-3-iid-ec-wseed0.csv', 'fedavg-nonlinear-figureeightv1-1iter-1e-2-5e-3-iid-ec-wseed1.csv', 'fedavg-nonlinear-figureeightv1-1iter-1e-2-5e-3-iid-ec-wseed2.csv', 'fedavg-nonlinear-figureeightv1-1iter-1e-2-5e-3-iid-ec-seed3.csv'], 'FedAvg-nonlinear-figureeightv1-1iter-1e-2-5e-3-iid-ec', 'black', '--'],
          # [['fedavg-nonlinear-figureeightv1-20iter-1e-2-1e-2-iid-ec-wseed0.csv', 'fedavg-nonlinear-figureeightv1-20iter-1e-2-1e-2-iid-ec-wseed1.csv', 'fedavg-nonlinear-figureeightv1-20iter-1e-2-1e-2-iid-ec-wseed2.csv', 'fedavg-nonlinear-figureeightv1-20iter-1e-2-1e-2-iid-ec-seed3.csv'], 'FedAvg-nonlinear-figureeightv1-20iter-1e-2-1e-2-iid-ec', 'black', '-'],
          # [['fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec-wseed0.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec-wseed1.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec-wseed2.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec-wseed3.csv'], 'FedAvg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec', 'black', '-'],
          # [['fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec-wseed0-1.1-2.0.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec-wseed1-1.1-2.0.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec-wseed2-1.1-2.0.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-3-iid-ec-wseed3-1.1-2.0.csv'], 'FedAvg-nonlinear-50iter-lr=1e-2-localkl=1e-3-iid-ec-1.1-2.0', 'black', '-'],
          # [['fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-4-iid-ec-wseed0.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-4-iid-ec-wseed1.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-4-iid-ec-wseed2.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-1e-4-iid-ec-wseed3.csv'], 'FedAvg-nonlinear-50iter-lr=1e-2-localkl=1e-4-iid-ec', 'black', 'dotted'],
          [['fedavg-nonlinear-figureeightv1-50iter-1e-2-2e-4-iid-ec-wseed0.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-2e-4-iid-ec-wseed1.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-2e-4-iid-ec-wseed2.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-2e-4-iid-ec-wseed3.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-2e-4-iid-ec-wseed4.csv'], 'FedAvg', 'black', '-'], # 'FedAvg-nonlinear-50iter-lr=1e-2-localkl=2e-4-iid-ec', 'black', '-'],
          # [['fedavg-nonlinear-figureeightv1-50iter-1e-2-3e-4-iid-ec-wseed0.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-3e-4-iid-ec-wseed1.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-3e-4-iid-ec-wseed2.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-3e-4-iid-ec-wseed3.csv'], 'FedAvg', 'black', '-'], # 'FedAvg-nonlinear-50iter-lr=1e-2-localkl=3e-4-iid-ec', 'black', '-'],
          # [['fedavg-nonlinear-figureeightv1-50iter-1e-2-5e-4-iid-ec-wseed0.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-5e-4-iid-ec-wseed1.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-5e-4-iid-ec-wseed2.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-5e-4-iid-ec-wseed3.csv'], 'FedAvg-nonlinear-50iter-lr=1e-2-localkl=5e-4-iid-ec', 'black', '-'],
          # [['fedavg-nonlinear-figureeightv1-50iter-1e-2-2e-3-iid-ec-wseed0.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-2e-3-iid-ec-wseed1.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-2e-3-iid-ec-wseed2.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-2e-3-iid-ec-seed3.csv'], 'FedAvg-nonlinear-figureeightv1-50iter-1e-2-2e-3-iid-ec', 'black', '-'],
          # [['fedavg-nonlinear-figureeightv1-50iter-1e-2-3e-3-iid-ec-wseed0.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-3e-3-iid-ec-wseed1.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-3e-3-iid-ec-wseed2.csv', 'fedavg-nonlinear-figureeightv1-50iter-1e-2-3e-3-iid-ec-seed3.csv'], 'FedAvg-nonlinear-figureeightv1-50iter-1e-2-3e-3-iid-ec', 'black', '-'],

          # [['fedprox-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-2-1e-4-iid-ec-seed0.csv', 'fedprox-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-2-1e-4-iid-ec-seed1.csv', 'fedprox-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-2-1e-4-iid-ec-seed2.csv', 'fedprox-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-2-1e-4-iid-ec-seed0.csv'], 'FedProx-nonlinear-figureeightv1-20iter-1e-2-1e-2-1e-4-iid-ec', 'orangered', '--'],
          # [['fedprox-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-2-1e-3-iid-ec-seed0.csv', 'fedprox-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-2-1e-3-iid-ec-seed1.csv', 'fedprox-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-2-1e-3-iid-ec-seed2.csv', 'fedprox-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-2-1e-3-iid-ec-seed0.csv'], 'FedProx-nonlinear-figureeightv1-20iter-1e-2-1e-2-1e-3-iid-ec', 'orangered', '-'],
          # [['fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1e-3-iid-ec-seed0.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1e-3-iid-ec-seed1.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1e-3-iid-ec-seed2.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1e-3-iid-ec-seed3.csv'], 'FedProx-nonlinear-figureeightv1-50iter-1e-2-1e-3-1e-3-iid-ec', 'orangered', '-'],
          # [['fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-4-iid-ec-seed0.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-4-iid-ec-seed1.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-4-iid-ec-seed2.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-4-iid-ec-seed3.csv'], 'FedProx-nonlinear-50iter-lr=1e-2-localkl=1e-4-mu=1e-4-iid-ec', 'orangered', '--'],
          [['fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-1e-3-iid-ec-seed0.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-1e-3-iid-ec-seed1.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-1e-3-iid-ec-seed2.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-1e-3-iid-ec-seed3.csv'], 'FedProx', 'orangered', '-'], # 'FedProx-nonlinear-50iter-lr=1e-2-localkl=2e-4-mu=1e-3-iid-ec', 'orangered', '-'],
          # [['fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1e-3-iid-ec-seed0.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1e-3-iid-ec-seed1.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1e-3-iid-ec-seed2.csv', 'fedprox-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1e-3-iid-ec-seed3.csv'], 'FedProx', 'orangered', '-'], # 'FedProx-nonlinear-50iter-lr=1e-2-localkl=3e-4-mu=1e-3-iid-ec', 'orangered', '-'],
          # [['fedprox-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-3e-3-1e-3-iid-ec-seed0.csv', 'fedprox-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-3e-3-1e-3-iid-ec-seed1.csv', 'fedprox-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-3e-3-1e-3-iid-ec-seed2.csv', 'fedprox-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-3e-3-1e-3-iid-ec-seed0.csv'], 'FedProx-nonlinear-figureeightv1-20iter-1e-2-3e-3-1e-3-iid-ec', 'orangered', '-'],

          # [['fedtrpo-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-2-2e-1-iid-ec-wseed0.csv', 'fedtrpo-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-2-2e-1-iid-ec-wseed1.csv', 'fedtrpo-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-2-2e-1-iid-ec-wseed2.csv', 'fedtrpo-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-2-2e-1-iid-ec-seed3.csv'], 'FedTRPO-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-2-2e-1-iid-ec', 'dodgerblue', '-'],
          # [['fedtrpo-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-3-1e-1-iid-ec-wseed0.csv', 'fedtrpo-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-3-1e-1-iid-ec-wseed1.csv', 'fedtrpo-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-3-1e-1-iid-ec-wseed2.csv', 'fedtrpo-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-3-1e-1-iid-ec-seed3.csv'], 'FedTRPO-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-1e-3-1e-1-iid-ec', 'dodgerblue', 'dotted'],
          # [['fedtrpo-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-2e-3-1e-1-iid-ec-wseed0.csv', 'fedtrpo-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-2e-3-1e-1-iid-ec-wseed1.csv', 'fedtrpo-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-2e-3-1e-1-iid-ec-wseed2.csv', 'fedtrpo-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-2e-3-1e-1-iid-ec-seed3.csv'], 'FedTRPO-nonlinear-figureeightv1-20iter-sqrt_kl-1e-2-2e-3-1e-1-iid-ec', 'dodgerblue', '--'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-wseed0-1.1.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-wseed1-1.1.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-wseed2-1.1.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-seed3-1.1.csv'], 'FedTRPO-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-1.1', 'purple', '--'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-3-4e-1-iid-ec-wseed0-1.1.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-3-4e-1-iid-ec-wseed1-1.1.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-3-4e-1-iid-ec-wseed2-1.1.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-3-4e-1-iid-ec-seed3-1.1.csv'], 'FedTRPO-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-3-4e-1-iid-ec-1.1', 'purple', '-'],

          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-wseed0-1.1-10.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-wseed1-1.1-10.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-wseed2-1.1-10.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-seed3-1.1-10.0.csv'], 'FedTRPO-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-1.1-10.0', 'dodgerblue', 'dotted'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-3e-1-iid-ec-wseed0-1.1-10.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-3e-1-iid-ec-wseed1-1.1-10.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-3e-1-iid-ec-wseed2-1.1-10.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-3e-1-iid-ec-seed3-1.1-10.0.csv'], 'FedTRPO-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-3e-1-iid-ec-1.1-10.0', 'dodgerblue', '--'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-wseed0-1.1-10.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-wseed1-1.1-10.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-wseed2-1.1-10.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-wseed3-1.1-10.0.csv'], 'FedTRPO-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-iid-ec-1.1-10.0', 'dodgerblue', '-'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1e-1-1e-4-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1e-1-1e-4-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1e-1-1e-4-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1e-1-1e-4-iid-ec-wseed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=1e-3-globalkl=1e-1-minkl=1e-4-iid-ec-1.1-2.0', 'dodgerblue', 'dotted'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-1e-1-1e-4-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-1e-1-1e-4-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-1e-1-1e-4-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-1e-1-1e-4-iid-ec-wseed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=2e-4-globalkl=1e-1-minkl=1e-4-iid-ec-1.1-2.0', 'dodgerblue', '--'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-1.5e-1-1e-4-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-1.5e-1-1e-4-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-1.5e-1-1e-4-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-1.5e-1-1e-4-iid-ec-wseed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=2e-4-globalkl=1.5e-1-minkl=1e-4-iid-ec-1.1-2.0', 'dodgerblue', 'dashed'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1e-1-1e-4-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1e-1-1e-4-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1e-1-1e-4-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1e-1-1e-4-iid-ec-wseed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=3e-4-globalkl=1e-1-minkl=1e-4-iid-ec-1.1-2.0', 'dodgerblue', 'dotted'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-2e-1-1e-2-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-2e-1-1e-2-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-2e-1-1e-2-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-2e-1-1e-2-iid-ec-wseed3-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-2e-1-1e-2-iid-ec-wseed4-1.1-2.0.csv'], 'FedPG-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=5e-4-globalkl=2e-1-minkl=1e-2-iid-ec-1.1-2.0', 'dodgerblue', 'dotted'],
          [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1.5e-1-1e-3-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1.5e-1-1e-3-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1.5e-1-1e-3-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1.5e-1-1e-3-iid-ec-wseed3-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1.5e-1-1e-3-iid-ec-wseed4-1.1-2.0.csv'], 'FedKL', 'dodgerblue', '-'], # 'FedPG-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=3e-4-globalkl=1.5e-1-minkl=1e-3-iid-ec-1.1-2.0', 'dodgerblue', '-'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1.5e-1-1e-4-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1.5e-1-1e-4-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1.5e-1-1e-4-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1.5e-1-1e-4-iid-ec-wseed3-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-1.5e-1-1e-4-iid-ec-wseed4-1.1-2.0.csv'], 'FedPG-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=3e-4-globalkl=1.5e-1-minkl=1e-4-iid-ec-1.1-2.0', 'dodgerblue', '--'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-4e-4-1.5e-1-1e-4-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-4e-4-1.5e-1-1e-4-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-4e-4-1.5e-1-1e-4-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-4e-4-1.5e-1-1e-4-iid-ec-wseed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=4e-4-globalkl=1.5e-1-minkl=1e-4-iid-ec-1.1-2.0', 'dodgerblue', '--'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-1.5e-1-1e-4-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-1.5e-1-1e-4-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-1.5e-1-1e-4-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-1.5e-1-1e-4-iid-ec-wseed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=5e-4-globalkl=1.5e-1-minkl=1e-4-iid-ec-1.1-2.0', 'dodgerblue', '-'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1.5e-1-1e-4-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1.5e-1-1e-4-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1.5e-1-1e-4-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-1.5e-1-1e-4-iid-ec-wseed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=1e-3-globalkl=1.5e-1-minkl=1e-4-iid-ec-1.1-2.0', 'dodgerblue', '--'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-1e-1-1e-4-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-1e-1-1e-4-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-1e-1-1e-4-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-1e-1-1e-4-iid-ec-wseed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=5e-4-globalkl=1e-1-minkl=1e-4-iid-ec-1.1-2.0', 'dodgerblue', '--'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-1.5e-1-1e-4-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-1.5e-1-1e-4-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-1.5e-1-1e-4-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-5e-4-1.5e-1-1e-4-iid-ec-wseed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=5e-4-globalkl=1.5e-1-minkl=1e-4-iid-ec-1.1-2.0', 'dodgerblue', '-'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-2e-1-1e-4-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-2e-1-1e-4-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-2e-1-1e-4-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-2e-1-1e-4-iid-ec-wseed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=1e-3-globalkl=2e-1-minkl=1e-4-iid-ec-1.1-2.0', 'dodgerblue', '-'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-1e-4-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-1e-4-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-1e-4-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-1e-4-iid-ec-wseed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-3-3e-1-1e-4-iid-ec-1.1-2.0', 'dodgerblue', '-'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-1-1e-4-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-1-1e-4-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-1-1e-4-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-1-1e-4-iid-ec-wseed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-1-1e-4-iid-ec-1.1-2.0', 'dodgerblue', '--'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-1-1e-5-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-1-1e-5-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-1-1e-5-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-1-1e-5-iid-ec-wseed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-1e-1-1e-5-iid-ec-1.1-2.0', 'dodgerblue', 'dotted'],
          # [['fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-2e-1-1e-4-fixed_sigma-iid-ec-wseed0-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-2e-1-1e-4-fixed_sigma-iid-ec-wseed1-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-2e-1-1e-4-fixed_sigma-iid-ec-wseed2-1.1-2.0.csv', 'fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-2e-1-1e-4-fixed_sigma-iid-ec-seed3-1.1-2.0.csv'], 'FedTRPO-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-2e-1-1e-4-fixed_sigma-iid-ec-1.1-2.0', 'dodgerblue', '--'],
          # [['fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-0.9999-iid-ec-seed0.csv', 'fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-0.9999-iid-ec-seed1.csv', 'fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-0.9999-iid-ec-seed2.csv', 'fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-0.9999-iid-ec-seed3.csv', 'fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-1e-4-0.9999-iid-ec-seed4.csv'], 'FMARL-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=1e-4-lambda=0.9999-iid-ec', 'green', 'dotted'],
          # [['fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-0.9999-iid-ec-seed0.csv', 'fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-0.9999-iid-ec-seed1.csv', 'fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-0.9999-iid-ec-seed2.csv', 'fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-0.9999-iid-ec-seed3.csv', 'fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-0.9999-iid-ec-seed4.csv'], 'FMARL', 'green', '-'], # 'FMARL-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=2e-4-lambda=0.9999-iid-ec', 'green', '-'],
          # [['fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-0.9999-iid-ec-seed0.csv', 'fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-0.9999-iid-ec-seed1.csv', 'fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-0.9999-iid-ec-seed2.csv', 'fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-0.9999-iid-ec-seed3.csv', 'fmarl-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-3e-4-0.9999-iid-ec-seed4.csv'], 'FMARL', 'green', '-'], # 'FMARL-nonlinear-50iter-sqrt_kl-lr=1e-2-localkl=3e-4-lambda=0.9999-iid-ec', 'green', '-'],
       ]
       pairs6 = [
           ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-1e-3-initstate-ec-seed0.csv', 'FedAvg', 'black', 'dotted'],
           ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-1e-2-initstate-ec-seed0.csv', 'FedAvg', 'black', '--'],
           ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-2e-3-initstate-ec-seed0.csv', 'FedAvg', 'green', '--'],
           ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-3e-3-initstate-ec-seed0.csv', 'FedAvg', 'purple', '--'],
           ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-1e-1-initstate-ec-seed0.csv', 'FedAvg', 'black', '-'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-3e-1-initstate-ec-seed0.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-3-3e-1-initstate-ec', 'black', '-'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-5e-2-3e-1-1e-4-initstate-ec-seed0.csv', 'FedPG', 'dodgerblue', '--'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-5e-2-4e-1-1e-4-initstate-ec-seed0.csv', 'FedPG', 'dodgerblue', '-'],

           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-1e-3-initstate-ec-seed0.csv', 'FedAvg $d_{local}=0.001$', 'black', 'dotted'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-5e-2-initstate-ec-seed0.csv', 'FedAvg $d_{local}=0.0001$', 'black', 'dotted'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-1e-2-initstate-ec-seed0.csv', 'FedAvg $d_{local}=0.01$', 'black', 'dotted'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-1e-1-initstate-ec-seed0.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-2-1e-1-initstate-ec', 'black', '--'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-5e-1-initstate-ec-seed0.csv', 'FedAvg', 'black', '-'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-2e-1-initstate-ec-seed0.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-2-2e-1-initstate-ec', 'black', '--'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-3e-1-initstate-ec-seed0.csv', 'FedAvg $d_{local}=0.3$', 'black', '-'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-1e0-initstate-ec-seed0.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-2-1e0-initstate-ec', 'black', '-'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-2-1e-2-3e-1-1e-4-initstate-ec-seed0.csv', 'FedPG $d_{local}=0.01,d_{local}=0.3$', 'dodgerblue', 'dotted'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-2-3e-1-6e-1-1e-4-initstate-ec-seed0.csv', 'FedPG $d_{local}=0.3,d_{global}=0.6$', 'dodgerblue', '-'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-2-5e-1-2e-1-1e-4-initstate-ec-seed0.csv', 'FedTRPO', 'dodgerblue', '--'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-2-5e-1-3e-1-1e-4-initstate-ec-seed0.csv', 'FedTRPO', 'dodgerblue', '-'],
       ]
       pairs7 = [
           ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-1e-3-dynamics-ec-seed1.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-3-1e-3-dynamics-ec', 'black', 'dotted'],
           ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-2e-3-dynamics-ec-seed1.csv', 'FedAvg', 'green', 'dotted'],
           ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-3e-3-dynamics-ec-seed1.csv', 'FedAvg', 'purple', 'dotted'],
           ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-1e-2-dynamics-ec-seed1.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-3-1e-2-dynamics-ec', 'black', '--'],
           ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-1e-1-dynamics-ec-seed1.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-3-1e-1-dynamics-ec', 'black', '-'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-3e-1-dynamics-ec-seed1.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-3-3e-1-dynamics-ec', 'black', '-'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-1e0-dynamics-ec-seed1.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-3-1e0-dynamics-ec', 'black', '-'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-5e-2-3e-1-1e-4-dynamics-ec-seed1.csv', 'FedPG', 'dodgerblue', '--'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-5e-2-4e-1-1e-4-dynamics-ec-seed1.csv', 'FedPG', 'dodgerblue', '-'],

           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-1e-3-dynamics-ec-seed1.csv', 'FedAvg', 'black', 'dotted'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-5e-2-dynamics-ec-seed1.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-2-1e-3-dynamics-ec', 'black', 'dotted'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-1e-2-dynamics-ec-seed1.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-2-1e-2-dynamics-ec', 'black', 'dotted'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-1e-1-dynamics-ec-seed1.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-2-1e-1-dynamics-ec', 'black', '--'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-5e-1-dynamics-ec-seed1.csv', 'FedAvg', 'black', '-'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-2e-1-dynamics-ec-seed1.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-2-2e-1-dynamics-ec', 'black', '--'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-3e-1-dynamics-ec-seed1.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-2-3e-1-dynamics-ec', 'black', '--'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-1e0-dynamics-ec-seed1.csv', 'FedAvg', 'black', '-'],           
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-2-1e0-5e-1-1e-4-dynamics-ec-seed1.csv', 'FedTRPO', 'dodgerblue', '-'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-2-1e-2-2e-1-1e-4-dynamics-ec-seed1.csv', 'FedTRPO-nonlinear-reacherv2-10iter-1e-2-1e-2-2e-1-1e-4-dynamics-ec', 'dodgerblue', 'dotted'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-2-3e-1-6e-1-1e-4-dynamics-ec-seed1.csv', 'FedTRPO-nonlinear-reacherv2-10iter-1e-2-3e-1-6e-1-1e-4-dynamics-ec', 'dodgerblue', '--'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-2-5e-1-2e-1-1e-4-dynamics-ec-seed1.csv', 'FedTRPO', 'dodgerblue', '--'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-2-5e-1-3e-1-1e-4-dynamics-ec-seed1.csv', 'FedTRPO', 'dodgerblue', '-'],
       ]
       pairs8 = [
           [['fedavg-nonlinear-reacherv2-6c-20iter-1e-3-1e-3-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-3-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-6c-50iter-1e-3-1e-3-both-ec-seed2.csv',], 'FedAvg', 'black', 'dotted'],
           # [['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-1e-2-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-3-1e-2-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-3-1e-2-both-ec-seed2.csv',], 'FedAvg', 'black', '--'],
           # [['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-2e-3-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-3-2e-3-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-3-2e-3-both-ec-seed2.csv',], 'FedAvg', 'green', '--'],
           # [['fedavg-nonlinear-reacherv2-6c-20iter-1e-3-3e-3-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-6c-20iter-1e-3-3e-3-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-6c-20iter-1e-3-3e-3-both-ec-seed2.csv',], 'FedAvg', 'purple', '--'],
           # [['fedavg-nonlinear-reacherv2-6c-10iter-1e-3-4e-3-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-3-4e-3-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-3-4e-3-both-ec-seed2.csv',], 'FedAvg', 'black', '--'],
           [['fedavg-nonlinear-reacherv2-6c-20iter-1e-3-1e-2-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-6c-50iter-1e-3-1e-2-both-ec-seed2.csv',], 'FedAvg', 'black', '--'],
           # [['fedavg-nonlinear-reacherv2-6c-20iter-1e-3-3e-2-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-3c-100iter-1e-3-3e-2-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-1c-20iter-1e-3-3e-2-both-ec-seed2.csv',], 'FedAvg', 'black', '-'],
           [['fedavg-nonlinear-reacherv2-6c-20iter-1e-3-1e-1-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-1-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-6c-50iter-1e-3-1e-1-both-ec-seed2.csv',], 'FedAvg', 'black', '-'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-5e-2-3e-1-1e-4-both-ec-seed2.csv', 'FedPG', 'dodgerblue', '--'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-5e-2-4e-1-1e-4-both-ec-seed2.csv', 'FedPG', 'dodgerblue', '-'],
           # [['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-3e-3-2e-1-1e-4-both-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-3e-3-2e-1-1e-4-both-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-3e-3-2e-1-1e-4-both-ec-seed2.csv',], 'FedPG', 'dodgerblue', '--'],
           # [['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-3e-3-3e-1-1e-4-both-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-3e-3-3e-1-1e-4-both-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-3-3e-3-3e-1-1e-4-both-ec-seed2.csv',], 'FedPG', 'dodgerblue', '--'],
           [['fedtrpo-nonlinear-reacherv2-6c-50iter-sqrt_kl-1e-3-1e-1-1e0-1e-4-both-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-1-8e-1-1e-4-both-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-6c-50iter-sqrt_kl-1e-3-1e-2-1e0-1e-4-both-ec-seed2.csv',], 'FedKL', 'dodgerblue', '--'],
           # [['fedtrpo-nonlinear-reacherv2-6c-20iter-sqrt_kl-1e-3-1e-1-5e-1-1e-4-both-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-1-5e-1-1e-4-both-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-1c-20iter-sqrt_kl-1e-3-1e-1-5e-1-1e-4-both-ec-seed2.csv',], 'FedPG', 'dodgerblue', 'dotted'],
           [['fedtrpo-nonlinear-reacherv2-6c-20iter-sqrt_kl-1e-3-1e-1-1e0-1e-4-both-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-1-6e-1-1e-4-both-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-6c-50iter-sqrt_kl-1e-3-1e-1-1e0-1e-4-both-ec-seed2.csv',], 'FedKL', 'dodgerblue', '-'],
           [['', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-4e-1-1e-4-both-ec-seed1.csv', '',], 'FedKL', 'purple', 'dotted'],
           [['', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-both-ec-seed1.csv', '',], 'FedKL', 'purple', '--'],
           # [['', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-8e-1-1e-4-both-ec-seed1.csv', '',], 'FedKL', 'purple', '-'],
           # [['fedtrpo-nonlinear-reacherv2-1c-20iter-sqrt_kl-1e-3-1e-1-8e-1-1e-4-both-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-1-8e-1-1e-4-both-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-1c-20iter-sqrt_kl-1e-3-1e-1-8e-1-1e-4-both-ec-seed2.csv',], 'FedPG', 'dodgerblue', '-'], 

           # [['fedavg-nonlinear-reacherv2-6c-50iter-1e-2-1e-3-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-6c-50iter-1e-2-1e-3-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-6c-50iter-1e-2-1e-3-both-ec-seed2.csv',], 'FedAvg', 'green', 'dotted'],
           # ['fedavg-nonlinear-reacherv2-6c-50iter-1e-2-1e-2-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-6c-50iter-1e-2-1e-2-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-6c-50iter-1e-2-1e-2-both-ec-seed2.csv',], 'FedAvg', 'green', '--'],
           # [['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-2e-3-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-2-2e-3-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-2-2e-3-both-ec-seed2.csv',], 'FedAvg', 'green', '--'],
           # [['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-3e-3-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-2-3e-3-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-2-3e-3-both-ec-seed2.csv',], 'FedAvg', 'green', '-'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-5e-2-both-ec-seed2.csv', 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-2-1e-3-both-ec', 'black', 'dotted'],
           # [['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-5e-3-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-2-5e-3-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-2-5e-3-both-ec-seed2.csv',], 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-2-5e-3-both-ec', 'green', '-'],
           # [['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-1e-1-both-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-2-1e-1-both-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-6c-10iter-1e-2-1e-1-both-ec-seed2.csv',], 'FedAvg-nonlinear-reacherv2-6c-10iter-1e-2-1e-1-both-ec', 'black', '--'],
           # ['fedavg-nonlinear-reacherv2-6c-10iter-1e-2-5e-1-both-ec-seed2.csv', 'FedAvg', 'black', '-'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-2-5e-2-2e-1-1e-4-both-ec-seed2.csv', 'FedTRPO', 'dodgerblue', '--'],
           # ['fedtrpo-nonlinear-reacherv2-6c-10iter-sqrt_kl-1e-2-5e-2-3e-1-1e-4-both-ec-seed2.csv', 'FedTRPO', 'dodgerblue', '-'],
       ]
       pairs9 = [
           # [['fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-initstate-ec-seed0-b.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-dynamics-ec-seed0-b.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-both-ec-seed0-b.csv'], '$\mathbf{B}_{\pi,\mu_{n},P_{n}}$ of FedAvg', 'black', '--'],
           # [['fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-initstate-ec-seed0-da.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-dynamics-ec-seed0-da.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-both-ec-seed0-da.csv'], '$\mathbf{D}_{\pi,\mu_{n},P_{n}} \mathbf{A}_{\pi,P_{n}}$ or FedAvg', 'black', '-'],
           [['fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed0-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed0-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed1-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed1-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed0-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed0-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed1-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed1-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed2-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed2-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed2-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed2-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed3-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed3-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed3-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed3-b.csv'], '$\mathbf{B}_{\pi,\mu_{n},P_{n}}$ of FedKL', 'red', '-', lambda n: np.mean(n)],
           [['fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed0-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed0-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed1-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed1-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed0-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed0-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed1-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed1-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed2-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed2-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed2-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed2-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed3-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed3-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed3-da.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed3-da.csv'], '$\mathbf{D}_{\pi,\mu_{n},P_{n}} \mathbf{A}_{\pi,P_{n}}$ of FedKL', 'dodgerblue', '--', lambda n: np.mean(n)],
           # [['fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed0-avg.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed0-avg.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed1-avg.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed1-avg.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed0-avg.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed0-avg.csv'], '$\sum_{k=1}^{N}p_{k} \mathbf{D}_{\pi,\mu_{n},P_{n}} \mathbf{A}_{\pi,P_{n}}$ of FedKL', 'dodgerblue', '-', lambda n: np.mean(n)],
           [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed0-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed0-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed1-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed1-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-iid-ec-seed0-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed0-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed2-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed2-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed2-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-iid-ec-seed2-evalh.csv'], 'FedKL', 'dodgerblue', '-', lambda n: np.mean(n)],
       ]
       if True:
           b_fns = pairs9[0][0]
           da_fns = pairs9[1][0]
           proc_func = np.mean
           # proc_func = lambda n: n[1] if len(n) > 1 else n[0]
           for i, b_fn in enumerate(b_fns):
               bs = load_data(b_fn, proc_func)
               das = load_data(da_fns[i], proc_func)
               print(bs)
               print(das)
               print(b_fn)
               print(da_fns[i])
               out = np.divide((das - bs), 1)
               with open(b_fn[:-5] + 'da-b.csv', 'w') as fp:
                   for o in out:
                       fp.write('%s\n' % o)
               print(b_fn[:-5] + 'da-b.csv')
       pairs10 = [
           [['fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-initstate-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-dynamics-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-both-ec-seed1.csv',], 'FedAvg', 'black', '-'],
           # [['fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-1e-3-initstate-ec-seed0.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-1e-3-dynamics-ec-seed0.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-1e-3-both-ec-seed1.csv',], 'FedProx', 'orangered', '-'],
           [['fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-1e-2-initstate-ec-seed0.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-1e-2-dynamics-ec-seed0.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-1e-2-both-ec-seed1.csv',], 'FedProx', 'orangered', '-'],
           [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-both-ec-seed1.csv',], 'FedKL', 'dodgerblue', '-'],
       ]
       pairs10 = [
         [['fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-initstate-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-initstate-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-initstate-ec-seed2.csv'], 'FedAvg', 'black', '-'],
         [['fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-1e0-initstate-ec-seed0.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-1e0-initstate-ec-seed1.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-1e0-initstate-ec-seed2.csv'], 'FedProx', 'orangered', '--'],
         [['fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-1e-1-initstate-ec-seed0.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-1e-1-initstate-ec-seed1.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-1e-1-initstate-ec-seed2.csv'], 'FedProx', 'orangered', '-'],
         [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-1e-1-1e-4-initstate-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-1e-1-1e-4-initstate-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-1e-1-1e-4-initstate-ec-seed2.csv'], 'FedKL', 'dodgerblue', 'dotted'],
         # [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-2e-1-1e-4-initstate-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-2e-1-1e-4-initstate-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-2e-1-1e-4-initstate-ec-seed2.csv'], 'FedKL', 'dodgerblue', '--'],
         # [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-3e-1-1e-4-initstate-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-3e-1-1e-4-initstate-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-3e-1-1e-4-initstate-ec-seed2.csv'], 'FedKL', 'dodgerblue', '-'],
       ]
       pairs100 = [
         [['fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-3-initstate-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-3-initstate-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-3-initstate-ec-seed2.csv'], 'FedAvg', 'black', '-'],
         [['fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-3-1e0-initstate-ec-seed0.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-3-1e0-initstate-ec-seed1.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-3-1e0-initstate-ec-seed2.csv'], 'FedProx', 'orangered', '--'],
         [['fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-3-1e-1-initstate-ec-seed0.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-3-1e-1-initstate-ec-seed1.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-3-1e-3-1e-1-initstate-ec-seed2.csv'], 'FedProx', 'orangered', '-'],
         [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-1e-1-1e-4-initstate-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-1e-1-1e-4-initstate-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-1e-1-1e-4-initstate-ec-seed2.csv'], 'FedKL', 'dodgerblue', 'dotted'],
         # [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-2e-1-1e-4-initstate-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-2e-1-1e-4-initstate-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-2e-1-1e-4-initstate-ec-seed2.csv'], 'FedKL', 'dodgerblue', '--'],
         # [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-3e-1-1e-4-initstate-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-3e-1-1e-4-initstate-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-3e-1-1e-4-initstate-ec-seed2.csv'], 'FedKL', 'dodgerblue', '-'],
       ]
       pairs1000 = [
         # [['fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-3-dynamics-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-3-dynamics-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-3-1e-3-dynamics-ec-seed2.csv'], 'FedAvg', 'black', '--'],
         [['fedavg-nonlinear-reacherv2-3c-20iter-1e-2-1e-2-dynamics-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-2-1e-2-dynamics-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-2-1e-2-dynamics-ec-seed2.csv'], 'FedAvg', 'black', '-'],
         [['fedprox-nonlinear-reacherv2-3c-20iter-1e-2-1e-2-1e-1-dynamics-ec-seed0.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-2-1e-2-1e-1-dynamics-ec-seed1.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-2-1e-2-1e-1-dynamics-ec-seed2.csv'], 'FedProx', 'orangered', '-'],
         # [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-1e-1-1e-4-dynamics-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-1e-1-1e-4-dynamics-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-1e-1-1e-4-dynamics-ec-seed2.csv'], 'FedKL', 'dodgerblue', 'dotted'],
         [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-2-1e-2-3e-1-1e-4-dynamics-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-2-1e-2-3e-1-1e-4-dynamics-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-2-1e-2-3e-1-1e-4-dynamics-ec-seed2.csv'], 'FedKL', 'dodgerblue', '-'],
         # [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-3e-1-1e-4-initstate-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-3e-1-1e-4-initstate-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-3-3e-1-1e-4-initstate-ec-seed2.csv'], 'FedKL', 'dodgerblue', '-'],
       ]
       pairs10000 = [
         # [['fedavg-nonlinear-reacherv2-3c-20iter-1e-2-1e-4-initstate-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-2-1e-4-initstate-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-2-1e-4-initstate-ec-seed2.csv'], 'FedAvg', 'black', 'dotted'],
         [['fedavg-nonlinear-reacherv2-3c-20iter-1e-2-1e-3-initstate-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-2-1e-3-initstate-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-2-1e-3-initstate-ec-seed2.csv'], 'FedAvg', 'black', '--'],
         [['fedavg-nonlinear-reacherv2-3c-20iter-1e-2-1e-2-initstate-ec-seed0.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-2-1e-2-initstate-ec-seed1.csv', 'fedavg-nonlinear-reacherv2-3c-20iter-1e-2-1e-2-initstate-ec-seed2.csv'], 'FedAvg', 'black', '-'],
         # [['fedprox-nonlinear-reacherv2-3c-20iter-1e-2-1e-3-1e-1-initstate-ec-seed0.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-2-1e-3-1e-1-initstate-ec-seed1.csv', 'fedprox-nonlinear-reacherv2-3c-20iter-1e-2-1e-3-1e-1-initstate-ec-seed2.csv'], 'FedProx', 'orangered', '-'],
         [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-2-1e-2-3e-1-1e-4-initstate-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-2-1e-2-3e-1-1e-4-initstate-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-2-1e-2-3e-1-1e-4-initstate-ec-seed2.csv'], 'FedKL', 'dodgerblue', 'dotted'],
         # [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-1e-1-1e-4-initstate-ec-seed0.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-1e-1-1e-4-initstate-ec-seed1.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-1e-1-1e-4-initstate-ec-seed2.csv'], 'FedKL', 'dodgerblue', '-'],
       ]
       pairs11 = [
           # [['fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed0-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed0-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed1-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed1-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed0-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed0-da-b.csv'], '$\mathbf{D}_{\pi,\mu_{n},P_{n}} \mathbf{A}_{\pi,P_{n}}-\mathbf{B}_{\pi,\mu_{n},P_{n}}$ of FedKL', 'red', '-', lambda n: np.mean(n)],
           [['fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed0-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed1-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed2-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-iid-ec-seed3-da-b.csv'], 'IID', 'orangered', 'dotted', '^'],
           [['fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed0-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed1-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed2-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed3-da-b.csv'], 'Initial State Distribution - $Q=60$', 'dodgerblue', 'dotted', 'o'],
           [['fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed0-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed1-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed2-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed3-da-b.csv'], 'Dynamics - action noise $\sim \mathcal{N}(0, 0.2)$', 'darkcyan', 'dotted', 'x'],
           [['fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed0-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed1-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed2-da-b.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed3-da-b.csv'], 'Dynamics - action noise $\sim \mathcal{N}(0, 0.4)$', 'orchid', 'dotted', 'd'],
       ]
       pairs12 = [
           [['fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed0-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed0-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-initstate-ec-seed1-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-dynamics-ec-seed1-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-iid-ec-seed0-evalh.csv', 'fedtrpo-nonlinear-reacherv2-3c-20iter-sqrt_kl-1e-3-1e-2-6e-1-1e-4-dynamics0.4-ec-seed0-evalh.csv'], 'FedKL', 'dodgerblue', '-', lambda n: np.mean(n)],
       ]
       ax1 = plt.subplot(3, 3, 1)
       ax2 = plt.subplot(3, 3, 2)
       ax3 = plt.subplot(3, 3, 3)
       ax4 = plt.subplot(3, 3, 4)
       ax5 = plt.subplot(3, 3, 5)
       ax6 = plt.subplot(3, 3, 6)
       ax7 = plt.subplot(3, 3, 7)
       ax8 = plt.subplot(3, 3, 8)
       ax9 = plt.subplot(3, 3, 9)
       if False:
           ax  = plt.subplot(1, 1, 1)
           ax4 = ax
           ax2 = ax
           if True:
               ax1 = plt.subplot(2, 2, 1)
               ax2 = plt.subplot(2, 2, 2)
               ax3 = plt.subplot(2, 2, 3)
               ax4 = plt.subplot(2, 2, 4)
           ax1 = plt.subplot(1, 1, 1)
           # plt.tight_layout()

       tasks = [[pairs1, ax4, (0.0, 500.0), (-90.0, 0.0), 'ReacherV2'], [pairs4, ax1, (0.0, 100.0), (0.0, 400.0), 'SUMO FigureEightV2 traffic control'], [pairs3, ax2, (0.0, 500.0), (-90.0, 0.0), 'ReacherV2']]
       tasks = [
           # fig 7., 8..
           # [[[p[0][0:3:1]] + p[1:] for p in pairs5], ax1, (0.0, 180.0), (150.0, 400.0), 'SUMO FigureEightV2 traffic control'],
           # [[[p[0][0:1]] + p[1:] for p in pairs5], ax1, (0.0, 180.0), (50.0, 400.0), 'SUMO FigureEightV2 traffic control: seed = 0'],
           # [[[p[0][1:2]] + p[1:] for p in pairs5], ax2, (0.0, 180.0), (50.0, 400.0), 'SUMO FigureEightV2 traffic control: seed = 1'],
           # [[[p[0][2:3]] + p[1:] for p in pairs5], ax3, (0.0, 180.0), (50.0, 400.0), 'SUMO FigureEightV2 traffic control: seed = 2'],
           # [[[p[0][3:4]] + p[1:] for p in pairs5], ax5, (0.0, 180.0), (50.0, 400.0), 'SUMO FigureEightV2 traffic control: seed = 3'],
           # [[[p[0][4:5]] + p[1:] for p in pairs5 if len(p[0]) >= 5], ax6, (0.0, 100.0), (0.0, 500.0), 'SUMO FigureEightV2 traffic control: seed = 4'],
           # [pairs6, ax1, (0.0, 100.0), (-30.0, -5.0), 'Reacher-v2 init-state seed0'],
           # [pairs7, ax2, (0.0, 100.0), (-30.0, -5.0), 'Reacher-v2 dynamics seed1'],
           # [pairs8, ax3, (0.0, 100.0), (-30.0, -5.0), 'Reacher-v2 both seed2'],
           # [[[p[0][0:1]] + p[1:] for p in pairs9], ax1, (0.0, 50.0), (0.0, 0.12), 'Reacher-v2 init-state seed0'],
           # [[[p[0][1:2]] + p[1:] for p in pairs9], ax2, (0.0, 50.0), (0.0, 0.12), 'Reacher-v2 dynamics seed1'],  # 0.006, 0.015.
           # [[[p[0][2:3]] + p[1:] for p in pairs9], ax3, (0.0, 50.0), (0.0, 0.12), 'Reacher-v2 init-state seed2'],
           # [[[p[0][3:4]] + p[1:] for p in pairs9], ax4, (0.0, 50.0), (0.0, 0.12), 'Reacher-v2 init-state seed2'],
           # [[[p[0][4:5]] + p[1:] for p in pairs9], ax3, (0.0, 50.0), (0.0, 0.12), 'Reacher-v2 IID seed0'],
           # [[[p[0][5:6]] + p[1:] for p in pairs9], ax4, (0.0, 50.0), (0.0, 0.12), 'Reacher-v2 dynamics with variance of noise=0.4 seed0'],
           # fig 4.
           # [[[p[0][1:4]] + p[1:] for p in pairs11], ax1, (0, 50), (0.0, 0.025), 'Reacher-v2 init-state seed0'],
           #
           # [[[p[0][0:1]] + p[1:] for p in pairs9], ax1, (0.0, 200.0), (-120.0, -5.0), 'Reacher-v2 init-state seed0'],
           # [[[p[0][1:2]] + p[1:] for p in pairs9], ax2, (0.0, 200.0), (-120.0, -5.0), 'Reacher-v2 dynamics seed1'],
           # [[[p[0][4:5]] + p[1:] for p in pairs9], ax3, (0.0, 200.0), (-120.0, -5.0), 'Reacher-v2 IID seed0'],
           # fig 5.
           # [[[p[0][5:6]] + p[1:] for p in pairs9[-1:]], ax1, (0.0, 200.0), (-120.0, -5.0), 'Reacher-v2 dynamics with variance of noise=0.4 seed0'],
           #
           # [[[p[0][0:1]] + p[1:] for p in pairs9][:1], ax1, (0.0, 50.0), (0.0, 0.2), 'Reacher-v2 init-state seed0'],
           # [[[p[0][1:2]] + p[1:] for p in pairs9][:1], ax1, (0.0, 50.0), (0.0, 0.2), 'Reacher-v2 init-state seed1'],
           # [[[p[0][4:5]] + p[1:] for p in pairs9][:1], ax1, (0.0, 50.0), (0.0, 0.2), 'Reacher-v2 init-state seed0'],
           # fig. 6.
           # [[[p[0][0:1]] + p[1:] for p in pairs10], ax1, (0.0, 200.0), (-30.0, -10.0), 'Reacher-v2 init-state seed0'],
           # [[[p[0][1:2]] + p[1:] for p in pairs10], ax2, (0.0, 200.0), (-30.0, -10.0), 'Reacher-v2 init-state seed1'],
           # [[[p[0][2:3]] + p[1:] for p in pairs10], ax3, (0.0, 200.0), (-60.0, -5.0), 'Reacher-v2 init-state seed2'],
           # [[[p[0][1:2]] + p[1:] for p in pairs8], ax4, (0.0, 200.0), (-60.0, -5.0), 'Reacher-v2 init-state averaged over three runs'],
           #
           [[[p[0][0:3]] + p[1:] for p in pairs10], ax1, (0.0, 200.0), (-30.0, -9.0), 'Reacher-v2 init-state seed0'],
           [[[p[0][0:3]] + p[1:] for p in pairs100], ax2, (0.0, 200.0), (-30.0, -9.0), 'Reacher-v2 init-state seed1'],
           [[[p[0][0:3]] + p[1:] for p in pairs10000], ax3, (0.0, 200.0), (-30.0, -9.0), 'Reacher-v2 init-state seed2'],
           [[[p[0][0:1]] + p[1:] for p in pairs10], ax4, (0.0, 200.0), (-30.0, -9.0), 'Reacher-v2 init-state seed2'],
           [[[p[0][1:2]] + p[1:] for p in pairs10], ax5, (0.0, 200.0), (-30.0, -9.0), 'Reacher-v2 init-state seed2'],
           [[[p[0][2:3]] + p[1:] for p in pairs10], ax6, (0.0, 200.0), (-30.0, -9.0), 'Reacher-v2 init-state seed2'],
           [[[p[0][0:1]] + p[1:] for p in pairs10000], ax7, (0.0, 200.0), (-30.0, -9.0), 'Reacher-v2 init-state seed2'],
           [[[p[0][1:2]] + p[1:] for p in pairs10000], ax8, (0.0, 200.0), (-30.0, -9.0), 'Reacher-v2 init-state seed2'],
           [[[p[0][2:3]] + p[1:] for p in pairs10000], ax9, (0.0, 200.0), (-30.0, -9.0), 'Reacher-v2 init-state seed2'],
       ]
       for task in tasks:
           pairs = task[0]
           ax = task[1]
           xlim = task[2]
           # xlim = (0.0, 300.0)
           ylim = task[3]
           # xlim = (0.0, 200.0)
           # ylim = (-120.0, -5.0)
           # ylim = (50.0, 400.0)
           title = task[4]
           handles, labels = [], []
           proc_func = np.mean
           # proc_func = lambda n: n[1] if len(n) > 1 else n[0]
           # h, = ax.plot(list(range(int(xlim[1] + 1))), [374.0]*int(xlim[1] + 1), color='purple')
           # handles.append(h)
           # ax.plot([1.0, 1.0], [0.0, 1.0], color='grey')
           # xlim = (0.0, 30.0)
           # ylim = (-50.0, -33.0)
           # for i in [9, 10, 13, 16, 17, 23, 24]:
           #     h, = ax.plot([i, i], [-120.0, 0.0], linestyle='--', color='red')
           # handles.append(h)
           # labels.append('$\Vert\mathbf{D}_{\pi,\mu_{n},P_{n}} \mathbf{A}_{\pi,P_{n}}\Vert_{F}<\Vert\mathbf{B}_{\pi,\mu_{n},P_{n}}\Vert_{F}$ occured')
           # labels.append('Human Driven (IDM)')
           for pair in pairs:
               if 'SUMO' in title and ('fmarl' in pair[0][0]):
                   # continue
                   pass
               marker = ''
               if len(pair) == 5:
                   marker = pair[4]
                   pass
               if type(pair[0]) is list:
                   h, label, datum = works(pair[0], pair[1], pair[2], pair[3], ax=ax, proc_func=proc_func, marker=marker)
               else:
                   h, label, datum = work(pair[0], pair[1], pair[2], pair[3], ax=ax, proc_func=proc_func, marker=marker)
               handles.append(h)
               labels.append(label)
           ax.set_xlim(xlim[0], xlim[1])
           ax.set_ylim(ylim[0], ylim[1])
           if False:
               ax.legend(handles=handles, labels=labels, loc='best', prop={'size': 30})
               ax.set_xlabel('# Round', fontsize=30)
               # ax.set_ylabel('Total Reward', fontsize=30)
               # ax.set_ylabel('Frobenius norm', fontsize=30)
               ax.set_ylabel('$\Vert\mathbf{D}_{\pi,\mu_{n},P_{n}} \mathbf{A}_{\pi,P_{n}}\Vert_{F}-\Vert\mathbf{B}_{\pi,\mu_{n},P_{n}}\Vert_{F}$', fontsize=30)
               ax.tick_params(axis='x', labelsize=30)
               ax.tick_params(axis='y', labelsize=30)
               # ax.title.set_text(title)
               from matplotlib.ticker import MaxNLocator
               ax.xaxis.set_major_locator(MaxNLocator(integer=True))
           ax.title.set_size(15)

run(0, 1000, True)
