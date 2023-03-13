# FedKL
===================================

Dependencies
------------

Please refer to requirements.txt. Dependencies can be installed by using the following command:

    pip install requirements.txt



File Structure
------------

- Algorithm and Model Implementations: model/
    - RL related: model/fl
    - FL related: model/rl
        - Agent wrapper: model/rl/agent.py
        - The core of FedKL local actor: model/rl/trpo.py
        - Critic: model/rl/critic.py
    - Optimizers: model/optimizer
- Customized RL Environments: environment/
- Implementation of federated client/device: client/


Example Usage
-----

To reproduce the result in our [FedKL](https://arxiv.org/abs/2204.08125) paper:

    ./main --pg=TRPO --fed=FedTRPO --lr=1e-2 --kl_targ=2e-4 --nm_targ=7e-2 --sigma=1e-3 --distance_metric=sqrt_kl --retry_min=-500 --n_local_iter=50 --parallel=10 --clients_per_round=7 --heterogeneity_type=iid --expose_critic --env=figureeightv1 --num_rounds=1000 --init_seed=5 --reward_history_fn=fedtrpo-nonlinear-figureeightv1-50iter-sqrt_kl-1e-2-2e-4-7e-2-1e-3-dynamics03-ec-wseed5-1.1-2.0.csv > fedtrpo50iter12247213-wseed5-1.1-2.0.log

Type `./main --help` for a list of all key flags.

## References
Please refer to our [FedKL](https://arxiv.org/abs/2204.08125) paper for more details as well as all references.
