# Image classification examples
#python main.py --num_clients 100 --dataset mnist --fusion fedavg --training_round 10 --local_epochs 3 --optimizer sgd --batch_size 64 --regularization 1e-5
#python main.py --num_clients 100 --dataset mnist --fusion dual_defense --training_round 20 --local_epochs 5 --batch_size 64 --attacker_strategy model_poisoning_ipm --attack_start_round 10 --attacker_ratio 0.3 --epsilon 0.01
#python main.py --num_clients 100 --dataset fmnist --fusion clipping_median --training_round 80 --local_epochs 5 --batch_size 64 --attacker_strategy gaussian --attack_start_round 40 --attacker_ratio 0.3 --epsilon 0.01

# CMAPSS RUL regression examples
# Basic CMAPSS FL training (FD001, 5 clients, no attack)
python main.py --dataset cmapss --cmapss_subset fd001 --num_clients 5 --fusion fedavg --training_round 50 --local_epochs 3 --batch_size 64 --window_size 30 --loss_type mse --rul_cap 130 --attacker_strategy gaussian --attack_start_round 20 --attacker_ratio 0.2

# CMAPSS with non-IID partition and LSTM-style training rounds
#python main.py --dataset cmapss --cmapss_subset fd001 --num_clients 10 --fusion fedavg --training_round 100 --local_epochs 5 --batch_size 128 --window_size 50 --partition_type noniid --loss_type huber --rul_cap 130

# CMAPSS FD002 (6 operating conditions, more clients)
#python main.py --dataset cmapss --cmapss_subset fd002 --num_clients 10 --fusion fedavg --training_round 80 --local_epochs 3 --batch_size 64 --window_size 30 --loss_type mse

# CMAPSS with defense against model poisoning
#python main.py --dataset cmapss --cmapss_subset fd001 --num_clients 10 --fusion clipping_median --training_round 50 --local_epochs 3 --batch_size 64 --attacker_strategy gaussian --attack_start_round 20 --attacker_ratio 0.2 --loss_type smooth_l1

python main.py --dataset cmapss --cmapss_subset fd002 --num_clients 5 --fusion fedavg --training_round 50 --local_epochs 3 --batch_size 64 --window_size 30 --loss_type mse --rul_cap 130 --attacker_strategy gaussian --attack_start_round 20 --attacker_ratio 0.2

