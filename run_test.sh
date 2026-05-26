# Image classification examples
#python main.py --num_clients 100 --dataset mnist --fusion fedavg --training_round 10 --local_epochs 3 --optimizer sgd --batch_size 64 --regularization 1e-5
#python main.py --num_clients 100 --dataset mnist --fusion dual_defense --training_round 20 --local_epochs 5 --batch_size 64 --attacker_strategy model_poisoning_ipm --attack_start_round 10 --attacker_ratio 0.3 --epsilon 0.01
#python main.py --num_clients 100 --dataset fmnist --fusion clipping_median --training_round 80 --local_epochs 5 --batch_size 64 --attacker_strategy gaussian --attack_start_round 40 --attacker_ratio 0.3 --epsilon 0.01

# CMAPSS RUL regression examples
python main.py --dataset cmapss --cmapss_subset fd001 --num_clients 10 --fusion dual_defense --training_round 80 --local_epochs 10 --batch_size 64 --window_size 50 --loss_type mse --rul_cap 130 --attacker_strategy gaussian --attack_start_round 40 --attacker_ratio 0.2
