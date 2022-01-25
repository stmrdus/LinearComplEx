python main.py --model InteractE --data WN18RR --batch 256 --train_strategy one_to_n --feat_drop 0.2 --hid_drop 0.3 --perm 4 --ker_sz 11 --lr 0.001
python main.py --model ComplEx --data WN18RR --batch 256 --train_strategy one_to_n --feat_drop 0.2 --hid_drop 0.3 --perm 4 --ker_sz 11 --lr 0.001
python main.py --model InteractE --data FB15k-237 --gpu 0 --name fb15k_237_run
python main.py --model ComplEx --data FB15k-237 --gpu 0 --name fb15k_237_run