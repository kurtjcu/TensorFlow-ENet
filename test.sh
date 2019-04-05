# python test_enet.py --checkpoint_dir=./log/train_original_ENet --logdir=./log/test_original_ENet
# python test_enet.py --checkpoint_dir=./log/train_original_MFB --logdir=./log/test_original_MFB
# python test_enet.py --checkpoint_dir=./log/train_original_ENet_combined_data --logdir=./log/test_original_ENet_combined_data
# python test_enet.py --checkpoint_dir=./log/train_original_MFB_combined_data --logdir=./log/test_original_MFB_combined_data

python test_enet.py --checkpoint_dir="./log/train_original_ENET" --logdir="./log/test/test_original_ENET"

python test_enet.py --checkpoint_dir="./log/ENET-ADAM-CYC-propper-M2.5e-3-L6.2e-4-10" --logdir="./log/test/test_ENET-ADAM-CYC-propper-M2.5e-3-L6.2e-4-10"

python test_enet.py --checkpoint_dir="./log/GS-CLR-V002/ENET-ADAMW-CYC-propper-M8e-3-L7.5e-4-1cycle" --logdir="./log/test/test_ENET-ADAMW-CYC-propper-M8e-3-L7.5e-4-1cycle"