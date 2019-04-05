#python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="./log/train_original_MFB_combined_data" --combine_dataset=True
#python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENet_combined_data" --combine_dataset=True

#python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="./log/train_original_MFB"
#python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENet" --batch_size=8

#python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENET-ADAMW" --batch_size=8 --optimizer_type="adamw"

#python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENET-ADAMW-CYC-propper-M2.5e-3-L6.2e-4-2000" --batch_size=8 --optimizer_type="adamw" --learning_rate_type="cyclic" --initial_learning_rate=6.2e-4 --max_learning_rate=2.5e-3

##########################
## Start grid search V001 - exploring cyclic learning rates

# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/ENET-ADAMW-CYC-propper-M2.5e-3-L6.2e-4-1cycle" --batch_size=8 --optimizer_type="adamw" --learning_rate_type="cyclic" --initial_learning_rate=6.2e-4 --max_learning_rate=2.5e-3 --learning_rate_step_size=6750
# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/ENET-ADAMW-CYC-propper-M2.5e-3-L6.2e-4-2cycle" --batch_size=8 --optimizer_type="adamw" --learning_rate_type="cyclic" --initial_learning_rate=6.2e-4 --max_learning_rate=2.5e-3 --learning_rate_step_size=3375
# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/ENET-ADAMW-CYC-propper-M2.5e-3-L6.2e-4-2epoch" --batch_size=8 --optimizer_type="adamw" --learning_rate_type="cyclic" --initial_learning_rate=6.2e-4 --max_learning_rate=2.5e-3 --learning_rate_step_size=45
# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/ENET-ADAMW-CYC-propper-M2.5e-3-L6.2e-4-10" --batch_size=8 --optimizer_type="adamw" --learning_rate_type="cyclic" --initial_learning_rate=6.2e-4 --max_learning_rate=2.5e-3 --learning_rate_step_size=675
# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/ENET-ADAMW-CYC-propper-M2.5e-3-L6.2e-4-100" --batch_size=8 --optimizer_type="adamw" --learning_rate_type="cyclic" --initial_learning_rate=6.2e-4 --max_learning_rate=2.5e-3 --learning_rate_step_size=67.5

# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/ENET-ADAM-CYC-propper-M2.5e-3-L6.2e-4-1cycle" --batch_size=8 --optimizer_type="adam" --learning_rate_type="cyclic" --initial_learning_rate=6.2e-4 --max_learning_rate=2.5e-3 --learning_rate_step_size=6750
# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/ENET-ADAM-CYC-propper-M2.5e-3-L6.2e-4-2cycle" --batch_size=8 --optimizer_type="adam" --learning_rate_type="cyclic" --initial_learning_rate=6.2e-4 --max_learning_rate=2.5e-3 --learning_rate_step_size=3375
# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/ENET-ADAM-CYC-propper-M2.5e-3-L6.2e-4-2epoch" --batch_size=8 --optimizer_type="adam" --learning_rate_type="cyclic" --initial_learning_rate=6.2e-4 --max_learning_rate=2.5e-3 --learning_rate_step_size=45
# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/ENET-ADAM-CYC-propper-M2.5e-3-L6.2e-4-10" --batch_size=8 --optimizer_type="adam" --learning_rate_type="cyclic" --initial_learning_rate=6.2e-4 --max_learning_rate=2.5e-3 --learning_rate_step_size=675
# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/ENET-ADAM-CYC-propper-M2.5e-3-L6.2e-4-100" --batch_size=8 --optimizer_type="adam" --learning_rate_type="cyclic" --initial_learning_rate=6.2e-4 --max_learning_rate=2.5e-3 --learning_rate_step_size=67.5

## winner = log/ENET-ADAM-CYC-propper-M2.5e-3-L6.2e-4-10 at 0.5170 Val _mean_IoU
## - end Gridsearch V001



###########################
## start Gridsearch V002 - Exploring cyclic rates  7.5e-4  to 8e-3

python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/GS-CLR-V002/ENET-ADAMW-CYC-propper-M8e-3-L7.5e-4-1cycle" --batch_size=8 --optimizer_type="adamw" --learning_rate_type="cyclic" --initial_learning_rate=7.5e-4 --max_learning_rate=8e-3 --learning_rate_step_size=6750
python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/GS-CLR-V002/ENET-ADAMW-CYC-propper-M8e-3-L7.5e-4-2cycle" --batch_size=8 --optimizer_type="adamw" --learning_rate_type="cyclic" --initial_learning_rate=7.5e-4 --max_learning_rate=8e-3 --learning_rate_step_size=3375
python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/GS-CLR-V002/ENET-ADAMW-CYC-propper-M8e-3-L7.5e-4-2epoch" --batch_size=8 --optimizer_type="adamw" --learning_rate_type="cyclic" --initial_learning_rate=7.5e-4 --max_learning_rate=8e-3 --learning_rate_step_size=45
python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/GS-CLR-V002/ENET-ADAMW-CYC-propper-M8e-3-L7.5e-4-10" --batch_size=8 --optimizer_type="adamw" --learning_rate_type="cyclic" --initial_learning_rate=7.5e-4 --max_learning_rate=8e-3 --learning_rate_step_size=675
python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/GS-CLR-V002/ENET-ADAMW-CYC-propper-M8e-3-L7.5e-4-100" --batch_size=8 --optimizer_type="adamw" --learning_rate_type="cyclic" --initial_learning_rate=7.5e-4 --max_learning_rate=8e-3 --learning_rate_step_size=67.5

python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/GS-CLR-V002/ENET-ADAM-CYC-propper-M8e-3-L7.5e-4-1cycle" --batch_size=8 --optimizer_type="adam" --learning_rate_type="cyclic" --initial_learning_rate=7.5e-4 --max_learning_rate=8e-3 --learning_rate_step_size=6750
python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/GS-CLR-V002/ENET-ADAM-CYC-propper-M8e-3-L7.5e-4-2cycle" --batch_size=8 --optimizer_type="adam" --learning_rate_type="cyclic" --initial_learning_rate=7.5e-4 --max_learning_rate=8e-3 --learning_rate_step_size=3375
python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/GS-CLR-V002/ENET-ADAM-CYC-propper-M8e-3-L7.5e-4-2epoch" --batch_size=8 --optimizer_type="adam" --learning_rate_type="cyclic" --initial_learning_rate=7.5e-4 --max_learning_rate=8e-3 --learning_rate_step_size=45
python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/GS-CLR-V002/ENET-ADAM-CYC-propper-M8e-3-L7.5e-4-10" --batch_size=8 --optimizer_type="adam" --learning_rate_type="cyclic" --initial_learning_rate=7.5e-4 --max_learning_rate=8e-3 --learning_rate_step_size=675
python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/GS-CLR-V002/ENET-ADAM-CYC-propper-M8e-3-L7.5e-4-100" --batch_size=8 --optimizer_type="adam" --learning_rate_type="cyclic" --initial_learning_rate=7.5e-4 --max_learning_rate=8e-3 --learning_rate_step_size=67.5

## winner = 
## - end Gridsearch V002