# python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="./log/train_original_MFB_combined_data" --combine_dataset=True
# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENet_combined_data" --combine_dataset=True

# python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="./log/train_original_MFB"
# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENet"


#python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENet" --batch_size=10

# INFO:tensorflow:Final Loss: 3.48885
# INFO:tensorflow:Final Training Accuracy: 0.856374
# INFO:tensorflow:Final Training Mean IOU: 0.518484
# INFO:tensorflow:Final Validation Accuracy: 0.829732
# INFO:tensorflow:Final Validation Mean IOU: 0.44978



#python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="./log/train_original_MFB" --batch_size=10

# INFO:tensorflow:Final Loss: 0.00733908
# INFO:tensorflow:Final Training Accuracy: 0.842647
# INFO:tensorflow:Final Training Mean IOU: 0.521835
# INFO:tensorflow:Final Validation Accuracy: 0.82084
# INFO:tensorflow:Final Validation Mean IOU: 0.474012




#python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="./log/train_original_MFB-ADAMW" --batch_size=8 --optimizer_type="adamw"

# INFO:tensorflow:Final Loss: 0.0200352
# INFO:tensorflow:Final Training Precision: 0.986666
# INFO:tensorflow:Final Training Accuracy: 0.84743
# INFO:tensorflow:Final Training Mean IOU: 0.529338
# INFO:tensorflow:Final Validation Precision: 0.997549
# INFO:tensorflow:Final Validation Accuracy: 0.8244
# INFO:tensorflow:Final Validation Mean IOU: 0.48248

#python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENET-ADAMW-CYC-propper-M2.5e-3-L6.2e-4" --batch_size=8 --optimizer_type="adamw" --learning_rate_type="cyclic"



#check adamw against original weighting
#python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENET-ADAMW" --batch_size=8 --optimizer_type="adamw"
##### Note #####
# loss rate went back up is this overfitting?
# INFO:tensorflow:Final Loss: 7.08987
# INFO:tensorflow:Final Training Precision: 0.987879
# INFO:tensorflow:Final Training Accuracy: 0.870615
# INFO:tensorflow:Final Training Mean IOU: 0.560335
# INFO:tensorflow:Final Validation Precision: 0.997521
# INFO:tensorflow:Final Validation Accuracy: 0.849578
# INFO:tensorflow:Final Validation Mean IOU: 0.484539




python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENET-ADAMW-norelu-on-last" --batch_size=8 --optimizer_type="adamw"

#python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENET-ADAMW-CYC-propper-M2.5e-3-L6.2e-4" --batch_size=8 --optimizer_type="adamw" --learning_rate_type="cyclic"
