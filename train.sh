# python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="./log/train_original_MFB_combined_data" --combine_dataset=True
# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENet_combined_data" --combine_dataset=True

# python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="./log/train_original_MFB"
# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENet"


# python train_enet.py --weighting="ENET" --num_epochs=300 --logdir="./log/train_original_ENet" --batch_size=9

# INFO:tensorflow:global step 12000: loss: 2.7022 (0.45 sec/step)    Current Streaming Accuracy: 0.8580    Current Mean IOU: 0.5245
# INFO:tensorflow:Final Loss: 2.7022
# INFO:tensorflow:Final Training Accuracy: 0.857984
# INFO:tensorflow:Final Training Mean IOU: 0.524496
# INFO:tensorflow:Final Validation Accuracy: 0.830779
# INFO:tensorflow:Final Validation Mean IOU: 0.447612
# INFO:tensorflow:Finished training! Saving model to disk now.
# INFO:tensorflow:Saving the images now...


#python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="./log/train_original_MFB" --batch_size=8

# INFO:tensorflow:Final Training Accuracy: 0.894851
# INFO:tensorflow:Final Training Mean IOU: 0.631599
# INFO:tensorflow:Final Validation Accuracy: 0.874859
# INFO:tensorflow:Final Validation Mean IOU: 0.537522
# INFO:tensorflow:Finished training! Saving model to disk now.
# INFO:tensorflow:Saving the images now...



#python train_enet.py --weighting="MFB" --num_epochs=300 --logdir="./log/train_original_MFB-ADAMW" --batch_size=8 --optimizer_type="adamw"

# INFO:tensorflow:Final Loss: 0.0146053
# INFO:tensorflow:Final Training Accuracy: 0.807911
# INFO:tensorflow:Final Training Mean IOU: 0.463249
# INFO:tensorflow:Final Validation Accuracy: 0.796311
# INFO:tensorflow:Final Validation Mean IOU: 0.464731
# INFO:tensorflow:Finished training! Saving model to disk now.

python train_enet.py --weighting="MFB" --num_epochs=1 --logdir="./log/train_original_MFB-ADAMW-30Epoch" --batch_size=10 --optimizer_type="adamw"