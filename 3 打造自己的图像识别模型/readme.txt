python 1\ data_convert.py \
-i dataset/pic \
-o dataset/tfrecord \
--train-shards 2 \
--validation-shards 2 \
--num-threads 2 
--dataset-name satellite

python slim/train_image_classifier.py \
--train_dir=dataset/train \
--dataset_name=satellite \
--dataset_split_name=train \
--dataset_dir=dataset/tfrecord \
--model_name=inception_v3 \
--checkpoint_path=dataset/pretrained/inception_v3.ckpt \
--checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
--trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
--max_number_of_steps=100000 \
--batch_size=32 \
--learning_rate=0.001 \
--learning_rate_decay_type=fixed \
--save_interval_secs=30 \
--save_summaries_secs=10 \
--log_every_n_steps=10 \
--optimizer=rmsprop \
--weight_decay=0.00004 \
--clone_on_cpu=True

python slim/eval_image_classifier.py \
--checkpoint_path=dataset/train \
--eval_dir=dataset/eval \
--dataset_name=satellite \
--dataset_split_name=validation \
--dataset_dir=dataset/tfrecord \
--model_name=inception_v3

tensorboard --logdir dataset/train

python slim/export_inference_graph.py \
--alsologtostderr \
--model_name=inception_v3 \
--output_file=dataset/graph/inception_v3_inference_graph.pb \
--dataset_name=satellite

python 2\ freeze_graph.py \
--input_graph=dataset/graph/inception_v3_inference_graph.pb \
--input_checkpoint=dataset/train/model.ckpt-55 \
--input_binary=true \
--output_node_names=InceptionV3/Predictions/Reshape_1 \
--output_graph=dataset/graph/freeze_graph.pb

python 3\ classify_image_inception_v3.py \
--model_path=dataset/graph/freeze_graph.pb \
--label_path=dataset/tfrecord/label.txt \
--image_file=dataset/pic/test/test_image_1.jpg