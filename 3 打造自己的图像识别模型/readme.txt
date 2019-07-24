python 1\ data_convert.py \
-i dataset/pic \
-o dataset/tfrecord \
--train-shards 2 \
--validation-shards 2 \
--num-threads 2 
--dataset-name satellite