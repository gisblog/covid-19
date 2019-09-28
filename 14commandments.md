**14 Commandments**

Transfer Learning – 14 commands to install and extend TensorFlow to retrain a classifier to classify your own images.

```
01. $:~# pip install --upgrade "tensorflow==1.7.*"
02. $:~# git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
03. $:~# cd tensorflow-for-poets-2
04. $:~/tensorflow-for-poets-2# curl http://download.tensorflow.org/example_images/flower_photos.tgz \
	  | tar --extract --gunzip --directory=tf_files
05. $:~/tensorflow-for-poets-2# IMAGE_SIZE=224
06. $:~/tensorflow-for-poets-2# ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}" # model
07. $:~/tensorflow-for-poets-2# tensorboard --logdir tf_files/training_summaries & # tensorboard provides measurements
08. $:~/tensorflow-for-poets-2# python -m scripts.retrain \
	--bottleneck_dir=tf_files/bottlenecks \
	--how_many_training_steps=4000 # default
	--model_dir=tf_files/models/ \
	--summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
	--output_graph=tf_files/retrained_graph.pb \
	--output_labels=tf_files/retrained_labels.txt \
	--architecture="${ARCHITECTURE}" \
	--image_dir=tf_files/flower_photos # train
09. $:~/tensorflow-for-poets-2# python -m scripts.label_image \
	--graph=tf_files/retrained_graph.pb  \
	--image=tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg # classify
  
10. $:~/tensorflow-for-poets-2# mkdir tf_files/flower_photos/flare # extend: min 20 images
11. $:~/tensorflow-for-poets-2# cd tf_files/flower_photos/flare
12. $:~/tensorflow-for-poets-2# curl --remote-name https://images-assets.nasa.gov/image/PIA17912/PIA17912~orig.jpg
13. $:~/tensorflow-for-poets-2# python -m scripts.retrain \
	--bottleneck_dir=tf_files/bottlenecks \
	--how_many_training_steps=4000 # default
	--model_dir=tf_files/models/ \
	--summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
	--output_graph=tf_files/retrained_graph.pb \
	--output_labels=tf_files/retrained_labels.txt \
	--architecture="${ARCHITECTURE}" \
	--image_dir=tf_files/flower_photos # retrain
14. $:~/tensorflow-for-poets-2# python -m scripts.label_image \
	--graph=tf_files/retrained_graph.pb  \
	--image=tf_files/flower_photos/flare/PIA17912~orig.jpg # classify custom image
```
