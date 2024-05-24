# A simple script to run experiments on the hidden layer size
model_sizes=(10 20 50 100 200 500 1000)

for model_size in "${model_sizes[@]}"
do
    python main.py train ./data/dataset --image-model-size $model_size --text-model-size $model_size
    python main.py eval ./data/dataset --image-model-size $model_size --text-model-size $model_size

    mkdir -p "./data/output/${model_size}"
    mv ./data/model "./data/output/${model_size}"
    mv ./data/train_loss_graph.png "./data/output/${model_size}"
done