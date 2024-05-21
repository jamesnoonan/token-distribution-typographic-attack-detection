# model_sizes='10 20 50 100 200 500 1000'

model_sizes='10 20 50 100 200'
for model_size in $model_sizes
do
    python main.py train ./data/datasets --image-model-size $model_size --text-model-size $model_size
    python main.py eval ./data/datasets --image-model-size $model_size --text-model-size $model_size
    python main.py graph ./data/model

    mkdir -p "./data/output/${model_size}"
    mv ./data/model "./data/output/${model_size}"
    mv ./data/train_loss_graph.png "./data/output/${model_size}"
done