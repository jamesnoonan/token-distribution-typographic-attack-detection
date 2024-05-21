import matplotlib.pyplot as plt
import csv

def load_csv(filename):
    csv_data = []
    with open(filename, 'r') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for row in data:
            csv_data.append(row)
    return csv_data

def compute_stats(data):
    llm_correct = 0
    attack_successful = 0
    image_pred_correct = 0
    text_pred_correct = 0
    both_pred_correct = 0

    successful_detection = 0

    for row in data:
        image_label, text_label, llm_pred, image_pred, text_pred = row
        if llm_pred.lower() == image_label.lower():
            llm_correct += 1
        if llm_pred.lower() == text_label.lower():
            attack_successful += 1
        if image_pred == image_label:
            image_pred_correct += 1
        if text_pred == text_label:
            text_pred_correct += 1
        if image_pred == image_label and text_pred == text_label:
            both_pred_correct += 1
        if (image_pred != text_pred) == (image_label != text_label):
            successful_detection += 1
    
    llm_accuracy = llm_correct / len(data)
    attack_success_rate = attack_successful / len(data)
    image_accuracy = image_pred_correct / len(data)
    text_accuracy = text_pred_correct / len(data)
    both_accuracy = both_pred_correct / len(data)
    successful_detection_rate = successful_detection / len(data)

    print("Cases: ", len(data))
    print("LLM Accuracy: ", f"{round((llm_accuracy) * 100, 2)}%")
    print("Attack Success Rate: ", f"{round((attack_success_rate) * 100, 2)}%")
    print("Image Accuracy: ", f"{round((image_accuracy) * 100, 2)}%")
    print("Text Accuracy: ", f"{round((text_accuracy) * 100, 2)}%")
    print("Both Accuracy: ", f"{round((both_accuracy) * 100, 2)}%")
    print("Successful Detection Rate: ", f"{round((successful_detection_rate) * 100, 2)}%")

    return llm_accuracy, image_accuracy, text_accuracy, both_accuracy

def graph_results(data):
    plt.figure(figsize=(6, 5))

    labels = ['MLLM Prediction', 'Image Model\n Prediction', 'Text Model\n Prediction']
    accuracies = [data[0] * 100, data[1] * 100, data[2] * 100]

    x = range(len(labels))
    plt.bar(x, accuracies, color='mediumseagreen')
    plt.xticks(x, labels, fontsize=13)
    plt.ylabel('Accuracy (%)', fontsize=13)
    plt.title('Model Accuracies on Attacked Images', fontsize=16)

    plt.savefig("data/accuracy_graph.png")

def graph_training(data_folder):
    train_acc = load_csv(f"{data_folder}/training_loss.csv")

    image_acc = [float(acc[0]) for acc in train_acc]
    caption_acc = [float(acc[1]) for acc in train_acc]

    plt.figure(figsize=(8, 5))

    plt.plot([x+1 for x in range(len(image_acc))], image_acc, color='cornflowerblue', linewidth=4)
    plt.plot([x+1 for x in range(len(caption_acc))], caption_acc, color='mediumseagreen', linewidth=4)
    plt.ylabel('Loss', fontsize=13)
    plt.xlabel('Epoch', fontsize=13)
    # plt.ylim(0, 102)
    plt.xlim(0, 40)
    # add padding to bottom of figure
    plt.subplots_adjust(bottom=0.12)

    plt.title('Training Loss', fontsize=16)

    plt.legend(['Image Prediction Model', 'Text Prediction Model'], loc='upper right')

    plt.savefig("./data/train_loss_graph.png")