import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_model(model, dataset, learning_rate=0.01, num_epochs=10):
    """
    Train a model on a dataset
 
    Args:
        model (torch.nn.Module): the model to train
        dataset (torch.utils.data.Dataset): the dataset to use
        learning_rate (float): the learning rate to use for the optimizer
        num_epochs (int): the number of epochs to train for
 
    Returns:
        dictionary: the trained state of the model
        list<float>: the training loss in each epoch
    """

    print("\n | Training model for {} epochs | \n".format(num_epochs))
    training_loss = []

    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        example_count = 0
        for i, (X_train, y_train) in enumerate(train_dataloader):
            y_predicted = model(X_train)

            loss = criterion(y_predicted, y_train)
            loss.backward()

            total_loss += loss.item()
            example_count += 1

            optimizer.step()
            optimizer.zero_grad()
        
        training_loss.append(total_loss / example_count)
        print(f'Epoch: {epoch+1}, loss = {loss.item():.4f}')
        print("\n")

    return model.state_dict(), training_loss

# Evaulate model
def eval_model(model, dataset):
    """
    Evalulate a model on a dataset
 
    Args:
        model (torch.nn.Module): the prediction model to use for evaluation
 
    Returns:
        float: the accuracy on the test set as a value from 0 to 1
    """

    test_dataloader = DataLoader(dataset, shuffle=True)

    with torch.no_grad():
        correct_count = 0
        example_count = 0

        for i, (X_test, y_test) in enumerate(test_dataloader):
            y_predicted = model(X_test)
            predicted_class = torch.argmax(y_predicted, dim=1)

            if (y_test[0] == predicted_class[0]):
                correct_count += 1
            example_count += 1
    test_acc = correct_count / example_count

    print("Test Accuracy: ", f"{round((test_acc) * 100, 2)}%")
    print("Example Count: ", example_count)

    return test_acc