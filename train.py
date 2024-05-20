import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Evaulate model
def eval_model(model, dataset):
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

def train_model(model, dataset, learning_rate=0.01, num_epochs=10):
    print("\n | Training model for {} epochs | \n".format(num_epochs))
    training_loss = []

    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        for i, (X_train, y_train) in enumerate(train_dataloader):
            y_predicted = model(X_train)

            loss = criterion(y_predicted, y_train)
            loss.backward()

            training_loss.append(loss)

            optimizer.step()
            optimizer.zero_grad()
        
        print(f'Epoch: {epoch+1}, loss = {loss.item():.4f}')
        print("\n")

    return model.state_dict(), training_loss
