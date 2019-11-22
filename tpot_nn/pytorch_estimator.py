import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


class PytorchLogisticRegression(ClassifierMixin, nn.Module):
    """Logistic Regression classifier, implemented in PyTorch, for use with
    TPOT.
    """

    class PytorchDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __getitem__(self, i):
            return (self.X[i,:], self.y[i,:])

        def __len__(self):
            return self.X.shape[0]

    def __init__(self, penalty='l2', num_epochs=5, batch_size=8, learning_rate=0.001):
        super().__init__()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """Based on code from
        https://www.kaggle.com/negation/pytorch-logistic-regression-tutorial
        """
        self.onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)
        y_onehot = self.onehot.fit_transform(y.reshape(-1,1))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_onehot, test_size=0.3
        )

        train_dset = self.PytorchDataset(X_train, y_train)
        test_dset = self.PytorchDataset(X_test, y_test)

        self.linear = nn.Linear(X.shape[1], y.shape[1])

        train_loader = DataLoader(train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dset, batch_size=self.batch_size, shuffle=True, num_workers=2)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        # Train the model
        for epoch in range(self.num_epochs):
            for i, (rows, labels) in enumerate(train_loader):
                rows = Variable(rows)
                labels = Variable(labels)

                optimizer.zero_grad()

                outputs = model(rows)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if (i+1) % 100 == 0:
                    print("Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f"
                          % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

        # Evaluate trained model on the test set
        correct = 0
        total = 0
        for rows, labels in test_loader:
            rows = Variable(rows)
            outputs = model(rows)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print("Model accuracy: %d %%" % (100 * correct / total))

    def forward(self, x):
        out = self.linear(x)
        return out

    def transform(self, X):
        pass

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class PytorchMLP(ClassifierMixin, nn.Module):
    """Multilayer Perceptron, implemented in PyTorch, for use with TPOT.
    """
    def __init__(self, num_epochs=5):
        super().__init__()
        self.num_epochs = num_epochs

    def fit(self, X, y):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def main():
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)

    lr = PytorchLogisticRegression()
    lr.fit(X, y)

if __name__=="__main__":
    main()