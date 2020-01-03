from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

import ipdb


class PytorchEstimator(ClassifierMixin):
    """Base class for Pytorch-based estimators (currently only classifiers) for
    use in TPOT.

    In the future, these will be merged into TPOT's main code base.
    """

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def get_params(self, deep=True):
        pass

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class PytorchClassifier(PytorchEstimator):
    def predict(self, X):
        return self.transform(X)


class LR(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LR, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # out = F.log_softmax(self.linear(x), dim=1)
        out = self.linear(x)
        return out


class PytorchLRClassifier(PytorchClassifier):
    """Logistic Regression classifier, implemented in PyTorch, for use with
    TPOT.
    """

    class PytorchDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __getitem__(self, i):
            return (self.X[i, :], self.y[i, :])

        def __len__(self):
            return self.X.shape[0]

    def __init__(
        self,
        penalty="l2",
        num_epochs=5,
        batch_size=8,
        learning_rate=0.001,
        num_classes=2,
    ):
        super().__init__()
        self.penalty = penalty
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes

    def get_params(self, deep=True):
        return {
            "penalty": self.penalty,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_classes": self.num_classes,
        }

    def fit(self, X, y):
        """Based on code from
        https://www.kaggle.com/negation/pytorch-logistic-regression-tutorial
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        input_size = X.shape[-1]
        num_classes = len(set(y))

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.int64)
        y_test = torch.tensor(y_test, dtype=torch.int64)

        train_dset = TensorDataset(X_train, y_train)
        test_dset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        test_loader = DataLoader(
            test_dset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

        # ipdb.set_trace()

        model = LR(input_size, num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

        # Train the model
        for epoch in range(self.num_epochs):
            for i, (rows, labels) in enumerate(train_loader):
                rows = Variable(rows)
                labels = Variable(labels)

                optimizer.zero_grad()
                # ipdb.set_trace()
                outputs = model(rows)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print(
                        "Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f"
                        % (
                            epoch + 1,
                            self.num_epochs,
                            i + 1,
                            len(train_dset) // self.batch_size,
                            loss.item(),
                        )
                    )

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


class PytorchMLP(PytorchEstimator):
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
    X, y = make_classification(
        n_features=100,
        n_redundant=2,
        n_informative=7,
        n_clusters_per_class=1,
        n_samples=10000,
    )

    lr = PytorchLRClassifier()
    lr.fit(X, y)

    predictions = lr.transform(X)


if __name__ == "__main__":
    predictions = main()
