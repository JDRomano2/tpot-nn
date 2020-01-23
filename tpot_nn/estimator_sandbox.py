from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from pmlb import fetch_data
import ipdb

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, assert_all_finite, check_array, check_is_fitted

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import ASGD, SGD, Adam
from torch.utils.data import Dataset, TensorDataset, DataLoader


# TODO: Change initialization attributes into abstract properties
# (i.e., via:
#   @property
#   @abstractmethod
#   def network(self):
#       return self._network
# )



def _pytorch_model_is_fully_initialized(clf: BaseEstimator):
    if all([
        hasattr(clf, 'network'),
        hasattr(clf, 'loss_function'),
        hasattr(clf, 'optimizer'),
        hasattr(clf, 'data_loader'), 
        hasattr(clf, 'train_dset_len'),
        hasattr(clf, 'device')
    ]):
        return True
    else:
        return False

def _get_cuda_device_if_available():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class PytorchClassifier(BaseEstimator, ClassifierMixin):
    @abstractmethod
    def _init_model(self, X, y):
        pass

    def fit(self, X, y):
        """Generalizable method for fitting a PyTorch estimator to a training
        set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Target vector relative to X.
        
        Returns
        -------
        self
            Fitted estimator.
        """
        
        self._init_model(X, y)

        assert _pytorch_model_is_fully_initialized(self)
        
        for epoch in range(self.num_epochs):
            for i, (samples, labels) in enumerate(self.data_loader):
                #ipdb.set_trace()
                samples.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.network(samples)

                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                if self.verbose and ((i + 1) % 100 == 0):
                    print(
                        "Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f"
                        % (
                            epoch + 1,
                            self.num_epochs,
                            i + 1,
                            self.train_dset_len // self.batch_size,
                            loss.item(),
                        )
                    )

        self.is_fitted_ = True
        return self

    def _validate_inputs(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False)

        assert_all_finite(X, y)

        if np.any(np.iscomplex(X)) or np.any(np.iscomplex(y)):
            raise ValueError("Complex data not supported")
        if np.issubdtype(X.dtype, np.object_) or np.issubdtype(y.dtype, np.object_):
            try:
                X = X.astype(float)
                y = y.astype(int)
            except TypeError:
                raise TypeError("argument must be a string.* number")

        return (X, y)
        
    def predict(self, X):

        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        predictions = np.empty(len(X), dtype=int)
        for i, rows in enumerate(X):
            rows = Variable(rows.view(-1, self.input_size))
            outputs = self.network(rows)

            _, predicted = torch.max(outputs.data, 1)
            predictions[i] = int(predicted)
        return predictions.reshape(-1, 1)

    def transform(self, X):
        return self.predict(X)


class _LR(nn.Module):
    def __init__(self, input_size, num_classes):
        super(_LR, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out

class PytorchLRClassifier(PytorchClassifier):
    def __init__(
        self,
        num_epochs=10,
        batch_size=16,
        learning_rate=0.02,
        weight_decay=1e-4,
        verbose=True
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.verbose = verbose

    def _init_model(self, X, y):
        device = _get_cuda_device_if_available()

        X, y = self._validate_inputs(X, y)

        self.input_size = X.shape[-1]
        self.num_classes = len(set(y))

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        train_dset = TensorDataset(X, y)

        # Set parameters of the network
        self.network = _LR(self.input_size, self.num_classes).to(device)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.data_loader = DataLoader(
            train_dset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        self.train_dset_len = len(train_dset)
        self.device = device

    def _more_tags(self):
        return {'non_deterministic': True, 'binary_only': True}


TEST_SKLEARN = False
TEST_PYTORCH = True

if __name__ == "__main__":

    import warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    # Good binary classification dataset with floating features and appx. equal
    # class balance. Very high accuracy attainable using LR (>0.99 accuracy)
    X, y = fetch_data('clean2', return_X_y=True)

    if True:
        # first two features are IDs for the molecule! The decision function will just learn to look at these...
        X = X[:,2:]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    if TEST_SKLEARN:
        clf_sklearn = LogisticRegression(penalty='l2', solver='sag', max_iter=1000)
        clf_sklearn.fit(X_train, y_train)
        print("SKLEARN ACCURACY: {0:.3f}".format(clf_sklearn.score(X_test, y_test)))
        #print(clf_sklearn.coef_)

    if TEST_PYTORCH:
        clf_torch = PytorchLRClassifier()
        clf_torch.fit(X_train, y_train)
        print("PYTORCH ACCURACY: {0:.3f}".format(clf_torch.score(X_test, y_test)))