import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  f = np.zeros((num_train, num_classes))  # f dimensions(N, C)
  correct_calss = 0
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    row_num = X[i]
    max_f = None
    for j in range(num_classes):
      f[i,j] =  np.dot(X[i,:], W[:,j])

      if  j == y[i]:
        correct_calss = f[i,j]

      if max_f < f[i,j]:
        max_f = f[i,j]

    dW += (np.outer(X[i,:].reshape(X.shape[1],1), np.exp(f[i, :]+max_f).reshape(f.shape[1],1)))/np.sum(np.exp(f[i]+max_f))
    adjust_correct_class_dw = np.zeros_like(W)
    adjust_correct_class_dw[:,y[i]] -= X[i,:]
    dW += adjust_correct_class_dw

    loss -= np.log(np.exp(correct_calss+max_f) / np.sum(np.exp(f[i]+max_f)))
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += reg * np.sum(W * W)
  dW /= num_train
  dW += 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
 """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = np.dot(X, W) # (N, C)
  dW_formula_of_Li =np.zeros_like(f) 

  f_max_row = np.amax(f, axis=1).reshape(f.shape[0],1)
  dW_formula_of_Li += np.exp(f + f_max_row) / np.sum(np.exp(f + f_max_row), axis=1).reshape(f.shape[0],1)
  dW_formula_of_Li[np.arange(X.shape[0]), y] -= 1
  dW = np.dot(X.T, dW_formula_of_Li)
  
  numerator = np.exp(f[np.arange(f.shape[0]), y].reshape(f.shape[0],1)+f_max_row)
  loss = np.sum(-np.log(numerator / np.sum(np.exp(f+f_max_row), axis=1).reshape(f.shape[0],1)))

  loss /= X.shape[0]
  loss += reg * np.sum(W * W)
  dW /= X.shape[0]
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

