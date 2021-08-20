import numpy as np
from random import shuffle
from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train=X.shape[0]
  num_classes=W.shape[1]
  num_dims=W.shape[0]
  for i in range(num_train):
      vector=X[i,:].dot(W)
      vector_exp=np.exp(vector)
      fraccion=vector_exp/np.sum(vector_exp)
      loss += -np.log(fraccion[y[i]])
        
      for j in range(num_dims):
          for k in range(num_classes):
              if k== y[i]:
                dW[j,k] += X.T[j,i]*(fraccion[k]-1)
              else:
                dW[j,k] += X.T[j,i]*fraccion[k]

  loss= loss/num_train
  loss+= 0.5*reg*np.sum(W*W)
    
  dW= dW/num_train
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
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
  num_train=X.shape[0]

  matriz=X.dot(W)
  matriz_exp=np.exp(matriz)
  fraccion=matriz_exp/np.sum(matriz_exp, axis=1, keepdims=True)
  log_fraccion=-np.log(fraccion[range(num_train),y])
  loss=np.sum(log_fraccion)
  loss/=num_train
  loss+=0.5*reg*np.sum(W*W)

  dscores = fraccion
  dscores[range(num_train), y] -= 1
  dW = np.dot(X.T, dscores)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

