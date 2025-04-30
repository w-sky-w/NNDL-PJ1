from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):        
        self.input = X  # Store input for use in backward pass
        # Compute linear transformation: XW + b
        output = np.dot(X, self.params['W']) + self.params['b']
        
        return output

    def backward(self, grad : np.ndarray):
        batch_size = self.input.shape[0]
        
        # Compute gradient with respect to weights: dL/dW = X^T * dL/dY
        dW = np.dot(self.input.T, grad)
        if self.weight_decay:
            # Add L2 regularization gradient if weight decay is enabled
            dW += self.weight_decay_lambda * self.params['W']
        # Compute gradient with respect to bias: sum of gradients across batch
        db = np.sum(grad, axis=0, keepdims=True)
        # Compute gradient with respect to input: dL/dX = dL/dY * W^T
        dX = np.dot(grad, self.params['W'].T)
        
        # Store gradients for optimizer
        self.grads['W'] = dW
        self.grads['b'] = db

        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding

        # Initialize filters: [out_channels, in_channels, kernel_height, kernel_width]
        self.W = initialize_method(size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.b = initialize_method(size=(out_channels,))
        
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.params = {'W': self.W, 'b': self.b}
        
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):        
        self.input = X  # Store input for backward pass
        batch_size, _, in_h, in_w = X.shape
        
        # Calculate output dimensions based on kernel size and stride
        out_h = (in_h - self.kernel_size[0]) // self.stride + 1
        out_w = (in_w - self.kernel_size[1]) // self.stride + 1
        
        # Initialize output tensor with zeros
        output = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        # Perform convolution operation
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h_out in range(out_h):
                    h_start = h_out * self.stride  # Calculate starting height position
                    for w_out in range(out_w):
                        w_start = w_out * self.stride  # Calculate starting width position
                        
                        # Extract current patch from input volume
                        patch = X[b, :, h_start:h_start+self.kernel_size[0], w_start:w_start+self.kernel_size[1]]
                        
                        # Compute convolution as element-wise multiplication and sum
                        # Add bias term for current output channel
                        output[b, c_out, h_out, w_out] = np.sum(patch * self.params['W'][c_out]) + self.params['b'][c_out]

        return output

    def backward(self, grads):      
        batch_size, _, out_h, out_w = grads.shape
        _, _, in_h, in_w = self.input.shape
        
        # Initialize gradient tensors
        dW = np.zeros_like(self.params['W'])
        db = np.zeros_like(self.params['b'])
        dX = np.zeros_like(self.input)
        
        # Compute gradients for each element
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h_out in range(out_h):
                    h_start = h_out * self.stride
                    for w_out in range(out_w):
                        w_start = w_out * self.stride
                        
                        # Get current input patch
                        patch = self.input[b, :, h_start:h_start+self.kernel_size[0], w_start:w_start+self.kernel_size[1]]
                        
                        # Compute gradient for kernel weights: dL/dW = X * dL/dY
                        dW[c_out] += patch * grads[b, c_out, h_out, w_out]
                        
                        # Compute gradient for bias: dL/db = dL/dY
                        db[c_out] += grads[b, c_out, h_out, w_out]
                        
                        # Compute gradient for input: dL/dX = dL/dY * W
                        dX[b, :, h_start:h_start+self.kernel_size[0], w_start:w_start+self.kernel_size[1]] += \
                            self.params['W'][c_out] * grads[b, c_out, h_out, w_out]
        
        # Add L2 regularization gradient if weight decay is enabled
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.params['W']
            
        # Store computed gradients
        self.grads['W'] = dW
        self.grads['b'] = db

        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.softmax_output = None
        self.labels = None
        self.batch_size = None
        self.grads = None

        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.batch_size = predicts.shape[0]
        self.labels = labels
        
        # Apply softmax if needed
        if self.has_softmax:
            self.softmax_output = softmax(predicts)
        else:
            self.softmax_output = predicts
            
        # Convert integer labels to one-hot encoded vectors
        y_one_hot = np.zeros((self.batch_size, self.max_classes))
        for i in range(self.batch_size):
            y_one_hot[i, labels[i]] = 1
        
        # Compute cross-entropy loss with numerical stability
        epsilon = 1e-10  # Small constant to prevent log(0)
        log_likelihood = -np.log(self.softmax_output + epsilon)
        # Average loss over batch
        loss = np.sum(y_one_hot * log_likelihood) / self.batch_size
        
        return loss
    
    def backward(self):
        # Convert labels to one-hot encoding
        y_one_hot = np.zeros((self.batch_size, self.max_classes))
        for i in range(self.batch_size):
            y_one_hot[i, self.labels[i]] = 1
        
        # Compute gradient of cross-entropy loss with respect to softmax outputs
        # dL/dx = (softmax(x) - y) / batch_size
        self.grads = (self.softmax_output - y_one_hot) / self.batch_size
        
        # Propagate gradients through the model
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, lambd=1e-4):
        super().__init__()
        self.lambd = lambd
        self.optimizable = False
        
    def __call__(self, model):
        return self.forward(model)
        
    def forward(self, model):
        """
        Calculate L2 regularization loss
        """
        reg_loss = 0
        for layer in model.layers:
            if hasattr(layer, 'params') and layer.optimizable:
                W = layer.params.get('W', None)
                if W is not None:
                    reg_loss += 0.5 * self.lambd * np.sum(W * W)
        return reg_loss
    
    def backward(self, model):
        """
        Add regularization gradients to each layer's weight gradients
        """
        for layer in model.layers:
            if hasattr(layer, 'params') and layer.optimizable:
                W = layer.params.get('W', None)
                if W is not None and layer.grads['W'] is not None:
                    layer.grads['W'] += self.lambd * W
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition