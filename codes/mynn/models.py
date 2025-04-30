from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, conv_params=None, fc_params=None, act_func='ReLU'):
        super().__init__()
        self.conv_params = conv_params
        self.fc_params = fc_params
        self.act_func = act_func
        
        if conv_params is not None and fc_params is not None:
            self.layers = []
            
            # Add convolutional layers
            for i, (in_channels, out_channels, kernel_size, stride, padding, weight_decay_lambda) in enumerate(conv_params):
                # Create a conv2D layer with given parameters
                conv_layer = conv2D(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding)
                
                if weight_decay_lambda is not None:
                    # Enable weight decay and set lambda if provided
                    conv_layer.weight_decay = True
                    conv_layer.weight_decay_lambda = weight_decay_lambda
                
                self.layers.append(conv_layer)
                
                # Add activation function after each conv layer except the last one
                if i < len(conv_params) - 1 or len(fc_params) > 0:
                    if act_func == 'ReLU':
                        self.layers.append(ReLU())
                    else:
                        raise NotImplementedError(f"Activation function {act_func} not implemented")
            
            # Add fully connected layers
            for i, (in_dim, out_dim, weight_decay_lambda) in enumerate(fc_params):
                # Create a Linear (fully connected) layer
                fc_layer = Linear(in_dim=in_dim, out_dim=out_dim)
                
                if weight_decay_lambda is not None:
                    # Enable weight decay and set lambda if provided
                    fc_layer.weight_decay = True
                    fc_layer.weight_decay_lambda = weight_decay_lambda
                
                self.layers.append(fc_layer)
                
                # Add activation function after each FC layer except the last one
                if i < len(fc_params) - 1:
                    if act_func == 'ReLU':
                        self.layers.append(ReLU())
                    else:
                        raise NotImplementedError(f"Activation function {act_func} not implemented")

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.layers is not None, 'Model has not been initialized yet. Use model.load_model to load a model or create a new model with conv_params and fc_params.'
        
        outputs = X
        self.shapes = [outputs.shape]  # Store the input shape for later reshape in backward
        
        for i, layer in enumerate(self.layers):
            layer_name = layer.__class__.__name__
            # If the current layer is Linear and input is not flat, flatten it
            if isinstance(layer, Linear) and len(outputs.shape) > 2:
                outputs = outputs.reshape(outputs.shape[0], -1)
            
            # Forward pass through the current layer
            outputs = layer(outputs)
            
            # Store the output shape after each layer for backward reshape
            self.shapes.append(outputs.shape)
        
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        reshape_indices = []
        
        # Identify the indices where reshape is needed (from conv2D to Linear)
        for i in range(len(self.layers)):
            if i > 0 and isinstance(self.layers[i-1], conv2D) and isinstance(self.layers[i], Linear):
                reshape_indices.append(i)
        
        # Backward pass through all layers in reverse order
        for i, layer in enumerate(reversed(self.layers)):
            layer_idx = len(self.layers) - 1 - i
            layer_name = layer.__class__.__name__
            
            # If this layer is right after a conv2D and before a Linear, reshape grads
            if layer_idx in reshape_indices:
                conv_output_shape = self.shapes[layer_idx]
                grads = grads.reshape(conv_output_shape)
            
            # If current layer is ReLU and previous is conv2D, ensure grads shape matches
            if isinstance(layer, ReLU):
                next_layer_idx = layer_idx - 1
                if next_layer_idx >= 0 and isinstance(self.layers[next_layer_idx], conv2D):
                    if len(grads.shape) != len(self.shapes[layer_idx]):
                        grads = grads.reshape(self.shapes[layer_idx])

            # Backward pass through the current layer
            grads = layer.backward(grads)
        
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        
        self.conv_params = param_list[0]
        self.fc_params = param_list[1]
        self.act_func = param_list[2]
        
        self.layers = []
        param_idx = 3
        
        # Rebuild convolutional layers
        if self.conv_params:
            for in_channels, out_channels, kernel_size, stride, padding, _ in self.conv_params:
                # Create a conv2D layer with saved parameters
                conv_layer = conv2D(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding)
                
                # Load weights and bias from saved model
                conv_layer.W = param_list[param_idx]['W']
                conv_layer.b = param_list[param_idx]['b']
                conv_layer.params['W'] = conv_layer.W
                conv_layer.params['b'] = conv_layer.b
                conv_layer.weight_decay = param_list[param_idx]['weight_decay']
                conv_layer.weight_decay_lambda = param_list[param_idx]['lambda']
                
                self.layers.append(conv_layer)
                param_idx += 1
                
                # Add activation layer after each conv layer
                if self.act_func == 'ReLU':
                    self.layers.append(ReLU())
                else:
                    raise NotImplementedError(f"Activation function {self.act_func} not implemented")
        
        # Rebuild fully connected layers
        if self.fc_params:
            # Remove the last activation if we're adding FC layers after conv layers
            if self.conv_params and self.layers[-1].__class__.__name__ == 'ReLU':
                self.layers.pop()
                
            for i, (in_dim, out_dim, _) in enumerate(self.fc_params):
                # Create a Linear layer with saved parameters
                fc_layer = Linear(in_dim=in_dim, out_dim=out_dim)
                
                # Load weights and bias from saved model
                fc_layer.W = param_list[param_idx]['W']
                fc_layer.b = param_list[param_idx]['b']
                fc_layer.params['W'] = fc_layer.W
                fc_layer.params['b'] = fc_layer.b
                fc_layer.weight_decay = param_list[param_idx]['weight_decay']
                fc_layer.weight_decay_lambda = param_list[param_idx]['lambda']
                
                self.layers.append(fc_layer)
                param_idx += 1
                
                # Add activation after each FC layer except the last one
                if i < len(self.fc_params) - 1 and self.act_func == 'ReLU':
                    self.layers.append(ReLU())
        
    def save_model(self, save_path):
        param_list = [self.conv_params, self.fc_params, self.act_func]
        
        for layer in self.layers:
            if layer.optimizable:
                # Save weights, bias, and regularization parameters for each optimizable layer
                param_list.append({
                    'W': layer.params['W'], 
                    'b': layer.params['b'], 
                    'weight_decay': layer.weight_decay, 
                    'lambda': layer.weight_decay_lambda
                })
        
        with open(save_path, 'wb') as f:
            # Serialize all parameters to file
            pickle.dump(param_list, f)