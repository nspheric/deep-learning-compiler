import torch
import torch.nn as nn
import torch.fx as fx
from model import VGG16

# this should be enough to do the basic optimizations
class DAG:
    def __init__(self, nodes):
        self.nodes = nodes
    def get_nodes(self):
        return self.nodes
    
class Node:
    def __init__(self, name, edges):
        self.name = name
        self.edges = edges
    def get_edges(self):
        return self.edges
        

# in a deep learning compiler the deep learning model
# is represented as a dag where the operations such as
# relu and convolution are the nodes and the edges are
# tensors, i.e., weights and bias

def torch_to_dag(model, input_tensor):
    activations = {}
    
    for name, module in model.named_modules():
        def hook_fn(mod, _input, _output, name=name):
            activations[name] = _input[0]
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d,
                               nn.ReLU, nn.MaxPool2d, nn.Linear,
                               nn.Dropout)):
            module.register_forward_hook(hook_fn)
            
    model(input_tensor)

    traced_model = fx.symbolic_trace(model)
    nodes = list()

    for node in traced_model.graph.nodes:  
        if node.op == 'call_module':
            layer_name = node.target
            layer = getattr(model, layer_name) 
            if isinstance(layer, nn.Conv2d):
                input_tensor = activations.get(layer_name)
                weights = layer.weight.data
                bias = layer.bias.data
                edges = [{"input_tensor": input_tensor},
                         {"weights": weights},
                         {"bias": bias}]
                nodes.append(Node("conv2d", edges))    
            elif isinstance(layer, nn.BatchNorm2d):
                input_tensor = activations.get(layer_name)
                edges = [{"input_tensor": input_tensor},
                         {"weights": layer.weight.data},
                         {"bias": layer.bias.data}]
                nodes.append(Node("batchnorm2d", edges))     
            elif isinstance(layer, nn.ReLU):
                input_tensor = activations.get(layer_name)
                edges = [{"input_tensor": input_tensor}]
                nodes.append(Node("relu", edges))    
            elif isinstance(layer, nn.MaxPool2d):
                input_tensor = activations.get(layer_name)
                edges = [{"input_tensor": input_tensor}]
                nodes.append(Node("max_pool2d", edges))   
            elif isinstance(layer, nn.Dropout):
                input_tensor = activations.get(layer_name)
                edges = [{"input_tensor": input_tensor}]
                nodes.append(Node("dropout", edges))
            else:
                input_tensor = activations.get(layer_name)
                weights = layer.weight.data
                bias = layer.bias.data
                edges = [{"weights": weights},
                         {"bias": bias}]
                nodes.append(Node("Linear", edges))

    dag = DAG(nodes)
    return dag

vgg = VGG16(10)
input_tensor = torch.randn(1,3,224,224)
dag = torch_to_dag(vgg, input_tensor)

for node in dag.get_nodes():
    print(node.get_edges())
    
