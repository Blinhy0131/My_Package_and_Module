############################################################################
'''
Version 1.2.1
Upgrade date: 2024/01/24
Now is avaliable to set max epoch for training


This is a bpnn function using pytorch built by github: Blinhy0131
you just need to input 4 thing 
1. input size  2.output size
3. training data  4. target data  //X_data and Y_data 

other parameter have the default value
default value please check Function Input Explanation below

Github link: https://github.com/Blinhy0131/My_Package_and_Module/py_module
'''
############################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error


#build model
class BPNN_network(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers, hidden_size,act_function):
        super(BPNN_network, self).__init__()

        # input layer
        self.input_layer = nn.Linear(in_features=input_size, out_features=hidden_size[0])
        self.hidden_layers = nn.ModuleList()
        self.act=eval(f"nn.{act_function[0]}")()

        # hidden layer
        for i in range(num_hidden_layers-1):
            self.hidden_layers.append(nn.Linear(in_features=hidden_size[i], out_features=hidden_size[i+1]))
            self.hidden_layers.append(eval(f"nn.{act_function[i+1]}")())  #act function

        # output layer
        self.output_layer = nn.Linear(in_features=hidden_size[-1], out_features=output_size)



    def forward(self, x):
        
        x=self.input_layer(x)
        x = self.act(x)

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)
        return x

'''
Function Input Explanation 

input_size                -> how many elements input to the model at once
output_size               -> how many elements output to the model at once
input_data                -> the data you want to input train model as input
target_data               -> the data you want to compare with model output
test_data=None            -> after training the data gonna test the model accurary, if none =input_data
test_target=None          -> after training the data gonna compare with model output, if none =input_data
save_model=False          -> if true, then save the model after training, file name will include the date and the time
num_hidden_layers=1       -> how many hidden layer in the network
hidden_size=10            -> how many neurons at the hidden layer. if input is a list, than it means each layer's neurons"
activations               -> the network activation function . if input is a list, than it means each layer's activation"
opt_type="Adam"           -> the optimizer type are gonna use at the model 
loss_type="MSELoss"       -> loss type of the model 
learning_rate=0.001       -> the optimizer learning rate
weight_decay=0            -> the optimizer weight decay
callback_each_epoch=True  -> if true will print the training loss at each epoch 
device='cpu'              -> what device are gonna use to train.  please check the cuda


'''

def bpnn_model(input_size, 
         output_size,
         input_data,
         target_data,
         test_data=None,
         test_target=None,
         save_model=False,
         max_epochs = 200,
         num_hidden_layers=1,
         hidden_size=10,
         activations="ReLU",
         opt_type="Adam",
         loss_type="MSELoss",
         learning_rate=0.001,
         weight_decay=0,
         callback_each_epoch=True,
         device='cpu'):
    
    # hidden size/activations if are not list than change 2 list
    if type(hidden_size)!=list:
        hidden_size=[hidden_size]*num_hidden_layers
        
    if type(activations) != list:
        activations = [activations]*num_hidden_layers
        
    if type(loss_type) == None:
        loss_dic={1:'MSELoss'}

    # Build model
    model = BPNN_network(input_size, output_size, num_hidden_layers, hidden_size, activations)
    model.to(device)
    
    # define loss and opt
    optimizer_args = {"lr":learning_rate, "weight_decay": weight_decay}
    opt  = eval(f"optim.{opt_type}")(model.parameters(), **optimizer_args)

    #define loss
    loss=eval(f"nn.{loss_type}")()

    # data process
    x_tensor = torch.tensor(input_data, dtype=torch.float32).view(-1,input_size).to(device)
    y_tensor = torch.tensor(target_data, dtype=torch.float32).view(-1,output_size).to(device)

    # train model
    for epoch in range(max_epochs):
        opt.zero_grad()
        # Forward pass
        y_pred = model(x_tensor)
        # Compute the loss
        loss_value = loss(y_pred, y_tensor)

        loss_value.backward()
        opt.step()
        if callback_each_epoch==True:
            print(f'Epoch [{epoch+1}/{max_epochs}], Loss: {loss_value.item():.4f}')


    #test model
    if test_data==None:
        y_p = model(x_tensor)
        test_target=target_data
    else:
        test_data = torch.tensor(test_data, dtype=torch.float32).view(-1,input_size).to(device)
        y_p = model(test_data)

    #save model
    if save_model==True:

        import datetime
        
        #save model
        model.eval()
        jit_model = torch.jit.trace(model,x_tensor)
        #get day time
        Dtime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #file name
        filename = f'model_{Dtime}.pth'
        #save_model
        torch.jit.save(jit_model, filename)

    y_p = y_p.detach().cpu().numpy()
    
    # if the value nan than mse will go wrong
    # so if the value is nan than return inf
    # it means mse is infinitely 
    if np.isnan(y_p[1]):
        return float('inf')
    
    mse=mean_squared_error(y_p, test_target)

    return mse


#testing model function is testing the parameter

def testing_model():
    
    x = np.arange(0, 10 , 1/1e3)
    y = np.cos(x)*np.sin(2*x)*np.exp(np.cos(6*x))

    mse=bpnn_model(input_data=x,
            target_data=y,
            input_size=1,
            output_size=1,
            num_hidden_layers=4,
            hidden_size=10,
            activations=["Tanh","ReLU","Tanh","ReLU"],
            opt_type="RMSprop",
            learning_rate=0.001
            )
    
    print(mse)

if __name__ == '__main__':
    testing_model()
