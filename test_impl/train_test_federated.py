import torch
import copy
from tqdm import tqdm


def federated_average(params):
    """
    Average the parameters from each client to update the global model.
    
    :param params: list of parameters from each client's model
    :return: averaged parameters from all clients
    """
    global_params = copy.deepcopy(params[0])
    for key in global_params.keys():
        for param in params[1:]:
            global_params[key] += param[key]
        global_params[key] = torch.div(global_params[key], len(params))
    return global_params


def train_local_model(model, device, dataloader, num_iterations, lr=0.001):
    """
    Train a local model for a given client.
    
    :param model: local model (copy of global model)
    :param device: device to train on (GPU/CPU)
    :param dataloader: training data for this client
    :param num_iterations: number of local training iterations
    :param lr: learning rate
    :return: (model parameters, average training loss)
    """
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    total_loss = 0.0
    
    for iteration in range(num_iterations):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            epoch_loss += loss.item() * data.size(0)
            loss.backward()
            optimizer.step()
        
        total_loss += epoch_loss / len(dataloader.dataset)
    
    avg_loss = total_loss / num_iterations
    return model.state_dict(), avg_loss


def federated_learning(global_model, client_dataloaders, device, 
                       num_epochs, local_iterations, lr=0.001):
    """
    Run federated learning across multiple clients.
    
    :param global_model: initial global model
    :param client_dataloaders: list of dataloaders, one per client
    :param device: device to train on (GPU/CPU)
    :param num_epochs: number of federated learning rounds
    :param local_iterations: number of local training iterations per round
    :param lr: learning rate
    :return: (trained global model, list of training losses per epoch)
    """
    num_clients = len(client_dataloaders)
    all_train_losses = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nFederated Learning Round: {epoch}/{num_epochs}")
        
        local_params = []
        local_losses = []
        
        # Train on each client
        for client_id in range(num_clients):
            print(f"  Training client {client_id + 1}/{num_clients}")
            
            # Create a copy of global model for this client
            local_model = copy.deepcopy(global_model)
            
            # Train locally
            params, loss = train_local_model(
                local_model, 
                device, 
                client_dataloaders[client_id],
                local_iterations,
                lr
            )
            
            local_params.append(params)
            local_losses.append(loss)
        
        # Aggregate parameters using federated averaging
        global_params = federated_average(local_params)
        global_model.load_state_dict(global_params)
        
        # Record average loss across clients
        avg_loss = sum(local_losses) / len(local_losses)
        all_train_losses.append(avg_loss)
        print(f"  Average training loss: {avg_loss:.4f}")
    
    return global_model, all_train_losses


# Example usage:
if __name__ == "__main__":
    """
    Example of how to use the federated learning framework:
    
    # 1. Define your model
    model = MyCNNModel()
    
    # 2. Prepare client dataloaders
    client_dataloaders = [
        DataLoader(client1_dataset, batch_size=32, shuffle=True),
        DataLoader(client2_dataset, batch_size=32, shuffle=True),
        # ... more clients
    ]
    
    # 3. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 4. Run federated learning
    trained_model, losses = federated_learning(
        global_model=model,
        client_dataloaders=client_dataloaders,
        device=device,
        num_epochs=10,
        local_iterations=2,
        lr=0.001
    )
    
    # 5. Evaluate or save your model
    torch.save(trained_model.state_dict(), 'federated_model.pth')
    """
    pass