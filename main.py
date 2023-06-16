import numpy as np
import torch
from torch.nn import Linear, Parameter, ReLU, Flatten
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.loader import DataLoader
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_data(num_samples, num_antennas, num_classes, snr_dB):
    # BPSK constellation
    constellation = np.array([-1, 1])

    # Generate transmit symbols
    transmit_symbols = np.random.choice(constellation, size=(num_samples, num_antennas))
    # Generate channel matrix
    channel_matrix = np.random.randn(num_samples, num_antennas, num_antennas)

    # Generate noise
    snr = 10 ** (snr_dB / 10)
    noise_variance = 1 / (2 * snr)
    noise = np.sqrt(noise_variance) * np.random.randn(num_samples, num_antennas)

    # Generate received signal
    received_signal = np.matmul(transmit_symbols[:, np.newaxis, :], channel_matrix) + noise[:, np.newaxis, :]

    # Normalize received signal (optional)
    # received_signal /= np.sqrt(np.mean(np.abs(received_signal) ** 2))

    # One-hot encode target labels
    target_labels = np.zeros((num_samples, num_antennas, num_classes))
    for i in range(num_samples):
        for j in range(num_antennas):
            target_labels[i, j, np.where(constellation == transmit_symbols[i, j])] = 1

    return received_signal, channel_matrix, transmit_symbols, target_labels


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col.to(torch.int64), x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row.to(torch.int64)] * deg_inv_sqrt[col.to(torch.int64)]

        # Step 4-5: Start propagating messages.
        out_gnconv = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out_gnconv += self.bias

        return out_gnconv

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class GNN(torch.nn.Module):
    def __init__(self, number_of_features):
        super().__init__()
        self.conv1 = GCNConv(number_of_features, 4)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(4, 8)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(8, 16)
        self.relu3 = ReLU()
        #self.flatit = Flatten()
        self.lin1 = Linear(128, 64)
        self.relu4 = ReLU()
        self.lin2 = Linear(64, 16)
        self.double()

    def forward(self, data):
        x, edge_index = data.x.double(), data.edge_index

        x = self.conv1(x, edge_index)
        x = self.relu1(x)
        x = self.conv2(x, edge_index)
        x = self.relu2(x)
        x = self.conv3(x, edge_index)
        x = self.relu3(x)
        #x = self.flatit(x)
        x = self.lin1(torch.reshape(x, (1, -1)))
        x = self.relu4(x)
        out = self.lin2(x)

        return out


num_samples = 10000
num_antennas = 4
num_classes = 2
snr_dB = 20

received_signal, channel_matrix, transmit_symbols, target_labels = generate_data(num_samples, num_antennas, num_classes, snr_dB)
graph_samples = []

for i in range(num_samples):
    adj_mat = np.concatenate((np.ones((num_antennas, num_antennas)) - np.eye(num_antennas), np.zeros((num_antennas, num_antennas))), axis=1)
    temp = np.concatenate((np.ones((num_antennas, num_antennas)), np.zeros((num_antennas, num_antennas))), axis=1)
    adj_mat = np.concatenate((adj_mat, temp), axis=0)
    node_features = np.concatenate((received_signal[i].reshape((-1, 1)), transmit_symbols[i].reshape((-1, 1))), axis=0)

    graph = nx.DiGraph()
    for n in range(2*num_antennas):
        graph.add_node(n, x=node_features[n])

    for n1 in range(num_antennas*2):
        for n2 in range(num_antennas*2):
            if adj_mat[n1, n2] == 1:
                graph.add_edge(n1, n2, edge_weight=1)

    pyg_graph = from_networkx(graph)
    graph_samples.append(pyg_graph)

val_received_signal, val_channel_matrix, val_transmit_symbols, val_target_labels = generate_data(1000, num_antennas, num_classes, snr_dB)
val_graph_samples = []

for i in range(1000):
    val_adj_mat = np.concatenate((np.ones((num_antennas, num_antennas)) - np.eye(num_antennas), np.zeros((num_antennas, num_antennas))), axis=1)
    temp = np.concatenate((np.ones((num_antennas, num_antennas)), np.zeros((num_antennas, num_antennas))), axis=1)
    val_adj_mat = np.concatenate((val_adj_mat, temp), axis=0)
    val_node_features = np.concatenate((received_signal[i].reshape((-1, 1)), transmit_symbols[i].reshape((-1, 1))), axis=0)

    val_graph = nx.DiGraph()
    for n in range(2*num_antennas):
        val_graph.add_node(n, x=val_node_features[n])

    for n1 in range(num_antennas*2):
        for n2 in range(num_antennas*2):
            if val_adj_mat[n1, n2] == 1:
                val_graph.add_edge(n1, n2, edge_weight=1)

    pyg_graph = from_networkx(val_graph)
    val_graph_samples.append(pyg_graph)


gnn = GNN(graph_samples[0].x.size(1))
loss_fn = torch.nn.MSELoss()
loss_fn_val = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=gnn.parameters(), lr=0.001)
#data_loader = DataLoader(graph_samples, batch_size=50, shuffle=True)
MSE = []
val_MSE = []
for ind in tqdm(range(100)):
    #H_predict = GNN(graph_samples[ind].x.double(), graph_samples[ind].edge_index)
    H_predict = gnn(graph_samples[ind])
    y = graph_samples[ind].x[0:num_antennas]
    x = graph_samples[ind].x[num_antennas:]
    Hx = torch.matmul(torch.reshape(H_predict, (num_antennas, num_antennas)), x)
    loss = loss_fn(y, Hx)
    MSE.append(loss.detach().numpy())
    loss.backward()
    optimizer.step()
    temp = []
    for i in range(len(val_graph_samples)):
        H_predict_val = gnn(val_graph_samples[ind])
        y_val = val_graph_samples[ind].x[0:num_antennas]
        x_val = val_graph_samples[ind].x[num_antennas:]
        Hx_val = torch.matmul(torch.reshape(H_predict_val, (num_antennas, num_antennas)), x_val)
        loss_val = loss_fn_val(y_val, Hx_val)
        temp.append(loss_val.detach().numpy())

    val_MSE.append(np.mean(np.array(temp)))

plt.plot(range(100), np.array(MSE).reshape((-1, 1)), label='Training Loss')
plt.plot(range(100), np.array(val_MSE).reshape(-1, 1), label='Validation Loss')
plt.grid()
plt.legend()
plt.title('MSE')
plt.show()
