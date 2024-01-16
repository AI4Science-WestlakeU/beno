from typing import List
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from transformer import *
import pdb

def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int = None,
        output_activation: nn.Module = nn.Identity,
        activation: nn.Module = nn.ReLU) -> nn.Module:

  # Size of each layer
  layer_sizes = [input_size] + hidden_layer_sizes
  if output_size:
    layer_sizes.append(output_size)

  # Number of layers
  nlayers = len(layer_sizes) - 1

  # Create a list of activation functions and
  # set the last element to output activation function
  act = [activation for i in range(nlayers)]
  act[-1] = output_activation

  # Create a torch sequential container
  mlp = nn.Sequential()
  for i in range(nlayers):
    mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i],
                                             layer_sizes[i + 1]))
    mlp.add_module("Act-" + str(i), act[i]())

  return mlp


class Encoder(nn.Module):


  def __init__(
          self,
          nnode_in_features: int,
          nnode_out_features: int, #latent dim=128
          nedge_in_features: int,
          nedge_out_features: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,):
 
    super(Encoder, self).__init__()
    # Encode node features as an MLP
    self.node_fn = nn.Sequential(*[build_mlp(nnode_in_features,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nnode_out_features),
                                   nn.LayerNorm(nnode_out_features)])
    # Encode edge features as an MLP
    self.edge_fn = nn.Sequential(*[build_mlp(nedge_in_features,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nedge_out_features),
                                   nn.LayerNorm(nedge_out_features)])
    
    self.node_fn_inbd = nn.Sequential(*[build_mlp(nnode_in_features,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nnode_out_features),
                                   nn.LayerNorm(nnode_out_features)])
    
    self.edge_fn_inbd = nn.Sequential(*[build_mlp(nedge_in_features,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nedge_out_features),
                                   nn.LayerNorm(nedge_out_features)])

  def forward(
          self,
          x: torch.tensor,
          edge_features: torch.tensor,
          x_inbd:torch.tensor,
          edge_inbd_features: torch.tensor):

    return self.node_fn(x), self.edge_fn(edge_features), self.node_fn_inbd(x_inbd), self.edge_fn_inbd(edge_inbd_features)


class InteractionNetwork(MessagePassing):
  def __init__(
      self,
      nnode_in: int,
      nnode_out: int,
      nedge_in: int,
      nedge_out: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
      boundary_dim: int,
      trans_layer:int,
      activation: nn.Module = nn.ReLU
  ):

    # Aggregate features from neighbors
    super(InteractionNetwork, self).__init__(aggr='mean')
    # Node MLP
    self.node_fn = nn.Sequential(*[build_mlp(nnode_in + nedge_out + boundary_dim,
                                            [mlp_hidden_dim
                                             for _ in range(nmlp_layers)],  
                                             nnode_out,activation=activation),
                                   nn.LayerNorm(nnode_out)])

    self.edge_fn = nn.Sequential(*[build_mlp(nnode_in + nnode_in + nedge_in,
                                             [mlp_hidden_dim
                                              for _ in range(nmlp_layers)],
                                             nedge_out,activation=activation),
                                   nn.LayerNorm(nedge_out)])

    #boundary
    self.boundary_fn = Transformer(enc_in = 3, d_model = boundary_dim, n_heads = 2, enc_layers = trans_layer)

  def forward(self,
              x: torch.tensor,
              edge_index: torch.tensor,
              edge_features: torch.tensor,
              boundary: torch.tensor,
              
              ):

    boundary = boundary.unsqueeze(0).float()
   
    boundary = self.boundary_fn(boundary)
   
    x_residual = x.clone()
    edge_features_residual = edge_features.clone()

    x, edge_features = self.propagate(
        edge_index=edge_index, x=x, edge_features=edge_features, boundary=boundary)


    return x + x_residual, edge_features + edge_features_residual 

  def message(self,
              x_i: torch.tensor,
              x_j: torch.tensor,
              edge_features: torch.tensor,
              boundary: torch.tensor
              ) -> torch.tensor:

    # Concat edge features with a final shape of [nedges, latent_dim*3]

    edge_features = torch.cat([x_i, x_j, edge_features], dim=-1)

    edge_features = self.edge_fn(edge_features)
    return edge_features

  def update(self,
             x_updated: torch.tensor,
             x: torch.tensor,
             edge_features: torch.tensor,
             boundary: torch.tensor
             ):
 
    # Concat node features with a final shape of
    boundary_all = boundary.repeat(x.shape[0], 1)
    x_updated = torch.cat([x_updated, x,boundary_all], dim=-1)
    x_updated = self.node_fn(x_updated)

    return x_updated, edge_features

class Processor(MessagePassing):


  def __init__(
      self,
      nnode_in: int,
      nnode_out: int,
      nedge_in: int,
      nedge_out: int,
      nmessage_passing_steps: int,
      nmlp_layers: int,
      mlp_hidden_dim: int,
      boundary_dim: int,
      trans_layer:int
  ):

    super(Processor, self).__init__(aggr='mean')
    # Create a stack of M Graph Networks GNs.
    self.gnn_stacks = nn.ModuleList([
        InteractionNetwork(
            nnode_in=nnode_in,
            nnode_out=nnode_out,
            nedge_in=nedge_in,
            nedge_out=nedge_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            boundary_dim=boundary_dim,
            trans_layer=trans_layer
        ) for _ in range(nmessage_passing_steps)])
    
    self.gnn_stacks_inbd = nn.ModuleList([
        InteractionNetwork(
            nnode_in=nnode_in,
            nnode_out=nnode_out,
            nedge_in=nedge_in,
            nedge_out=nedge_out,
            nmlp_layers=nmlp_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            boundary_dim=boundary_dim,
            trans_layer=trans_layer
        ) for _ in range(nmessage_passing_steps)])


  def forward(self,
              x: torch.tensor,
              edge_index: torch.tensor,
              edge_features: torch.tensor,
              boundary:torch.tensor,
              x_inbd: torch.tensor,
              edge_inbd_index: torch.tensor,
              edge_inbd_features: torch.tensor,
              boundary_inbd:torch.tensor):  

    for gnn in self.gnn_stacks:
      x, edge_features= gnn(x, edge_index, edge_features,boundary )  
      
    for gnn in self.gnn_stacks_inbd:
      x_inbd, edge_inbd_features = gnn(x_inbd, edge_inbd_index, edge_inbd_features,boundary_inbd)  
    
    return x, edge_features, x_inbd, edge_inbd_features

class Decoder(nn.Module):


  def __init__(
          self,
          nnode_in: int,
          nnode_out: int,
          nmlp_layers: int,
          mlp_hidden_dim: int):

    super(Decoder, self).__init__()
    self.node_fn = build_mlp(
        nnode_in, [mlp_hidden_dim for _ in range(nmlp_layers)], nnode_out)
    self.node_fn_inbd = build_mlp(
        nnode_in, [mlp_hidden_dim for _ in range(nmlp_layers)], nnode_out)

  def forward(self,
              x: torch.tensor,
              x_inbd: torch.tensor):
    u1=self.node_fn(x)
    u2=self.node_fn_inbd(x_inbd)
    u=u1+u2

    return u


class HeteroGNS(nn.Module):
  def __init__(
      self,
      nnode_in_features: int,
      nnode_out_features: int,
      nedge_in_features: int,
      latent_dim: int = 128,
      nmessage_passing_steps: int = 10,
      nmlp_layers: int = 2,
      mlp_hidden_dim: int = 128,
      activation: nn.Module = nn.ELU,
      boundary_dim: int=128,
      trans_layer: int =3,
  ):

    super(HeteroGNS, self).__init__()
    self._encoder = Encoder(
        nnode_in_features=nnode_in_features,
        nnode_out_features=latent_dim,
        nedge_in_features=nedge_in_features,
        nedge_out_features=latent_dim,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,

        
    )
    self._processor = Processor(
        nnode_in=latent_dim,
        nnode_out=latent_dim,
        nedge_in=latent_dim,
        nedge_out=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        boundary_dim = boundary_dim,
        trans_layer = trans_layer,
    )
    self._decoder = Decoder(
        nnode_in=latent_dim,
        nnode_out=nnode_out_features,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim,
        
    )

  def forward(self,data):

    x, edge_index, edge_features,boundary = data['G1'].x, data['G1'].edge_index, data['G1'].edge_features,data['G1'].boundary
    x_inbd, edge_inbd_index, edge_inbd_features,boundary_inbd = data['G2'].x, data['G2'].edge_index,data['G2'].edge_features,data['G2'].boundary

    x, edge_features,x_inbd,edge_inbd_features = self._encoder(x, edge_features,x_inbd,edge_inbd_features)

    x, edge_features, x_inbd, edge_inbd_features = self._processor(x, edge_index, edge_features, boundary,x_inbd,edge_inbd_index, edge_inbd_features,boundary_inbd)

    u = self._decoder(x,x_inbd)

    return u