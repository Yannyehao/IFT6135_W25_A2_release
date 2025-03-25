import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

########################################################################################
########################################################################################

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """
         Implements a single-layer LSTM cell.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden features.
            bias (bool): Whether to use bias in linear layers. Default: True.
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size  
        self.hidden_size = hidden_size   
        self.bias = bias
        
        # The combined input dimension will be input_size + hidden_size
        combined_dim = input_size + hidden_size  # int
        
        # Separate linear layers for each gate:
        self.input_gate = nn.Linear(combined_dim, hidden_size, bias=bias)       # [W_i, U_i] and b_i
        self.forget_gate = nn.Linear(combined_dim, hidden_size, bias=bias)      # [W_f, U_f] and b_f
        self.output_gate = nn.Linear(combined_dim, hidden_size, bias=bias)      # [W_o, U_o] and b_o
        self.candidate_cell = nn.Linear(combined_dim, hidden_size, bias=bias)   # [W_c, U_c] and b_c
        
        self.reset_parameters()

    def reset_parameters(self):
        """ Initializes weights using Xavier and biases to zero. """
        for layer in [self.input_gate, self.forget_gate, self.candidate_cell, self.output_gate]:
            nn.init.xavier_uniform_(layer.weight)
            if self.bias:
                nn.init.zeros_(layer.bias)

    def forward(self, x, hx):
        """
        Computes a single step of LSTM.

        Args:
            x (torch.Tensor): Input at time t of shape (batch_size, input_size).
            hx (torch.Tensor, torch.Tensor): Tuple of hidden states (h, c) at time t-1, each of shape (batch_size, hidden_size).

        Returns:
            h_t (torch.Tensor): New hidden state (batch_size, hidden_size).
            c_t (torch.Tensor): New cell state (batch_size, hidden_size).
        """
        h, c = hx  # (batch_size, hidden_size), (batch_size, hidden_size)
        
        # Concatenate input and previous hidden state along the feature dimension
        combined = torch.cat([x, h], dim=1)  # (batch_size, input_size + hidden_size)

        # ==========================
        # TODO: Write your code here
        # ==========================
        
        # Compute the input, forget, output gates, and candidate cell
        i_t = torch.sigmoid(self.input_gate(combined)) 
        f_t = torch.sigmoid(self.forget_gate(combined))
        o_t = torch.sigmoid(self.output_gate(combined))
        g_t = torch.tanh(self.candidate_cell(combined))
        
        # Update the cell and hidden states
        c_t = f_t * c + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

########################################################################################
########################################################################################

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, dropout=0.0):
        """
        Implements a multi-layer LSTM by stacking multiple LSTM cells.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden features.
            num_layers (int): Number of stacked LSTM layers.
            bias (bool): Whether to use bias in linear layers. Default: True.
            dropout (float): Dropout probability applied between layers.
            device (torch.device, optional): Device for computations.
            dtype (torch.dtype, optional): Data type.
        """
        super(LSTM, self).__init__()
        self.input_size = input_size        
        self.hidden_size = hidden_size      
        self.num_layers = num_layers        
        self.bias = bias
        self.dropout = dropout

        # Create LSTM layers
        self.layers = nn.ModuleList([
            LSTMCell(input_size if layer == 0 else hidden_size, hidden_size, bias) # input_size for the first layer, hidden_size for the rest
            for layer in range(num_layers)
        ])

        # Dropout layer to apply between LSTM layers (except the last)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, hx=None):
        """
        Processes an entire sequence using stacked LSTM layers.

        If hx is not provided, both h_0 and c_0 are set to zero tensors 
        of same type and on the same device as the input.

        Args:
            x (torch.Tensor): Input sequence (batch_size, seq_len, input_size).
            hx (tuple, optional): Initial hidden states (h_0, c_0) for each layer, each of shape (num_layers, batch_size, hidden_size).

        Returns:
            outputs (torch.Tensor): Sequence outputs (batch_size, seq_len, hidden_size).
            h_n (torch.Tensor): Final hidden states (num_layers, batch_size, hidden_size).
            c_n (torch.Tensor): Final cell states (num_layers, batch_size, hidden_size).
        """
        batch_size, seq_len, _ = x.size() 
        
        # Initialize hidden and cell states if not provided.
        if hx is None:
            # ==========================
            # TODO: Write your code here
            # ==========================
            device = x.device
            dtype = x.dtype
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)
        else:
            h0, c0 = hx  # (num_layers, batch_size, hidden_size), (num_layers, batch_size, hidden_size)

        output = x  # (batch_size, seq_len, hidden_size) for layer 0 input
        final_hidden_states = []
        final_cell_states = []

        # Process the input sequence layer by layer
        for layer_idx, cell in enumerate(self.layers):
            h_t = h0[layer_idx]  # (batch_size, hidden_size)
            c_t = c0[layer_idx]  # (batch_size, hidden_size)
            layer_outputs = []
            
            # Iterate over the time steps
            for t in range(seq_len):
                # ==========================
                # TODO: Write your code here
                # ==========================
                # Extract x_t from "output" tensor, and compute h_t, c_t using the LSTM "cell" based on x_t, h_t, and c_t

                x_t = output[:, t, :]  # (batch_size, input_size) if layer_idx == 0, (batch_size, hidden_size) otherwise
                h_t, c_t = cell(x_t,(h_t,c_t))  # (batch_size, hidden_size), (batch_size, hidden_size)

                layer_outputs.append(h_t.unsqueeze(1))  # (batch_size, 1, hidden_size)
            
            # Concatenate time outputs: (batch_size, seq_len, hidden_size)
            output = torch.cat(layer_outputs, dim=1)  # (batch_size, seq_len, hidden_size)
            final_hidden_states.append(h_t.unsqueeze(0))  # (1, batch_size, hidden_size)
            final_cell_states.append(c_t.unsqueeze(0))  # (1, batch_size, hidden_size)

            # Apply dropout between layers (except for the last layer)
            if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                output = self.dropout_layer(output)  # (batch_size, seq_len, hidden_size)
                
        h_n = torch.cat(final_hidden_states, dim=0)  # (num_layers, batch_size, hidden_size)
        c_n = torch.cat(final_cell_states, dim=0)  # (num_layers, batch_size, hidden_size)

        # Return the full sequence output, and the final (h, c) states
        return output, (h_n, c_n)

########################################################################################
########################################################################################

class LSTMLM(nn.Module):
    """
    This is a Long Short-Term Memory network for language modeling. 
    This module returns for each position in the sequence the log-probabilities of the next token. 
    The module also returns the hidden states of the LSTM.
    """
    def __init__(
        self, 
        vocabulary_size: int, 
        embedding_size: int, 
        hidden_size: int,  
        num_layers: int,  
        dropout: float=0.0,
        padding_index: int=None,
        bias_lstm: bool=True,
        bias_classifier: bool=True,
        share_embeddings: bool=False
        ):

        super(LSTMLM, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocabulary_size, embedding_size, padding_idx=padding_index)

        self.lstm = LSTM(
            input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, 
            bias=bias_lstm, dropout=dropout, 
        )

        self.classifier = nn.Linear(hidden_size, vocabulary_size, bias=bias_classifier)
        if share_embeddings:
            # Tying classifier and embedding weights (similar to GPT-1)
            assert embedding_size == hidden_size, "Embedding size and hidden size must be equal to share embeddings"
            self.classifier.weight = self.embedding.weight

    def forward(self, x, hidden_states=None):
        """
        LSTM forward pass.

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        hidden_states (`tuple` of size 2)
            The (initial) hidden state. This is a tuple containing
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)

        Returns
        -------
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocabulary_size)`)
            A tensor containing the logits of the next token for all positions in each sequence of the batch.

        hidden_states (`tuple` of size 2)
            The final hidden state. This is a tuple containing
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
        """

        # ==========================
        # TODO: Write your code here
        # ==========================

        
       
        x_embed = self.embedding(x)
        
        lstm_output, (h_n, c_n) = self.lstm(x_embed, hidden_states)
        
        logits = self.classifier(lstm_output)
        
        return logits, (h_n, c_n)

########################################################################################
########################################################################################

if __name__ == "__main__":
    
    # Data
    vocabulary_size=4
    batch_size, sequence_length = 10, 5
    sequences = torch.Tensor(batch_size, sequence_length+1).uniform_(1, vocabulary_size).long() # (batch_size, sequence_length+1)
    mask = torch.ones(batch_size, sequence_length, dtype=torch.long) # (batch_size, sequence_length)
    for i in range(batch_size) :
        seq_len = torch.randint(low=2, high=sequence_length, size=(1,))[0]
        mask[i,seq_len:] = 0
        sequences[i,seq_len:] = 0
    # next sentence prediction
    inputs = sequences[:,:-1] # (batch_size, sequence_length)
    targets = sequences[:,1:] # (batch_size, sequence_length)
    
    # Model
    model = LSTMLM(
        vocabulary_size = vocabulary_size, 
        embedding_size = 6, 
        hidden_size = 5, 
        num_layers = 2, 
        dropout = 0.0,
        padding_index = None,
        bias_lstm=True,
        bias_classifier = True,
        share_embeddings = False
    )
    logits, (h, c) = model(inputs)
    print(logits.shape, h.shape, c.shape)
