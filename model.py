import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from math import ceil


class MLP(nn.Module):
    def __init__(self, input_size: int, class_num: int, hidden_dims: List[int] = None):
        """Multi-Layer Perceptron with configurable hidden layers."""
        super(MLP, self).__init__()
        if hidden_dims is None:
            hidden_dims = [ceil(input_size * 1.2), ceil(input_size * 0.5)]

        modules = []
        in_size = input_size
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_size, out_features=h_dim),
                    nn.ReLU()
                )
            )
            in_size = h_dim

        self.encoder = nn.Sequential(*modules)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Sequential(nn.Linear(hidden_dims[-1], ceil(class_num * 10)), nn.ReLU())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.encoder(input)
        result = self.dropout(result)
        return self.fc(result)


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        """Bahdanau attention mechanism."""
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, query: torch.Tensor, values: torch.Tensor) -> 'tuple[torch.Tensor, torch.Tensor]':
        scores = self.V(torch.tanh(self.W1(query) + self.W2(values)))
        scores = scores.squeeze(2).unsqueeze(1)
        alphas = F.softmax(scores, dim=-1)
        context = torch.bmm(alphas, values).squeeze(1)
        return context, alphas


class MVIB(nn.Module):
    def __init__(self, input_dims: List[int], class_num: int):
        """Multiview Information Bottleneck model with attention mechanism."""
        super(MVIB, self).__init__()
        self.n_view = len(input_dims)
        self.X_nets = nn.ModuleList([MLP(input_size=input_dims[i], class_num=class_num) for i in range(self.n_view)])
        self.dropout = nn.Dropout(p=0.5)
        classifier_input_dim = class_num * 10
        self.attention_fusion = BahdanauAttention(classifier_input_dim)
        self.classifier = nn.Linear(classifier_input_dim, class_num)

    def forward(self, inputs: List[torch.Tensor]) -> 'tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]]':
        assert len(inputs) == self.n_view, f"Expected {self.n_view} inputs, but got {len(inputs)}"
        features = [self.X_nets[i](input_i) for i, input_i in enumerate(inputs)]
        z = torch.stack(features).transpose(0, 1)
        context, attn_weights = self.attention_fusion(z, z)
        out = self.classifier(context)
        return inputs, out, context, features


# import torch.nn as nn
# from typing import List
# from math import ceil
# import torch
# import torch.nn.functional as F


# class MLP(nn.Module):
#     def __init__(self, input_size: int, class_num: int, hidden_dims: List = None):
#         super(MLP, self).__init__()
#         modules = []
#         if hidden_dims is None:
#             hidden_dims = [ceil(input_size*1.2), ceil(input_size*0.5)]
        
#         # Build Encoder
#         in_size = input_size
#         for h_dim in hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Linear(in_size,out_features=h_dim),
#                     nn.ReLU()
#                 )
#             )
#             in_size = h_dim
#         self.encoder = nn.Sequential(*modules)
#         self.dropout = nn.Dropout(p=0.5)
#         self.fc = nn.Sequential(nn.Linear(hidden_dims[-1], ceil(class_num*10)), nn.ReLU())

#     def forward(self, input):
#         result = self.encoder(input)
#         result = self.dropout(result)
#         return self.fc(result)
    

# class MVIB(nn.Module):
#     def __init__(self, input_dims, class_num) -> None:
#         super(MVIB, self).__init__()
#         self.n_view = len(input_dims)
#         self.X_nets = nn.ModuleList([MLP(input_size=input_dims[i], class_num=class_num) for i in range(self.n_view)])
#         self.dropout = nn.Dropout(p=0.5)
#         classifier_input_dim = class_num*10
#         self.attention_fusion = BahdanauAttention(classifier_input_dim)
#         self.classifier = nn.Linear(classifier_input_dim, class_num)

#     def forward(self, inputs):
#         assert len(inputs) == self.n_view, f"Expected {self.n_views} inputs, but got {len(inputs)}"
#         features = []
#         for i, input_i in enumerate(inputs):
#             X_net_i = self.X_nets[i]
#             z_i = X_net_i(input_i)
#             features.append(z_i)
#         z = torch.stack(features).transpose(0,1)
#         context, attn_weights = self.attention_fusion(z, z)
#         out    = self.classifier(context)
#         return inputs, out, context, features
    
# class BahdanauAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(BahdanauAttention, self).__init__()
#         self.W1 = nn.Linear(hidden_size, hidden_size)
#         self.W2 = nn.Linear(hidden_size, hidden_size)
#         self.V = nn.Linear(hidden_size, 1)
#         self.W3 = nn.Linear(hidden_size, 1)

#     def forward(self, query, values): 
#         # Additive attention 
#         scores = self.V(torch.tanh(self.W1(query) + self.W2(values))) 
#         scores = scores.squeeze(2).unsqueeze(1) 
#         alphas = F.softmax(scores, dim=-1)
#         context = torch.bmm(alphas, values)
#         context = context.squeeze(1)
#         return context, alphas


# # class BahdanauAttention(nn.Module):
# #     def __init__(self, hidden_size):
# #         super(BahdanauAttention, self).__init__()
# #         self.W1 = nn.Linear(hidden_size, hidden_size)
# #         self.W2 = nn.Linear(hidden_size, hidden_size)
# #         self.V = nn.Linear(hidden_size, 1)
# #         self.W3 = nn.Linear(hidden_size, 1)

# #     def forward(self, query, values): 
# #         # Additive attention 
# #         scores = self.V(torch.tanh(self.W1(query) + self.W2(values))) 
# #         scores = scores.squeeze(2).unsqueeze(1) # [B, M, 1] -> [B, 1, M] 
# #         # scores = scores.unsqueeze(1)
# #         # Dot-Product Attention: score(s_t, h_i) = s_t^T h_i
# #         # Query [B, 1, D] * Values [B, D, M] -> Scores [B, 1, M]
# #         # scores = torch.bmm(query, values.permute(0,2,1))

# #         # Cosine Similarity: score(s_t, h_i) = cosine_similarity(s_t, h_i)
# #         # scores = F.cosine_similarity(query, values, dim=2).unsqueeze(1)

# #         # Mask out invalid positions.
# #         # scores.data.masked_fill_(mask.unsqueeze(1) == 0, -float('inf'))

# #         # Attention weights
# #         alphas = F.softmax(scores, dim=-1)

# #         # The context vector is the weighted sum of the values.
# #         # values = values.unsqueeze(1)
# #         context = torch.bmm(alphas, values)
# #         context = context.squeeze(1)
# #         # context shape: [B, 1, D], alphas shape: [B, 1, M]
# #         return context, alphas