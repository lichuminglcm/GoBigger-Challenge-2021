import torch
import torch.nn as nn
from ding.torch_utils import MLP, get_lstm

class RelationGCN(nn.Module):

    def __init__(
            self,
            hidden_shape: int,
            activation=nn.ReLU(),
    ) -> None:
        super(RelationGCN, self).__init__()
        # activation
        self.act = activation
        # layers
        self.thorn_relation_layers = MLP(
            2 * hidden_shape, hidden_shape, hidden_shape, layer_num=1, activation=activation
        )
        self.clone_relation_layers = MLP(
            3 * hidden_shape, hidden_shape, hidden_shape, layer_num=1, activation=activation
        )
        self.agg_relation_layers = MLP(
            4 * hidden_shape, hidden_shape, hidden_shape, layer_num=1, activation=activation
        )


    def forward(self, food_relation, thorn_relation, clone, clone_relation, thorn_mask, clone_mask):
        b, t, c = clone.shape[0], thorn_relation.shape[2], clone.shape[1]
        # encode thorn relation
        thorn_relation = torch.cat([clone.unsqueeze(2).repeat(1, 1, t, 1), thorn_relation], dim=3) # [b,n_clone,n_thorn,c]
        thorn_relation = self.thorn_relation_layers(thorn_relation) * thorn_mask.view(b, 1, t, 1) # [b,n_clone,n_thorn,c]
        thorn_relation = thorn_relation.max(2).values # [b,n_clone,c]
        # encode clone relation
        clone_relation = torch.cat([clone.unsqueeze(2).repeat(1, 1, c, 1), clone.unsqueeze(1).repeat(1, c, 1, 1), clone_relation], dim=3) # [b,n_clone,n_clone,c]
        clone_relation = self.clone_relation_layers(clone_relation) * clone_mask.view(b, 1, c, 1) # [b,n_clone,n_clone,c]
        clone_relation = clone_relation.max(2).values # [b,n_clone,c]
        # encode aggregated relation
        agg_relation = torch.cat([clone, food_relation, thorn_relation, clone_relation], dim=2)
        clone = clone + self.agg_relation_layers(agg_relation)
        return clone

class StructedGcnConv(nn.Module):

    def __init__(
            self,
            scalar_shape: int,
            food_shape: int,
            food_relation_shape: int,
            thorn_relation_shape: int,
            clone_shape: int,
            clone_relation_shape: int,
            hidden_shape: int,
            encode_shape: int,
            rnn: bool = False,
            activation=nn.ReLU(),
    ) -> None:
        super(StructedGcnConv, self).__init__()
        # scalar encoder
        self.scalar_encoder = MLP(
            scalar_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # food encoder
        layers = []
        kernel_size = [5, 3, 3, 3, 1]
        stride = [4, 2, 2, 2, 1]
        shape = [hidden_shape // 4, hidden_shape // 2, hidden_shape // 2, hidden_shape, hidden_shape]
        input_shape = food_shape
        for i in range(len(kernel_size)):
            layers.append(nn.Conv2d(input_shape, shape[i], kernel_size[i], stride[i], kernel_size[i] // 2))
            layers.append(activation)
            input_shape = shape[i]
        self.food_encoder = nn.Sequential(*layers)
        # food relation encoder
        self.food_relation_encoder = MLP(
            food_relation_shape, hidden_shape // 2, hidden_shape, layer_num=2, activation=activation
        )
        # thorn relation encoder
        self.thorn_relation_encoder = MLP(
            thorn_relation_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # clone encoder
        self.clone_encoder = MLP(
            clone_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # clone relation encoder
        self.clone_relation_encoder = MLP(
            clone_relation_shape, hidden_shape // 4, hidden_shape, layer_num=2, activation=activation
        )
        # gcn
        self.gcn_1 = RelationGCN(
            hidden_shape, activation=activation
        )
        self.gcn_2 = RelationGCN(
            hidden_shape, activation=activation
        )
        self.agg_encoder = MLP(
            3 * hidden_shape, hidden_shape, encode_shape, layer_num=2, activation=activation
        )

    def reduce(self, mask):
        max_len = max(mask.sum(1).max().long(), 1)
        return mask[:,:max_len], max_len

    def forward(self, scalar, food, food_relation, thorn_relation, thorn_mask, clone, clone_relation, clone_mask, reduce=False):
        # reduce obs
        if reduce:
            clone_mask, clone_len = self.reduce(clone_mask)
            thorn_mask, thorn_len = self.reduce(thorn_mask)
            clone = clone[:, :clone_len]
            food_relation = food_relation[:, :clone_len]
            thorn_relation = thorn_relation[:, :clone_len, :thorn_len]
            clone_relation = clone_relation[:, :clone_len, :clone_len]
        # encode scalar
        scalar = self.scalar_encoder(scalar) # [b,c]
        # encode food
        food = self.food_encoder(food) # [b,c,h,w]
        food = food.reshape(*food.shape[:2], -1).max(-1).values # [b,c]
        # encode food relation
        food_relation = self.food_relation_encoder(food_relation) # [b,c]
        # encode thorn relation
        thorn_relation = self.thorn_relation_encoder(thorn_relation) # [b,n_clone,n_thorn, c]
        # encode clone
        clone = self.clone_encoder(clone) # [b,n_clone,c]
        # encode clone relation
        clone_relation = self.clone_relation_encoder(clone_relation) # [b,n_clone,n_clone,c]
        # aggregate all relation
        clone = self.gcn_1(food_relation, thorn_relation, clone, clone_relation, thorn_mask, clone_mask)
        clone = self.gcn_2(food_relation, thorn_relation, clone, clone_relation, thorn_mask, clone_mask)
        clone = clone * clone_mask.unsqueeze(2) # [b,n_clone,c]
        clone = clone.max(1).values # [b,c]

        return self.agg_encoder(torch.cat([scalar, food, clone], dim=1))

class GoBiggerHybridAction(nn.Module):
    r"""
    Overview:
        The GoBiggerHybridAction model.
    Interfaces:
        ``__init__``, ``forward``, ``compute_encoder``, ``compute_critic``
    """
    mode = ['compute_encoder', 'compute_critic']
    def __init__(
            self,
            scalar_shape: int,
            food_shape: int,
            food_relation_shape: int,
            thorn_relation_shape: int,
            clone_shape: int,
            clone_relation_shape: int,
            hidden_shape: int,
            encode_shape: int,
            action_shape: int,
            rnn: bool = False,
            activation=nn.ReLU(),
    ) -> None:
        super(GoBiggerHybridAction, self).__init__()
        # encoder
        self.encoder = StructedGcnConv(scalar_shape, food_shape, food_relation_shape, thorn_relation_shape, clone_shape, clone_relation_shape, hidden_shape, encode_shape, rnn, activation)
        self.encoder = nn.DataParallel(self.encoder)
        # head
        self.q = nn.Sequential(
                    MLP(encode_shape + action_shape, encode_shape, encode_shape, layer_num=1, activation=activation),
                    nn.Linear(encode_shape, 1),
                )

    def forward(self, inputs, mode):
        assert mode in self.mode, "not support forward mode: {}/{}".format(mode, self.mode)
        return getattr(self, mode)(inputs)

    def compute_encoder(self, inputs):
        scalar = inputs['scalar']
        food = inputs['food']
        food_relation = inputs['food_relation']
        thorn_relation = inputs['thorn_relation']
        thorn_mask = inputs['thorn_mask']
        clone = inputs['clone']
        clone_relation = inputs['clone_relation']
        clone_mask = inputs['clone_mask']
        return {'encode': self.encoder(scalar, food, food_relation, thorn_relation, thorn_mask, clone, clone_relation, clone_mask)}

    def compute_critic(self, inputs):
        action = inputs['action'] #[b,c] or #[b,n_a,c]
        encode = inputs['encode'] #[b,c]
        if len(action.shape) == 3:
            encode = encode.unsqueeze(1).repeat(1, action.shape[1], 1) #[b,n_a,c]
        return {'q_value': self.q(torch.cat([encode, action], dim=-1)).squeeze(-1)} #[b,] or #[b,n_a,]
