import torch
import torch.nn as nn


class AttentionMechanism(nn.Module):
    def __init__(self, input_size=100, output_size=100):
        super(AttentionMechanism, self).__init__()
        self.attention = nn.Linear(input_size, output_size)
        nn.init.xavier_normal_(self.attention.weight)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query_embedding, key_embeddings):
        hidden_state = self.relu(self.attention(key_embeddings))
        attention_weights = self.softmax(
            torch.matmul(query_embedding, hidden_state.transpose(0, 1)))
        attention_scores = torch.matmul(attention_weights, key_embeddings)
        return query_embedding + attention_scores


class MessageAggregation(nn.Module):
    def __init__(self, embed_dim=100):
        super(MessageAggregation, self).__init__()
        self.embed_dim = embed_dim
        self.W1 = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.xavier_normal_(self.W1.weight)
        nn.init.xavier_normal_(self.W2.weight)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, embedding1, all_embeddings2):
        output = self.W1(embedding1)

        for embedding2 in all_embeddings2:
            output = output + self.W1(embedding2)
            elementwise_product = embedding1 * embedding2
            output = output + self.W2(elementwise_product)

        return self.leaky_relu(output)


class AttentiveNCF(nn.Module):
    def __init__(self, embed_dim):
        super(AttentiveNCF, self).__init__()
        self.attention_mechanism = AttentionMechanism(embed_dim, embed_dim)
        self.message_aggregation = MessageAggregation(embed_dim)

    def forward(self, embedding1, all_embeddings2):
        attn_embedding = self.attention_mechanism(embedding1, all_embeddings2)
        output_embedding = self.message_aggregation(attn_embedding,
                                                    all_embeddings2)
        return output_embedding


class RecommenderNet(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=64, num_ancf_layers=1):
        super(RecommenderNet, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.num_ancf_layers = num_ancf_layers

        self.user_embeddings = nn.Embedding(n_users, embed_dim)
        self.item_embeddings = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_normal_(self.user_embeddings.weight)
        nn.init.xavier_normal_(self.item_embeddings.weight)

        self.user_ANCFLayers = nn.ModuleList(
            [AttentiveNCF(embed_dim) for _ in range(num_ancf_layers)])
        self.item_ANCFLayers = nn.ModuleList([
            AttentiveNCF(embed_dim) for _ in range(num_ancf_layers)])

    def forward(self, user_id, item_id, all_user_ids_tensor,
                all_item_ids_tensor):
        all_user_embeddings = self.user_embeddings(all_user_ids_tensor)
        all_item_embeddings = self.item_embeddings(all_item_ids_tensor)

        user_embedding = all_user_embeddings[user_id]
        item_embedding = all_item_embeddings[item_id]

        og_user_embedding = user_embedding.clone()
        og_item_embedding = item_embedding.clone()

        for ancf_layer in self.user_ANCFLayers:
            user_embedding = ancf_layer(user_embedding, all_item_embeddings)
            og_user_embedding = torch.cat((og_user_embedding, user_embedding),
                                          dim=1)

        for ancf_layer in self.item_ANCFLayers:
            item_embedding = ancf_layer(item_embedding, all_user_embeddings)
            og_item_embedding = torch.cat((og_item_embedding, item_embedding),
                                          dim=1)

        return torch.matmul(torch.transpose(og_user_embedding, 0, 1),
                            og_item_embedding)
