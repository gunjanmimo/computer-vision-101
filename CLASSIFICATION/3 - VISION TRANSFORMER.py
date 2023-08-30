import torch
import torch.nn as nn


class PatchEmbeddingLayer(nn.Module):
    def __init__(
        self,
        in_channel: int,
        patch_size: int,
        embedding_dim: int,
        num_patches: int,
        batch_size: int,
    ) -> None:
        super(PatchEmbeddingLayer, self).__init__()
        self.in_channel = in_channel
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.conv_layer = nn.Conv2d(
            in_channels=in_channel,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=2)
        self.class_token_embedding = nn.Parameter(
            torch.rand(1, 1, embedding_dim), requires_grad=True
        )
        self.position_embedding = nn.Parameter(
            torch.rand(1, num_patches + 1, embedding_dim), requires_grad=True
        )

    def forward(self, x):
        image_through_conv = self.conv_layer(x)
        # reshape
        image_through_conv = image_through_conv.permute(0, 2, 3, 1)
        # flatten - it is embedded image
        flatten_image = self.flatten_layer(image_through_conv)

        batch_size = x.size(0)  # Get the actual batch size dynamically

        class_token_embedding = self.class_token_embedding.expand(batch_size, -1, -1)
        position_embedding = self.position_embedding.expand(batch_size, -1, -1)

        embedded_image_with_class_token_embedding = torch.cat(
            (class_token_embedding, flatten_image), dim=1
        )

        final_embedding = embedded_image_with_class_token_embedding + position_embedding
        return final_embedding


class MultiHeadedAttentionBlcok(nn.Module):
    def __init__(
        self, embedding_dim: int = 768, num_head: int = 12, attn_dropout=0
    ) -> None:
        super(MultiHeadedAttentionBlcok, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_head = num_head

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multi_head_attention_layer = nn.MultiheadAttention(
            num_heads=num_head, embed_dim=embedding_dim, batch_first=True
        )

    def forward(self, x):
        x = self.layer_norm(x)
        output, _ = self.multi_head_attention_layer(
            query=x, key=x, value=x, need_weights=False
        )
        return output


class MachineLearningPerceptronBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_size: int, mlp_dropout: float) -> None:
        super(MachineLearningPerceptronBlock, self).__init__()
        self.embedding_dim = embedding_dim
        self.mlp_size = mlp_size
        self.mlp_dropout = mlp_dropout

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=mlp_dropout),
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim=768,
        mlp_dropout=0.1,
        attn_dropout=0.0,
        mlp_size=3072,
        num_heads=12,
    ):
        super().__init__()

        self.msa_block = MultiHeadedAttentionBlcok(
            embedding_dim=embedding_dim,
            num_head=num_heads,
            attn_dropout=attn_dropout,
        )

        self.mlp_block = MachineLearningPerceptronBlock(
            embedding_dim=embedding_dim,
            mlp_size=mlp_size,
            mlp_dropout=mlp_dropout,
        )

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x

        return x


class ViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        batch_size: int,
        img_width: int,
        img_height: int,
        in_channels=3,
        patch_size=16,
        embedding_dim=768,
        num_transformer_layers=12,  # from table 1 above
        mlp_dropout=0.1,
        attn_dropout=0.0,
        mlp_size=3072,
        num_heads=12,
    ):
        super().__init__()
        num_patches = int((img_width * img_height) / patch_size**2)

        self.patch_embedding_layer = PatchEmbeddingLayer(
            in_channel=in_channels,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            batch_size=batch_size,
            num_patches=num_patches,
        )

        self.transformer_encoder = nn.Sequential(
            *[
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    mlp_dropout=mlp_dropout,
                    attn_dropout=attn_dropout,
                    mlp_size=mlp_size,
                    num_heads=num_heads,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes),
        )

    def forward(self, x):
        return self.classifier(
            self.transformer_encoder(self.patch_embedding_layer(x))[:, 0]
        )
