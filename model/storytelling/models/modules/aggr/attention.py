import torch
from torch import nn
# from allennlp.nn.util import masked_softmax
import torch.nn.functional as F


class Attention(nn.Module):
    r"""
    A PyTorch module to compute bottom-up top-down attention
    (`Anderson et al. 2017 <https://arxiv.org/abs/1707.07998>`_). Used in
    :class:`~updown.modules.updown_cell.UpDownCell`

    Parameters
    ----------
    query_size: int
        Size of the query vector, typically the output of Attention LSTM in
        :class:`~updown.modules.updown_cell.UpDownCell`.
    image_feature_size: int
        Size of the bottom-up image features.
    projection_size: int
        Size of the projected image and textual features before computing bottom-up top-down
        attention weights.
    """

    def __init__(self, image_feature_size, goal_feature_size, projection_size):
        super(Attention, self).__init__()

        self._image_features_projection_layer = nn.Linear(
            image_feature_size, projection_size, bias=False
        )
        self._goal_features_projection_layer = nn.Linear(
            goal_feature_size, projection_size, bias=False
        )
        self._attention_layer = nn.Linear(projection_size, 1, bias=False)

    def forward(
        self,
        image_features,
        goal_features,
        image_features_mask = None,
    ):
        # Image features are projected by a method call, which is decorated using LRU cache, to
        # save some computation. Refer method docstring.
        # shape: (batch_size, num_boxes, projection_size)
        projected_image_features = self._image_features_projection_layer(image_features)

        projected_goal_features = self._goal_features_projection_layer(goal_features).unsqueeze(1)

        # shape: (batch_size, num_boxes, 1)
        attention_logits = self._attention_layer(
            torch.tanh(projected_image_features + projected_goal_features)
        )

        # shape: (batch_size, num_boxes)
        attention_logits = attention_logits.squeeze(-1)

        # `\alpha`s as importance weights for boxes (rows) in the `image_features`.
        # shape: (batch_size, num_boxes)
        if image_features_mask is not None:
            attention_weights = masked_softmax(attention_logits, image_features_mask, dim=-1)
        else:
            attention_weights = F.softmax(attention_logits, dim=-1)

        return attention_weights

