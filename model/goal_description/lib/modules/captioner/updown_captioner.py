import functools
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchtext.vocab import GloVe
from allennlp.data import Vocabulary
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import add_sentence_boundary_token_ids, sequence_cross_entropy_with_logits

# from updown.config import Config
from lib.modules.captioner.modules.updown_cell import UpDownCell
from lib.modules.captioner.modules.cbs import ConstrainedBeamSearch
from lib.modules.captioner.utils.decoding import select_best_beam, select_best_beam_with_constraints


class UpDownCaptioner(nn.Module):
    r"""
    Image captioning model using bottom-up top-down attention, as in
    `Anderson et al. 2017 <https://arxiv.org/abs/1707.07998>`_. At training time, this model
    maximizes the likelihood of ground truth caption, given image features. At inference time,
    given image features, captions are decoded using beam search.

    Extended Summary
    ----------------
    This captioner is basically a recurrent language model for caption sequences. Internally, it
    runs :class:`~updown.modules.updown_cell.UpDownCell` for multiple time-steps. If this class is
    analogous to an :class:`~torch.nn.LSTM`, then :class:`~updown.modules.updown_cell.UpDownCell`
    would be analogous to :class:`~torch.nn.LSTMCell`.

    Parameters
    ----------
    vocabulary: allennlp.data.Vocabulary
        AllenNLP’s vocabulary containing token to index mapping for captions vocabulary.
    image_feature_size: int
        Size of the bottom-up image features.
    embedding_size: int
        Size of the word embedding input to the captioner.
    hidden_size: int
        Size of the hidden / cell states of attention LSTM and language LSTM of the captioner.
    attention_projection_size: int
        Size of the projected image and textual features before computing bottom-up top-down
        attention weights.
    max_caption_length: int, optional (default = 20)
        Maximum length of caption sequences for language modeling. Captions longer than this will
        be truncated to maximum length.
    beam_size: int, optional (default = 1)
        Beam size for finding the most likely caption during decoding time (evaluation).
    use_cbs: bool, optional (default = False)
        Whether to use :class:`~updown.modules.cbs.ConstrainedBeamSearch` for decoding.
    min_constraints_to_satisfy: int, optional (default = 2)
        Minimum number of constraints to satisfy for CBS, used for selecting the best beam. This
        will be ignored when ``use_cbs`` is False.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        image_feature_size: int,
        embedding_size: int,
        hidden_size: int,
        attention_projection_size: int,
        max_caption_length: int = 20,
        beam_size: int = 1,
        use_cbs: bool = False,
        min_constraints_to_satisfy: int = 2,
    ) -> None:
        super().__init__()
        self._vocabulary = vocabulary

        self.image_feature_size = image_feature_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_projection_size = attention_projection_size

        self._max_caption_length = max_caption_length
        self._use_cbs = use_cbs
        self._min_constraints_to_satisfy = min_constraints_to_satisfy

        # Short hand variable names for convenience
        _vocab_size = vocabulary.get_vocab_size()
        self._pad_index = vocabulary.get_token_index("@@UNKNOWN@@")
        self._boundary_index = vocabulary.get_token_index("@@BOUNDARY@@")
        self._separate_index = vocabulary.get_token_index("@@SEPARATE@@")

        # Initialize embedding layer with GloVe embeddings and freeze it if the specified size
        # is 300. CBS cannot be supported for any other embedding size, using CBS is optional
        # with embedding size 300. So in either cases, embeddig size is the deciding factor.
        if self.embedding_size == 300:
            glove_vectors = self._initialize_glove()
            self._embedding_layer = nn.Embedding.from_pretrained(
                glove_vectors, freeze=True, padding_idx=self._pad_index
            )
        else:
            self._embedding_layer = nn.Embedding(
                _vocab_size, embedding_size, padding_idx=self._pad_index
            )
            assert not use_cbs, "CBS is not supported without Frozen GloVe embeddings (300d), "
            f"found embedding size to be {self.embedding_size}."

        self._updown_cell = UpDownCell(
            image_feature_size, embedding_size, hidden_size, attention_projection_size
        )

        if self.embedding_size == 300:
            # Tie the input and output word embeddings when using frozen GloVe embeddings.
            # In this case, project hidden states to GloVe dimension (with a non-linearity).
            self._output_projection = nn.Sequential(
                nn.Linear(hidden_size, self.embedding_size), nn.Tanh()
            )
            self._output_layer = nn.Linear(self.embedding_size, _vocab_size, bias=False)
            self._output_layer.weight = self._embedding_layer.weight
        else:
            # Else don't tie them when learning embeddings during training.
            # In this case, project hidden states directly to output vocab space.
            self._output_projection = nn.Identity()  # type: ignore
            self._output_layer = nn.Linear(hidden_size, _vocab_size)

        self._log_softmax = nn.LogSoftmax(dim=1)

        # We use beam search to find the most likely caption during inference.
        BeamSearchClass = ConstrainedBeamSearch if use_cbs else BeamSearch
        self._beam_search = BeamSearchClass(
            self._boundary_index,
            max_steps=max_caption_length,
            beam_size=beam_size,
            per_node_beam_size=beam_size // 2,
        )


    def _initialize_glove(self) -> torch.Tensor:
        r"""
        Initialize embeddings of all the tokens in a given
        :class:`~allennlp.data.vocabulary.Vocabulary` by their GloVe vectors.

        Extended Summary
        ----------------
        It is recommended to train an :class:`~updown.models.updown_captioner.UpDownCaptioner` with
        frozen word embeddings when one wishes to perform Constrained Beam Search decoding during
        inference. This is because the constraint words may not appear in caption vocabulary (out of
        domain), and their embeddings will never be updated during training. Initializing with frozen
        GloVe embeddings is helpful, because they capture more meaningful semantics than randomly
        initialized embeddings.

        Returns
        -------
        torch.Tensor
            GloVe Embeddings corresponding to tokens.
        """
        glove = GloVe(name="42B", dim=300)
        glove_vectors = torch.zeros(self._vocabulary.get_vocab_size(), 300)

        for word, i in self._vocabulary.get_token_to_index_vocabulary().items():
            if word in glove.stoi:
                glove_vectors[i] = glove.vectors[glove.stoi[word]]
            elif word != self._pad_index:
                # Initialize by random vector.
                glove_vectors[i] = 2 * torch.randn(300) - 1

        return glove_vectors

    def forward(  # type: ignore
        self,
        image_features: torch.Tensor,
        caption_tokens: Optional[torch.Tensor] = None,
        fsm: torch.Tensor = None,
        num_constraints: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        r"""
        Given bottom-up image features, maximize the likelihood of paired captions during
        training. During evaluation, decode captions given image features using beam search.

        Parameters
        ----------
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes * image_feature_size)``. ``num_boxes`` for
            each instance in a batch might be different. Instances with lesser boxes are padded
            with zeros up to ``num_boxes``.
        caption_tokens: torch.Tensor, optional (default = None)
            A tensor of shape ``(batch_size, max_caption_length)`` of tokenized captions. This
            tensor does not contain ``@@BOUNDARY@@`` tokens yet. Captions are not provided
            during evaluation.
        fsm: torch.Tensor, optional (default = None)
            A tensor of shape ``(batch_size, num_states, num_states, vocab_size)``: finite state
            machines per instance, represented as adjacency matrix. For a particular instance
            ``[_, s1, s2, v] = 1`` shows a transition from state ``s1`` to ``s2`` on decoding
            ``v`` token (constraint). Would be ``None`` for regular beam search decoding.
        num_constraints: torch.Tensor, optional (default = None)
            A tensor of shape ``(batch_size, )`` containing the total number of given constraints
            for CBS. Would be ``None`` for regular beam search decoding.

        Returns
        -------
        Dict[str, torch.Tensor]
            Decoded captions and/or per-instance cross entropy loss, dict with keys either
            ``{"predictions"}`` or ``{"loss"}``.
        """
        batch_size, num_boxes, image_feature_size = image_features.size()

        # Initialize states at zero-th timestep.
        states = None

        if self.training and caption_tokens is not None:
            # Add "@@BOUNDARY@@" tokens to caption sequences.
            caption_tokens, _ = add_sentence_boundary_token_ids(
                caption_tokens,
                (caption_tokens != self._pad_index),
                self._boundary_index,
                self._boundary_index,
            )
            batch_size, max_caption_length = caption_tokens.size()

            # shape: (batch_size, max_caption_length)
            tokens_mask = caption_tokens != self._pad_index

            # The last input from the target is either padding or the boundary token.
            # Either way, we don't have to process it.
            num_decoding_steps = max_caption_length - 1

            step_logits: List[torch.Tensor] = []
            for timestep in range(num_decoding_steps):
                # shape: (batch_size,)
                input_tokens = caption_tokens[:, timestep]

                # shape: (batch_size, num_classes)
                output_logits, states = self._decode_step(image_features, input_tokens, states)

                # list of tensors, shape: (batch_size, 1, vocab_size)
                step_logits.append(output_logits.unsqueeze(1))

            # shape: (batch_size, num_decoding_steps)
            logits = torch.cat(step_logits, 1)

            # Skip first time-step from targets for calculating loss.
            output_dict = {
                "loss": self._get_loss(
                    logits, caption_tokens[:, 1:].contiguous(), tokens_mask[:, 1:].contiguous()
                )
            }
        else:
            num_decoding_steps = self._max_caption_length
            start_predictions = image_features.new_full((batch_size,), self._boundary_index).long()

            # Add image features as a default argument to match callable signature acceptable by
            # beam search class (previous predictions and states only).
            beam_decode_step = functools.partial(self._decode_step, image_features)

            # shape (all_top_k_predictions): (batch_size, net_beam_size, num_decoding_steps)
            # shape (log_probabilities): (batch_size, net_beam_size)
            if self._use_cbs:
                all_top_k_predictions, log_probabilities = self._beam_search.search(
                    start_predictions, states, beam_decode_step, fsm
                )
                best_beam = select_best_beam_with_constraints(
                    all_top_k_predictions,
                    log_probabilities,
                    num_constraints,
                    self._min_constraints_to_satisfy,
                )
            else:
                all_top_k_predictions, log_probabilities = self._beam_search.search(
                    start_predictions, states, beam_decode_step
                )
                best_beam = select_best_beam(all_top_k_predictions, log_probabilities)

            # shape: (batch_size, num_decoding_steps)
            output_dict = {"predictions": best_beam}

        return output_dict

    def _decode_step(
        self,
        image_features: torch.Tensor,
        previous_predictions: torch.Tensor,
        states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Given image features, tokens predicted at previous time-step and LSTM states of the
        :class:`~updown.modules.updown_cell.UpDownCell`, take a decoding step. This is also
        called by the beam search class.

        Parameters
        ----------
        image_features: torch.Tensor
            A tensor of shape ``(batch_size, num_boxes, image_feature_size)``.
        previous_predictions: torch.Tensor
            A tensor of shape ``(batch_size * net_beam_size, )`` containing tokens predicted at
            previous time-step -- one for each beam, for each instances in a batch.
            ``net_beam_size`` is 1 during teacher forcing (training), ``beam_size`` for regular
            :class:`allennlp.nn.beam_search.BeamSearch` and ``beam_size * num_states`` for
            :class:`updown.modules.cbs.ConstrainedBeamSearch`

        states: [Dict[str, torch.Tensor], optional (default = None)
            LSTM states of the :class:`~updown.modules.updown_cell.UpDownCell`. These are
            initialized as zero tensors if not provided (at first time-step).
        """
        net_beam_size = 1

        # Expand and repeat image features while doing beam search (during inference).
        if not self.training and image_features.size(0) != previous_predictions.size(0):

            batch_size, num_boxes, image_feature_size = image_features.size()
            net_beam_size = int(previous_predictions.size(0) / batch_size)

            # Add (net) beam dimension and repeat image features.
            image_features = image_features.unsqueeze(1).repeat(1, net_beam_size, 1, 1)

            # shape: (batch_size * net_beam_size, num_boxes, image_feature_size)
            image_features = image_features.view(
                batch_size * net_beam_size, num_boxes, image_feature_size
            )

        # shape: (batch_size * net_beam_size, )
        current_input = previous_predictions

        # shape: (batch_size * net_beam_size, embedding_size)
        token_embeddings = self._embedding_layer(current_input)

        # shape: (batch_size * net_beam_size, hidden_size)
        updown_output, states = self._updown_cell(image_features, token_embeddings, states)

        # shape: (batch_size * net_beam_size, vocab_size)
        updown_output = self._output_projection(updown_output)
        output_logits = self._output_layer(updown_output)

        # Return logits while training, to further calculate cross entropy loss.
        # Return logprobs during inference, because beam search needs them.
        # Note:: This means NO BEAM SEARCH DURING TRAINING.
        outputs = output_logits + 1e-10 if self.training else self._log_softmax(output_logits) + 1e-10

        return outputs, states  # type: ignore

    def _get_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        r"""
        Compute cross entropy loss of predicted caption (logits) w.r.t. target caption. The cross
        entropy loss of caption is cross entropy loss at each time-step, summed.

        Parameters
        ----------
        logits: torch.Tensor
            A tensor of shape ``(batch_size, max_caption_length - 1, vocab_size)`` containing
            unnormalized log-probabilities of predicted captions.
        targets: torch.Tensor
            A tensor of shape ``(batch_size, max_caption_length - 1)`` of tokenized target
            captions.
        target_mask: torch.Tensor
            A mask over target captions, elements where mask is zero are ignored from loss
            computation. Here, we ignore ``@@UNKNOWN@@`` token (and hence padding tokens too
            because they are basically the same).

        Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size, )`` containing cross entropy loss of captions, summed
            across time-steps.
        """

        # shape: (batch_size, )
        target_lengths = torch.sum(target_mask, dim=-1).float()

        # shape: (batch_size, )
        return target_lengths * sequence_cross_entropy_with_logits(
            logits, targets, target_mask, average=None
        )
