from datasets import load_from_disk
from transformers import DataCollatorWithPadding
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
import random
from transformers.models.bart.modeling_bart import shift_tokens_right, BartPretrainedModel, _make_causal_mask
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput, Seq2SeqLMOutput
import math
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import torch
from copy import deepcopy
from transformers.models.bart.modeling_bart import shift_tokens_right
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers import AdamW
from transformers import get_scheduler
from transformers import AutoModel
from torch.nn import functional as F
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
logger = logging.get_logger(__name__)

BATCH_SIZE = 2
LAMBDA = 1.0
LR = 0.0001
NUM_epochs = 1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenized_datasets_task1 = load_from_disk("./DataForModel/dataset-task1")
tokenized_datasets_task2 = load_from_disk("./DataForModel/dataset-task2")
tokenized_datasets_task3 = load_from_disk("./DataForModel/dataset-task3")
tokenized_datasets_task4 = load_from_disk("./DataForModel/dataset-task4")

train_data_task1, validation_data_task1 = tokenized_datasets_task1.train_test_split(test_size=0.2, shuffle=True,
                                                                                    seed=42).values()
train_data_task1, test_data_task1 = train_data_task1.train_test_split(test_size=0.2, shuffle=True, seed=42).values()

train_data_task2, validation_data_task2 = tokenized_datasets_task2.train_test_split(test_size=0.2, shuffle=True,
                                                                                    seed=42).values()
train_data_task2, test_data_task2 = train_data_task2.train_test_split(test_size=0.2, shuffle=True, seed=42).values()

train_data_task3, validation_data_task3 = tokenized_datasets_task3.train_test_split(test_size=0.2, shuffle=True,
                                                                                    seed=42).values()
train_data_task3, test_data_task3 = train_data_task3.train_test_split(test_size=0.2, shuffle=True, seed=42).values()

train_data_task4, validation_data_task4 = tokenized_datasets_task4.train_test_split(test_size=0.2, shuffle=True,
                                                                                    seed=42).values()
train_data_task4, test_data_task4 = train_data_task4.train_test_split(test_size=0.2, shuffle=True, seed=42).values()

print(train_data_task1)

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

collate_function = DataCollatorWithPadding(tokenizer)

train_datasetLoaderTask1 = DataLoader(train_data_task1, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=collate_function)

valid_datasetLoaderTask1 = DataLoader(validation_data_task1, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=collate_function)

test_datasetLoaderTask1 = DataLoader(test_data_task1, batch_size=BATCH_SIZE, shuffle=True,
                                     collate_fn=collate_function)

train_datasetLoaderTask2 = DataLoader(train_data_task2, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=collate_function)

valid_datasetLoaderTask2 = DataLoader(validation_data_task2, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=collate_function)

test_datasetLoaderTask2 = DataLoader(test_data_task2, batch_size=BATCH_SIZE, shuffle=True,
                                     collate_fn=collate_function)

train_datasetLoaderTask3 = DataLoader(train_data_task3, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=collate_function)

valid_datasetLoaderTask3 = DataLoader(validation_data_task3, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=collate_function)

test_datasetLoaderTask3 = DataLoader(test_data_task3, batch_size=BATCH_SIZE, shuffle=True,
                                     collate_fn=collate_function)

train_datasetLoaderTask4 = DataLoader(train_data_task4, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=collate_function)

valid_datasetLoaderTask4 = DataLoader(validation_data_task4, batch_size=BATCH_SIZE, shuffle=True,
                                      collate_fn=collate_function)

test_datasetLoaderTask4 = DataLoader(test_data_task4, batch_size=BATCH_SIZE, shuffle=True,
                                     collate_fn=collate_function)

model = AutoModel.from_pretrained("fnlp/bart-base-chinese")

config = AutoConfig.from_pretrained('fnlp/bart-base-chinese')
# print(config)

shared, encoder, decoder = model._modules['shared'], model._modules['encoder'], model._modules['decoder']
layers = decoder._modules["layers"]


# print(layers)

class StylizedBartAttention(nn.Module):
    def __init__(self, layer_id):
        super(StylizedBartAttention, self).__init__()
        layer = layers._modules[str(layer_id)]

        self.embed_dim = layer._modules['encoder_attn'].embed_dim
        self.num_heads = layer._modules['encoder_attn'].num_heads
        self.dropout = layer._modules['encoder_attn'].dropout
        self.head_dim = layer._modules['encoder_attn'].head_dim

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = layer._modules['encoder_attn'].is_decoder

        self.k_proj = layer._modules['encoder_attn']._modules['k_proj']
        self.v_proj = layer._modules['encoder_attn']._modules['v_proj']
        # self.q_proj = layer._modules['encoder_attn']._modules['q_proj']

        self.q_proj_dict = nn.ModuleDict({
            '0': layer._modules['encoder_attn']._modules['q_proj'],
            '1': nn.Linear(self.embed_dim, self.embed_dim, bias=True),
            '2': nn.Linear(self.embed_dim, self.embed_dim, bias=True),
            '3': nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        })

        for k, v in self.q_proj_dict.items():
            if k != 0:
                nn.init.xavier_uniform_(v.weight, gain=1.0)

        self.out_proj = layer._modules['encoder_attn']._modules['out_proj']

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self, hidden_states, key_value_states=None,
            past_key_value=None, attention_mask=None, layer_head_mask=None,
            output_attentions=False, task_id='0'):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, _ = hidden_states.size()

        query_states = self.q_proj_dict[str(task_id)](hidden_states) * self.scaling
        # query_states = self.q_proj(hidden_states) * self.scaling

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len,
                                                                                 src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class StylizedDecoderLayer(nn.Module):
    def __init__(self, config, layer_id):
        super(StylizedDecoderLayer, self).__init__()
        self.embed_dim = config.d_model

        layer = layers._modules[str(layer_id)]
        self.self_attn = layer._modules['self_attn']
        self.dropout = config.dropout
        self.activation_fn = layer._modules['activation_fn']
        self.activation_dropout = config.activation_dropout

        # self.self_attn_layer_norm = layer._modules['self_attn_layer_norm']
        self_attn_layer_norm_for_task0 = nn.LayerNorm(self.embed_dim)
        self_attn_layer_norm_for_task1 = nn.LayerNorm(self.embed_dim)
        self_attn_layer_norm_for_task2 = nn.LayerNorm(self.embed_dim)
        self_attn_layer_norm_for_task3 = nn.LayerNorm(self.embed_dim)

        state_dict_temp = layer._modules['self_attn_layer_norm'].state_dict()
        self_attn_layer_norm_for_task0.load_state_dict(state_dict_temp)

        self.self_attn_layer_norm_dict = nn.ModuleDict({
            '0': self_attn_layer_norm_for_task0,
            '1': self_attn_layer_norm_for_task1,
            '2': self_attn_layer_norm_for_task2,
            '3': self_attn_layer_norm_for_task3
        })

        # stylized attention
        self.encoder_attn = StylizedBartAttention(layer_id=layer_id)

        # self.encoder_attn_layer_norm = layer._modules['encoder_attn_layer_norm']
        # encoder attention normalization layer
        encoder_attn_layer_norm_task0 = nn.LayerNorm(self.embed_dim)
        encoder_attn_layer_norm_task1 = nn.LayerNorm(self.embed_dim)
        encoder_attn_layer_norm_task2 = nn.LayerNorm(self.embed_dim)
        encoder_attn_layer_norm_task3 = nn.LayerNorm(self.embed_dim)

        state_dict_temp = layer._modules['encoder_attn_layer_norm'].state_dict()
        encoder_attn_layer_norm_task0.load_state_dict(state_dict_temp)

        self.encoder_attn_layer_norm_dict = nn.ModuleDict({
            '0': encoder_attn_layer_norm_task0,
            '1': encoder_attn_layer_norm_task1,
            '2': encoder_attn_layer_norm_task2,
            '3': encoder_attn_layer_norm_task3
        })

        # fully connected layers
        self.fc1 = layer._modules['fc1']
        self.fc2 = layer._modules['fc2']

        # self.final_layer_norm = layer._modules['final_layer_norm']
        # final normalization layer
        final_layer_norm_task0 = nn.LayerNorm(self.embed_dim)
        final_layer_norm_task1 = nn.LayerNorm(self.embed_dim)
        final_layer_norm_task2 = nn.LayerNorm(self.embed_dim)
        final_layer_norm_task3 = nn.LayerNorm(self.embed_dim)

        state_dict_temp = layer._modules['final_layer_norm'].state_dict()
        final_layer_norm_task0.load_state_dict(state_dict_temp)

        self.final_layer_norm_dict = nn.ModuleDict({
            '0': final_layer_norm_task0,
            '1': final_layer_norm_task1,
            '2': final_layer_norm_task2,
            '3': final_layer_norm_task3
        })

    def forward(
            self, hidden_states, attention_mask=None,
            encoder_hidden_states=None, encoder_attention_mask=None,
            layer_head_mask=None, cross_attn_layer_head_mask=None,
            past_key_value=None, output_attentions=False,
            use_cache=True, task_id=0):

        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        hidden_states = self.self_attn_layer_norm_dict[str(task_id)](hidden_states)
        # hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                task_id=task_id
            )

            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            hidden_states = self.encoder_attn_layer_norm_dict[str(task_id)](hidden_states)
            # hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        hidden_states = self.final_layer_norm_dict[str(task_id)](hidden_states)
        # hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class StylizedDecoder(nn.Module):
    def __init__(self, config, embed_tokens):
        super(StylizedDecoder, self).__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = decoder._modules["embed_tokens"]

        self.embed_positions = decoder._modules["embed_positions"]

        self.layers = nn.ModuleList([StylizedDecoderLayer(config, i) for i in range(6)])
        self.layernorm_embedding = decoder._modules["layernorm_embedding"]
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(inputs_embeds.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
            self, input_ids: torch.LongTensor = None, attention_mask=None, encoder_hidden_states=None,
            encoder_attention_mask=None, head_mask=None, cross_attn_head_mask=None,
            past_key_values=None, inputs_embeds=None, use_cache=None,
            output_attentions=None, output_hidden_states=None, return_dict=None, task_id=0):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module, task_id):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache, task_id)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer, task_id=task_id),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    task_id=task_id
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

            # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class ProjectModel(nn.Module):
    def __init__(self, config):
        super(ProjectModel, self).__init__()

        shared, encoder, decoder = model._modules['shared'], model._modules['encoder'], model._modules['decoder']
        self.shared = shared
        self.encoder = encoder
        self.decoder = StylizedDecoder(config, self.shared)
        self.config = config

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, head_mask=None, decoder_head_mask=None,
                encoder_outputs=None, past_key_values=None, inputs_embeds=None,
                decoder_inputs_embeds=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, task_id=0
                ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task_id=task_id
        )
        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class ModelSeq2SeqGeneration(nn.Module):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__()
        self.model = ProjectModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.config = config

        # # Initialize weights and apply final processing
        # self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self, input_ids=None, attention_mask=None,
            decoder_input_ids=None, decoder_attention_mask=None, head_mask=None,
            decoder_head_mask=None, cross_attn_head_mask=None, encoder_outputs=None,
            past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None,
            labels=None, use_cache=None, output_attentions=None, output_hidden_states=None,
            return_dict=None, task_id=0):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask, head_mask=head_mask,
                             decoder_head_mask=decoder_head_mask,
                             encoder_outputs=encoder_outputs, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds,
                             decoder_inputs_embeds=None, use_cache=use_cache, output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states, return_dict=return_dict, task_id=task_id
                             )

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # print(masked_lm_loss)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


model = ModelSeq2SeqGeneration(config)
print(model)
model.to(device)
# print(model)
optimizer = AdamW(model.parameters(), lr=LR)

num_training_steps = NUM_epochs * len(train_datasetLoaderTask1)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

#
progress_bar = tqdm(range(num_training_steps))

# print(torch.cuda.memory_summary())

def Add_random_mask(batch_of_data):
    for i in range(BATCH_SIZE):
        label_tensor = batch_of_data['input_ids'][i]
        attention_mask_tensor = batch_of_data['attention_mask'][i]
        for i2 in range(1, label_tensor.shape[0]):
            if attention_mask_tensor[i2] == 0:
                break
            else:
                if i2 < 510:
                    if attention_mask_tensor[i2 + 1] != 0 and (random.random() < 0.15):
                        label_tensor[i2] = 103

output_list = []
for epoch in range(NUM_epochs):
    for i, data in enumerate(train_datasetLoaderTask1):
        loss_per10 = 0
        model.train()
        data = {k: v.to(device) for k, v in data.items()}
        # labels = data['labels']
        # labels_attention_mask = data['labels_attention_mask']
        # input_ids = data['input_ids']
        # attention_mask = data['attention_mask']

        loss = model(input_ids=data['input_ids'], attention_mask=data['attention_mask'],
                     labels=data['labels'], task_id=0)[0]

        if i % 3 == 0:
            data2 = None
            task_id = 1
            for i2, data2_ in enumerate(train_datasetLoaderTask2):
                data2 = data2_
                Add_random_mask(data2)
                break
        elif i % 3 == 1:
            data2 = None
            task_id = 2
            for i2, data2_ in enumerate(train_datasetLoaderTask3):
                data2 = data2_
                Add_random_mask(data2)
                break
        elif i % 3 == 2:
            data2 = None
            task_id = 3
            for i2, data2_ in enumerate(train_datasetLoaderTask4):
                data2 = data2_
                Add_random_mask(data2)
                break


        data2 = {k: v.to(device) for k, v in data2.items()}

        loss2 = model(input_ids=data2['input_ids'], attention_mask=data2['attention_mask'],
                     labels=data2['labels'], task_id=task_id)[0]

        loss_total = loss + 2 * loss2

        loss_total.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        loss_per10 = loss_per10 + loss_total.to('cpu').detach().numpy()
        if (i % 1000 == 0) and (i != 0):
            output_list.append(loss_per10)
            y = output_list
            x = list(range(len(output_list)))
            plt.plot(x, y)
            plt.show()

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': output_list,
        'step': i,
    }, '/modelcheckpoint/model_lambda2_epoch' + str(epoch) + '.pt')
