//https://github.com/ToluClassics/candle-tutorial?tab=readme-ov-file#31-roberta
use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};

use crate::models::modelling_outputs::{SequenceClassifierOutput, TokenClassifierOutput, QuestionAnsweringModelOutput};
use crate::models::model_utils::{Dropout, HiddenAct, Linear, HiddenActLayer, LayerNorm, PositionEmbeddingType};
use crate::models::model_utils::binary_cross_entropy_with_logit;
use serde::Deserialize;

pub const FLOATING_DTYPE: DType = DType::F32;
pub const LONG_DTYPE: DType = DType::I64;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DebertaV2Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    hidden_act: String,
    hidden_dropout_prob: f64,
    attention_probs_dropout_prob: f64,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    initializer_range: f64,
    layer_norm_eps: f64,
    relative_attention: bool,
    max_relative_positions: i32,
    pad_token_id: usize,
    position_biased_input: bool,
    pos_att_type: Option<Vec<String>>,
    pooler_dropout: f64,
    pooler_hidden_act: String,
    pooler_hidden_size: usize,
    model_type: Option<String>,
}

impl Default for DebertaV2Config {
    fn default() -> Self {
        Self {
            vocab_size: 128100,
            hidden_size: 1536,
            num_hidden_layers: 24,
            num_attention_heads: 24,
            intermediate_size: 6144,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-7,
            relative_attention: false,
            max_relative_positions: -1,
            pad_token_id: 0,
            position_biased_input: true,
            pos_att_type: None,
            pooler_dropout: 0.0,
            pooler_hidden_act: "gelu".to_string(),
            pooler_hidden_size: 1536,
            model_type: Some("deberta-v2".to_string()),
        }
    }
}


fn cumsum_2d(mask: &Tensor, dim: u8, device: &Device) -> Result<Tensor> {
    let mask = mask.to_vec2::<u8>()?;

    let rows = mask.len();
    let cols = mask[0].len();

    let mut result = mask.clone();

    match dim {
        0 => {
            // Cumulative sum along rows
            for i in 0..rows {
                for j in 1..cols {
                    result[i][j] += result[i][j - 1];
                }
            }
        }
        1 => {
            // Cumulative sum along columns
            for j in 0..cols {
                for i in 1..rows {
                    result[i][j] += result[i - 1][j];
                }
            }
        }
        _ => panic!("Dimension not supported"),
    }

    let result = Tensor::new(result, &device)?;

    Ok(result)
}

pub fn create_position_ids_from_input_ids(
    input_ids: &Tensor,
    padding_idx: u32,
    past_key_values_length: u8,
) -> Result<Tensor> {
    let mask = input_ids.ne(padding_idx)?;
    let incremental_indices = cumsum_2d(&mask, 0, input_ids.device())?;

    let incremental_indices = incremental_indices
        .broadcast_add(&Tensor::new(&[past_key_values_length], input_ids.device())?)?;

    Ok(incremental_indices)
}

fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
}

fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), "weight")?;
    let bias = vb.get(size2, "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let (weight, bias) = match (vb.get(size, "weight"), vb.get(size, "bias")) {
        (Ok(weight), Ok(bias)) => (weight, bias),
        (Err(err), _) | (_, Err(err)) => {
            if let (Ok(weight), Ok(bias)) = (vb.get(size, "gamma"), vb.get(size, "beta")) {
                (weight, bias)
            } else {
                return Err(err);
            }
        }
    };
    Ok(LayerNorm::new(weight, bias, eps))
}

pub struct DebertaV2Embeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    pub padding_idx: u32,
}

impl DebertaV2Embeddings {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = if config.position_biased_input {
            Some(embedding(
                config.max_position_embeddings,
                config.hidden_size,
                vb.pp("position_embeddings"),
            )?)
        } else {
            None
        };
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let padding_idx = config.pad_token_id as u32;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
            padding_idx,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
    ) -> Result<Tensor> {
        let position_ids = match position_ids {
            Some(ids) => ids.to_owned(),
            None => {
                if Option::is_some(&inputs_embeds) {
                    let position_ids =
                        self.create_position_ids_from_input_embeds(inputs_embeds.unwrap())?;
                    position_ids
                } else {
                    let position_ids =
                        create_position_ids_from_input_ids(input_ids, self.padding_idx, 1)?;
                    position_ids
                }
            }
        };

        let inputs_embeds: Tensor = match inputs_embeds {
            Some(embeds) => embeds.to_owned(),
            None => {
                let embeds = self.word_embeddings.forward(input_ids)?;
                embeds
            }
        };

        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let mut embeddings = (inputs_embeds + token_type_embeddings)?;

        if let Some(position_embeddings) = &self.position_embeddings {
            embeddings = embeddings.broadcast_add(&position_embeddings.forward(&position_ids)?)?
        }

        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings)?;

        Ok(embeddings)
    }

    pub fn create_position_ids_from_input_embeds(&self, input_embeds: &Tensor) -> Result<Tensor> {
        let input_shape = input_embeds.dims3()?;
        let seq_length = input_shape.1;

        let mut position_ids = Tensor::arange(
            self.padding_idx + 1,
            seq_length as u32 + self.padding_idx + 1,
            &Device::Cpu,
        )?;

        position_ids = position_ids
            .unsqueeze(0)?
            .expand((input_shape.0, input_shape.1))?;
        Ok(position_ids)
    }
}

struct DebertaV2SelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    relative_attention: bool,
    max_relative_positions: i32,
}

impl DebertaV2SelfAttention {
    fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let dropout = Dropout::new(config.attention_probs_dropout_prob);
        let hidden_size = config.hidden_size;
        let query = linear(hidden_size, all_head_size, vb.pp("query"))?;
        let value = linear(hidden_size, all_head_size, vb.pp("value"))?;
        let key = linear(hidden_size, all_head_size, vb.pp("key"))?;
        Ok(Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            relative_attention: config.relative_attention,
            max_relative_positions: config.max_relative_positions,
        })
    }

    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let mut new_x_shape = xs.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(self.num_attention_heads);
        new_x_shape.push(self.attention_head_size);
        let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
        xs.contiguous()
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        let attention_scores = query_layer.matmul(&key_layer.t()?)?;
        let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
        let attention_probs =
            { candle_nn::ops::softmax(&attention_scores, candle_core::D::Minus1)? };
        let attention_probs = self.dropout.forward(&attention_probs)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
        let context_layer = context_layer.flatten_from(candle_core::D::Minus2)?;
        Ok(context_layer)
    }
}

struct DebertaV2SelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl DebertaV2SelfOutput {
    fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

struct DebertaV2Attention {
    self_attention: DebertaV2SelfAttention,
    self_output: DebertaV2SelfOutput,
}

impl DebertaV2Attention {
    fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let self_attention = DebertaV2SelfAttention::load(vb.pp("self"), config)?;
        let self_output = DebertaV2SelfOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            self_attention,
            self_output,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward(hidden_states)?;
        let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
        Ok(attention_output)
    }
}

struct DebertaV2Intermediate {
    dense: Linear,
    intermediate_act: HiddenActLayer,
}

impl DebertaV2Intermediate {
    fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            intermediate_act: HiddenActLayer::new(&config.hidden_act),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}

struct DebertaV2Output {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl DebertaV2Output {
    fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);
        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

struct DebertaV2Layer {
    attention: DebertaV2Attention,
    intermediate: DebertaV2Intermediate,
    output: DebertaV2Output,
}

impl DebertaV2Layer {
    fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let attention = DebertaV2Attention::load(vb.pp("attention"), config)?;
        let intermediate = DebertaV2Intermediate::load(vb.pp("intermediate"), config)?;
        let output = DebertaV2Output::load(vb.pp("output"), config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self
            .output
            .forward(&intermediate_output, &attention_output)?;
        Ok(layer_output)
    }
}

struct DebertaV2Encoder {
    layers: Vec<DebertaV2Layer>,
}

impl DebertaV2Encoder {
    fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| DebertaV2Layer::load(vb.pp(&format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        Ok(DebertaV2Encoder { layers })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states)?
        }
        Ok(hidden_states)
    }
}

pub struct DebertaV2Model {
    embeddings: DebertaV2Embeddings,
    encoder: DebertaV2Encoder,
    pub device: Device,
}

impl DebertaV2Model {
    pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let (embeddings, encoder) = match (
            DebertaV2Embeddings::load(vb.pp("embeddings"), config),
            DebertaV2Encoder::load(vb.pp("encoder"), config),
        ) {
            (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
            (Err(err), _) | (_, Err(err)) => {
                if let Some(model_type) = &config.model_type {
                    if let (Ok(embeddings), Ok(encoder)) = (
                        DebertaV2Embeddings::load(vb.pp(&format!("{model_type}.embeddings")), config),
                        DebertaV2Encoder::load(vb.pp(&format!("{model_type}.encoder")), config),
                    ) {
                        (embeddings, encoder)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            embeddings,
            encoder,
            device: vb.device().clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let embedding_output = self
            .embeddings
            .forward(input_ids, token_type_ids, None, None)?;
        let sequence_output = self.encoder.forward(&embedding_output)?;
        Ok(sequence_output)
    }
}

