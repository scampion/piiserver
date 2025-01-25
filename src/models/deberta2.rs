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

