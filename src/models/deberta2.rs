//https://github.com/ToluClassics/candle-tutorial?tab=readme-ov-file#31-roberta
use crate::models::model_utils::Linear;
use crate::models::model_utils::LayerNorm;
use crate::models::model_utils::Dropout;
use crate::models::model_utils::HiddenAct;
use crate::models::model_utils::HiddenActLayer;
use candle_nn::Activation;
use candle_nn::Conv1d;
use candle_nn::Conv1dConfig;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, embedding, Embedding, VarBuilder};
//use crate::models::modelling_outputs::{SequenceClassifierOutput, TokenClassifierOutput, QuestionAnsweringModelOutput};
//use crate::models::model_utils::{HiddenAct, HiddenActLayer, Linear, LayerNorm, PositionEmbeddingType};


// use crate::models::model_utils::binary_cross_entropy_with_logit;
use serde::Deserialize;

pub const FLOATING_DTYPE: DType = DType::F32;
pub const LONG_DTYPE: DType = DType::I64;


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

fn layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<crate::models::model_utils::LayerNorm> {
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
    Ok(crate::models::model_utils::LayerNorm::new(weight, bias, eps))
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


fn build_relative_position(
    query_layer: &Tensor,
    key_layer: &Tensor,
    bucket_size: i64,
    max_position: i64,
) -> Result<Tensor> {
    let query_size = query_layer.dim(query_layer.dims().len() - 2)?;
    let key_size = key_layer.dim(key_layer.dims().len() - 2)?;

    // Create tensors for query and key IDs
    let q_ids = Tensor::arange(0, query_size as i64, &Device::Cpu)?.to_dtype(DType::I64)?;
    let k_ids = Tensor::arange(0, key_size as i64, &Device::Cpu)?.to_dtype(DType::I64)?;

    // Compute relative positions
    let rel_pos_ids = q_ids.unsqueeze(1)?.broadcast_sub(&k_ids.unsqueeze(0)?)?;

    // Apply bucketization if bucket_size and max_position are provided
    let rel_pos_ids = if bucket_size > 0 && max_position > 0 {
        make_log_bucket_position(&rel_pos_ids, bucket_size, max_position)?
    } else {
        rel_pos_ids
    };

    // Ensure the tensor is of type i64
    let rel_pos_ids = rel_pos_ids.to_dtype(DType::I64)?;

    // Truncate to the query size and add a batch dimension
    let rel_pos_ids = rel_pos_ids.narrow(0, 0, query_size)?;
    let rel_pos_ids = rel_pos_ids.unsqueeze(0)?;

    Ok(rel_pos_ids)
}

fn make_log_bucket_position(rel_pos_ids: &Tensor, bucket_size: i64, max_position: i64) -> Result<Tensor> {
    // Placeholder for the bucketing logic
    // You would implement the specific bucketing logic here
    Ok(rel_pos_ids.clone())
}


#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DebertaV2Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: i64,
    pub type_vocab_size: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f64,
    pub relative_attention: bool,
    pub max_relative_positions: i64,
    pub pad_token_id: usize,
    pub position_biased_input: bool,
    pub pos_att_type: Option<Vec<String>>,
    pub pooler_dropout: f32,
    pub pooler_hidden_act: String,
    pub legacy: bool,
    pub position_buckets: i64,
    pub norm_rel_ebd: Option<String>,
    pub conv_kernel_size: usize,
    pub share_att_key: bool,
}

impl Default for DebertaV2Config {
    fn default() -> Self {
        DebertaV2Config {
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
            legacy: true,
            position_buckets: -1,
            norm_rel_ebd: None,
            conv_kernel_size: 0,
            share_att_key: true,

        }
    }
}



pub struct DebertaV2Embeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    embed_proj: Option<Linear>,
    layer_norm: LayerNorm,
    dropout: Dropout,
    pub padding_idx: u32,
    position_biased_input: bool,
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
                config.max_position_embeddings as usize,
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

        let embed_proj = if config.hidden_size != config.intermediate_size {
            Some(linear(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("embed_proj"),
            )?)
        } else {
            None
        };

        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        let padding_idx = config.pad_token_id as u32;

        let position_biased_input = config.position_biased_input;

        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            embed_proj,
            layer_norm,
            dropout: Dropout::new(config.hidden_dropout_prob),
            padding_idx,
            position_biased_input
        }
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: Option<&Tensor>,
        mask: Option<&Tensor>,
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

        let position_embeddings = if let Some(position_embeddings) = &self.position_embeddings {
            position_embeddings.forward(&position_ids.to_dtype(DType::U32)?)?
        } else {
            Tensor::zeros_like(&inputs_embeds)?
        };

        let mut embeddings = inputs_embeds;
        if self.position_biased_input {
            embeddings = embeddings.add(&position_embeddings)?;
        }

        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        embeddings = embeddings.add(&token_type_embeddings)?;


        if let Some(embed_proj) = &self.embed_proj {
            embeddings = embed_proj.forward(&embeddings)?;
        }

        embeddings = self.layer_norm.forward(&embeddings)?;

        if let Some(mask) = mask {
            let mut mask = mask.clone();
            if mask.dims().len() != embeddings.dims().len() {
                if mask.dims().len() == 4 {
                    mask = mask.squeeze(1)?.squeeze(1)?;
                }
                mask = mask.unsqueeze(2)?;
            }
            mask = mask.to_dtype(embeddings.dtype())?;
            embeddings = embeddings.mul(&mask)?;
        }

        let embeddings = self.dropout.forward(&embeddings)?;
        Ok(embeddings)
    }

    pub fn create_position_ids_from_input_embeds(&self, input_embeds: &Tensor) -> Result<Tensor> {
        let input_shape = input_embeds.dims3()?;
        let seq_length = input_shape.1;

        println!("seq_length: {:?}", seq_length);
        let mut position_ids = Tensor::arange(
            self.padding_idx + 1,
            seq_length as u32 + self.padding_idx + 1,
            &Device::Cpu,
        )?;

        println!("position_ids: {:?}", position_ids);

        position_ids = position_ids
            .unsqueeze(0)?
            .expand((input_shape.0, input_shape.1))?;
        Ok(position_ids)
    }
}

pub struct DebertaV2Intermediate {
    dense: Linear,
    intermediate_act: HiddenActLayer,
}

impl DebertaV2Intermediate {
    fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
        let intermediate_act = match config.hidden_act.as_str() {
            "gelu" => HiddenActLayer::new(HiddenAct::Gelu),
            "tanh" => HiddenActLayer::new(HiddenAct::Tanh),
            "relu" => HiddenActLayer::new(HiddenAct::Relu),
            _ => HiddenActLayer::new(HiddenAct::Gelu),
        };

        Ok(Self {
            dense,
            intermediate_act,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let ys = self.intermediate_act.forward(&hidden_states)?;
        Ok(ys)
    }
}



pub struct DebertaV2Output {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout
}


impl DebertaV2Output {
    fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let dense = linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("dense"),
        )?;

        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        let dropout = Dropout::new(config.hidden_dropout_prob as f64);

        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.layer_norm.forward(&(hidden_states + input_tensor)?)
    }
}

pub struct DebertaV2Layer {
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

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
        output_attentions: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Forward pass through the attention layer
        let (attention_output, att_matrix) = self.attention.forward(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states,
            relative_pos,
            rel_embeddings,
        );

        // Forward pass through the intermediate layer
        let intermediate_output = self.intermediate.forward(&attention_output)?;

        // Forward pass through the output layer
        let layer_output = self.output.forward(&intermediate_output, &attention_output)?;

        // Return the layer output and attention matrix (if requested)
        if output_attentions {
            Ok((layer_output, att_matrix))
        } else {
            Ok((layer_output, None))
        }

    }
}


pub struct DebertaV2Encoder {
    layer: Vec<DebertaV2Layer>,
    relative_attention: bool,
    max_relative_positions: i64,
    position_buckets: i64,
    rel_embeddings: Option<Embedding>,
    norm_rel_ebd: Vec<String>,
    layer_norm: Option<LayerNorm>,
    conv: Option<ConvLayer>,
    gradient_checkpointing: bool,
}

impl DebertaV2Encoder {
    fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let layer = (0..config.num_hidden_layers)
            .map(|index| DebertaV2Layer::load(vb.pp(&format!("layer.{index}")), config))
            .collect();
        // Ok(DebertaV2Encoder { layers })
        let layer: Vec<DebertaV2Layer> = (0..config.num_hidden_layers)
            .map(|index| {
                DebertaV2Layer::load(vb.pp(&format!("layer.{index}")), config)
                    .map_err(|e| format!("Failed to load layer {index}: {e}"))
            })
            .collect()?;

        let relative_attention = config.relative_attention;
        let max_relative_positions: i64 = if config.max_relative_positions < 1 {
            config.max_position_embeddings
        } else {
            config.max_relative_positions
        };

        let position_buckets = config.position_buckets;
        let pos_ebd_size = if position_buckets > 0 {
            position_buckets * 2
        } else {
            max_relative_positions * 2
        };

        let rel_embeddings = if relative_attention {
            Some(embedding(
                pos_ebd_size,
                config.hidden_size,
                vb.pp("rel_embeddings"),
            )?)
        } else {
            None
        };

        let norm_rel_ebd = config
            .norm_rel_ebd
            .as_ref()
            .map(|s| s.to_lowercase().split('|').map(|x| x.trim().to_string()).collect())
            .unwrap_or_else(|| vec!["none".to_string()]);

        let layer_norm = if norm_rel_ebd.contains(&"layer_norm".to_string()) {
            Some(layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb.pp("layer_norm"),
            )?)
        } else {
            None
        };


        let conv = if config.conv_kernel_size > 0 {
            Some(ConvLayer::load(vb, config))
        } else {
            None
        };

        Ok(Self {
            layer,
            relative_attention,
            max_relative_positions,
            position_buckets,
            rel_embeddings,
            norm_rel_ebd,
            layer_norm,
            conv,
            gradient_checkpointing: false,
        })
    }

    pub fn get_rel_embedding(&self) -> Option<Tensor> {
        if let Some(rel_embeddings) = &self.rel_embeddings {
            // Access the embeddings tensor directly since Embedding doesn't have a weight() method
            let mut embeddings = rel_embeddings.embeddings().clone();
            if self.norm_rel_ebd.contains(&"layer_norm".to_string()) {
                //embeddings = self.layer_norm.as_ref().unwrap().forward(&embeddings)?;
                embeddings = &self.layer_norm.forward(&embeddings);
            }
            Some(embeddings)
        } else {
            None
        }
    }

    pub fn get_attention_mask(&self, attention_mask: &Tensor) -> Result<Tensor> {
        let dims = attention_mask.dims();
        if dims.len() <= 2 {
            let extended_attention_mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;
            Ok(extended_attention_mask * extended_attention_mask.squeeze(-2)?.unsqueeze(-1)?)
        } else if dims.len() == 3 {
            Ok(attention_mask.unsqueeze(1)?)
        } else {
            Ok(attention_mask.clone())
        }
    }

    pub fn get_rel_pos(
        &self,
        hidden_states: &Tensor,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> Option<Tensor> {
        if self.relative_attention && relative_pos.is_none() {
            let q_states = query_states.unwrap_or(hidden_states);
            Some(build_relative_position(
                q_states,
                hidden_states,
                self.position_buckets,
                self.max_relative_positions,
            ))
        } else {
            relative_pos.cloned()
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        output_hidden_states: bool,
        output_attentions: bool,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
    ) -> (Tensor, Option<Vec<Tensor>>, Option<Vec<Tensor>>) {
        let input_mask = if attention_mask.dims() <= 2 {
            attention_mask.clone()
        } else {
            //attention_mask.sum_dim_intlist(&[-2], false, Kind::Bool) > 0
            attention_mask.sum_keepdim(-2)?.gt(0)?
        };

        let attention_mask = self.get_attention_mask(attention_mask);
        let relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos);

        let mut all_hidden_states = if output_hidden_states {
            Some(vec![hidden_states.clone()])
        } else {
            None
        };

        let mut all_attentions = if output_attentions {
            Some(Vec::new())
        } else {
            None
        };

        let mut next_kv = hidden_states.clone();
        let rel_embeddings = self.get_rel_embedding();

        for (i, layer_module) in self.layer.iter().enumerate() {
            let (output_states, attn_weights) = layer_module.forward(
                &next_kv,
                &attention_mask,
                query_states,
                relative_pos.as_ref(),
                rel_embeddings.as_ref(),
                output_attentions,
            );

            if output_attentions {
                all_attentions.as_mut().unwrap().push(attn_weights.unwrap());
            }

            if i == 0 && self.conv.is_some() {
                next_kv = self.conv.as_ref().unwrap().forward(hidden_states, &output_states, &input_mask);
            } else {
                next_kv = output_states.clone();
            }

            if output_hidden_states {
                all_hidden_states.as_mut().unwrap().push(next_kv.clone());
            }

            if query_states.is_some() {
                query_states = Some(&output_states);
            }
        }

        (next_kv, all_hidden_states, all_attentions)
    }
}

struct DebertaV2Attention {
    self_attention: DisentangledSelfAttention,
    output: DebertaV2SelfOutput
}

impl DebertaV2Attention {
    fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let self_attention = DisentangledSelfAttention::load(vb.pp("self"), config)?;
        let output = DebertaV2SelfOutput::load(vb.pp("output"), config)?;
        Ok(Self {
            self_attention,
            output,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        output_attentions: bool,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> (Tensor, Option<Tensor>) {
        let (self_output, att_matrix) = self.self_attention.forward(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states,
            relative_pos,
            rel_embeddings,
        );

        let query_states = query_states.unwrap_or(hidden_states);
        let attention_output = self.output.forward(&self_output, query_states);

        if output_attentions {
            (attention_output, att_matrix)
        } else {
            (attention_output, None)
        }
    }
}


pub struct DebertaV2SelfOutput {
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
        let hidden_states = self.layer_norm.forward(&(hidden_states + input_tensor)?)?;
        Ok(hidden_states)
    }
}



#[derive(Debug)]
pub struct DisentangledSelfAttentionConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub attention_head_size: Option<usize>,
    pub share_att_key: bool,
    pub pos_att_type: Vec<String>,
    pub relative_attention: bool,
    pub position_buckets: Option<usize>,
    pub max_relative_positions: Option<usize>,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: usize,
}

#[derive(Debug)]
pub struct DisentangledSelfAttention {
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    query_proj: Linear,
    key_proj: Linear,
    value_proj: Linear,
    share_att_key: bool,
    pos_att_type: Option<Vec<String>>,
    relative_attention: bool,
    position_buckets: Option<usize>,
    max_relative_positions: usize,
    pos_ebd_size: i64,
    pos_dropout: Dropout,
    pos_key_proj: Option<Linear>,
    pos_query_proj: Option<Linear>,
    dropout: Dropout,
}

impl DisentangledSelfAttention {
    // pub fn new(config: DisentangledSelfAttentionConfig, vb: VarBuilder) -> Result<Self, Box<dyn std::error::Error>> {
    fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;

        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;


        if hidden_size % num_attention_heads != 0 {
            return Err(format!(
                "The hidden size ({}) is not a multiple of the number of attention heads ({})",
                hidden_size, num_attention_heads
            ).into());
        }

        let query_proj = linear(hidden_size, all_head_size, vb.pp("query_proj"))?;
        let key_proj = linear(hidden_size, all_head_size, vb.pp("key_proj"))?;
        let value_proj = linear(hidden_size, all_head_size, vb.pp("value_proj"))?;

        let share_att_key = config.share_att_key;
        let pos_att_type = config.pos_att_type;
        let relative_attention = config.relative_attention;

        let position_buckets = config.position_buckets;
        let max_relative_positions = config.max_relative_positions.unwrap_or(config.max_position_embeddings);
        let pos_ebd_size = if position_buckets > 0 {
            position_buckets
        } else {
            max_relative_positions
        };

        let pos_dropout = Dropout::new(config.hidden_dropout_prob);

        let pos_key_proj = if !share_att_key && pos_att_type.as_ref().map_or(false, |v| v.contains(&"c2p".to_string())) {
            Some(linear(hidden_size, all_head_size, vb.pp("pos_key_proj"))?)
        } else {
            None
        };

        let pos_query_proj = if !share_att_key && pos_att_type.as_ref().map_or(false, |v| v.contains(&"p2c".to_string())) {
            Some(linear(hidden_size, all_head_size, vb.pp("pos_query_proj"))?)
        } else {
            None
        };

        let dropout = Dropout::new(config.attention_probs_dropout_prob);

        Ok(Self {
            num_attention_heads,
            attention_head_size,
            all_head_size,
            query_proj,
            key_proj,
            value_proj,
            share_att_key,
            pos_att_type,
            relative_attention,
            position_buckets,
            max_relative_positions,
            pos_ebd_size,
            pos_dropout,
            pos_key_proj,
            pos_query_proj,
            dropout,
        })
    }

    fn transpose_for_scores(&self, x: Tensor, attention_heads: usize) -> Result<Tensor> {
        let mut new_x_shape = x.dims().to_vec();
        new_x_shape.pop();
        new_x_shape.push(attention_heads);
        new_x_shape.push(self.attention_head_size);
        let x = x.reshape(&new_x_shape)?;
        x.permute(&[0, 2, 1, 3])?.contiguous()?.reshape(&[-1, x.dim(1)?, x.dim(3)?])
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        output_attentions: bool,
        query_states: Option<&Tensor>,
        relative_pos: Option<&Tensor>,
        rel_embeddings: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let query_states = query_states.unwrap_or(hidden_states);
        let query_layer = self.transpose_for_scores(self.query_proj.forward(query_states)?, self.num_attention_heads)?;
        let key_layer = self.transpose_for_scores(self.key_proj.forward(hidden_states)?, self.num_attention_heads)?;
        let value_layer = self.transpose_for_scores(self.value_proj.forward(hidden_states)?, self.num_attention_heads)?;

        let mut rel_att = None;
        let scale_factor = 1 + self.pos_att_type.iter().filter(|&x| x == "c2p" || x == "p2c").count();
        let scale = (query_layer.dim(2)? as f64).sqrt() * scale_factor as f64;
        let mut attention_scores = query_layer.matmul(&key_layer.transpose(1, 2)?)? / scale;

        if self.relative_attention {
            if let Some(rel_embeddings) = rel_embeddings {
                let rel_embeddings = self.pos_dropout.forward(rel_embeddings)?;
                rel_att = self.disentangled_attention_bias(&query_layer, &key_layer, relative_pos, &rel_embeddings, scale_factor)?;
            }
        }

        if let Some(rel_att) = rel_att {
            attention_scores = attention_scores.add(&rel_att)?;
        }

        let attention_scores = attention_scores?.reshape(&[
            -1,
            self.num_attention_heads,
            attention_scores.dim(1)?,
            attention_scores.dim(2)?,
        ])?;

        let attention_mask = attention_mask.to_dtype(DType::Bool)?;
        let attention_scores = attention_scores.masked_fill(&attention_mask.logical_not()?, f32::MIN)?;

        let attention_probs = attention_scores.softmax(2)?;
        let attention_probs = self.dropout.forward(&attention_probs)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.reshape(&[
            -1,
            self.num_attention_heads,
            context_layer.dim(1)?,
            context_layer.dim(2)?,
        ])?;

        let context_layer = context_layer.permute(&[0, 2, 1, 3])?.contiguous()?;
        let new_context_layer_shape = context_layer.dims()[..context_layer.dims().len() - 2]
            .iter()
            .chain(&[self.all_head_size])
            .cloned()
            .collect::<Vec<_>>();
        let context_layer = context_layer.reshape(&new_context_layer_shape)?;

        if !output_attentions {
            Ok((context_layer, None))
        } else {
            Ok((context_layer, Some(attention_probs)))
        }
    }

    fn disentangled_attention_bias(
        &self,
        query_layer: &Tensor,
        key_layer: &Tensor,
        relative_pos: Option<&Tensor>,
        rel_embeddings: &Tensor,
        scale_factor: usize,
    ) -> Result<Tensor> {
        let relative_pos = if let Some(relative_pos) = relative_pos {
            relative_pos.clone()
        } else {
            self.build_relative_position(query_layer, key_layer)?
        };

        let relative_pos = if relative_pos.dims().len() == 2 {
            relative_pos.unsqueeze(0)?.unsqueeze(0)?
        } else if relative_pos.dims().len() == 3 {
            relative_pos.unsqueeze(1)?
        } else if relative_pos.dims().len() != 4 {
            return Err(format!(
                "Relative position ids must be of dim 2 or 3 or 4. {}",
                relative_pos.dims().len()
            ).into());
        } else {
            relative_pos
        };

        let att_span = self.pos_ebd_size;
        let relative_pos = relative_pos.to_dtype(DType::I64)?;

        let rel_embeddings = rel_embeddings.narrow(0, 0, att_span * 2)?.unsqueeze(0)?;

        let pos_query_layer = if self.share_att_key {
            self.transpose_for_scores(self.query_proj.forward(&rel_embeddings)?, self.num_attention_heads)?
                .repeat(&[query_layer.dim(0)? / self.num_attention_heads, 1, 1])?
        } else if self.pos_att_type.contains(&"p2c".to_string()) {
            self.transpose_for_scores(self.pos_query_proj.as_ref().unwrap().forward(&rel_embeddings)?, self.num_attention_heads)?
                .repeat(&[query_layer.dim(0)? / self.num_attention_heads, 1, 1])?
        } else {
            Tensor::zeros(&[0], DType::F32, query_layer.device())?
        };

        let pos_key_layer = if self.share_att_key {
            self.transpose_for_scores(self.key_proj.forward(&rel_embeddings)?, self.num_attention_heads)?
                .repeat(&[query_layer.dim(0)? / self.num_attention_heads, 1, 1])?
        } else if self.pos_att_type.contains(&"c2p".to_string()) {
            self.transpose_for_scores(self.pos_key_proj.as_ref().unwrap().forward(&rel_embeddings)?, self.num_attention_heads)?
                .repeat(&[query_layer.dim(0)? / self.num_attention_heads, 1, 1])?
        } else {
            Tensor::zeros(&[0], DType::F32, query_layer.device())?
        };

        let mut score = Tensor::zeros(&[0], DType::F32, query_layer.device())?;

        if self.pos_att_type.contains(&"c2p".to_string()) {
            let scale = (pos_key_layer.dim(2)? as f64).sqrt() * scale_factor as f64;
            let c2p_att = query_layer.matmul(&pos_key_layer.transpose(1, 2)?)?;
            let c2p_pos = relative_pos.clamp(0, att_span * 2 - 1)?;
            let c2p_att = c2p_att.gather(&c2p_pos.squeeze(0)?.expand(&[query_layer.dim(0)?, query_layer.dim(1)?, relative_pos.dim(3)?])?, 2)?;
            score = score.add(&(c2p_att / scale))?;
        }

        if self.pos_att_type.contains(&"p2c".to_string()) {
            let scale = (pos_query_layer.dim(2)? as f64).sqrt() * scale_factor as f64;
            let r_pos = self.build_rpos(query_layer, key_layer, &relative_pos)?;
            let p2c_pos = r_pos.neg()?.add(att_span)?.clamp(0, att_span * 2 - 1)?;
            let p2c_att = key_layer.matmul(&pos_query_layer.transpose(1, 2)?)?;
            let p2c_att = p2c_att.gather(&p2c_pos.squeeze(0)?.expand(&[query_layer.dim(0)?, key_layer.dim(1)?, key_layer.dim(1)?])?, 2)?.transpose(1, 2)?;
            score = score.add(&(p2c_att / scale))?;
        }

        Ok(score)
    }

    fn build_relative_position(&self, query_layer: &Tensor, key_layer: &Tensor) -> Result<Tensor> {
        // Implement the logic to build relative positions
        // This is a placeholder and should be replaced with actual logic
        Ok(Tensor::zeros(&[query_layer.dim(1)?, key_layer.dim(1)?], DType::I64, query_layer.device())?)
    }

    fn build_rpos(&self, query_layer: &Tensor, key_layer: &Tensor, relative_pos: &Tensor) -> Result<Tensor> {
        // Implement the logic to build relative positions
        // This is a placeholder and should be replaced with actual logic
        Ok(Tensor::zeros(&[query_layer.dim(1)?, key_layer.dim(1)?], DType::I64, query_layer.device())?)
    }
}


pub struct ConvLayerConfig {
    conv_kernel_size: usize,
    conv_groups: usize,
    conv_act: Activation,
    hidden_size: usize,
    layer_norm_eps: f64,
    hidden_dropout_prob: f64,
}

pub struct ConvLayer {
    conv: Conv1d,
    conv_act: Activation,
    layer_norm: LayerNorm,
    dropout: Dropout,
    config: ConvLayerConfig,
}

impl ConvLayer {
    fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
        let conv = Conv1d::new(
            config.hidden_size,
            config.hidden_size,
            config.conv_kernel_size,
            Conv1dConfig {
                padding: (config.conv_kernel_size - 1) / 2,
                groups: config.conv_groups,
                ..Default::default()
            },
            vb.pp("conv"),
        )
            .expect("Failed to create Conv1d layer");

        let layer_norm = layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("layer_norm"),
        )
            .expect("Failed to create LayerNorm");

        let dropout = Dropout::new(config.hidden_dropout_prob);

        Self {
            conv,
            conv_act: config.conv_act,
            layer_norm,
            dropout,
            config,
        }
    }

    fn forward(&self, hidden_states: &Tensor, residual_states: &Tensor, input_mask: Option<&Tensor>) ->  Result<Tensor>  {
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();

        // Permute and apply convolution
        let out = self.conv.forward(&hidden_states.permute((0, 2, 1))?)?.permute((0, 2, 1))?;

        // Apply mask if provided
        let out = if let Some(mask) = input_mask {
            let rmask = mask.eq(0.0)?;
            out.masked_fill(&rmask.unsqueeze(-1)?.expand(&out.shape())?, 0.0)?
        } else {
            out
        };

        // Apply activation and dropout
        let out = self.conv_act.forward(&self.dropout.forward(&out)?)?;

        // Add residual and apply layer normalization
        let layer_norm_input = residual_states.add(&out)?;
        let output = self.layer_norm.forward(&layer_norm_input)?;

        // Apply input mask to output if provided
        let output_states = if let Some(mask) = input_mask {
            let mask = if mask.dims().len() != layer_norm_input.dims().len() {
                if mask.dims().len() == 4 {
                    mask.squeeze(1)?.squeeze(1)?
                } else {
                    mask.unsqueeze(2)?
                }
            } else {
                mask.clone()
            };
            output.mul(&mask.to_dtype(dtype)?)?
        } else {
            output
        };

        Ok(output_states)
    }
}




// impl DebertaV2Output {
//     pub fn new(config: DebertaV2Config, vb: VarBuilder) -> Result<Self> {
//         let dense = Linear::new(
//             vb.pp("dense"),
//             config.intermediate_size,
//             config.hidden_size,
//         )?;
//         let layer_norm = LayerNorm::new(
//             vb.pp("layer_norm"),
//             config.hidden_size,
//             config.layer_norm_eps,
//         )?;
//         let dropout = Dropout::new(config.hidden_dropout_prob);
//
//         Ok(Self {
//             dense,
//             layer_norm,
//             dropout,
//             config,
//         })
//     }
// }
//
// //impl Module for DebertaV2Output {
// impl DebertaV2Output {
//     fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
//         let hidden_states = self.dense.forward(hidden_states)?;
//         let hidden_states = self.dropout.forward(&hidden_states)?;
//         let hidden_states = self.layer_norm.forward(&(hidden_states + input_tensor)?)?;
//         Ok(hidden_states)
//     }
// }
//
//


// struct DebertaV2SelfAttention {
//     query: Linear,
//     key: Linear,
//     value: Linear,
//     dropout: Dropout,
//     num_attention_heads: usize,
//     attention_head_size: usize,
//     relative_attention: bool,
//     max_relative_positions: i32,
// }
//
// impl DebertaV2SelfAttention {
//     fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
//         let attention_head_size = config.hidden_size / config.num_attention_heads;
//         let all_head_size = config.num_attention_heads * attention_head_size;
//         let dropout = Dropout::new(config.attention_probs_dropout_prob);
//         let hidden_size = config.hidden_size;
//         let query = linear(hidden_size, all_head_size, vb.pp("query_proj"))?;
//         let value = linear(hidden_size, all_head_size, vb.pp("value_proj"))?;
//         let key = linear(hidden_size, all_head_size, vb.pp("key_proj"))?;
//         Ok(Self {
//             query,
//             key,
//             value,
//             dropout,
//             num_attention_heads: config.num_attention_heads,
//             attention_head_size,
//             relative_attention: config.relative_attention,
//             max_relative_positions: config.max_relative_positions,
//         })
//     }
//
//     fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
//         let mut new_x_shape = xs.dims().to_vec();
//         new_x_shape.pop();
//         new_x_shape.push(self.num_attention_heads);
//         new_x_shape.push(self.attention_head_size);
//         let xs = xs.reshape(new_x_shape.as_slice())?.transpose(1, 2)?;
//         xs.contiguous()
//     }
//
//     fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
//         let query_layer = self.query.forward(hidden_states)?;
//         let key_layer = self.key.forward(hidden_states)?;
//         let value_layer = self.value.forward(hidden_states)?;
//
//         let query_layer = self.transpose_for_scores(&query_layer)?;
//         let key_layer = self.transpose_for_scores(&key_layer)?;
//         let value_layer = self.transpose_for_scores(&value_layer)?;
//
//         let attention_scores = query_layer.matmul(&key_layer.t()?)?;
//         let attention_scores = (attention_scores / (self.attention_head_size as f64).sqrt())?;
//         let attention_probs =
//             { candle_nn::ops::softmax(&attention_scores, candle_core::D::Minus1)? };
//         let attention_probs = self.dropout.forward(&attention_probs)?;
//
//         let context_layer = attention_probs.matmul(&value_layer)?;
//         let context_layer = context_layer.transpose(1, 2)?.contiguous()?;
//         let context_layer = context_layer.flatten_from(candle_core::D::Minus2)?;
//         Ok(context_layer)
//     }
// }
//
// struct DebertaV2SelfOutput {
//     dense: Linear,
//     layer_norm: LayerNorm,
//     dropout: Dropout,
// }
//
// impl DebertaV2SelfOutput {
//     fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
//         let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
//         let layer_norm = layer_norm(
//             config.hidden_size,
//             config.layer_norm_eps,
//             vb.pp("LayerNorm"),
//         )?;
//         let dropout = Dropout::new(config.hidden_dropout_prob);
//         Ok(Self {
//             dense,
//             layer_norm,
//             dropout,
//         })
//     }
//
//     fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
//         let hidden_states = self.dense.forward(hidden_states)?;
//         let hidden_states = self.dropout.forward(&hidden_states)?;
//         self.layer_norm.forward(&(hidden_states + input_tensor)?)
//     }
// }
//
// struct DebertaV2Attention {
//     self_attention: DebertaV2SelfAttention,
//     self_output: DebertaV2SelfOutput,
// }
//
// impl DebertaV2Attention {
//     fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
//         let self_attention = DebertaV2SelfAttention::load(vb.pp("self"), config)?;
//         let self_output = DebertaV2SelfOutput::load(vb.pp("output"), config)?;
//         Ok(Self {
//             self_attention,
//             self_output,
//         })
//     }
//
//     fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
//         let self_outputs = self.self_attention.forward(hidden_states)?;
//         let attention_output = self.self_output.forward(&self_outputs, hidden_states)?;
//         Ok(attention_output)
//     }
// }
//
// struct DebertaV2Intermediate {
//     dense: Linear,
//     intermediate_act: HiddenActLayer,
// }
//
// impl DebertaV2Intermediate {
//     fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
//         let dense = linear(config.hidden_size, config.intermediate_size, vb.pp("dense"))?;
//         Ok(Self {
//             dense,
//             intermediate_act: HiddenActLayer::new(config.hidden_act),
//         })
//     }
//
//     fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
//         let hidden_states = self.dense.forward(hidden_states)?;
//         let ys = self.intermediate_act.forward(&hidden_states)?;
//         Ok(ys)
//     }
// }
//
// struct DebertaV2Output {
//     dense: Linear,
//     layer_norm: LayerNorm,
//     dropout: Dropout,
// }
//
// impl DebertaV2Output {
//     fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
//         let dense = linear(config.intermediate_size, config.hidden_size, vb.pp("dense"))?;
//         let layer_norm = layer_norm(
//             config.hidden_size,
//             config.layer_norm_eps,
//             vb.pp("LayerNorm"),
//         )?;
//         let dropout = Dropout::new(config.hidden_dropout_prob);
//         Ok(Self {
//             dense,
//             layer_norm,
//             dropout,
//         })
//     }
//
//     fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
//         let hidden_states = self.dense.forward(hidden_states)?;
//         let hidden_states = self.dropout.forward(&hidden_states)?;
//         self.layer_norm.forward(&(hidden_states + input_tensor)?)
//     }
// }
//
// struct DebertaV2Layer {
//     attention: DebertaV2Attention,
//     intermediate: DebertaV2Intermediate,
//     output: DebertaV2Output,
// }
//
// impl DebertaV2Layer {
//     fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
//         let attention = DebertaV2Attention::load(vb.pp("attention"), config)?;
//         let intermediate = DebertaV2Intermediate::load(vb.pp("intermediate"), config)?;
//         let output = DebertaV2Output::load(vb.pp("output"), config)?;
//         Ok(Self {
//             attention,
//             intermediate,
//             output,
//         })
//     }
//
//     fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
//         let attention_output = self.attention.forward(hidden_states)?;
//         let intermediate_output = self.intermediate.forward(&attention_output)?;
//         let layer_output = self
//             .output
//             .forward(&intermediate_output, &attention_output)?;
//         Ok(layer_output)
//     }
// }
//
// struct DebertaV2Encoder {
//     layers: Vec<DebertaV2Layer>,
// }
//
// impl DebertaV2Encoder {
//     fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
//         let layers = (0..config.num_hidden_layers)
//             .map(|index| DebertaV2Layer::load(vb.pp(&format!("layer.{index}")), config))
//             .collect::<Result<Vec<_>>>()?;
//         Ok(DebertaV2Encoder { layers })
//     }
//
//     fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
//         let mut hidden_states = hidden_states.clone();
//         for layer in self.layers.iter() {
//             hidden_states = layer.forward(&hidden_states)?
//         }
//         Ok(hidden_states)
//     }
// }
//
// pub struct DebertaV2Model {
//     embeddings: DebertaV2Embeddings,
//     encoder: DebertaV2Encoder,
//     pub device: Device,
// }
//
// impl DebertaV2Model {
//     pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
//         let (embeddings, encoder) = match (
//             DebertaV2Embeddings::load(vb.pp("embeddings"), config),
//             DebertaV2Encoder::load(vb.pp("encoder"), config),
//         ) {
//             (Ok(embeddings), Ok(encoder)) => (embeddings, encoder),
//             (Err(err), _) | (_, Err(err)) => {
//                 println!("Error in loading embeddings and encoder");
//                 if let Some(model_type) = &config.model_type {
//                     if let (Ok(embeddings), Ok(encoder)) = (
//                         DebertaV2Embeddings::load(vb.pp(&format!("{model_type}.embeddings")), config),
//                         DebertaV2Encoder::load(vb.pp(&format!("{model_type}.encoder")), config),
//                     ) {
//                         (embeddings, encoder)
//                     } else {
//                         return Err(err);
//                     }
//                 } else {
//                     return Err(err);
//                 }
//             }
//         };
//         Ok(Self {
//             embeddings,
//             encoder,
//             device: vb.device().clone(),
//         })
//     }
//
//     pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
//         let embedding_output = self
//             .embeddings
//             .forward(input_ids, token_type_ids, None, None)?;
//         let sequence_output = self.encoder.forward(&embedding_output)?;
//         Ok(sequence_output)
//     }
// }
//
//
// pub struct DebertaV2ForTokenClassification {
//     debertav2: DebertaV2Model,
//     dropout: Dropout,
//     classifier: Linear,
//     pub device: Device,
// }
//
// impl DebertaV2ForTokenClassification {
//     pub fn load(vb: VarBuilder, config: &DebertaV2Config) -> Result<Self> {
//         let classifier_dropout = config.classifier_dropout;
//         let (debertav2, classifier) = match (
//             DebertaV2Model::load(vb.pp("debertaV2"), config),
//
//             if Option::is_some(&config._num_labels) {
//                 linear(config.hidden_size, config._num_labels.unwrap(), vb.pp("classifier"))
//             } else if Option::is_some(&config.id2label) {
//                 let num_labels = &config.id2label.as_ref().unwrap().len();
//                 linear(config.hidden_size, num_labels.clone(), vb.pp("classifier"))
//             } else {
//                 candle_core::bail!("cannnot find the number of classes to map to")
//             }
//
//         ) {
//             (Ok(debertav2), Ok(classifier)) => (debertav2, classifier),
//             (Err(err), _) | (_, Err(err)) => {
//                 return Err(err);
//             }
//         };
//         Ok(Self {
//             debertav2,
//             dropout: Dropout::new(classifier_dropout.unwrap_or_else(|| 0.2)),
//             classifier,
//             device: vb.device().clone(),
//         })
//     }
//
//     pub fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, labels: Option<&Tensor>) -> Result<TokenClassifierOutput> {
//         let outputs = self
//             .debertav2
//             .forward(input_ids, token_type_ids)?;
//         let outputs = self.dropout.forward(&outputs)?;
//
//         let logits = self.classifier.forward(&outputs)?;
//
//         println!("{:?}", logits);
//         let mut loss: Tensor = Tensor::new(vec![0.0], &self.device)?;
//
//         match labels {
//             Some(labels) => {
//                 loss = candle_nn::loss::cross_entropy(&logits.flatten_to(1)?, &labels.flatten_to(1)?)?;
//             }
//             None => {}
//         }
//
//         Ok(TokenClassifierOutput {
//             loss :Some(loss),
//             logits,
//             hidden_states :None,
//             attentions : None
//         })
//
//
//     }
//
// }
