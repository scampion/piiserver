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

