use std::ops::Deref;
use anyhow::{anyhow, Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Cache, Repo, RepoType};
use tokenizers::Tokenizer;
use piiserver::models::roberta::{RobertaConfig, RobertaModel};
use piiserver::models::deberta2::{DebertaV2Config, DebertaV2Model, FLOATING_DTYPE};

fn build_model_and_tokenizer() -> Result<(DebertaV2Model, Tokenizer)> {
    let device = Device::Cpu;
    let default_model = "iiiorg/piiranha-v1-detect-personal-information".to_string();
    let default_revision = "main".to_string();
    let (model_id, revision) = (default_model, default_revision);
    let repo = Repo::with_revision(model_id, RepoType::Model, revision);
    let offline = false;

    let (config_filename, tokenizer_filename, weights_filename) = if offline {
        let cache = Cache::default().repo(repo);
        (
            cache
                .get("config.json")
                .ok_or(anyhow!("Missing config file in cache"))?,
            cache
                .get("tokenizer.json")
                .ok_or(anyhow!("Missing tokenizer file in cache"))?,
            cache
                .get("model.safetensors")
                .ok_or(anyhow!("Missing weights file in cache"))?,
        )
    } else {
        let api = Api::new()?;
        let api = api.repo(repo);
        (
            api.get("config.json")?,
            api.get("tokenizer.json")?,
            api.get("model.safetensors")?,
        )
    };

    println!("config_filename: {}", config_filename.display());

    let config = std::fs::read_to_string(config_filename)?;
    let config: DebertaV2Config = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    println!("Weights: {}", weights_filename.display());
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_filename], FLOATING_DTYPE, &device)?
    };
    let vbd = vb.push_prefix("deberta");
    let model = DebertaV2Model::load(vbd, &config)?;
    Ok((model, tokenizer))
}

fn main() -> Result<()> {
    let (model, _tokenizer) = build_model_and_tokenizer()?;
    let device = &model.device;

    let encoding = _tokenizer.encode("Hello, world!", false).unwrap();

    let input_ids2 = &[
        [0u32, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2],
        [0u32, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2],
    ];

    let input_ids = encoding.get_ids().as_ref();
    let token_ids = input_ids.zeros_like()?;

    println!("token_ids: {:?}", token_ids.to_vec2::<u32>()?);
    println!("input_ids: {:?}", input_ids.to_vec2::<u32>()?);

    let output = model.forward(&input_ids, &token_ids)?;
    // let output = output.squeeze(0)?;

    println!("output: {:?}", output.i((.., 0))?.dims2());

    let logits = &[[0.1_f32, 0.2], [0.5, 0.6]];
    let logits = Tensor::new(logits, &device)?;

    println!("logits: {:?}", logits.i((.., 0))?.to_vec1::<f32>()?);
    println!("logits: {:?}", logits.i((.., 1))?.to_vec1::<f32>()?);



    Ok(())
}