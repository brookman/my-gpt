use anyhow::{Ok, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    embedding, loss, ops::softmax, AdamW, Embedding, Module, Optimizer, VarBuilder, VarMap,
};
use itertools::Itertools;
use rand::{
    distributions::{Distribution, Uniform, WeightedIndex},
    thread_rng,
};
use std::{fs, time::Instant};

use crate::encoder::Encoder;

mod encoder;

const BATCH_SIZE: usize = 32;
const BLOCK_SIZE: usize = 8;
const TRAIN_ITERS: usize = 3000;
const EVAL_INTERVAL: usize = 300;
const EVAL_ITERS: usize = 100;
const LEARNING_RATE: f64 = 1e-2;

fn main() -> Result<()> {
    let start = Instant::now();

    let device: Device = Device::new_cuda(0)?;
    let device: Device = Device::Cpu;

    println!("device: {:?}", device);

    let text = fs::read_to_string("data/input.txt")?;
    let vocab = text.chars().unique().sorted().collect::<Vec<_>>();
    let encoder = Encoder::new(&vocab);

    let encoded_text = encoder.encode(text)?;

    let (training_data, validation_data) = split(encoded_text, 0.9, &device)?;

    println!("training_data: {training_data}");
    println!("validation_data: {validation_data}");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = embedding(vocab.len(), vocab.len(), vb)?;

    let mut opt = AdamW::new_lr(varmap.all_vars(), LEARNING_RATE)?;

    for i in 0..TRAIN_ITERS {
        let batch = get_batch(&training_data, BLOCK_SIZE, BATCH_SIZE, &device)?;
        let (_logits, loss) = forward(&model, &batch)?;
        opt.backward_step(&loss)?;

        if i % EVAL_INTERVAL == 0 {
            let mut eval_losses: Vec<Tensor> = Vec::new();
            for _ in 0..EVAL_ITERS {
                let batch = get_batch(&validation_data, BLOCK_SIZE, BATCH_SIZE, &device)?;
                let (_logits, loss) = forward(&model, &batch)?;
                eval_losses.push(loss);
            }
            let eval_loss = Tensor::stack(&eval_losses, 0)?.mean(0)?;

            println!("iter: {i}, training_loss: {loss}, eval_loss: {eval_loss}");
        }
    }

    let training_time = start.elapsed().as_millis();
    let start = Instant::now();

    for _ in 0..100 {
        let generated = generate(&model, 500, &device)?;
        let generated_text = encoder.decode_tensor(generated)?;

        println!("generated_text: {generated_text}");
    }

    let gen_time = start.elapsed().as_millis();

    println!("training took {training_time} ms");
    println!("generation took {gen_time} ms");

    Ok(())
}

fn split(data: Vec<u8>, split: f64, device: &Device) -> Result<(Tensor, Tensor)> {
    let len = data.len();

    let n = (f64::clamp(split, 0.0, 1.0) * len as f64) as usize;
    let training_data = Tensor::from_iter(data[..n].iter().map(|u| *u as u32), device)?;
    let validation_data = Tensor::from_iter(data[n..].iter().map(|u| *u as u32), device)?;
    Ok((training_data, validation_data))
}

fn get_batch(
    data: &Tensor,
    block_size: usize,
    batch_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let len = data.dims1()?;
    let indices = random_indices(0, len - block_size, batch_size);

    let x_slices: Vec<Tensor> = indices
        .iter()
        .map(|&i| slice(data, i, i + block_size as u32, device))
        .collect::<Result<Vec<_>, _>>()?;

    let y_slices: Vec<Tensor> = indices
        .iter()
        .map(|&i| slice(data, i + 1, i + block_size as u32 + 1, device))
        .collect::<Result<Vec<_>, _>>()?;

    Ok((Tensor::stack(&x_slices, 0)?, Tensor::stack(&y_slices, 0)?))
}

fn random_indices(min: usize, max: usize, n: usize) -> Vec<u32> {
    let mut rng = thread_rng();
    let dist = Uniform::new(min, max);
    (0..n).map(|_| dist.sample(&mut rng) as u32).collect()
}

fn slice(tensor_1d: &Tensor, start_index: u32, end_index: u32, device: &Device) -> Result<Tensor> {
    let len = (end_index - start_index) as usize;
    let indidces = (start_index..end_index).collect::<Vec<_>>();
    let index_tensor = Tensor::from_slice(&indidces, len, device)?;
    let slice = tensor_1d.index_select(&index_tensor, 0)?;
    Ok(slice)
}

fn forward(model: &Embedding, batch: &(Tensor, Tensor)) -> Result<(Tensor, Tensor)> {
    let logits = model.forward(&batch.0)?;
    let (b, t, c) = logits.shape().dims3()?;
    let loss = loss::cross_entropy(&logits.reshape((b * t, c))?, &batch.1.reshape(b * t)?)?;
    Ok((logits, loss))
}

fn generate(model: &Embedding, number_of_tokens: usize, device: &Device) -> Result<Tensor> {
    let mut idx = Tensor::zeros((1, 1), DType::U8, device)?;

    for _ in 0..number_of_tokens {
        let logits = model.forward(&idx)?;
        let (_, t, _) = logits.shape().dims3()?;
        let last = Tensor::from_slice(&[t as u32 - 1], 1, device)?;
        let logits = logits.index_select(&last, 1)?;

        // println!("logits {logits}");

        let probs = softmax(&logits, 2)?;
        let probs: Vec<f32> = probs.flatten_all()?.to_vec1()?;
        // println!("probs {:?}", probs);

        let mut rng = thread_rng();
        let dist = WeightedIndex::new(probs).unwrap();
        let sample_index = dist.sample(&mut rng);
        let sample_tensor = Tensor::from_slice(&[sample_index as u8], (1, 1), device)?;

        idx = Tensor::cat(&[idx, sample_tensor], 1)?;
    }

    Ok(idx)
}
