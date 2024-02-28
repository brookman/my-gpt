use anyhow::{Context, Result};
use candle_core::{Device, NdArray, Tensor};
use itertools::Itertools;
use std::{collections::HashMap, fs};

struct Encoder<'a> {
    i_to_c: HashMap<u8, &'a char>,
    c_to_i: HashMap<&'a char, u8>,
}

impl<'a> Encoder<'a> {
    fn new(vocab: &'a [char]) -> Self {
        let i_to_c = vocab
            .iter()
            .enumerate()
            .map(|(i, c)| (i as u8, c))
            .collect::<HashMap<_, _>>();
        let c_to_i = vocab
            .iter()
            .enumerate()
            .map(|(i, c)| (c, i as u8))
            .collect::<HashMap<_, _>>();
        Self { i_to_c, c_to_i }
    }

    fn encode(&self, s: String) -> Result<Vec<u8>> {
        s.chars()
            .map(|c| self.c_to_i.get(&c).copied().context("Can't map"))
            .into_iter()
            .collect()
    }

    fn decode(&self, vec: Vec<u8>) -> Result<String> {
        let x = vec
            .iter()
            .map(|i| self.i_to_c.get(i).context("Can't map"))
            .collect::<Result<Vec<_>>>()?;
        Ok(x.iter().map(|c| *c).join(""))
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;

    let text = fs::read_to_string("data/input.txt")?;
    let vocab = text.chars().unique().sorted().collect::<Vec<_>>();
    let encoder = Encoder::new(&vocab);

    let encoded_text = encoder.encode(text)?;
    let len = encoded_text.len();

    let data = Tensor::from_vec(encoded_text, len, &device)?;
    println!("data shape {:?}", data.shape());

    let n = (0.9 * len as f64) as usize;

    Ok(())
}
