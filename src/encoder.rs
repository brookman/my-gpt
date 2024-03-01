use anyhow::{Context, Result};
use candle_core::Tensor;
use std::collections::HashMap;

pub struct Encoder<'a> {
    i_to_c: HashMap<u8, &'a char>,
    c_to_i: HashMap<&'a char, u8>,
}

impl<'a> Encoder<'a> {
    pub fn new(vocab: &'a [char]) -> Self {
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

    pub fn encode(&self, s: String) -> Result<Vec<u8>> {
        s.chars()
            .map(|c| {
                self.c_to_i
                    .get(&c)
                    .copied()
                    .context("Can't map from char to token")
            })
            .collect()
    }

    pub fn decode(&self, vec: Vec<u8>) -> Result<String> {
        vec.iter()
            .map(|i| {
                self.i_to_c
                    .get(i)
                    .copied()
                    .context("Can't map from token to char")
            })
            .collect::<Result<String>>()
    }

    pub fn decode_tensor(&self, tensor: Tensor) -> Result<String> {
        tensor
            .flatten_all()?
            .to_vec1::<u8>()?
            .iter()
            .map(|i| {
                self.i_to_c
                    .get(i)
                    .copied()
                    .context("Can't map from token to char")
            })
            .collect::<Result<String>>()
    }
}
