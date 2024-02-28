use itertools::Itertools;
use std::{
    collections::{HashMap, HashSet},
    fs,
};

struct Encoder<'a> {
    i_to_c:HashMap<u8,&'a char>,
    c_to_i:HashMap<&'a char, u8>,
}

impl<'a>  Encoder<'a> {
    
}

fn main() {
    let text = fs::read_to_string("data/input.txt").unwrap();
    let vocab = text.chars().unique().sorted().collect::<Vec<_>>();
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

    fn encode(s: String) -> Vec<u8> {
        s.chars().map(|c|c_to_i[])
    }
    println!("{:?}", vocab);
}
