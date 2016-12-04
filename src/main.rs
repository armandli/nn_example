extern crate nn_example;
extern crate rulinalg;

use std::path::Path;

use rulinalg::matrix::BaseMatrix;
use rulinalg::matrix::BaseMatrixMut;
use rulinalg::matrix::Matrix;

use nn_example::*;

fn main(){
    let input = "./spiral_dataset.txt";
    println!("reading input file: {}", input);

    let input_file = Path::new(input);

    let input = match load_input(input_file) {
        Ok(v) => v,
        Err(e) => panic!(e),
    };  

    println!("input {}*{}", input.0.rows(), input.0.cols());

    let (trainx, trainy, testx, testy) = create_dataset(&input.0, input.1);

    println!("{}*{} {}*{} {}*{} {}*{}", trainx.rows(), trainx.cols(), trainy.rows(), trainy.cols(), testx.rows(), testx.cols(), testy.rows(), testy.cols());

    println!("training NN1");

    let NN1 = FFN1::train(&trainx, &trainy, 2000, 1e-0, 1e-3);
    let score = NN1.test(&testx, &testy);

    println!("score 1: {}", score);

    let NN2 = FFN2::train(&trainx, &trainy, 2000, 1e-0, 1e-3);
    let score = NN2.test(&testx, &testy);

    println!("score 2: {}", score);
}
