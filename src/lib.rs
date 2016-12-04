extern crate rand;
extern crate rulinalg;

use std::fs::File;
use std::io::{Read, Error};
use std::path::Path;
use std::collections::HashSet;

use rand::Rng;
use rand::ThreadRng;

use rulinalg::matrix::BaseMatrix;
use rulinalg::matrix::BaseMatrixMut;
use rulinalg::matrix::Matrix;

pub fn load_input(path: &Path) -> Result<(Matrix<f64>, usize), Error> {
    let mut f = try!(File::open(path));
    let mut buf = String::new();

    f.read_to_string(&mut buf).expect("cannot read input file to string");

    let v = buf.split_whitespace().map(|value| {
        let value = match value.parse::<f64>() {
            Ok(v) => v,
            Err(_) => panic!("invalid f64 value detected"),
        };
        value
    }).collect::<Vec<f64>>();

    let mut i = 2;
    let mut max_category = 0;
    while i < v.len() {
        if v[i] as usize > max_category {
            max_category = v[i] as usize;
        }

        i += 3;
    }

    Ok((Matrix::new(v.len() / 3, 3, v), max_category + 1))
}

//create (trainx, trainy, testx, testy)
pub fn create_dataset(data: &Matrix<f64>, ycategory: usize) -> (Matrix<f64>, Matrix<f64>, Matrix<f64>, Matrix<f64>) {
    let mut test_rows: HashSet<usize> = HashSet::new();
    let mut rng = rand::thread_rng();
    let n = data.rows();
    let m = data.cols();

    for _ in 0..n / 5 {
        while true {
            let idx = rng.gen::<usize>() % n;
            if !test_rows.contains(&idx) {
                test_rows.insert(idx);
                break;
            }
        }
    }

    let mut trainx: Vec<f64> = Vec::new();
    let mut trainy: Vec<f64> = Vec::new();
    let mut testx: Vec<f64> = Vec::new();
    let mut testy: Vec<f64> = Vec::new();

    for i in 0..n {
        if test_rows.contains(&i) {
            let row = data.get_row(i).expect("data matrix index out of bound");
            for i in 0..m-1 {
                testx.push(row[i]);
            }
            testx.push(1.);
            for i in 0..ycategory {
                if row[m-1] as usize == i as usize {
                    testy.push(1.);
                } else {
                    testy.push(0.);
                }
            }
        } else {
            let row = data.get_row(i).expect("data matrix index out of bound");
            for i in 0..m-1 {
                trainx.push(row[i]);
            }
            trainx.push(1.);
            for i in 0..ycategory {
                if row[m-1] as usize == i as usize {
                    trainy.push(1.);
                } else {
                    trainy.push(0.);
                }
            }
        }
    }

    (Matrix::new(trainx.len() / m, m, trainx), Matrix::new(trainy.len() / ycategory, ycategory, trainy), Matrix::new(testx.len() / m, m, testx), Matrix::new(testy.len() / ycategory, ycategory, testy))
}

fn random_matrix(row: usize, col: usize) -> Matrix<f64> {
    let mut rng = rand::thread_rng();
    let f = |x, y| {
        rng.gen::<f64>().abs() % 1. - 0.5
    };

    Matrix::from_fn(row, col, f)
}

pub trait IFFN {
    fn train(trainx: &Matrix<f64>, trainy: &Matrix<f64>, iter: u32, step: f64, reg: f64) -> Self;
    fn test(&self, testx: &Matrix<f64>, testy: &Matrix<f64>) -> f64;
}

//FFN1 cannot handle linearly non-separable dataset because it does not have an activation function
pub struct FFN1 {
    W: Matrix<f64>,
}

impl IFFN for FFN1 {
    fn train(trainx: &Matrix<f64>, trainy: &Matrix<f64>, iter: u32, step: f64, reg: f64) -> FFN1 {
        //random initialization
        let mut W = random_matrix(trainx.cols(), trainy.cols());

        for k in 0..iter {
            //evaluate score
            let mut Y = trainx * &W;
    
            //convert to softmax
            //formula: p = (e ^ fi) / sum_j(e ^ fj)
            for v in Y.mut_data().iter_mut() {
                *v = v.exp();
            };
            for i in 0..Y.rows() {
                let mut row = Y.get_row_mut(i).expect("invalid index on Y");
                let sum = row.iter().fold(0., |s, v| {
                    s + v
                });
                for v in row {
                    let val = *v;
                    *v = val / sum;
                }
            }
    
            //compute loss: average cross entropy loss and L2 regularization
            //formula: L = 1 / N * sum(Li) + 1/2 * l * sum_k(sum_l(Wk,l ^ 2))
            //where Li is the -Log(p) where p is the softmax value of the correct class
            let mut data_loss = 0.;
            for i in 0..Y.rows() {
                let yr = trainy.get_row(i).expect("invalid index on trainy");
                let mut idx = 0;
                for i in 0..trainy.cols() {
                    if yr[i] == 1. {
                        idx = i;
                    }
                }
                let xr = Y.get_row(i).expect("invalid index on Y");
                data_loss += xr[idx].log(std::f64::consts::E) * -1.;
            }
            data_loss /= trainx.rows() as f64;
            let mut reg_loss = W.data().iter().fold(0., |s, v| {
                s + v * v
            });
            reg_loss *= 0.5 * reg;
            let loss = data_loss + reg_loss;

            if k % 10 == 0 {
                println!("loss = {}", loss);
            }

            //compute gradient with back propagation
            let mut dY = Y - trainy;
            for v in dY.mut_data().iter_mut() {
                *v /= trainx.rows() as f64;
            }
            let mut dW = trainx.transpose() * dY;
            for (dw, w) in dW.mut_data().iter_mut().zip(W.data().iter()) {
                *dw += reg * w;
            }

            //parameter update
            for v in dW.mut_data().iter_mut() {
                let val = *v;
                *v = val * -1. * step;
            }
            W += dW;
        }

        FFN1{ W: W }
    }

    fn test(&self, testx: &Matrix<f64>, testy: &Matrix<f64>) -> f64 {
        let Y = testx * &self.W;
        let mut corrects = 0;
        for i in 0..Y.rows() {
            let yr = Y.get_row(i).expect("invalid index on Y");
            let tr = testy.get_row(i).expect("invalid index on testy");

            let mut maxi = 0;
            let mut ci = 0;
            for i in 0..Y.cols() {
                if yr[i] > yr[maxi] {
                    maxi = i;
                }
                if tr[i] > tr[ci] {
                    ci = i;
                }
            }
            if maxi == ci {
                corrects += 1;
            }
        }

        corrects as f64 / testx.rows() as f64
    }
}

pub struct FFN2 {
    Wxh: Matrix<f64>,
    Who: Matrix<f64>,
}

fn relu(m: &mut Matrix<f64>) {
    for v in m.mut_data().iter_mut() {
        let val = *v;
        *v = val.max(0.);
    }
}

impl IFFN for FFN2 {
    fn train(trainx: &Matrix<f64>, trainy: &Matrix<f64>, iter: u32, step: f64, reg: f64) -> FFN2 {
        //initialization
        let HDIM = 100;
        let mut Wxh = random_matrix(trainx.cols(), HDIM);
        let mut Who = random_matrix(HDIM, trainx.cols());

        for k in 0..iter {
            //forward path
            let mut H = trainx * &Wxh;
            relu(&mut H);
            let mut Y = &H * &Who;
    
            //softmax
            for v in Y.mut_data().iter_mut() {
                let val = *v;
                *v = val.exp();
            }
            for i in 0..Y.rows() {
                let mut yr = Y.get_row_mut(i).expect("invalid index on Y");
                let sum = yr.iter().fold(0., |s, v| {
                    s + v
                });
                for v in yr.iter_mut() {
                    let val = *v;
                    *v = val / sum;
                }
            }
    
            //compute loss
            let mut data_loss = 0.;
            for i in 0..Y.rows() {
                let yr = trainy.get_row(i).expect("invalid index on trainy");
                let mut idx = 0;
                for i in 0..trainy.cols() {
                    if yr[i] == 1. {
                        idx = i;
                    }
                }
                let xr = Y.get_row(i).expect("invalid index on Y");
                data_loss += xr[idx].log(std::f64::consts::E) * -1.;
            }
            data_loss /= trainx.rows() as f64;
            let mut reg_loss = Wxh.data().iter().fold(0., |s, v| {
                s + v * v
            });
            reg_loss += Who.data().iter().fold(0., |s, v| {
                s + v * v
            });
            reg_loss *= 0.5 * reg;
            let loss = data_loss + reg_loss;

            if k % 10 == 0 {
                println!("loss = {}", loss);
            }
    
            //backpropagation
            let mut dY = Y - trainy;
            for v in dY.mut_data().iter_mut() {
                *v /= trainx.rows() as f64;
            }
            let mut dWho = H.transpose() * &dY;
            for (dw, w) in dWho.mut_data().iter_mut().zip(Who.data().iter()) {
                *dw += reg * w;
            }
            let mut dH = dY * Who.transpose();
            for (dh, h) in dH.mut_data().iter_mut().zip(H.data().iter()) {
                if *h <= 0. {
                    *dh = 0.;
                }
            }
            let mut dWxh = trainx.transpose() * dH;
    
            //parameter update
            for v in dWho.mut_data().iter_mut() {
                let val = *v;
                *v = val * -1. * step;
            }
            Who += dWho;
            for v in dWxh.mut_data().iter_mut() {
                let val = *v;
                *v = val * -1. * step;
            }
            Wxh += dWxh;
        }

        FFN2{ Wxh: Wxh, Who: Who }
    }

    fn test(&self, testx: &Matrix<f64>, testy: &Matrix<f64>) -> f64 {
        let mut H = testx * &self.Wxh;
        relu(&mut H);
        let Y = H * &self.Who;

        let mut corrects = 0;
        for i in 0..Y.rows() {
            let yr = Y.get_row(i).expect("invalid index on Y");
            let tr = testy.get_row(i).expect("invalid index on testy");

            let mut maxi = 0;
            let mut ci = 0;
            for i in 0..Y.cols() {
                if yr[i] > yr[maxi] {
                    maxi = i;
                }
                if tr[i] > tr[ci] {
                    ci = i;
                }
            }
            if maxi == ci {
                corrects += 1;
            }
        }

        corrects as f64 / testx.rows() as f64
    }
}
