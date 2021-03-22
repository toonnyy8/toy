use rand::{self, prelude::SliceRandom};

trait Layer {
    fn forward(&self, inps: &Vec<f32>) -> Vec<f32>;
}

#[derive(Debug, Clone)]
struct Linear {
    inp_size: usize,
    out_size: usize,
    use_bias: bool,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

impl Linear {
    fn new(inp_size: usize, out_size: usize, use_bias: bool) -> Linear {
        let weights: Vec<f32> = (0..inp_size * out_size)
            .map(|idx| rand::random::<f32>())
            .collect::<Vec<f32>>();
        let biases = (0..out_size)
            .map(|idx| rand::random::<f32>())
            .collect::<Vec<f32>>();
        Linear {
            inp_size,
            out_size,
            use_bias,
            weights,
            biases,
        }
    }

    fn get_weights(&self) -> Vec<f32> {
        [self.weights.clone(), self.biases.clone()].concat()
    }

    fn set_weights(&mut self, weights: &Vec<f32>) {
        assert_eq!(weights.len(), self.weight_num(), "weight number error");
        self.weights = weights[0..self.inp_size * self.out_size].to_vec();
        if self.use_bias {
            self.biases = weights[self.inp_size * self.out_size..].to_vec();
        }
    }

    fn weight_num(&self) -> usize {
        self.inp_size * self.out_size + if self.use_bias { self.out_size } else { 0 }
    }
}
impl Layer for Linear {
    fn forward(&self, inps: &Vec<f32>) -> Vec<f32> {
        assert_eq!(inps.len(), self.inp_size, "input size error");

        (0..self.out_size)
            .map(|out_idx| {
                inps.iter().enumerate().fold(
                    if self.use_bias {
                        self.biases[out_idx]
                    } else {
                        0.
                    },
                    |acc, (inp_idx, inp)| {
                        acc + inp * self.weights[self.out_size * out_idx + inp_idx]
                    },
                )
            })
            .collect::<Vec<f32>>()
    }
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + x.exp())
}

#[derive(Debug, Clone)]
struct Sigmoid {}
impl Sigmoid {
    fn new() -> Self {
        Self {}
    }
}
impl Layer for Sigmoid {
    fn forward(&self, inps: &Vec<f32>) -> Vec<f32> {
        inps.iter().map(|&inp| sigmoid(inp)).collect()
    }
}

#[derive(Debug, Clone)]
struct Net {
    layer1: Linear,
    layer1_activation: Sigmoid,
    layer2: Linear,
    layer2_activation: Sigmoid,
}
impl Net {
    fn new() -> Self {
        Self {
            layer1: Linear::new(2, 2, true),
            layer1_activation: Sigmoid::new(),
            layer2: Linear::new(2, 1, true),
            layer2_activation: Sigmoid::new(),
        }
    }
    fn get_weights(&self) -> Vec<f32> {
        [self.layer1.get_weights(), self.layer2.get_weights()].concat()
    }
    fn set_weights(&mut self, weights: &Vec<f32>) {
        assert_eq!(
            weights.len(),
            self.layer1.weight_num() + self.layer2.weight_num(),
            "weight number error",
        );
        self.layer1
            .set_weights(&weights[0..self.layer1.weight_num()].to_vec());
        self.layer2.set_weights(
            &weights[self.layer1.weight_num()..self.layer1.weight_num() + self.layer2.weight_num()]
                .to_vec(),
        );
    }
}
impl Layer for Net {
    fn forward(&self, inps: &Vec<f32>) -> Vec<f32> {
        let outs = self.layer1.forward(inps);
        let outs = self.layer1_activation.forward(&outs);
        let outs = self.layer2.forward(&outs);
        let outs = self.layer2_activation.forward(&outs);
        outs
    }
}

fn xor_loss(net: &Net) -> f32 {
    let loss00 = f32::powi(net.forward(&vec![0., 0.])[0] - 0., 2);
    let loss01 = f32::powi(net.forward(&vec![0., 1.])[0] - 1., 2);
    let loss10 = f32::powi(net.forward(&vec![1., 0.])[0] - 1., 2);
    let loss11 = f32::powi(net.forward(&vec![1., 1.])[0] - 0., 2);
    (loss00 + loss01 + loss10 + loss11) / 4.
}
struct Particle {
    net: Net,
    pbest: Vec<f32>,
    pbest_loss: f32,
    v: Vec<f32>,
}

impl Particle {
    fn new() -> Self {
        let net = Net::new();
        let pbest = net.get_weights();
        let pbest_loss = xor_loss(&net);
        let v = (0..pbest.len()).map(|idx| rand::random::<f32>()).collect();
        Self {
            net,
            pbest,
            pbest_loss,
            v,
        }
    }

    fn update(&mut self, v_max: f32, w: f32, c1: f32, c2: f32, gbest: &Vec<f32>) {
        let weights = self.net.get_weights();
        self.v = (0..self.v.len())
            .map(|idx| {
                let _v = self.v[idx] * w
                    + (self.pbest[idx] - weights[idx]) * c1 * rand::random::<f32>()
                    + (gbest[idx] - weights[idx]) * c2 * rand::random::<f32>();
                if _v > v_max {
                    v_max
                } else if _v < -v_max {
                    -v_max
                } else {
                    _v
                }
            })
            .collect::<Vec<f32>>();

        let new_weights = weights
            .iter()
            .enumerate()
            .map(|(idx, weight)| weight + self.v[idx])
            .collect::<Vec<f32>>();
        self.net.set_weights(&new_weights);

        let loss = xor_loss(&self.net);
        if loss <= self.pbest_loss {
            self.pbest_loss = loss;
            self.pbest = new_weights;
        }
    }
}

struct Swarm {
    particle_vec: Vec<Particle>,
    v_max: f32,
    w: f32,
    c1: f32,
    c2: f32,
    gbest: Vec<f32>,
    gbest_loss: f32,
}

impl Swarm {
    fn new(particle_num: usize, v_max: f32, w: f32, c1: f32, c2: f32) -> Self {
        let particle_vec = (0..particle_num)
            .map(|idx| Particle::new())
            .collect::<Vec<_>>();
        let mut gbest = particle_vec[0].pbest.clone();
        let mut gbest_loss = particle_vec[0].pbest_loss;
        for particle in &particle_vec {
            let pbest_loss = particle.pbest_loss;
            if pbest_loss < gbest_loss {
                gbest_loss = pbest_loss;
                gbest = particle.pbest.clone();
            }
        }
        Self {
            particle_vec,
            v_max,
            w,
            c1,
            c2,
            gbest,
            gbest_loss,
        }
    }

    fn train(&mut self, epoch: usize) {
        for _ in 0..epoch {
            for particle in &mut self.particle_vec {
                particle.update(self.v_max, self.w, self.c1, self.c2, &self.gbest);
            }
            for particle in &self.particle_vec {
                let pbest_loss = particle.pbest_loss;
                if pbest_loss < self.gbest_loss {
                    self.gbest_loss = pbest_loss;
                    self.gbest = particle.pbest.clone();
                }
            }
            println!("{}", self.gbest_loss)
        }
    }
}

fn main() {
    let mut particle_num = 20;
    let mut v_max = 2.;
    let mut w = 2.;
    let mut c1 = 2.;
    let mut c2 = 2.;
    let mut epoch = 100;
    let args: Vec<String> = std::env::args().collect();
    for idx in 0..args.len() {
        match args[idx].as_str() {
            "-p" => particle_num = args[idx + 1].parse::<usize>().unwrap(),
            "-v" => v_max = args[idx + 1].parse::<f32>().unwrap(),
            "-w" => w = args[idx + 1].parse::<f32>().unwrap(),
            "-c1" => c1 = args[idx + 1].parse::<f32>().unwrap(),
            "-c2" => c2 = args[idx + 1].parse::<f32>().unwrap(),
            "-e" => epoch = args[idx + 1].parse::<usize>().unwrap(),
            _ => {}
        };
    }
    Swarm::new(particle_num, v_max, w, c1, c2).train(epoch);
}
