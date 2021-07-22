#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
use num_traits;
use rand::Rng;
use rand_distr::{Distribution, Normal, NormalError};
use std::ops;

#[derive(Debug, Clone)]
struct Tensor<Ax: Axis, Dtype: num_traits::NumOps>
where
    [usize; Ax::dim]: Sized,
{
    shape: [usize; Ax::dim],
    data: Vec<Dtype>,
    _ax: std::marker::PhantomData<Ax>,
}

impl<Ax: Axis, Dtype: num_traits::NumOps + num_traits::NumCast + Copy> Tensor<Ax, Dtype>
where
    [usize; Ax::dim]: Sized,
{
    fn rand(shape: &[usize; Ax::dim]) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0., 1.).unwrap();

        let size = shape.iter().fold(1, |a, b| a * b);
        let data = (0..size)
            .map(|idx| num_traits::cast(normal.sample(&mut rng)).unwrap())
            .collect::<Vec<Dtype>>();

        Self {
            shape: shape.clone(),
            data,
            _ax: std::marker::PhantomData,
        }
    }
}

impl<Ax: Axis, Dtype: num_traits::NumOps + num_traits::NumCast + Copy> ops::Add<&Tensor<Ax, Dtype>>
    for &Tensor<Ax, Dtype>
where
    [usize; Ax::dim]: Sized,
{
    type Output = Tensor<Ax, Dtype>;

    fn add(self, other: &Tensor<Ax, Dtype>) -> Tensor<Ax, Dtype> {
        let data = (0..self.data.len())
            .map(|idx| self.data[idx] + other.data[idx])
            .collect();
        Tensor::<Ax, Dtype> {
            shape: self.shape.clone(),
            data,
            _ax: std::marker::PhantomData,
        }
    }
}

trait Axis {
    const dim: usize;
}
struct Nil {}
impl Axis for Nil {
    const dim: usize = 0;
}
struct I<Ax: Axis = Nil> {
    _ax: std::marker::PhantomData<Ax>,
}
impl<Ax: Axis> Axis for I<Ax> {
    const dim: usize = Ax::dim + 1;
}

fn main() {
    let t = &Tensor::<I<I>, f32>::rand(&[2, 3]);
    let t2 = t + t;
    println!("{:?}", t2.data)
}
