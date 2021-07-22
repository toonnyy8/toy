#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
use num_traits;
use rand_distr::{Distribution, Normal};
use std::ops;

struct Tensor<Ax: Axis, Dtype: num_traits::NumOps>
where
    [usize; Ax::DIM]: Sized,
{
    shape: [usize; Ax::DIM],
    data: Vec<Dtype>,
    _ax: std::marker::PhantomData<Ax>,
}

impl<Ax: Axis, Dtype: num_traits::NumOps + num_traits::NumCast + Copy> Tensor<Ax, Dtype>
where
    [usize; Ax::DIM]: Sized,
{
    fn rand(shape: &[usize; Ax::DIM]) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0., 1.).unwrap();

        let size = shape.iter().fold(1, |a, b| a * b);
        let data = (0..size)
            .map(|_| num_traits::cast(normal.sample(&mut rng)).unwrap())
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
    [usize; Ax::DIM]: Sized,
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
    const DIM: usize;
}
struct Nil {}
impl Axis for Nil {
    const DIM: usize = 0;
}
struct I<Ax: Axis = Nil> {
    _ax: std::marker::PhantomData<Ax>,
}
impl<Ax: Axis> Axis for I<Ax> {
    const DIM: usize = Ax::DIM + 1;
}

fn main() {
    let t = &Tensor::<I<I>, f32>::rand(&[2, 3]);
    let t2 = t + t;
    println!("{:?}", t2.data)
}
