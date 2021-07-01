#![feature(const_evaluatable_checked)]
#![feature(const_generics)]
#![feature(generic_associated_types)]
// use std::ops::Add;

// use ndarray::prelude::*;
// use ndarray::Array;
// use ndarray_rand::rand_distr::Uniform;
// use ndarray_rand::RandomExt;

// trait Layer {}
// struct Input {}
// struct Linear<const I: usize, const O: usize> {
//     w: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 2]>>,
//     b: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>,
// }

// impl<const I: usize, const O: usize> Linear<I, O> {
//     fn new() -> Linear<I, O> {
//         Linear::<I, O> {
//             w: Array::random((I, O), Uniform::new(-1., 1.)),
//             b: Array::random(O, Uniform::new(-1., 1.)),
//         }
//     }
//     fn forward(
//         self: &Self,
//         x: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>>,
//     ) -> ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>> {
//         x.dot(&self.w) + &self.b
//     }
// }

// struct Seq<const I: usize, const O: usize> {
//     // layers: Vec<Layer>,
// }

// impl<const I: usize, const O: usize> Seq<I, O> {
//     fn new(layer: Linear<I, O>) -> Seq<I, O> {
//         todo!()
//     }
//     fn chain<const P: usize>(self: Self, layer: Linear<O, P>) -> Seq<I, P> {
//         todo!()
//     }
// }

// fn main() {
//     let w = Array::random((2, 2), Uniform::new(-1., 1.));
//     let b = Array::random(2, Uniform::new(-1., 1.));

//     // Seq::new(Linear::<2, 3>::new())
//     //     .chain(Linear::<3, 2>::new())
//     //     .chain(Linear::<2, 3>::new());

//     println!("{:8.4}", b.dot(&w));
// }

trait Layer {
    // fn forward();
    // type Last<L>: Layer;
}
struct ReLU {}
impl Layer for ReLU {}
impl ReLU {
    fn new() -> ReLU {
        todo!()
    }
}
struct Sigmoid {}
impl Layer for Sigmoid {}
impl Sigmoid {
    fn new() -> Sigmoid {
        todo!()
    }
}

struct Linear<const I: usize, const O: usize> {}
impl<const I: usize, const O: usize> Layer for Linear<I, O> {
    // fn forward() {}
    // type Last<L> = Linear<I, O>;
}
impl<const I: usize, const O: usize> Linear<I, O> {
    fn new() -> Linear<I, O> {
        todo!()
    }
}

trait Sequence {}
struct Slinear<const I: usize, const O: usize, S: Sequence> {
    _s: std::marker::PhantomData<S>,
}
impl<const I: usize, const O: usize> Slinear<I, O, Input> {
    fn new() -> Slinear<I, O, Input> {
        todo!()
    }
    fn chain<S: Sequence>(layer: S) -> Slinear<I, O, Input> {
        todo!()
    }
}
impl<const I: usize, const O: usize, S: Sequence> Sequence for Slinear<I, O, S> {}
struct Input {}
impl Sequence for Input {}
struct Seq<L: Layer, S: Sequence> {
    _l: std::marker::PhantomData<L>,
    _s: std::marker::PhantomData<S>,
    layer: L,
    seq: S,
}
impl<S: Sequence, L: Layer> Sequence for Seq<L, S> {}
impl<L: Layer> Seq<L, Input> {
    fn new(layer: L) -> Seq<L, Input> {
        todo!()
    }
}
impl<S: Sequence, L: Layer> Seq<L, S> {
    fn chain<L2: Layer>(self, layer: L2) -> Seq<L2, Seq<L, S>> {
        todo!()
    }
}
fn main() {
    let model = Seq::new(Linear::<2, 2>::new())
        .chain(Sigmoid::new())
        .chain(Linear::<2, 2>::new())
        .chain(Sigmoid::new());
}
