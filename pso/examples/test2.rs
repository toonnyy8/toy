//#![feature(generic_associated_types)]
//#![feature(const_generics)]

trait Layer {
    fn apply(&self, params: Option<Tensor>);
    fn init(&self) -> Option<Tensor>;
}
struct Tensor {
    data: Vec<f32>,
}
impl Tensor {
    fn new() -> Self {
        Self { data: vec![0.] }
    }
}

struct Linear<Axes> {
    _axes: std::marker::PhantomData<Axes>,
}
impl<Axes> Linear<Axes> {
    fn new() -> Self {
        Self {
            _axes: std::marker::PhantomData,
        }
    }
}
impl<Axes> Layer for Linear<Axes> {
    fn apply(&self, params: Option<Tensor>) {}
    fn init(&self) -> Option<Tensor> {
        None
    }
}

struct Seq {}
impl Layer for Seq {
    fn apply(&self, params: Option<Tensor>) {}
    fn init(&self) -> Option<Tensor> {
        None
    }
}
impl Seq {
    fn new<L: Layer>(layer: &L) -> Self {
        Self {}
    }

    fn srl<L: Layer>(&self, layer: &L) -> Self {
        Seq::new(layer)
    }
}

struct Prl<const N: usize> {}

impl<const N: usize> Prl<N> {
    fn new() -> Self {
        Self {}
    }
    fn prl(&self) -> Self {
        Prl::new()
    }
    fn to_seq<L: Fn([Tensor; N]) -> Tensor>(&self, fan_in: &L) -> Seq {
        Seq {}
    }
}
trait TensorPKG {}
struct TensorPackage<T, Tpkg: TensorPKG = Nil> {
    t: T,
    tpkg: Option<Tpkg>,
}
struct Nil {}
impl TensorPKG for Nil {}
impl Nil {
    fn new() -> Self {
        Self {}
    }
}
impl<T, Tpkg: TensorPKG> TensorPackage<T, Tpkg> {
    fn new(t: T) -> Self {
        Self { t, tpkg: None }
    }

    fn add<T2>(self, t: T2) -> TensorPackage<T2, Self> {
        TensorPackage::<T2, Self> {
            t,
            tpkg: Some(self),
        }
    }
}
impl<T, Tpkg: TensorPKG> TensorPKG for TensorPackage<T, Tpkg> {}
fn main() {
    let lin = &Linear::<i32>::new();
    let params = lin.init();
    Linear::<i32>::apply(&lin, params);
    let t1 = Tensor::new();
    let t2 = Tensor::new();
    let t3 = Tensor::new();
    let tpkg = TensorPackage::<Tensor, Nil>::new(t1).add(t2).add(t3);

    let m1 = Prl::<2>::new()
        .prl()
        .prl()
        .to_seq(&|xs: [Tensor; 2]| Tensor { data: vec![0.0] });
    let m2 = Seq::new(lin).srl(lin).srl(lin);
    let a = 1;
}
