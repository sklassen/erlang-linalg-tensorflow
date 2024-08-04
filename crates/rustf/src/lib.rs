extern crate rustler;
extern crate tensorflow;

//use rustler::{Encoder, Env,Term};

//use rustler::{NifEnv, NifTerm, NifError, NifDecoder, NifEncoder, NifResult};
use tensorflow::Tensor;
use tensorflow::Scope;
use tensorflow::DataType::{Float,Int32};

//use rustler::{NifResult, Atom, ResourceArc};
use rustler::ResourceArc;
use std::sync::RwLock;

rustler::init!("linalg_tf", 
    [version,to_tensor,from_tensor,transpose,inv],
    load = load
    );

fn load(env: rustler::Env, _:rustler::Term) -> bool {
    rustler::resource!(TensorResource, env);
    true
}

pub struct TensorResource {
    payload: RwLock<Tensor<f32>>
}
// ResourceArc::new(TensorResource{payload: RwLock::new(1)})
// res.payload.read()


#[rustler::nif]
fn to_tensor(m: Vec<Vec<f32>>) -> ResourceArc<TensorResource> {
    //println!("to_tensor {:?}",m);
    let t = v2t(m);
    ResourceArc::new(TensorResource{payload: RwLock::new(t)})
}

#[rustler::nif]
fn from_tensor(res: ResourceArc<TensorResource>) -> Vec<Vec<f32>> {
    let t = res.payload.read().unwrap();
    //println!("from_tensor {:?}",t);
    t2v(t.clone())
}

#[rustler::nif]
fn transpose(res: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {

    let t1 = res.payload.read().unwrap();
    //println!("in: {:?}", t1.shape());
    //let t1 = Tensor::new(&[2, 2]).with_values(&[1.0f32,2.0f32,3.0f32,4.0f32]).unwrap();

    let f = |t:Tensor<f32>| -> Result<Tensor<f32>, tensorflow::Status> {
        let mut scope = Scope::new_root_scope();
        let session = tensorflow::Session::new(&tensorflow::SessionOptions::new(), &scope.graph())?;

        let in1 = tensorflow::ops::Placeholder::new()
            .dtype(Float)
            .build(&mut scope.with_op_name("input1"))?;

        let perm = tensorflow::ops::Const::new()
            .value([1, 0]) // swap rows and columns
            .dtype(Int32)
            .build(&mut scope)?;

        let _ = tensorflow::ops::Transpose::new().T(Float).Tperm(Int32).build(
            in1,
            perm,
            &mut scope.with_op_name("transpose"))?;

        let mut step = tensorflow::SessionRunArgs::new();

        step.add_feed(
            &scope.graph().operation_by_name_required("input1")?,
            0,
            &t);

        // fetch final result
        let result = step
            .request_fetch(&scope.graph()
            .operation_by_name_required("transpose")?, 0);

        // Run the operation
        session.run(&mut step)?;

        let ans: Tensor<f32> = step.fetch(result)?;
        Ok(ans)
    };

    match f(t1.clone()) {
        Ok(val) => { 
            //println!("out: {:?}", val.shape());
            ResourceArc::new(TensorResource{payload: RwLock::new(val)})
        },
        Err(_) => panic!("opps")
    }
}

#[rustler::nif]
fn inv(res: ResourceArc<TensorResource>) -> ResourceArc<TensorResource> {

    let t1 = res.payload.read().unwrap();
    //println!("in: {:?}", t1.shape());
    //let t1 = Tensor::new(&[2, 2]).with_values(&[1.0f32,2.0f32,3.0f32,4.0f32]).unwrap();

    let f = |t:Tensor<f32>| -> Result<Tensor<f32>, tensorflow::Status> {
        let scope = Scope::new_root_scope();
        let session = tensorflow::Session::new(&tensorflow::SessionOptions::new(), &scope.graph())?;

        let in1 = tensorflow::ops::Placeholder::new()
            .dtype(Float)
            .build(&mut scope.with_op_name("input1"))?;

        let _ = tensorflow::ops::Inv::new().T(Float).build(
            in1,
            &mut scope.with_op_name("inv"))?;

        let mut step = tensorflow::SessionRunArgs::new();

        step.add_feed(
            &scope.graph().operation_by_name_required("input1")?,
            0,
            &t);

        // fetch final result
        let result = step
            .request_fetch(&scope.graph()
            .operation_by_name_required("inv")?, 0);

        // Run the operation
        session.run(&mut step)?;

        let ans: Tensor<f32> = step.fetch(result)?;
        Ok(ans)
    };

    match f(t1.clone()) {
        Ok(val) => { 
            //println!("out: {:?}", val.shape());
            ResourceArc::new(TensorResource{payload: RwLock::new(val)})
        },
        Err(_) => panic!("opps")
    }
}

mod atoms {
    rustler::atoms! {
        null,
    }
}


#[rustler::nif]
fn version() -> Vec<u8> {
    let error="error version";
    match tensorflow::version() {
        Ok(version) => version.chars().map(|x| x as u8).collect(),
        Err(_) => error.chars().map(|x| x as u8).collect()
    }
}

// Private convert erlang to tenor
//vec![vec![d.get(&[0, 0]),d.get(&[0, 1])],vec![d.get(&[1, 0]),d.get(&[1, 1])]]
fn t2v(d: Tensor<f32>) -> Vec<Vec<f32>> {
    //println!("t2v {:?}",d);
    //println!("shape {:?}",d.shape());
    //vec![vec![d.get(&[0, 0]),d.get(&[0, 1])],vec![d.get(&[1, 0]),d.get(&[1, 1])]]
    let shape = d.shape();
    match (shape[0],shape[1]) {
        (Some(n),Some(m)) => 
            (0u64..=(n as u64-1)).collect::<Vec<u64>>().iter().map(|i| (0u64..=(m as u64-1)).collect::<Vec<u64>>().iter().map(|j|d.get(&[*i, *j])).collect()).collect()
            ,
        _ => vec![vec![]]
    }
}

fn v2t(m: Vec<Vec<f32>>) -> Tensor<f32> {
    let nrows=m.len() as u64;
    let ncols=m[0].len() as u64;

    let values: Vec<f32> = m.into_iter().flatten().collect();

    Tensor::new(&[nrows, ncols]).with_values(&values).unwrap()
}

