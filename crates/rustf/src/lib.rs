extern crate rustler;
extern crate tensorflow;

use rustler::{Env,Term,ListIterator,ResourceArc,Error,Encoder,NifResult};

use tensorflow::Tensor;
use tensorflow::Scope;
use tensorflow::DataType::{Float,Int32};

use std::sync::RwLock;

use log::debug;

rustler::init!("linalg_tf", 
    [version,to_tensor,from_tensor,transpose,diag,inv,matmul,svd],
    load = load
    );

fn load(env: rustler::Env, _:rustler::Term) -> bool {
    rustler::resource!(TensorResource, env);
    true
}

pub struct TensorResource {
    payload: RwLock<Tensor<f32>>
}

#[rustler::nif]
fn to_tensor(term: Term) -> NifResult<ResourceArc<TensorResource>> {
    let v: Vec<Vec<f32>> = term.decode::<ListIterator>()?.map(from_vector).collect::<NifResult<Vec<Vec<f32>>>>()?;
    let t = matrix_to_tensor(v);
    Ok(ResourceArc::new(TensorResource{payload: RwLock::new(t)}))
}

#[rustler::nif]
fn from_tensor(env: Env, res: ResourceArc<TensorResource>) -> NifResult<Term> {
    match res.payload.read() {
        Ok(t) => match t.dims() {
                  [_] => Ok(vector_to_term(env,tenor_to_vector(t.clone()))),
                  [_,_] => Ok(matrix_to_term(env,tenor_to_matrix(t.clone()))),
                  _ => todo!(),
        },
        Err(_) => Err(Error::RaiseAtom("null")),
    }
}

#[rustler::nif]
fn transpose(res: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {

    let t1 = res.payload.read().unwrap();
    //debug!("in: {:?}", t1.shape());
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
        Ok(val) => Ok(ResourceArc::new(TensorResource{payload: RwLock::new(val)})),
        Err(_) => Err(Error::RaiseAtom("null")),
    }
}

#[rustler::nif]
fn diag(res: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {

    let t1 = res.payload.read().unwrap();
    //debug!("in: {:?}", t1.shape());
    //let t1 = Tensor::new(&[2, 2]).with_values(&[1.0f32,2.0f32,3.0f32,4.0f32]).unwrap();

    let f = |t:Tensor<f32>| -> Result<Tensor<f32>, tensorflow::Status> {
        let scope = Scope::new_root_scope();
        let session = tensorflow::Session::new(&tensorflow::SessionOptions::new(), &scope.graph())?;

        let in1 = tensorflow::ops::Placeholder::new()
            .dtype(Float)
            .build(&mut scope.with_op_name("input1"))?;

        println!("diag n {} {:?}",t.dims().len(),t.dims());
        let _ = match t.dims().len() {
            1=>
            tensorflow::ops::Diag::new().T(Float).build(
                in1,
                &mut scope.with_op_name("diag"))?,

            2=>
            tensorflow::ops::DiagPart::new().T(Float).build(
                in1,
                &mut scope.with_op_name("diag"))?,
            _ => todo!(),
        };

        let mut step = tensorflow::SessionRunArgs::new();

        step.add_feed(
            &scope.graph().operation_by_name_required("input1")?,
            0,
            &t);

        // fetch final result
        let result = step
            .request_fetch(&scope.graph()
            .operation_by_name_required("diag")?, 0);

        // Run the operation
        session.run(&mut step)?;

        let ans: Tensor<f32> = step.fetch(result)?;
        println!("diag {:?}",ans);
        Ok(ans)
    };

    match f(t1.clone()) {
        Ok(val) => Ok(ResourceArc::new(TensorResource{payload: RwLock::new(val)})),
        Err(_) => Err(Error::RaiseAtom("null")),
    }
}

#[rustler::nif]
fn inv(res: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {

    let t1 = res.payload.read().unwrap();

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
        Ok(val) => Ok(ResourceArc::new(TensorResource{payload: RwLock::new(val)})),
        Err(_) => Err(Error::RaiseAtom("null")),
    }
}

#[rustler::nif]
fn matmul(res_a: ResourceArc<TensorResource>,res_b: ResourceArc<TensorResource>) -> NifResult<ResourceArc<TensorResource>> {

    let a = res_a.payload.read().unwrap();
    let b = res_b.payload.read().unwrap();

    let f = |a:Tensor<f32>,b:Tensor<f32>| -> Result<Tensor<f32>, tensorflow::Status> {
        let scope = Scope::new_root_scope();
        let session = tensorflow::Session::new(&tensorflow::SessionOptions::new(), &scope.graph())?;

        let in1 = tensorflow::ops::Placeholder::new()
            .dtype(Float)
            .build(&mut scope.with_op_name("input1"))?;

        let in2 = tensorflow::ops::Placeholder::new()
            .dtype(Float)
            .build(&mut scope.with_op_name("input2"))?;

        let _ = tensorflow::ops::MatMul::new().T(Float).build(
            in1,
            in2,
            &mut scope.with_op_name("matmul"))?;

        let mut step = tensorflow::SessionRunArgs::new();

        step.add_feed(
            &scope.graph().operation_by_name_required("input1")?,
            0,
            &a);

        step.add_feed(
            &scope.graph().operation_by_name_required("input2")?,
            0,
            &b);

        // fetch final result
        let result = step
            .request_fetch(&scope.graph()
            .operation_by_name_required("matmul")?, 0);

        // Run the operation
        session.run(&mut step)?;

        let ans: Tensor<f32> = step.fetch(result)?;
        Ok(ans)
    };

    match f(a.clone(),b.clone()) {
        Ok(val) => Ok(ResourceArc::new(TensorResource{payload: RwLock::new(val)})),
        Err(_) => Err(Error::RaiseAtom("null")),
    }
}

#[rustler::nif]
fn svd(res: ResourceArc<TensorResource>) -> NifResult<(ResourceArc<TensorResource>, ResourceArc<TensorResource>, ResourceArc<TensorResource>)> {

    let t1 = res.payload.read().unwrap();

    let f = |t:Tensor<f32>| -> Result<(Tensor<f32>,Tensor<f32>,Tensor<f32>), tensorflow::Status> {
        let scope = Scope::new_root_scope();
        let session = tensorflow::Session::new(&tensorflow::SessionOptions::new(), &scope.graph())?;

        let in1 = tensorflow::ops::Placeholder::new()
            .dtype(Float)
            .build(&mut scope.with_op_name("input1"))?;

        let _ = tensorflow::ops::Svd::new().T(Float).build_instance(
            in1.into(),
            &mut scope.with_op_name("svd"))?;

        let mut step = tensorflow::SessionRunArgs::new();
        step.set_request_metadata(true);

        step.add_feed(
            &scope.graph().operation_by_name_required("input1")?,
            0,
            &t);

        // fetch final result
        let s = step
            .request_fetch(&scope.graph()
            .operation_by_name_required("svd")?, 0);

        let u = step
            .request_fetch(&scope.graph()
            .operation_by_name_required("svd")?, 1);

        let v = step
            .request_fetch(&scope.graph()
            .operation_by_name_required("svd")?, 2);


        // Run the operation
        session.run(&mut step)?;

        let s  = step.fetch(s)?;
        let u  = step.fetch(u)?;
        let v  = step.fetch(v)?;

        debug!("s = {:?}",s);
        debug!("u = {:?}",u);
        debug!("v = {:?}",v);

        Ok((s,u,v))
    };

    match f(t1.clone()) {
        Ok((s,u,v)) => Ok((
                ResourceArc::new(TensorResource{payload: RwLock::new(s)}),
                ResourceArc::new(TensorResource{payload: RwLock::new(u)}),
                ResourceArc::new(TensorResource{payload: RwLock::new(v)}),
        )),
        Err(_) => Err(Error::RaiseAtom("null")),
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

// Private Functions
fn tenor_to_vector(d: Tensor<f32>) -> Vec<f32> {
    //debug!("tenor_to_vector {:?}",d);
    debug!("shape {:?}",d.shape());
    debug!("tensor {:?}",d);
    //vec![vec![d.get(&[0, 0]),d.get(&[0, 1])],vec![d.get(&[1, 0]),d.get(&[1, 1])]]
    let shape = d.shape();
    match shape[0] {
        Some(n) => 
            (0u64..=(n as u64-1)).collect::<Vec<u64>>().iter().map(|i|d.get(&[*i])).collect()
            ,
        _ => vec![]
    }
}

fn tenor_to_matrix(d: Tensor<f32>) -> Vec<Vec<f32>> {
    //debug!("tenor_to_matrix {:?}",d);
    debug!("shape {:?}",d.shape());
    debug!("tensor {:?}",d);
    //vec![vec![d.get(&[0, 0]),d.get(&[0, 1])],vec![d.get(&[1, 0]),d.get(&[1, 1])]]
    let shape = d.shape();
    match (shape[0],shape[1]) {
        (Some(n),Some(m)) => 
            (0u64..=(n as u64-1)).collect::<Vec<u64>>().iter().map(|i| (0u64..=(m as u64-1)).collect::<Vec<u64>>().iter().map(|j|d.get(&[*i, *j])).collect()).collect(),
        _ => vec![vec![]]
    }
}

fn vector_to_term(env: Env, m: Vec<f32>) -> Term {
    let terms = Vec::from(m.as_slice());
    terms.encode(env)
}

fn matrix_to_term(env: Env, m: Vec<Vec<f32>>) -> Term {
    let ncols=m.len();
    let mut terms = Vec::new();
    for r in m.as_slice().chunks(ncols) {
        terms.push(Vec::from(r))
    }
    terms[0].encode(env)
}

fn matrix_to_tensor(m: Vec<Vec<f32>>) -> Tensor<f32> {
    let nrows=m.len() as u64;
    let ncols=m[0].len() as u64;

    let values: Vec<f32> = m.into_iter().flatten().collect();

    Tensor::new(&[nrows, ncols]).with_values(&values).unwrap()
}

fn from_number(term: Term) -> NifResult<f32> {
    match term.decode::<f32>() {
        Ok(f) => Ok(f),
        Err(_) => match term.decode::<i32>() {
            Ok(i) => Ok(i as f32),
            Err(_) => Err(Error::BadArg),
        }
    }
}

fn from_vector(term: Term) -> NifResult<Vec<f32>> {
    let array :Vec<f32> = match term.list_length() {
        Ok(_) => term.decode::<ListIterator>()?.map(from_number).collect::<NifResult<Vec<f32>>>()?,
        Err(_) => vec![from_number(term)?],
    };
    Ok(array)
}

