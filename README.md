# erlang-linalg-tensorflow
An Erlang NIF using Rust's Rustler and Tensorflow

Assuming you have rustup, erlang and rebar3. 

```
rebar3 shell
```

Should build and launch 

```
Erlang/OTP 25 [erts-13.2.2.5] [source] [64-bit] [smp:8:8] [ds:8:8:10] [async-threads:1] [jit:ns]

Eshell V13.2.2.5  (abort with ^G)
1> linalg_tf:to_tensor([[1.0,2.0],[3.0,4.0]]).                
#Ref<0.1234474187.3896901636.45871>
2> linalg_tf:from_tensor(#Ref<0.1234474187.3896901636.45871>).
[[1.0,2.0],[3.0,4.0]]
3> linalg_tf:transpose(#Ref<0.1234474187.3896901636.45882>).
#Ref<0.1234474187.3896901636.45902>
9> linalg_tf:from_tensor(#Ref<0.1234474187.3896901636.45902>).
[[1.0,3.0],[2.0,4.0]]
```