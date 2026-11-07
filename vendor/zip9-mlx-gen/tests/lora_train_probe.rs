//! sc-3042 SPIKE de-risk probe — proves the *functional* mlx-rs training surface works in the
//! pinned fork (e59ffd8) for the LoRA-training pattern this epic needs.
//!
//! WHY this exists: the mlx-gen model crates do NOT use mlx-rs's `Module`/`ModuleParameters`
//! system (hand-rolled `&self` forwards over raw `Array`s — see `src/adapters.rs:6-12`). So the
//! Python reference's `nn.value_and_grad(model, …)` path is unavailable. Training must instead use
//! the *functional* autograd: `keyed_value_and_grad` over an explicit `HashMap<Rc<str>, Array>` of
//! trainable LoRA factors, stepped by `Optimizer::update_single`. This probe fits a tiny LoRA pair
//! `(a:[r,in], b:[out,r])` to reproduce a target linear map `y = x·Wᵀ` using EXACTLY that toolkit
//! (same forward orientation as `Adapter::Lora` — `((x·aᵀ)·bᵀ)·scale`), and asserts the loss
//! collapses. GREEN here = the autograd + AdamW + grad-clip mechanism is sound; the spike can build
//! the real Z-Image trainer on it.
//!
//!   cargo test -p mlx-gen --test integration lora_train_probe:: -- --nocapture

use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::error::Result as MlxResult;
use mlx_rs::ops::{matmul, multiply, subtract};
use mlx_rs::optimizers::{clip_grad_norm, AdamW, Optimizer};
use mlx_rs::transforms::keyed_value_and_grad;
use mlx_rs::{array, random, Array};

/// Scalar f32 array helper.
fn s(v: f32) -> Array {
    array!(v)
}

#[test]
fn lora_functional_autograd_converges() {
    let (in_f, out_f, rank) = (8i32, 6i32, 4i32);
    let scale = 1.0f32;
    let n = 16i32;

    // Fixed data: the target is itself a rank-`rank` LoRA map `y = ((x·A_trueᵀ)·B_trueᵀ)`, so a
    // rank-`rank` adapter CAN represent it exactly → a correct optimizer must drive loss → ~0.
    // (A full-rank target would plateau at the rank-deficiency floor and prove nothing.)
    let kx = random::key(11).unwrap();
    let kat = random::key(22).unwrap();
    let kbt = random::key(44).unwrap();
    let x = random::normal::<f32>(&[n, in_f], None, None, Some(&kx)).unwrap();
    let a_true = random::normal::<f32>(&[rank, in_f], None, None, Some(&kat)).unwrap();
    let b_true = random::normal::<f32>(&[out_f, rank], None, None, Some(&kbt)).unwrap();
    let y = matmul(matmul(&x, a_true.t()).unwrap(), b_true.t()).unwrap(); // [n, out]

    // Trainable factors in the SAVE orientation: a=[rank,in] small-normal, b=[out,rank] zeros
    // (the Python `_MlxLoRALinear` init — A~N(0,0.02), B=0 → adapter starts as identity).
    let ka = random::key(33).unwrap();
    let mut a = multiply(
        random::normal::<f32>(&[rank, in_f], None, None, Some(&ka)).unwrap(),
        s(0.02),
    )
    .unwrap();
    let mut b = Array::zeros::<f32>(&[out_f, rank]).unwrap();
    mlx_rs::transforms::eval([&a, &b]).unwrap();

    let mut opt = AdamW::new(5e-2);
    let total_elems = (n * out_f) as f32;

    let key_a: Rc<str> = Rc::from("a");
    let key_b: Rc<str> = Rc::from("b");

    let mut first_loss = f32::NAN;
    let mut last_loss = f32::NAN;

    for step in 0..300 {
        let mut params: HashMap<Rc<str>, Array> = HashMap::new();
        params.insert(key_a.clone(), a.clone());
        params.insert(key_b.clone(), b.clone());

        let xc = x.clone();
        let yc = y.clone();
        // Loss over the trainable params: pred = ((x·aᵀ)·bᵀ)·scale, MSE vs y.
        let loss_fn = move |p: HashMap<Rc<str>, Array>, _: i32| -> MlxResult<Vec<Array>> {
            let a = &p["a"];
            let b = &p["b"];
            let xa = matmul(&xc, a.t())?; // [n, rank]
            let pred = matmul(&xa, b.t())?; // [n, out]
            let pred = multiply(&pred, s(scale))?;
            let diff = subtract(&pred, &yc)?;
            let sumsq = diff.square()?.sum(None)?;
            Ok(vec![multiply(&sumsq, s(1.0 / total_elems))?])
        };

        let mut vg = keyed_value_and_grad(loss_fn);
        let (val, grads) = vg(params, 0).unwrap();
        let loss = val[0].item::<f32>();
        if step == 0 {
            first_loss = loss;
        }
        last_loss = loss;

        // Global-norm clip (proves the API), then AdamW step per-parameter.
        let (clipped, _norm) = clip_grad_norm(&grads, 1.0).unwrap();
        for (k, g) in clipped.iter() {
            if k.as_ref() == "a" {
                opt.update_single(k, g.as_ref(), &mut a).unwrap();
            } else {
                opt.update_single(k, g.as_ref(), &mut b).unwrap();
            }
        }
        mlx_rs::transforms::eval([&a, &b]).unwrap();
    }

    println!("[probe] loss {first_loss:.6} -> {last_loss:.6}");
    assert!(first_loss.is_finite() && last_loss.is_finite());
    // A rank-`rank` adapter can fit a rank-`rank` target exactly: loss must collapse toward 0.
    assert!(
        last_loss < 1e-3 && last_loss < first_loss * 1e-3,
        "LoRA functional autograd should converge to ~0: {first_loss:.6} -> {last_loss:.6}"
    );
}
