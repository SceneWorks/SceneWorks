use sceneworks_core::{HealthContract, API_PREFIX};

fn main() {
    let health = HealthContract::default();

    println!(
        "SceneWorks Rust backend scaffold ready at {}{}",
        API_PREFIX,
        health.route()
    );
}
