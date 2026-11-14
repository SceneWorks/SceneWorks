fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&sceneworks_options::upgrade::metadata()).unwrap()
    );
}
