# LoRA closed-loop study instruments

These development-only modules are included by the worker unit-test crate under the existing macOS/test cfg. They are outside shipped source, and their module names, ignored learned tests, parameters, output formats and study evidence remain unchanged.

The manual selectors remain `lora_eval_harness::harness::eval_lora_outputs` and `lora_train_driver::driver::*`. Learned execution remains opt-in through the existing study controller and never runs in ordinary CI.

See [the protocol](../../../../docs/sc-6541/closed-loop-protocol.md) and [epic 2123 study](../../../../docs/epic-2123-ab.md). Historical documents naming `src/lora_eval_harness.rs` or `src/lora_train_driver.rs` describe the former location; these files retain those instrument bodies verbatim.
