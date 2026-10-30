//! Regression coverage for failures that surface only when a lazy preview graph is read back.
//!
//! This is its own test binary because the pmetal injection hook is process-global. Keeping the
//! binary single-test prevents a concurrent MLX unit test from draining the synthetic error.

#[cfg(debug_assertions)]
mod debug_hook {
    use std::ffi::CString;

    use mlx_rs::Array;

    extern "C" {
        /// Debug-only pmetal hook that makes the next host-blocking Metal evaluation return an error.
        fn mlx_pmetal_test_inject_command_buffer_error(message: *const std::os::raw::c_char);
    }

    #[test]
    fn deferred_device_error_is_returned_instead_of_panicking_at_readback() {
        let latents = Array::zeros::<f32>(&[1, 4, 1, 1]).unwrap();
        let message = CString::new("synthetic preview readback failure").unwrap();
        unsafe { mlx_pmetal_test_inject_command_buffer_error(message.as_ptr()) };

        let error = mlx_gen::preview::project_latents(&latents, &[[0.0; 3]; 4], [0.0; 3])
            .expect_err("the injected deferred error must remain recoverable");
        assert!(error
            .to_string()
            .contains("preview projection readback failed"));

        let recovered = mlx_gen::preview::project_latents(&latents, &[[0.0; 3]; 4], [0.0; 3])
            .expect("the injected error must be drained after one readback");
        assert_eq!(recovered.pixels, [0, 0, 0]);
    }
}
