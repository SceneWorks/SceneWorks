use super::{get_or_load, get_or_load_by_key, get_or_load_local, lock_resetting};
use crate::WorkerError;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

struct DropProbe(Arc<AtomicUsize>);
impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn single_slot_retains_first_load_when_caller_has_no_key() {
    let mut cache = None;
    let model = get_or_load(&mut cache, |_| true, || Ok("first snapshot")).unwrap();
    assert_eq!(*model, "first snapshot");
    let model = get_or_load(
        &mut cache,
        |_| true,
        || panic!("unkeyed warm hit must not reload"),
    )
    .unwrap();
    assert_eq!(*model, "first snapshot");
}

#[test]
fn replacement_load_keeps_old_alive_and_failure_retries_without_losing_it() {
    let drops = Arc::new(AtomicUsize::new(0));
    let mut cache = Some(("old", DropProbe(drops.clone())));
    let attempts = Cell::new(0);
    let error = get_or_load(
        &mut cache,
        |(key, _)| *key == "new",
        || {
            attempts.set(attempts.get() + 1);
            assert_eq!(
                drops.load(Ordering::SeqCst),
                0,
                "old model lives through load"
            );
            Err(WorkerError::Engine("failed loader".into()))
        },
    )
    .err()
    .unwrap();
    assert!(matches!(error, WorkerError::Engine(ref message) if message == "failed loader"));
    assert_eq!(cache.as_ref().unwrap().0, "old");
    get_or_load(
        &mut cache,
        |(key, _)| *key == "new",
        || {
            attempts.set(attempts.get() + 1);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            Ok(("new", DropProbe(drops.clone())))
        },
    )
    .unwrap();
    assert_eq!(attempts.get(), 2);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    drop(cache);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn key_policy_keeps_each_discriminator_and_mutable_warm_model() {
    let mut cache = None;
    let loads = Cell::new(0);
    // Path tuple, backend/mode and quant tier remain caller-owned discriminators.
    for (key, expected) in [
        (("det-a", "pose-a", "mlx", Some(8)), 1),
        (("det-a", "pose-a", "mlx", Some(8)), 1),
        (("det-b", "pose-a", "mlx", Some(8)), 2),
        (("det-b", "pose-b", "mlx", Some(8)), 3),
        (("det-b", "pose-b", "ort", Some(8)), 4),
        (("det-b", "pose-b", "ort", None), 5),
    ] {
        let model = get_or_load(
            &mut cache,
            |(resident, _)| *resident == key,
            || {
                loads.set(loads.get() + 1);
                Ok((key, 0usize))
            },
        )
        .unwrap();
        model.1 += 1;
        assert_eq!(loads.get(), expected);
    }
    assert_eq!(cache.unwrap().1, 1);
}

#[test]
fn poisoned_single_slot_resets_before_load_and_keeps_existing_sticky_policy() {
    let drops = Arc::new(AtomicUsize::new(0));
    let cache = Mutex::new(Some(DropProbe(drops.clone())));
    let panic = std::panic::catch_unwind(|| {
        let mut guard = cache.lock().unwrap();
        let _ = get_or_load(
            &mut guard,
            |_| false,
            || {
                assert_eq!(drops.load(Ordering::SeqCst), 0);
                panic!("loader panicked while holding old resident");
            },
        );
    });
    assert!(panic.is_err());
    assert!(cache.is_poisoned());
    for expected_drops in 1..=2 {
        let mut guard = lock_resetting(&cache, |slot| *slot = None);
        assert!(guard.is_none());
        assert_eq!(drops.load(Ordering::SeqCst), expected_drops);
        get_or_load(&mut guard, |_| true, || Ok(DropProbe(drops.clone()))).unwrap();
    }
    assert!(
        cache.is_poisoned(),
        "do not silently change poison certification policy"
    );
    let mut guard = lock_resetting(&cache, |slot| *slot = None);
    let error = get_or_load(
        &mut guard,
        |_| true,
        || Err(WorkerError::Engine("retry".into())),
    );
    assert!(error.is_err());
    assert!(guard.is_none());
    get_or_load(&mut guard, |_| true, || Ok(DropProbe(drops.clone()))).unwrap();
    drop(guard);
    drop(cache);
    assert_eq!(drops.load(Ordering::SeqCst), 4);
}

#[test]
fn multi_entry_factor_path_cache_retains_other_keys_and_clears_all_on_poison() {
    let drops = Arc::new(AtomicUsize::new(0));
    let cache = Mutex::new(HashMap::new());
    {
        let mut guard = lock_resetting(&cache, HashMap::clear);
        for key in [(2, "first"), (4, "first"), (2, "second")] {
            get_or_load_by_key(&mut guard, key, || Ok(DropProbe(drops.clone()))).unwrap();
        }
        get_or_load_by_key(&mut guard, (2, "first"), || panic!("warm factor/path key")).unwrap();
        assert!(
            get_or_load_by_key(&mut guard, (4, "missing"), || Err(WorkerError::Engine(
                "failed".into()
            )))
            .is_err()
        );
        assert_eq!(guard.len(), 3);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
    }
    let capacity = cache.lock().unwrap().capacity();
    assert!(std::panic::catch_unwind(|| {
        let _guard = cache.lock().unwrap();
        panic!("upscaler job panicked");
    })
    .is_err());
    let guard = lock_resetting(&cache, HashMap::clear);
    assert!(guard.is_empty());
    assert_eq!(guard.capacity(), capacity, "preserve HashMap::clear policy");
    assert_eq!(drops.load(Ordering::SeqCst), 3);
}

#[test]
fn concurrent_users_serialize_loading_and_model_use_under_the_returned_guard() {
    let cache = Mutex::new(None);
    let loads = AtomicUsize::new(0);
    let barrier = Barrier::new(8);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                barrier.wait();
                let mut guard = lock_resetting(&cache, |slot| *slot = None);
                let model = get_or_load(
                    &mut guard,
                    |_| true,
                    || {
                        loads.fetch_add(1, Ordering::SeqCst);
                        std::thread::yield_now();
                        Ok(0usize)
                    },
                )
                .unwrap();
                *model += 1;
                std::thread::yield_now();
            });
        }
    });
    assert_eq!(loads.load(Ordering::SeqCst), 1);
    assert_eq!(*cache.lock().unwrap(), Some(8));
}

#[test]
fn local_non_send_model_preserves_build_borrow_failed_load_and_drop_thread() {
    struct Affine {
        thread: std::thread::ThreadId,
        drops: Rc<Cell<usize>>,
    }
    impl Drop for Affine {
        fn drop(&mut self) {
            assert_eq!(self.thread, std::thread::current().id());
            self.drops.set(self.drops.get() + 1);
        }
    }
    std::thread::spawn(|| {
        let drops = Rc::new(Cell::new(0));
        let cell = RefCell::new(None);
        let make = || Affine {
            thread: std::thread::current().id(),
            drops: drops.clone(),
        };
        drop(get_or_load_local(&cell, |(key, _)| *key == 1, || Ok((1, make()))).unwrap());
        let warm =
            get_or_load_local(&cell, |(key, _)| *key == 1, || panic!("local warm hit")).unwrap();
        assert_eq!(
            cell.borrow().as_ref().unwrap().0,
            1,
            "use keeps a shared borrow"
        );
        drop(warm);
        assert!(get_or_load_local(
            &cell,
            |(key, _)| *key == 2,
            || {
                assert_eq!(
                    cell.borrow().as_ref().unwrap().0,
                    1,
                    "build has no outstanding borrow"
                );
                Err(WorkerError::Engine("failed local loader".into()))
            }
        )
        .is_err());
        assert_eq!(cell.borrow().as_ref().unwrap().0, 1);
        drop(
            get_or_load_local(
                &cell,
                |(key, _)| *key == 2,
                || {
                    assert_eq!(drops.get(), 0);
                    assert_eq!(cell.borrow().as_ref().unwrap().0, 1);
                    Ok((2, make()))
                },
            )
            .unwrap(),
        );
        assert_eq!(drops.get(), 1);
        drop(cell);
        assert_eq!(drops.get(), 2);
    })
    .join()
    .unwrap();
}
