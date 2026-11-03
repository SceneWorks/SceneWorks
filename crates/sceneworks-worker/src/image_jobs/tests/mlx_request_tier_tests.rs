use super::*;

#[tokio::test]
async fn request_tiers_preserve_the_first_fitting_prepared_candidate() {
    let mut seen = Vec::new();
    let selected =
        choose_mlx_request_tier(["bf16", "q8", "q4"].map(PathBuf::from).to_vec(), |dir| {
            seen.push(dir.clone());
            std::future::ready(Ok(if dir == Path::new("bf16") {
                MlxTierFit::TooBig(WorkerError::InvalidPayload("request exceeds budget".into()))
            } else {
                MlxTierFit::Fits((dir, "prepared exact adapter/decoder composition"))
            }))
        })
        .await
        .unwrap();
    assert_eq!(
        selected,
        (
            PathBuf::from("q8"),
            "prepared exact adapter/decoder composition"
        )
    );
    assert_eq!(seen, [PathBuf::from("bf16"), PathBuf::from("q8")]);
}

#[tokio::test]
async fn request_tiers_never_downtier_a_contract_or_source_failure() {
    let mut calls = 0;
    let error = choose_mlx_request_tier::<(), _>(["q8", "q4"].map(PathBuf::from).to_vec(), |_| {
        calls += 1;
        std::future::ready(Err(WorkerError::InvalidPayload(
            "artifact changed during contract query".into(),
        )))
    })
    .await
    .unwrap_err();
    assert_eq!(calls, 1);
    assert!(error.to_string().contains("artifact changed"));
}

#[tokio::test]
async fn request_tiers_report_the_last_evaluated_constraint_and_honor_a_single_explicit_tier() {
    for directories in [
        vec![PathBuf::from("q8")],
        vec![PathBuf::from("q8"), PathBuf::from("q4")],
    ] {
        let expected = directories.last().unwrap().display().to_string();
        let mut seen = Vec::new();
        let error = choose_mlx_request_tier::<(), _>(directories.clone(), |dir| {
            seen.push(dir.clone());
            std::future::ready(Ok(MlxTierFit::TooBig(WorkerError::InvalidPayload(
                format!(
                    "{} request needs 7.25 GiB but only 6.00 GiB is safely available",
                    dir.display(),
                ),
            ))))
        })
        .await
        .unwrap_err()
        .to_string();
        assert_eq!(seen, directories);
        assert!(error.contains(&format!("{expected} tier:")), "{error}");
        assert!(error.contains("7.25 GiB"), "{error}");
        assert!(!error.contains("30 GB"));
        assert!(!error.contains("Lower the output resolution"));
    }
}
