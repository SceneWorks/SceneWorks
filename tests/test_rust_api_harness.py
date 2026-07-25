from rust_api_harness import enable_api_listening_log, listening_url_from_log


def test_listening_url_from_text_and_json_tracing_output():
    assert (
        listening_url_from_log(
            "INFO sceneworks_rust_api: SceneWorks API listening "
            "event=\"api_listening\" address=127.0.0.1:43125"
        )
        == "http://127.0.0.1:43125"
    )
    assert (
        listening_url_from_log(
            '{"event":"api_listening","address":"127.0.0.1:51234"}'
        )
        == "http://127.0.0.1:51234"
    )


def test_api_listening_event_survives_restrictive_ambient_logging():
    env = {"RUST_LOG": "warn"}
    enable_api_listening_log(env)
    assert env["RUST_LOG"] == "warn,sceneworks_rust_api::server=info"
