# SceneWorks Core

Shared Rust domain contracts and services used by the Axum API, native workers,
MCP server and desktop application. Core owns request validation, persistent stores,
model/checkpoint facts and media/workflow metadata. It depends on SQLite and media
libraries and on the pinned inference gen-core contracts.
