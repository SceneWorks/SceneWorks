//! Area-owned route declarations. Host/auth/CORS/body/fairness layers are applied once by lib.rs.
use super::*;

mod assets;
mod audio;
mod catalogs;
mod characters;
mod films;
mod generation;
mod jobs;
mod library;
mod models;
mod person;
mod projects;
mod system;
mod timelines;
mod training;
mod workers;

pub(crate) fn routes() -> Router<AppState> {
    Router::new()
        .merge(system::routes())
        .merge(training::routes())
        .merge(catalogs::routes())
        .merge(projects::routes())
        .merge(assets::routes())
        .merge(audio::routes())
        .merge(jobs::routes())
        .merge(models::routes())
        .merge(characters::routes())
        .merge(timelines::routes())
        .merge(films::routes())
        .merge(person::routes())
        .merge(generation::routes())
        .merge(library::routes())
        .merge(workers::routes())
}
