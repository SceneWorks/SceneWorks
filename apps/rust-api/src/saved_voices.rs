//! HTTP handlers for the per-project **saved voices** registry (Voice Clone, sc-13517).
//!
//! Register: resolve the chosen library audio asset → compute its Chatterbox-VE speaker embedding on
//! a blocking thread (the worker's embed path) → persist name + reference asset id + embedding, and
//! run the near-duplicate check (the embedding's real consumer). List / delete round out the CRUD.
//! Picking a saved voice in the Studio supplies `referenceAudioAssetId` to the existing audio-job
//! pipeline, so no new generation wiring is needed here.

use super::*;

use sceneworks_core::voice_store::{SavedVoiceCreateInput, DEFAULT_VOICE_DEDUP_THRESHOLD};
use sceneworks_worker::voice_register::{embed_reference_clip, VoiceEmbedError};

fn map_voice_embed_error(error: VoiceEmbedError) -> ApiError {
    match error {
        // A missing install or an undecodable reference clip is client-actionable.
        VoiceEmbedError::WeightsMissing(message) | VoiceEmbedError::Decode(message) => {
            ApiError::bad_request(message)
        }
        // An embedder load/run failure is a server-side fault.
        VoiceEmbedError::Embed(message) => ApiError::internal(message),
    }
}

pub(crate) async fn list_saved_voices(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
) -> Result<Json<Vec<Value>>, ApiError> {
    Ok(Json(
        project_call(state, move |store| store.list_saved_voices(&project_id)).await?,
    ))
}

pub(crate) async fn create_saved_voice(
    State(state): State<AppState>,
    Path(project_id): Path<String>,
    ApiJson(payload): ApiJson<SavedVoiceCreateRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let name = payload.name.trim().to_owned();
    if name.is_empty() {
        return Err(ApiError::bad_request("Voice name must not be empty"));
    }
    let asset_id = payload.reference_audio_asset_id.trim().to_owned();
    if asset_id.is_empty() {
        return Err(ApiError::bad_request(
            "A reference audio asset id is required",
        ));
    }
    let threshold = payload
        .dedup_threshold
        .unwrap_or(DEFAULT_VOICE_DEDUP_THRESHOLD);

    // 1. Resolve the reference clip to an absolute on-disk path (blocking store call).
    let wav_path = project_call(state.clone(), {
        let project_id = project_id.clone();
        let asset_id = asset_id.clone();
        move |store| store.resolve_asset_media_path(&project_id, &asset_id)
    })
    .await?;

    // 2. Compute the Chatterbox-VE speaker embedding on a blocking thread (real inference).
    let data_dir = state.settings.data_dir.clone();
    let embedding = tokio::task::spawn_blocking(move || embed_reference_clip(&data_dir, &wav_path))
        .await
        .map_err(|error| ApiError::internal(format!("voice embed task panicked: {error}")))?
        .map_err(map_voice_embed_error)?;

    // 3. Persist + run near-duplicate detection (blocking store call).
    let (voice, duplicate) = project_call(state, move |store| {
        store.create_saved_voice(
            &project_id,
            SavedVoiceCreateInput {
                name,
                reference_audio_asset_id: asset_id,
                embedding,
            },
            threshold,
        )
    })
    .await?;

    // Surface the dedup consumer's result inline so the UI can warn on register.
    let mut body = voice;
    if let Some(object) = body.as_object_mut() {
        object.insert(
            "nearDuplicate".to_owned(),
            match duplicate {
                Some(dup) => json!({
                    "id": dup.id,
                    "name": dup.name,
                    "similarity": dup.similarity,
                }),
                None => Value::Null,
            },
        );
    }
    Ok((StatusCode::CREATED, Json(body)))
}

pub(crate) async fn delete_saved_voice(
    State(state): State<AppState>,
    Path((project_id, voice_id)): Path<(String, String)>,
) -> Result<Json<sceneworks_core::voice_store::SavedVoiceMutationResult>, ApiError> {
    Ok(Json(
        project_call(state, move |store| {
            store.delete_saved_voice(&project_id, &voice_id)
        })
        .await?,
    ))
}
