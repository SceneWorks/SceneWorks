//! sc-22998: the commercial-use verdict and the conditional-component seam, judged on the LIVE
//! builtin YuE2 entry plus YuE1 fixtures shaped exactly like the YuE1 epic's (sc-19373) catalog rows,
//! which are not on this branch yet (epic sc-22988 acceptance test 5: V1 fixtures, no claim of live
//! V1 inference).

use super::*;
use crate::builtin_manifests::BUILTIN_MANIFESTS;
use serde_json::json;

fn builtin_models() -> Vec<Value> {
    let (_, contents) = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .expect("builtin.models.jsonc is embedded");
    let manifest: Value =
        serde_json::from_str(&crate::jsonc::strip_jsonc_comments(contents)).expect("parses");
    manifest["models"].as_array().expect("models").clone()
}

fn builtin_yue2() -> Value {
    builtin_models()
        .into_iter()
        .find(|model| model["id"] == "yue2")
        .expect("yue2 is in the builtin catalog")
}

/// The YuE1 epic's catalog row shape (feature/sc-19373: family `yue`, ids
/// `yue_{en,zh,jp_kr}_{cot,icl}`, SceneWorks re-hosts, Apache-2.0 so no NC flag).
fn yue1_fixture(id: &str) -> Value {
    json!({
        "id": id,
        "family": "yue",
        "type": "audio",
        "downloads": [{
            "provider": "huggingface",
            "repo": format!("SceneWorks/{}-candle", id.replace('_', "-")),
            "revision": "5842b8bc97d2a6dddc427a444920050dd3860949",
            "variant": "q4",
            "default": true,
            "files": ["q4/*"]
        }]
    })
}

fn catalog_with_yue1() -> Vec<Value> {
    let mut catalog = builtin_models();
    catalog.push(yue1_fixture("yue_en_cot"));
    catalog.push(yue1_fixture("yue_zh_icl"));
    catalog
}

#[test]
fn commercial_route_refuses_yue2_and_points_to_eligible_yue1_entries_only() {
    // Operates on the real resolution point: an id looked up in a catalog that holds the LIVE
    // builtin yue2 entry. Mutation that reds this: drop `commercialUse` + `nonCommercial` from the
    // manifest entry (verdict becomes Eligible), or change `alternativeFamily` away from `yue`
    // (alternatives go empty).
    let catalog = catalog_with_yue1();
    let verdict = commercial_use_verdict(&catalog, "yue2").expect("yue2 resolves");
    let CommercialUseVerdict::Refused {
        model_id,
        reason,
        alternative_family,
        alternatives,
        note,
    } = verdict
    else {
        panic!("a commercial-use route must refuse YuE2, got {verdict:?}");
    };
    assert_eq!(model_id, "yue2");
    assert!(reason.contains("CC BY-NC 4.0"), "{reason}");
    assert_eq!(alternative_family.as_deref(), Some("yue"));
    assert_eq!(alternatives, ["yue_en_cot", "yue_zh_icl"]);
    let note = note.expect("the pointer carries its rights caveat");
    assert!(
        note.contains("does not clear rights"),
        "the V1 pointer must not claim unrelated rights clearance: {note}"
    );
}

#[test]
fn a_yue1_entry_is_eligible_and_never_resolves_to_yue2() {
    let catalog = catalog_with_yue1();
    assert_eq!(
        commercial_use_verdict(&catalog, "yue_en_cot").unwrap(),
        CommercialUseVerdict::Eligible {
            model_id: "yue_en_cot".to_owned()
        }
    );
}

#[test]
fn an_unknown_or_ambiguous_id_is_an_error_never_another_model() {
    // Swapped-model negative fixtures: a V1 id this catalog does not hold must not be answered with
    // the V2 entry that shares its prefix, and V2 must not be answered with a V1 row.
    let catalog = builtin_models();
    assert_eq!(
        commercial_use_verdict(&catalog, "yue_en_cot"),
        Err(CommercialUseError::UnknownModel("yue_en_cot".to_owned()))
    );
    assert_eq!(
        commercial_use_verdict(&catalog, "yue"),
        Err(CommercialUseError::UnknownModel("yue".to_owned()))
    );
    let mut duplicated = catalog_with_yue1();
    duplicated.push(builtin_yue2());
    assert_eq!(
        commercial_use_verdict(&duplicated, "yue2"),
        Err(CommercialUseError::AmbiguousModel("yue2".to_owned()))
    );
}

#[test]
fn alternatives_exclude_restricted_family_members_and_self_pointers() {
    // A YuE1 fixture that is itself non-commercial is not offered; a block pointing at its own family
    // offers nothing (it would hand the refused weights back).
    let mut restricted_v1 = yue1_fixture("yue_jp_kr_cot");
    restricted_v1["nonCommercial"] = json!(true);
    let mut catalog = catalog_with_yue1();
    catalog.push(restricted_v1);
    let CommercialUseVerdict::Refused { alternatives, .. } =
        commercial_use_verdict(&catalog, "yue2").unwrap()
    else {
        panic!("refused");
    };
    assert!(!alternatives.contains(&"yue_jp_kr_cot".to_owned()));

    let mut self_pointer = builtin_yue2();
    self_pointer["commercialUse"]["alternativeFamily"] = json!("yue2");
    let CommercialUseVerdict::Refused { alternatives, .. } =
        commercial_use_verdict(&[self_pointer], "yue2").unwrap()
    else {
        panic!("refused");
    };
    assert!(alternatives.is_empty());
}

#[test]
fn non_commercial_flag_refuses_even_without_or_against_a_commercial_use_block() {
    let flagged = json!({"id": "nc", "family": "x", "nonCommercial": true});
    assert!(matches!(
        commercial_use_verdict(std::slice::from_ref(&flagged), "nc"),
        Ok(CommercialUseVerdict::Refused { .. })
    ));
    let contradictory = json!({
        "id": "nc", "family": "x", "nonCommercial": true,
        "commercialUse": {"eligible": true}
    });
    assert!(matches!(
        commercial_use_verdict(&[contradictory], "nc"),
        Ok(CommercialUseVerdict::Refused { .. })
    ));
}

#[test]
fn cover_components_are_refused_while_blocked_with_reason_and_unblock() {
    // The live entry: SheetSage2 + MERT-v2-FullSong are declared for covers and BLOCKED on the owner's
    // licensing decision. Mutation that reds this: delete either `blocked` record (the purpose then
    // returns rows), or drop `cover` from a component's `requiredFor`.
    let yue2 = builtin_yue2();
    let Err(ConditionalComponentsError::Blocked { purpose, blocked }) =
        conditional_component_downloads(&yue2, "cover")
    else {
        panic!("the cover closure must be refused while blocked");
    };
    assert_eq!(purpose, "cover");
    let ids = blocked
        .iter()
        .map(|(id, _, _)| id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(ids, ["yue2_sheetsage2", "yue2_mert_v2_fullsong"]);
    for (_, reason, unblock) in &blocked {
        assert!(
            reason.starts_with("blocked: owner licensing decision for SheetSage2/MERT port code"),
            "{reason}"
        );
        assert!(!unblock.is_empty());
    }
    // Base generation declares no conditional components at all.
    assert_eq!(
        conditional_component_downloads(&yue2, "generation"),
        Err(ConditionalComponentsError::NotDeclared {
            purpose: "generation".to_owned()
        })
    );
}

#[test]
fn an_unblocked_cover_closure_yields_exact_pinned_co_requisite_rows() {
    // The seam itself, with the owner's gate lifted on a copy: every pinned identity is carried
    // through verbatim and the rows can never be taken for a primary download.
    let mut yue2 = builtin_yue2();
    for component in yue2["conditionalComponents"].as_array_mut().unwrap() {
        component.as_object_mut().unwrap().remove("blocked");
    }
    let rows = conditional_component_downloads(&yue2, "cover").expect("unblocked");
    assert_eq!(rows.len(), 2);
    for (row, component) in rows
        .iter()
        .zip(yue2["conditionalComponents"].as_array().unwrap())
    {
        for key in ["provider", "repo", "revision", "files", "componentId"] {
            assert_eq!(row[key], component[key], "{key}");
        }
        assert_eq!(row["coRequisite"], json!(true));
    }
    // One blocked component still refuses the whole purpose.
    let mut half = yue2.clone();
    half["conditionalComponents"][1]["blocked"] = json!({"reason": "r", "unblock": "u"});
    assert!(matches!(
        conditional_component_downloads(&half, "cover"),
        Err(ConditionalComponentsError::Blocked { .. })
    ));
}
