//! Structural rules are parsed as Rust/TOML, independently of the human-readable matrix.
//! No provider is loaded or executed. Parsing includes inactive platform code and include! files.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use quote::ToTokens;
use syn::parse::{Parse, ParseStream};
use syn::visit::{self, Visit};
use syn::{Expr, Item, Pat, Token};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn parse(source: &str) -> syn::File {
    syn::parse_file(source).expect("architecture input must be valid Rust, including inactive cfgs")
}

struct StringEnum {
    name: syn::Ident,
    variants: BTreeMap<String, String>,
}
impl Parse for StringEnum {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        input.call(syn::Attribute::parse_outer)?;
        input.parse::<syn::Visibility>()?;
        input.parse::<Token![enum]>()?;
        let name = input.parse()?;
        let body;
        syn::braced!(body in input);
        let mut variants = BTreeMap::new();
        while !body.is_empty() {
            body.call(syn::Attribute::parse_outer)?;
            let variant: syn::Ident = body.parse()?;
            body.parse::<Token![=>]>()?;
            let wire: syn::LitStr = body.parse()?;
            if variants.insert(variant.to_string(), wire.value()).is_some() {
                return Err(body.error("duplicate string-enum variant"));
            }
            if !body.is_empty() {
                body.parse::<Token![,]>()?;
            }
        }
        Ok(Self { name, variants })
    }
}

pub(super) fn enum_variants(source: &str, name: &str) -> BTreeMap<String, String> {
    parse(source)
        .items
        .into_iter()
        .filter_map(|item| match item {
            Item::Macro(item) if item.mac.path.is_ident("string_enum") => {
                Some(syn::parse2::<StringEnum>(item.mac.tokens).expect("string_enum syntax"))
            }
            _ => None,
        })
        .find(|item| item.name == name)
        .unwrap_or_else(|| panic!("missing string_enum {name}"))
        .variants
}

fn pattern_variants(pattern: &Pat, enum_name: &str, out: &mut BTreeSet<String>) {
    match pattern {
        Pat::Path(path) => {
            let parts: Vec<_> = path.path.segments.iter().collect();
            if parts.len() >= 2 && parts[parts.len() - 2].ident == enum_name {
                out.insert(parts.last().unwrap().ident.to_string());
            }
        }
        Pat::Or(or) => {
            for case in &or.cases {
                pattern_variants(case, enum_name, out);
            }
        }
        Pat::Paren(paren) => pattern_variants(&paren.pat, enum_name, out),
        // Unknown(String) is intentionally excluded: this guard checks known contracts, not
        // the forward-compatibility catchall required by the non_exhaustive external enum.
        _ => {}
    }
}

fn job_type_match(expr: &Expr) -> bool {
    matches!(expr, Expr::Field(field)
        if matches!(&field.member, syn::Member::Named(name) if name == "job_type")
        && matches!(field.base.as_ref(), Expr::Path(path) if path.path.is_ident("job")))
}

fn function(source: &str, name: &str) -> syn::ItemFn {
    parse(source)
        .items
        .into_iter()
        .find_map(|item| match item {
            Item::Fn(item) if item.sig.ident == name => Some(item),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing function {name}"))
}

pub(super) fn dispatch_variants(source: &str, function_name: &str) -> BTreeSet<String> {
    dispatch_variants_for(source, function_name, true, false)
}

fn dispatch_variants_for(
    source: &str,
    function_name: &str,
    mac: bool,
    candle: bool,
) -> BTreeSet<String> {
    struct Matches {
        sets: Vec<BTreeSet<String>>,
        mac: bool,
        candle: bool,
    }
    impl<'ast> Visit<'ast> for Matches {
        fn visit_expr_match(&mut self, node: &'ast syn::ExprMatch) {
            if job_type_match(&node.expr) {
                let mut variants = BTreeSet::new();
                for arm in &node.arms {
                    let active = arm.attrs.iter().all(|attr| {
                        assert!(
                            !attr.path().is_ident("cfg_attr"),
                            "conditional dispatch cfg needs an explicit architecture decision"
                        );
                        !attr.path().is_ident("cfg")
                            || cfg_meta(
                                &attr.parse_args().expect("cfg syntax"),
                                self.mac,
                                self.candle,
                                false,
                            )
                            .expect("dispatch cfg must be decidable for native and neither lanes")
                    });
                    // A guarded arm can refuse every instance; inactive cfg arms must not fill
                    // a hole on another platform. Unknown(String) remains the external catchall.
                    if active && arm.guard.is_none() {
                        pattern_variants(&arm.pat, "JobType", &mut variants);
                    }
                }
                self.sets.push(variants);
            }
            visit::visit_expr_match(self, node);
        }
    }
    let function = function(source, function_name);
    let mut matches = Matches {
        sets: Vec::new(),
        mac,
        candle,
    };
    matches.visit_block(&function.block);
    assert_eq!(
        matches.sets.len(),
        1,
        "{function_name} must have one job-type match"
    );
    matches.sets.pop().unwrap()
}

#[test]
fn dispatch_and_both_routing_oracles_decide_every_known_job_kind() {
    let expected = enum_variants(super::CONTRACTS, "JobType")
        .into_keys()
        .collect::<BTreeSet<_>>();
    let gaps = include_str!("../../sceneworks-core/src/jobs_store/routing/gaps.rs");
    for (source, name) in [
        (super::WORKER, "run_utility_job_inner"),
        (gaps, "mac_rust_supported"),
        (gaps, "candle_supported"),
    ] {
        for (mac, candle) in [(true, false), (true, true), (false, true)] {
            assert_eq!(
                dispatch_variants_for(source, name, mac, candle),
                expected,
                "{name}: each native lane needs an explicit decision for every known job kind"
            );
        }
    }
    // The neither-backend worker intentionally has no handlers for these native-only jobs.
    // Preserve that boundary rather than broadening runtime policy to satisfy a source guard.
    let native_only: BTreeSet<String> = [
        "PoseDetect",
        "KpsExtract",
        "ImageUpscale",
        "DatasetUpscale",
        "ImageSegment",
        "ImageDepth",
        "VideoUpscale",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(
        dispatch_variants_for(super::WORKER, "run_utility_job_inner", false, false),
        expected.difference(&native_only).cloned().collect()
    );
}

#[test]
fn worker_entry_delegates_to_the_guarded_dispatcher() {
    struct Calls(usize);
    impl<'ast> Visit<'ast> for Calls {
        fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
            if matches!(call.func.as_ref(), Expr::Path(path) if path.path.is_ident("run_utility_job_inner"))
            {
                self.0 += 1;
            }
            visit::visit_expr_call(self, call);
        }
    }
    let mut calls = Calls(0);
    calls.visit_item_fn(&function(super::WORKER, "run_utility_job"));
    assert_eq!(
        calls.0, 1,
        "the load-progress wrapper must execute the checked dispatcher"
    );
}

#[test]
fn dispatch_ast_ignores_comments_strings_nested_matches_and_guarded_arms() {
    let source = r#"async fn run_utility_job() {
        // JobType::Missing => {}
        let misleading = "JobType::Missing => {}";
        match job.job_type {
            JobType::First | JobType::Second => {
                match other { JobType::Missing => (), _ => () }
            },
            JobType::Guarded if false => (),
            #[cfg(any())] JobType::Inactive => (),
            _ => (),
        }
    }"#;
    assert_eq!(
        dispatch_variants(source, "run_utility_job"),
        BTreeSet::from(["First".into(), "Second".into()])
    );
    let missing =
        super::WORKER.replacen("JobType::PromptRefine =>", "_removed_prompt_refine =>", 1);
    assert_ne!(missing, super::WORKER);
    assert!(!dispatch_variants(&missing, "run_utility_job_inner").contains("PromptRefine"));
}

fn capability_errors(contracts: &str, jobs_store: &str) -> Vec<String> {
    let jobs = enum_variants(contracts, "JobType");
    let capabilities = enum_variants(contracts, "WorkerCapability");
    let wires: BTreeSet<_> = capabilities.values().collect();
    let required = function(jobs_store, "required_capability");
    let matched = match required
        .block
        .stmts
        .last()
        .expect("required capability body")
    {
        syn::Stmt::Expr(Expr::Match(expr), _) if job_type_match(&expr.expr) => expr,
        _ => panic!("required_capability must match the job kind"),
    };
    let mut errors = Vec::new();
    // These jobs intentionally advertise another contract. Pin the AST mapping as well as the
    // vocabulary: changing a comment or keeping a removed target in a string cannot satisfy it.
    for (job, capability) in [
        ("ControlTraining", "LoraTrain"),
        ("DatasetSubjectMask", "ImageSegment"),
    ] {
        let arm = matched.arms.iter().find(|arm| {
            let mut variants = BTreeSet::new();
            pattern_variants(&arm.pat, "JobType", &mut variants);
            arm.guard.is_none() && variants.contains(job)
        });
        let target = format!("WorkerCapability :: {capability} . as_str ()");
        if !arm.is_some_and(|arm| arm.body.to_token_stream().to_string() == target) {
            errors.push(format!("{job} must explicitly require {capability}"));
        }
    }
    for (job, wire) in &jobs {
        if !["ControlTraining", "DatasetSubjectMask", "VectorGenerate"].contains(&job.as_str())
            && !wires.contains(wire)
        {
            errors.push(format!(
                "{job} has no capability contract for its default {wire} route"
            ));
        }
    }
    #[derive(Default)]
    struct Referenced(BTreeSet<String>);
    impl<'ast> Visit<'ast> for Referenced {
        fn visit_path(&mut self, path: &'ast syn::Path) {
            let parts: Vec<_> = path.segments.iter().collect();
            if parts.len() == 2 && parts[0].ident == "WorkerCapability" {
                self.0.insert(parts[1].ident.to_string());
            }
            visit::visit_path(self, path);
        }
    }
    let mut references = Referenced::default();
    references.visit_item_fn(&required);
    for name in references.0 {
        if !capabilities.contains_key(&name) {
            errors.push(format!(
                "required_capability refers to missing capability {name}"
            ));
        }
    }
    for (job, expected) in [
        (
            "VectorGenerate",
            vec!["VectorImageToSvg", "VectorTextToSvg"],
        ),
        ("PersonDetect", vec!["PersonDetectPreview"]),
        ("PersonTrack", vec!["PersonTrackPreview"]),
        ("ModelDownload", vec!["AudioGenerate", "ImageGenerate"]),
    ] {
        let mut references = Referenced::default();
        for arm in &matched.arms {
            let mut variants = BTreeSet::new();
            pattern_variants(&arm.pat, "JobType", &mut variants);
            if variants.contains(job) {
                references.visit_expr(&arm.body);
            }
        }
        if references.0 != expected.into_iter().map(str::to_owned).collect() {
            errors.push(format!("{job}: exceptional capability aliases drifted"));
        }
    }
    for capability in [
        "VectorImageToSvg",
        "VectorTextToSvg",
        "PersonDetectPreview",
        "PersonTrackPreview",
    ] {
        if !capabilities.contains_key(capability) {
            errors.push(format!("missing exceptional capability {capability}"));
        }
    }
    errors
}

#[test]
fn job_and_capability_contracts_agree_without_changing_runtime_admission() {
    let jobs = include_str!("../../sceneworks-core/src/jobs_store.rs");
    assert_eq!(
        capability_errors(super::CONTRACTS, jobs),
        Vec::<String>::new()
    );
    let no_image = super::CONTRACTS.replacen(
        "ImageGenerate => \"image_generate\"",
        "ImageGenerate => \"new_unadvertised_kind\"",
        1,
    );
    assert!(!capability_errors(&no_image, jobs).is_empty());
    let wrong_alias = jobs.replacen(
        "JobType::ControlTraining => WorkerCapability::LoraTrain.as_str()",
        "JobType::ControlTraining => WorkerCapability::ImageGenerate.as_str()",
        1,
    );
    assert_ne!(wrong_alias, jobs);
    assert!(!capability_errors(super::CONTRACTS, &wrong_alias).is_empty());
}

fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
    {
        let path = entry.unwrap().path();
        if path.is_dir() {
            paths.extend(rust_sources(&path));
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            paths.push(path);
        }
    }
    paths.sort();
    paths
}

fn read_manifest(path: &Path) -> toml::Value {
    std::fs::read_to_string(path).unwrap().parse().unwrap()
}

fn dependency_tables(manifest: &toml::Value) -> Vec<&toml::value::Table> {
    let mut tables = Vec::new();
    for kind in ["dependencies", "dev-dependencies", "build-dependencies"] {
        if let Some(table) = manifest.get(kind).and_then(toml::Value::as_table) {
            tables.push(table);
        }
    }
    if let Some(targets) = manifest.get("target").and_then(toml::Value::as_table) {
        for target in targets.values() {
            tables.extend(dependency_tables(target));
        }
    }
    tables
}

fn internal_edges(manifests: &[(String, toml::Value)]) -> BTreeSet<(String, String)> {
    let names: BTreeSet<_> = manifests.iter().map(|(name, _)| name.as_str()).collect();
    let mut edges = BTreeSet::new();
    for (name, manifest) in manifests {
        for deps in dependency_tables(manifest) {
            for (alias, value) in deps {
                let inherited;
                let value = if value.get("workspace").and_then(toml::Value::as_bool) == Some(true) {
                    inherited = read_manifest(&root().join("Cargo.toml"));
                    &inherited["workspace"]["dependencies"][alias]
                } else {
                    value
                };
                let dependency = value
                    .get("package")
                    .and_then(toml::Value::as_str)
                    .unwrap_or(alias);
                if names.contains(dependency) {
                    edges.insert((name.clone(), dependency.to_owned()));
                }
            }
        }
    }
    edges
}

fn workspace_manifests() -> Vec<(String, toml::Value)> {
    let root = root();
    let workspace = read_manifest(&root.join("Cargo.toml"));
    workspace["workspace"]["members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|member| {
            let manifest = read_manifest(&root.join(member.as_str().unwrap()).join("Cargo.toml"));
            (
                manifest["package"]["name"].as_str().unwrap().to_owned(),
                manifest,
            )
        })
        .collect()
}

#[test]
fn cargo_dependency_direction_is_an_executable_graph_rule() {
    let expected: BTreeSet<(String, String)> = [
        ("sceneworks-desktop", "sceneworks-core"),
        ("sceneworks-rust-api", "sceneworks-core"),
        ("sceneworks-rust-api", "sceneworks-image-quality"),
        ("sceneworks-rust-api", "sceneworks-mcp"),
        ("sceneworks-rust-api", "sceneworks-worker"),
        ("sceneworks-rust-worker", "sceneworks-worker"),
        ("sceneworks-image-quality", "sceneworks-core"),
        ("sceneworks-mcp", "sceneworks-core"),
        ("sceneworks-memory-adapter", "sceneworks-core"),
        ("sceneworks-worker", "sceneworks-core"),
        ("sceneworks-worker", "sceneworks-image-quality"),
    ]
    .into_iter()
    .map(|(from, to)| (from.into(), to.into()))
    .collect();
    let mut manifests = workspace_manifests();
    assert_eq!(internal_edges(&manifests), expected, "internal crate edges changed: reconsider layering explicitly (including dev/build/target dependencies)");
    // Core -> gen-core and memory-adapter -> core are valid. A renamed, target-specific
    // dependency back to the worker is not: inspect Cargo's package field, not its alias.
    let core = manifests
        .iter_mut()
        .find(|(name, _)| name == "sceneworks-core")
        .unwrap();
    let target: toml::Value = "[target.'cfg(windows)'.dependencies]\ninnocent_alias = { package = \"sceneworks-worker\", path = \"../sceneworks-worker\" }".parse().unwrap();
    core.1
        .as_table_mut()
        .unwrap()
        .insert("target".into(), target["target"].clone());
    assert_ne!(internal_edges(&manifests), expected);
}

#[test]
fn common_dependencies_inherit_versions_and_preserve_platform_feature_boundaries() {
    let workspace = read_manifest(&root().join("Cargo.toml"));
    let common = &workspace["workspace"]["dependencies"];
    let mut users: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (name, manifest) in workspace_manifests() {
        for deps in dependency_tables(&manifest) {
            for (dependency, declaration) in deps {
                if declaration.get("path").is_none() && declaration.get("git").is_none() {
                    users
                        .entry(dependency.clone())
                        .or_default()
                        .insert(name.clone());
                }
                if common.get(dependency).is_some() {
                    assert_eq!(
                        declaration.get("workspace").and_then(toml::Value::as_bool),
                        Some(true),
                        "{name}: {dependency} must inherit its workspace version"
                    );
                    assert!(
                        declaration.get("version").is_none(),
                        "{name}: independent {dependency} version"
                    );
                }
            }
        }
        if name == "sceneworks-rust-api" {
            // F-180's original test-only premise does not describe catalog result materialization.
            // Preserve the RAII staging dependency used by that production handler.
            assert!(manifest["dependencies"].get("tempfile").is_some());
            assert!(manifest["dev-dependencies"].get("tempfile").is_none());
        }
        if name == "sceneworks-core" {
            assert!(
                manifest["dependencies"]["sceneworks-gen-core"]
                    .get("git")
                    .is_some(),
                "core legitimately consumes backend-neutral gen-core"
            );
            assert!(
                manifest["dependencies"]["sha2"].get("features").is_none(),
                "sha2/asm must stay target-specific"
            );
            assert_eq!(
                manifest["target"]
                    ["cfg(all(target_arch = \"aarch64\", target_vendor = \"apple\"))"]
                    ["dependencies"]["sha2"]["features"]
                    .as_array()
                    .unwrap(),
                &[toml::Value::String("asm".into())]
            );
        }
        if name == "sceneworks-worker" {
            assert!(
                manifest["dependencies"]["candle-gen"]
                    .get("optional")
                    .is_none(),
                "candle training contracts are backend-neutral"
            );
            assert!(
                manifest["target"]["cfg(not(target_os = \"macos\"))"]["dependencies"]
                    .get("candle-gen")
                    .is_none()
            );
            let features = manifest["features"]["backend-candle"].as_array().unwrap();
            assert!(!features.contains(&toml::Value::String("dep:candle-gen".into())));
            assert!(features.contains(&toml::Value::String("dep:runtime-cuda".into())));
            assert_eq!(
                manifest["target"]["cfg(not(target_os = \"macos\"))"]["dependencies"]
                    ["runtime-cuda"]["optional"]
                    .as_bool(),
                Some(true)
            );
        }
    }
    for (dependency, members) in users {
        if members.len() > 1 {
            assert!(
                common.get(&dependency).is_some(),
                "{dependency} is shared by {members:?} but has no workspace-owned declaration"
            );
        }
    }
}

fn worker_paths(source: &str) -> BTreeSet<String> {
    fn imports(
        tree: &syn::UseTree,
        prefix: Vec<String>,
        aliases: &mut BTreeMap<String, Vec<String>>,
        paths: &mut BTreeSet<String>,
        pending: &mut Vec<(String, Vec<String>)>,
    ) {
        match tree {
            syn::UseTree::Path(path) => {
                let mut prefix = prefix;
                prefix.push(path.ident.to_string());
                imports(&path.tree, prefix, aliases, paths, pending);
            }
            syn::UseTree::Group(group) => {
                for tree in &group.items {
                    imports(tree, prefix.clone(), aliases, paths, pending);
                }
            }
            syn::UseTree::Name(name) => {
                let mut full = prefix;
                if name.ident != "self" {
                    full.push(name.ident.to_string());
                }
                if full.first().is_some_and(|name| name == "sceneworks_worker") {
                    paths.insert(full.join("::"));
                    let binding = if name.ident == "self" {
                        full.last().unwrap().clone()
                    } else {
                        name.ident.to_string()
                    };
                    aliases.insert(binding, full);
                } else {
                    let binding = if name.ident == "self" {
                        full.last().cloned().unwrap_or_else(|| "self".into())
                    } else {
                        name.ident.to_string()
                    };
                    pending.push((binding, full));
                }
            }
            syn::UseTree::Rename(rename) => {
                let mut full = prefix;
                if rename.ident != "self" {
                    full.push(rename.ident.to_string());
                }
                if full.first().is_some_and(|name| name == "sceneworks_worker") {
                    paths.insert(full.join("::"));
                    aliases.insert(rename.rename.to_string(), full);
                } else {
                    pending.push((rename.rename.to_string(), full));
                }
            }
            syn::UseTree::Glob(_) => {
                if prefix
                    .first()
                    .is_some_and(|name| name == "sceneworks_worker")
                {
                    paths.insert(format!("{}::*", prefix.join("::")));
                } else {
                    pending.push(("*".into(), prefix));
                }
            }
        }
    }
    #[derive(Default)]
    struct Imports {
        aliases: BTreeMap<String, Vec<String>>,
        paths: BTreeSet<String>,
        pending: Vec<(String, Vec<String>)>,
    }
    impl<'ast> Visit<'ast> for Imports {
        fn visit_item_use(&mut self, item: &'ast syn::ItemUse) {
            imports(
                &item.tree,
                Vec::new(),
                &mut self.aliases,
                &mut self.paths,
                &mut self.pending,
            );
        }
        fn visit_item_extern_crate(&mut self, item: &'ast syn::ItemExternCrate) {
            if item.ident == "sceneworks_worker" {
                let alias = item
                    .rename
                    .as_ref()
                    .map(|(_, alias)| alias)
                    .unwrap_or(&item.ident);
                self.aliases
                    .insert(alias.to_string(), vec!["sceneworks_worker".into()]);
                self.paths.insert("sceneworks_worker".into());
            }
        }
    }
    // Walk syntax tokens too: syn deliberately treats json!, format!, etc. as opaque. A worker
    // call inside such a macro must be guarded just like an ordinary ExprPath. Rust's token tree
    // keeps comments out, literals opaque and nested groups explicit; no substring matching.
    fn tokens(
        stream: proc_macro2::TokenStream,
        aliases: &BTreeMap<String, Vec<String>>,
        paths: &mut BTreeSet<String>,
    ) {
        use proc_macro2::TokenTree;
        let trees: Vec<_> = stream.into_iter().collect();
        let mut index = 0;
        while index < trees.len() {
            if let TokenTree::Group(group) = &trees[index] {
                tokens(group.stream(), aliases, paths);
            }
            if let TokenTree::Ident(ident) = &trees[index] {
                let name = ident.to_string();
                let mut full = if name == "sceneworks_worker" {
                    vec![name]
                } else {
                    aliases.get(&name).cloned().unwrap_or_default()
                };
                if !full.is_empty() {
                    let mut end = index;
                    while end + 3 < trees.len() {
                        match (&trees[end + 1], &trees[end + 2], &trees[end + 3]) {
                            (
                                TokenTree::Punct(a),
                                TokenTree::Punct(b),
                                TokenTree::Ident(segment),
                            ) if a.as_char() == ':' && b.as_char() == ':' => {
                                full.push(segment.to_string());
                                end += 3;
                            }
                            _ => break,
                        }
                    }
                    if end > index {
                        paths.insert(full.join("::"));
                    }
                    index = end;
                }
            }
            index += 1;
        }
    }
    let file = parse(source);
    let mut imports = Imports::default();
    imports.visit_file(&file);
    // Resolve re-exports/chained aliases to the same worker owner. Otherwise
    // `use self::restore as rebound; rebound::new_api()` would bypass the boundary.
    loop {
        let before = imports.aliases.len();
        for (binding, path) in &imports.pending {
            let path = path
                .iter()
                .skip_while(|segment| ["self", "super", "crate"].contains(&segment.as_str()))
                .collect::<Vec<_>>();
            let Some(first) = path.first() else { continue };
            let Some(mut expanded) = imports.aliases.get(first.as_str()).cloned() else {
                continue;
            };
            expanded.extend(path.into_iter().skip(1).cloned());
            if binding == "*" {
                imports.paths.insert(format!("{}::*", expanded.join("::")));
            } else {
                imports.paths.insert(expanded.join("::"));
                imports.aliases.entry(binding.clone()).or_insert(expanded);
            }
        }
        if imports.aliases.len() == before {
            break;
        }
    }
    tokens(
        file.into_token_stream(),
        &imports.aliases,
        &mut imports.paths,
    );
    imports.paths
}

#[test]
fn api_uses_only_reviewed_worker_entry_points() {
    let root = root();
    let mut actual = BTreeSet::new();
    for path in rust_sources(&root.join("apps/rust-api/src")) {
        let source = std::fs::read_to_string(&path).unwrap();
        let display = path
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        for dependency in worker_paths(&source) {
            actual.insert(format!("{display} {dependency}"));
        }
    }
    let expected: BTreeSet<_> = include_str!("../tests/architecture/api_worker_paths.txt")
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_owned)
        .collect();
    assert!(
        !actual.is_empty(),
        "API must exercise its reviewed worker embedding boundary"
    );
    let added: Vec<_> = actual.difference(&expected).collect();
    let removed: Vec<_> = expected.difference(&actual).collect();
    assert!(added.is_empty() && removed.is_empty(), "API/worker boundary changed: new={added:?}; stale={removed:?}. Review shared ownership before updating the explicit entry-point inventory.");
}

#[test]
fn api_boundary_guard_tracks_aliases_and_macro_calls_but_ignores_prose() {
    let source = r#"
        // sceneworks_worker::comment_only();
        use sceneworks_worker::iris_restore as restore;
        use self::restore as rebound;
        fn call() {
            let prose = "sceneworks_worker::string_only()";
            json!({"result": rebound::new_private_api()});
            sceneworks_worker::new_unreviewed_entry();
        }
    "#;
    assert_eq!(
        worker_paths(source),
        BTreeSet::from([
            "sceneworks_worker::iris_restore".into(),
            "sceneworks_worker::iris_restore::new_private_api".into(),
            "sceneworks_worker::new_unreviewed_entry".into(),
        ])
    );
}

fn cfg_meta(meta: &syn::Meta, mac: bool, candle: bool, test: bool) -> Option<bool> {
    use syn::Meta;
    match meta {
        Meta::Path(path) if path.is_ident("native_lane") => Some(mac || candle),
        Meta::Path(path) if path.is_ident("test") => Some(test),
        Meta::NameValue(pair) => {
            let Expr::Lit(lit) = &pair.value else {
                return None;
            };
            let syn::Lit::Str(value) = &lit.lit else {
                return None;
            };
            if pair.path.is_ident("target_os") && value.value() == "macos" {
                Some(mac)
            } else if pair.path.is_ident("feature") && value.value() == "backend-candle" {
                Some(candle)
            } else {
                None
            }
        }
        Meta::List(list) => {
            let args = list
                .parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, Token![,]>::parse_terminated,
                )
                .ok()?;
            let values = args
                .iter()
                .map(|arg| cfg_meta(arg, mac, candle, test))
                .collect::<Option<Vec<_>>>()?;
            if list.path.is_ident("any") {
                Some(values.iter().any(|value| *value))
            } else if list.path.is_ident("all") {
                Some(values.iter().all(|value| *value))
            } else if list.path.is_ident("not") && values.len() == 1 {
                Some(!values[0])
            } else {
                None
            }
        }
        _ => None,
    }
}

fn native_alias_copies(meta: &syn::Meta) -> usize {
    fn canonical(meta: &syn::Meta) -> bool {
        match meta {
            syn::Meta::Path(path) => path.is_ident("native_lane"),
            syn::Meta::List(list) => {
                let Ok(args) = list.parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, Token![,]>::parse_terminated,
                ) else {
                    return false;
                };
                if list.path.is_ident("not") && args.len() == 1 {
                    return canonical(&args[0]);
                }
                list.path.is_ident("any")
                    && args.len() == 2
                    && args.iter().any(
                        |arg| matches!(arg, syn::Meta::Path(path) if path.is_ident("native_lane")),
                    )
                    && args
                        .iter()
                        .any(|arg| matches!(arg, syn::Meta::Path(path) if path.is_ident("test")))
            }
            _ => false,
        }
    }
    let equivalent = |inverse: bool, include_test: bool| {
        [false, true].into_iter().all(|mac| {
            [false, true].into_iter().all(|candle| {
                [false, true].into_iter().all(|test| {
                    cfg_meta(meta, mac, candle, test)
                        == Some((mac || candle || (include_test && test)) != inverse)
                })
            })
        })
    };
    if !canonical(meta)
        && [false, true].into_iter().any(|inverse| {
            [false, true]
                .into_iter()
                .any(|include_test| equivalent(inverse, include_test))
        })
    {
        return 1;
    }
    if let syn::Meta::List(list) = meta {
        if let Ok(args) = list
            .parse_args_with(syn::punctuated::Punctuated::<syn::Meta, Token![,]>::parse_terminated)
        {
            return args.iter().map(native_alias_copies).sum();
        }
    }
    0
}

#[test]
fn cfg_rules_use_the_checked_native_alias_and_keep_helpers_in_their_lanes() {
    struct Attributes {
        copies: usize,
        inverse_dead: usize,
    }
    impl<'ast> Visit<'ast> for Attributes {
        fn visit_attribute(&mut self, attr: &'ast syn::Attribute) {
            if attr.path().is_ident("cfg") || attr.path().is_ident("cfg_attr") {
                let copies = native_alias_copies(&attr.meta);
                if copies > 0 {
                    eprintln!("expanded native cfg: {}", attr.to_token_stream());
                }
                self.copies += copies;
            }
            if attr.path().is_ident("cfg_attr") {
                let args = attr
                    .parse_args_with(
                        syn::punctuated::Punctuated::<syn::Meta, Token![,]>::parse_terminated,
                    )
                    .unwrap();
                if args.iter().skip(1).any(|meta| matches!(meta, syn::Meta::List(list) if list.path.is_ident("allow") && list.tokens.to_string().split_whitespace().any(|token| token == "dead_code"))) {
                    self.inverse_dead += 1;
                }
            }
        }
    }
    let worker = Path::new(env!("CARGO_MANIFEST_DIR"));
    let scoped = [
        "video_jobs/seedvr2.rs",
        "video_admission.rs",
        "image_jobs/stream.rs",
        "image_jobs/strict_control.rs",
        "image_jobs/geometry.rs",
        "image_jobs/routing.rs",
        "image_jobs/artifact_resolution.rs",
        "image_jobs/quantization.rs",
        "image_jobs/load_spec.rs",
        "image_jobs/adapters.rs",
        "image_jobs/load_policy.rs",
        "image_jobs/memory.rs",
        "image_jobs/sampling.rs",
        "image_jobs/conditioning.rs",
        "image_jobs/render.rs",
        "image_jobs/mlx.rs",
        "image_jobs/candle.rs",
        "image_jobs/metrics.rs",
        "image_jobs/events.rs",
        "image_decode.rs",
    ];
    for path in rust_sources(&worker.join("src")) {
        let mut attributes = Attributes {
            copies: 0,
            inverse_dead: 0,
        };
        attributes.visit_file(&parse(&std::fs::read_to_string(&path).unwrap()));
        assert_eq!(
            attributes.copies,
            0,
            "{} duplicates the native-lane predicate",
            path.display()
        );
        if scoped
            .iter()
            .any(|relative| path == worker.join("src").join(relative))
        {
            assert_eq!(
                attributes.inverse_dead,
                0,
                "{} suppresses lane-dependent dead code instead of gating its callers/items",
                path.display()
            );
        }
    }
    for source in [
        "#[cfg(any(test, target_os = \"macos\", feature = \"backend-candle\"))] fn copy() {}",
        "#[cfg(all(not(target_os = \"macos\"), not(feature = \"backend-candle\")))] fn copy() {}",
        "#[cfg(any(target_os = \"macos\", feature = \"backend-candle\"))] fn copy() {}",
        "#[cfg(any(test, any(target_os = \"macos\", all(not(target_os = \"macos\"), feature = \"backend-candle\"))))] fn copy() {}",
    ] {
        let mut attributes = Attributes { copies: 0, inverse_dead: 0 };
        attributes.visit_file(&parse(source));
        assert_eq!(attributes.copies, 1, "new predicate copies must fail even when nested under test");
    }
}

#[allow(dead_code)] // build.rs main is not invoked by these non-inference contract tests.
#[path = "../build.rs"]
mod worker_build_script;

#[test]
fn build_script_alias_matches_the_full_platform_feature_truth_table() {
    for os in ["macos", "windows", "linux", "freebsd"] {
        for candle in [false, true] {
            assert_eq!(
                worker_build_script::native_lane(os, candle),
                os == "macos" || candle,
                "{os} candle={candle}"
            );
        }
    }
    assert_eq!(
        cfg!(native_lane),
        cfg!(any(target_os = "macos", feature = "backend-candle")),
        "Cargo must actually emit the alias for this compilation"
    );
}

#[test]
fn native_stream_helper_and_training_hash_import_follow_their_callers() {
    let image = parse(include_str!("image_jobs.rs"));
    let module_attrs = |name: &str| {
        image
            .items
            .iter()
            .find_map(|item| match item {
                Item::Mod(item) if item.ident == name => Some(&item.attrs),
                _ => None,
            })
            .expect("actual native image module exists")
    };
    let stream = module_attrs("stream");
    let mlx_module = module_attrs("mlx");
    let candle_module = module_attrs("candle");
    let helper = function(
        include_str!("image_jobs/stream.rs"),
        "drive_gen_items_scored_reported",
    );
    let mac_caller = function(include_str!("image_jobs/mlx.rs"), "generate_stream_using");
    let candle_caller = function(
        include_str!("image_jobs/candle.rs"),
        "generate_candle_registry_stream_using",
    );
    for caller in [&mac_caller, &candle_caller] {
        assert!(
            caller
                .block
                .to_token_stream()
                .to_string()
                .contains("drive_gen_items_scored_reported"),
            "the actual shared-helper call must remain in {}",
            caller.sig.ident
        );
    }
    let training = parse(include_str!("training_jobs.rs"));
    let hash_import = training.items.iter().find_map(|item| match item {
        Item::Use(item) if matches!(&item.tree, syn::UseTree::Path(path) if path.ident == "sha2") => Some(item),
        _ => None,
    }).expect("training SHA-256 import exists");
    let portable_test = training
        .items
        .iter()
        .find_map(|item| {
            let Item::Mod(module) = item else {
                return None;
            };
            if module.ident != "tests" {
                return None;
            }
            module
                .content
                .as_ref()?
                .1
                .iter()
                .find_map(|item| match item {
                    Item::Fn(item)
                        if item.sig.ident
                            == "ltx_prepared_bundle_path_is_resolved_under_the_dataset_root" =>
                    {
                        Some(item)
                    }
                    _ => None,
                })
        })
        .expect("portable prepared-bundle byte fixture exists");
    assert!(portable_test
        .block
        .to_token_stream()
        .to_string()
        .contains("Sha256 :: digest"));
    let enabled = |attrs: &[syn::Attribute], mac, candle, test| {
        attrs
            .iter()
            .filter(|attr| attr.path().is_ident("cfg"))
            .all(|attr| {
                cfg_meta(&attr.parse_args::<syn::Meta>().unwrap(), mac, candle, test)
                    .expect("known platform/test predicate")
            })
    };
    for mac in [false, true] {
        for candle in [false, true] {
            for test in [false, true] {
                let native =
                    worker_build_script::native_lane(if mac { "macos" } else { "linux" }, candle);
                let context = format!("mac={mac} candle={candle} test={test}");
                assert_eq!(
                    enabled(stream, mac, candle, test),
                    native,
                    "stream include: {context}"
                );
                assert_eq!(
                    enabled(candle_module, mac, candle, test),
                    native,
                    "Candle owner module boundary: {context}"
                );
                assert_eq!(
                    enabled(mlx_module, mac, candle, test),
                    native,
                    "MLX owner module boundary: {context}"
                );
                assert_eq!(
                    enabled(stream, mac, candle, test) && enabled(&helper.attrs, mac, candle, test),
                    native,
                    "shared scored/report helper must exist for every native caller: {context}"
                );
                assert_eq!(
                    enabled(mlx_module, mac, candle, test)
                        && enabled(&mac_caller.attrs, mac, candle, test),
                    mac,
                    "MLX eligibility unchanged: {context}"
                );
                assert_eq!(
                    enabled(candle_module, mac, candle, test)
                        && enabled(&candle_caller.attrs, mac, candle, test),
                    !mac && candle,
                    "Candle eligibility unchanged: {context}"
                );
                assert_eq!(
                    enabled(&hash_import.attrs, mac, candle, test),
                    native || test,
                    "SHA-256 needed by native witness or ordinary tests only: {context}"
                );
                assert!(
                    enabled(&portable_test.attrs, mac, candle, test),
                    "portable byte-fixture test remains portable"
                );
            }
        }
    }
}

fn helper_fingerprint(function: &syn::ItemFn) -> String {
    #[derive(Default)]
    struct Bindings(BTreeMap<String, String>);
    impl<'ast> Visit<'ast> for Bindings {
        fn visit_pat_ident(&mut self, pat: &'ast syn::PatIdent) {
            let next = format!("binding_{}", self.0.len());
            self.0.entry(pat.ident.to_string()).or_insert(next);
            visit::visit_pat_ident(self, pat);
        }
    }
    fn normalize(
        tokens: proc_macro2::TokenStream,
        bindings: &BTreeMap<String, String>,
    ) -> proc_macro2::TokenStream {
        use proc_macro2::TokenTree;
        tokens
            .into_iter()
            .map(|token| match token {
                TokenTree::Ident(ident) => bindings
                    .get(&ident.to_string())
                    .map_or(TokenTree::Ident(ident.clone()), |name| {
                        TokenTree::Ident(proc_macro2::Ident::new(name, ident.span()))
                    }),
                TokenTree::Group(group) => TokenTree::Group(proc_macro2::Group::new(
                    group.delimiter(),
                    normalize(group.stream(), bindings),
                )),
                token => token,
            })
            .collect()
    }
    let mut bindings = Bindings::default();
    bindings.visit_signature(&function.sig);
    bindings.visit_block(&function.block);
    normalize(function.block.to_token_stream(), &bindings.0).to_string()
}

// This inventory must shrink when another tracked consolidation removes a copy. Reintroducing
// it at the old path must then fail just like a copy in a new file.
const LEGACY_HELPER_COPIES: &[(&str, &str)] = &[("apps/desktop/src/settings.rs", "normalize_host")];

#[test]
fn new_copies_of_shared_core_helpers_are_rejected_even_when_renamed() {
    // Each implementation has a shared core owner; compatibility wrappers delegate there.
    let canonical = [
        (
            "crates/sceneworks-core/src/model_artifacts/artifact_selection.rs",
            "safe_download_dir",
        ),
        (
            "crates/sceneworks-core/src/model_artifacts/artifact_selection.rs",
            "pattern_matches",
        ),
        (
            "crates/sceneworks-core/src/model_artifacts/artifact_selection.rs",
            "allow_pattern_matches",
        ),
        ("crates/sceneworks-core/src/display_util.rs", "format_bytes"),
        ("crates/sceneworks-core/src/display_util.rs", "quote_path"),
        (
            "crates/sceneworks-core/src/payload_util.rs",
            "json_size_to_u64",
        ),
        ("crates/sceneworks-core/src/env_util.rs", "env_string"),
        (
            "crates/sceneworks-core/src/env_util.rs",
            "env_trimmed_string",
        ),
        ("crates/sceneworks-core/src/env_util.rs", "env_path_or"),
        ("crates/sceneworks-core/src/time.rs", "utc_now"),
        ("crates/sceneworks-core/src/time.rs", "civil_from_days"),
        ("crates/sceneworks-core/src/ids.rs", "is_safe_id"),
        ("crates/sceneworks-core/src/ids.rs", "is_safe_id_with_max"),
        (
            "crates/sceneworks-core/src/safe_path.rs",
            "portable_relative_path_parts",
        ),
        (
            "crates/sceneworks-core/src/safe_path.rs",
            "relative_normal_components",
        ),
        (
            "crates/sceneworks-core/src/safe_path.rs",
            "path_normal_components",
        ),
        (
            "crates/sceneworks-core/src/safe_path.rs",
            "is_link_or_reparse_point",
        ),
        ("crates/sceneworks-core/src/safe_path.rs", "confine_under"),
        (
            "crates/sceneworks-core/src/safe_path.rs",
            "resolve_local_file",
        ),
        (
            "crates/sceneworks-core/src/safe_path.rs",
            "is_safe_media_path",
        ),
        (
            "crates/sceneworks-core/src/file_lock.rs",
            "open_regular_lock_file",
        ),
        ("crates/sceneworks-core/src/slug.rs", "slugify"),
        ("crates/sceneworks-core/src/slug.rs", "slugify_with"),
        (
            "crates/sceneworks-core/src/api_control.rs",
            "post_json_with_client",
        ),
        ("crates/sceneworks-core/src/safetensors.rs", "read_header"),
        (
            "crates/sceneworks-core/src/safetensors.rs",
            "read_file_header",
        ),
        ("crates/sceneworks-core/src/safetensors.rs", "dtype_bits"),
        (
            "crates/sceneworks-core/src/builtin_manifests.rs",
            "builtin_manifest",
        ),
        ("crates/sceneworks-core/src/fsutil.rs", "write_atomic"),
        ("crates/sceneworks-core/src/fsutil.rs", "write_atomic_with"),
        (
            "crates/sceneworks-core/src/fsutil.rs",
            "atomic_create_supported",
        ),
        (
            "crates/sceneworks-core/src/fsutil.rs",
            "write_new_unpublished_with",
        ),
        (
            "crates/sceneworks-core/src/fsutil.rs",
            "payload_entry_metadata",
        ),
        (
            "crates/sceneworks-core/src/fsutil.rs",
            "sync_parent_directory",
        ),
        (
            "crates/sceneworks-core/src/media_process.rs",
            "ffmpeg_program",
        ),
        (
            "crates/sceneworks-core/src/media_process.rs",
            "resolve_ffmpeg_program",
        ),
        (
            "crates/sceneworks-core/src/media_process.rs",
            "bounded_tail",
        ),
        ("crates/sceneworks-core/src/media_process.rs", "run_command"),
        (
            "crates/sceneworks-core/src/image_decode.rs",
            "decode_image_any",
        ),
        (
            "crates/sceneworks-core/src/image_decode.rs",
            "decode_image_bytes_any",
        ),
        (
            "crates/sceneworks-core/src/image_decode.rs",
            "decode_image_with",
        ),
        (
            "crates/sceneworks-core/src/image_decode.rs",
            "decode_image_bounded",
        ),
        (
            "crates/sceneworks-core/src/media_convert.rs",
            "ingest_image",
        ),
        (
            "crates/sceneworks-core/src/media_convert.rs",
            "ingest_image_with_naming",
        ),
        (
            "crates/sceneworks-core/src/slug.rs",
            "slugify_worker_id_part",
        ),
        ("crates/sceneworks-core/src/hashing.rs", "hex"),
        ("crates/sceneworks-core/src/hashing.rs", "hex_prefix"),
        ("crates/sceneworks-core/src/hashing.rs", "sha256_bytes"),
        (
            "crates/sceneworks-core/src/hashing.rs",
            "update_sha256_with",
        ),
        ("crates/sceneworks-core/src/hashing.rs", "update_sha256"),
        (
            "crates/sceneworks-core/src/hashing.rs",
            "sha256_reader_with",
        ),
        (
            "crates/sceneworks-core/src/hashing.rs",
            "try_sha256_reader_with",
        ),
        ("crates/sceneworks-core/src/hashing.rs", "sha256_reader"),
        (
            "crates/sceneworks-core/src/hashing.rs",
            "sha256_reader_retry",
        ),
        ("crates/sceneworks-core/src/hashing.rs", "sha256_file"),
        ("crates/sceneworks-core/src/hashing.rs", "random_hex"),
        (
            "crates/sceneworks-core/src/hf_home.rs",
            "safe_repo_dir_name",
        ),
        (
            "crates/sceneworks-core/src/lora_url.rs",
            "safe_lora_filename",
        ),
        (
            "crates/sceneworks-core/src/video_request.rs",
            "ltx_frame_count",
        ),
        (
            "crates/sceneworks-core/src/credentials.rs",
            "normalize_host",
        ),
        (
            "crates/sceneworks-core/src/credentials.rs",
            "normalize_scheme",
        ),
    ];
    let root = root();
    let helpers: Vec<_> = canonical
        .into_iter()
        .map(|(path, name)| {
            let item = function(&std::fs::read_to_string(root.join(path)).unwrap(), name);
            (path, name, helper_fingerprint(&item))
        })
        .collect();
    struct Functions(Vec<syn::ItemFn>);
    impl<'ast> Visit<'ast> for Functions {
        fn visit_item_fn(&mut self, function: &'ast syn::ItemFn) {
            self.0.push(function.clone());
            visit::visit_item_fn(self, function);
        }
        fn visit_impl_item_fn(&mut self, function: &'ast syn::ImplItemFn) {
            self.0.push(syn::ItemFn {
                attrs: function.attrs.clone(),
                vis: function.vis.clone(),
                sig: function.sig.clone(),
                block: Box::new(function.block.clone()),
            });
            visit::visit_impl_item_fn(self, function);
        }
        fn visit_trait_item_fn(&mut self, function: &'ast syn::TraitItemFn) {
            if let Some(block) = &function.default {
                self.0.push(syn::ItemFn {
                    attrs: function.attrs.clone(),
                    vis: syn::Visibility::Inherited,
                    sig: function.sig.clone(),
                    block: Box::new(block.clone()),
                });
            }
            visit::visit_trait_item_fn(self, function);
        }
    }
    fn delegates_to_core(function: &syn::ItemFn, name: &str, in_core: bool) -> bool {
        // The async file adapter owns heartbeat/load progress around one blocking core call.
        // It may not quietly grow a local streaming implementation again.
        if function.sig.asyncness.is_some() && name == "sha256_file" {
            #[derive(Default)]
            struct HashAdapter {
                core: usize,
                spawn: usize,
                heartbeat: usize,
                loops: bool,
            }
            impl<'ast> Visit<'ast> for HashAdapter {
                fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
                    if let Expr::Path(target) = call.func.as_ref() {
                        let path = target.path.to_token_stream().to_string();
                        match path.as_str() {
                            "sceneworks_core :: hashing :: sha256_file" => self.core += 1,
                            "crate :: load_progress :: spawn_blocking" => self.spawn += 1,
                            "heartbeat_while_blocking" => self.heartbeat += 1,
                            _ => {}
                        }
                    }
                    visit::visit_expr_call(self, call);
                }
                fn visit_expr_for_loop(&mut self, node: &'ast syn::ExprForLoop) {
                    self.loops = true;
                    visit::visit_expr_for_loop(self, node);
                }
                fn visit_expr_while(&mut self, node: &'ast syn::ExprWhile) {
                    self.loops = true;
                    visit::visit_expr_while(self, node);
                }
                fn visit_expr_loop(&mut self, node: &'ast syn::ExprLoop) {
                    self.loops = true;
                    visit::visit_expr_loop(self, node);
                }
            }
            let mut adapter = HashAdapter::default();
            adapter.visit_block(&function.block);
            return adapter.core == 1
                && adapter.spawn == 1
                && adapter.heartbeat == 1
                && !adapter.loops;
        }
        if function.sig.asyncness.is_some() && name == "sync_parent_directory" {
            // A checked-in blocking adapter owns join/error conversion, while core owns all I/O.
            // Match its complete body, including the spawn boundary; extra I/O or loops fail.
            let expected: syn::ItemFn = syn::parse_quote! {
                async fn sync_parent_directory(path: &Path) -> WorkerResult<()> {
                    let path = path.to_path_buf();
                    crate::load_progress::spawn_blocking(move || -> WorkerResult<()> {
                        Ok(sceneworks_core::fsutil::sync_parent_directory(&path)?)
                    })
                    .await
                    .map_err(|error| task_join_error("download directory sync", error))?
                }
            };
            return helper_fingerprint(function) == helper_fingerprint(&expected);
        }
        let expression = match function.block.stmts.as_slice() {
            [syn::Stmt::Expr(expression, _)] => expression,
            [syn::Stmt::Local(local)]
                if name == "sync_parent_directory" && matches!(local.pat, syn::Pat::Wild(_)) =>
            {
                let Some(initializer) = &local.init else {
                    return false;
                };
                initializer.expr.as_ref()
            }
            _ => return false,
        };
        let expression = if let Expr::Return(returned) = expression {
            let Some(value) = &returned.expr else {
                return false;
            };
            value.as_ref()
        } else {
            expression
        };
        // Domain error adaptation does not duplicate the shared implementation.
        let expression = if let Expr::MethodCall(call) = expression {
            if call.method != "map_err" {
                return false;
            }
            call.receiver.as_ref()
        } else {
            expression
        };
        let Expr::Call(call) = expression else {
            return false;
        };
        let Expr::Path(target) = call.func.as_ref() else {
            return false;
        };
        let store_atomic_facade = in_core
            && name == "write_atomic"
            && target.path.to_token_stream().to_string() == "crate :: store_util :: atomic_write";
        (store_atomic_facade
            || target
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == name))
            && target.path.segments.first().is_some_and(|segment| {
                segment.ident == "sceneworks_core" || (in_core && segment.ident == "crate")
            })
    }
    let mut copies = Vec::new();
    let mut legacy_found = BTreeSet::new();
    let mut canonical_seen: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    for member in read_manifest(&root.join("Cargo.toml"))["workspace"]["members"]
        .as_array()
        .unwrap()
    {
        for path in rust_sources(&root.join(member.as_str().unwrap()).join("src")) {
            let display = path
                .strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let mut functions = Functions(Vec::new());
            functions.visit_file(&parse(&std::fs::read_to_string(&path).unwrap()));
            for item in functions.0 {
                let fingerprint = helper_fingerprint(&item);
                for (owner, name, body) in &helpers {
                    let legacy = LEGACY_HELPER_COPIES.contains(&(display.as_str(), *name))
                        && item.sig.ident == *name;
                    if legacy && fingerprint == *body {
                        legacy_found.insert((display.clone(), (*name).to_owned()));
                    }
                    let canonical = display == *owner && item.sig.ident == *name;
                    if canonical {
                        *canonical_seen.entry((owner, name)).or_default() += 1;
                    }
                    let forwarder = delegates_to_core(
                        &item,
                        name,
                        display.starts_with("crates/sceneworks-core/"),
                    );
                    if !canonical
                        && !legacy
                        && (fingerprint == *body || (item.sig.ident == *name && !forwarder))
                    {
                        copies.push(format!(
                            "{display}::{} copies {owner}::{name}",
                            item.sig.ident
                        ));
                    }
                }
            }
        }
    }
    assert!(copies.is_empty(), "new helper copies: {copies:?}");
    for (owner, name, _) in &helpers {
        assert_eq!(canonical_seen.get(&(*owner, *name)), Some(&1), "{owner}::{name} must have exactly one definition, including nested functions and associated methods");
    }
    let recorded_legacy = LEGACY_HELPER_COPIES
        .iter()
        .map(|(path, name)| ((*path).to_owned(), (*name).to_owned()))
        .collect();
    assert_eq!(legacy_found, recorded_legacy, "a recorded helper copy was removed or changed; delete its legacy exception so the old location cannot silently absorb a future copy");
    let source = function(
        &std::fs::read_to_string(root.join(helpers[0].0)).unwrap(),
        helpers[0].1,
    );
    let renamed = source
        .to_token_stream()
        .to_string()
        .replace("safe_download_dir", "innocent_new_helper")
        .replace("value", "input")
        .replace("output", "result");
    let renamed: syn::ItemFn = syn::parse_str(&renamed).unwrap();
    assert_eq!(
        helper_fingerprint(&source),
        helper_fingerprint(&renamed),
        "renaming the function and bindings cannot hide a helper copy"
    );
    let forwarding: syn::ItemFn = syn::parse_quote! {
        pub fn safe_download_dir(value: &str) -> String {
            sceneworks_core::model_artifacts::safe_download_dir(value)
        }
    };
    assert!(
        delegates_to_core(&forwarding, "safe_download_dir", false),
        "a compatibility delegate is not a helper implementation copy"
    );
    let adapter = function(
        &std::fs::read_to_string(root.join("crates/sceneworks-worker/src/downloads.rs")).unwrap(),
        "sha256_file",
    );
    assert!(delegates_to_core(&adapter, "sha256_file", false));
    let mut copied_stream = adapter.clone();
    copied_stream.block.stmts.insert(
        0,
        syn::parse_quote! { for byte in bytes { consume(byte); } },
    );
    assert!(
        !delegates_to_core(&copied_stream, "sha256_file", false),
        "a lifecycle wrapper cannot contain a copied streaming loop"
    );
    let mapped: syn::ItemFn = syn::parse_quote! {
        fn random_hex(bytes: usize) -> Result<String, Error> {
            crate::hashing::random_hex(bytes).map_err(Error::from)
        }
    };
    assert!(delegates_to_core(&mapped, "random_hex", true));
    let sync_adapter = function(
        &std::fs::read_to_string(root.join("crates/sceneworks-worker/src/downloads.rs")).unwrap(),
        "sync_parent_directory",
    );
    assert!(delegates_to_core(
        &sync_adapter,
        "sync_parent_directory",
        false
    ));
    let mut extra_io = sync_adapter.clone();
    extra_io.block.stmts.insert(
        0,
        syn::parse_quote! { std::fs::write(path, b"unexpected")?; },
    );
    assert!(
        !delegates_to_core(&extra_io, "sync_parent_directory", false),
        "a blocking adapter cannot hide additional file I/O"
    );
    let best_effort: syn::ItemFn = syn::parse_quote! {
        fn sync_parent_directory(path: &Path) { let _ = sceneworks_core::fsutil::sync_parent_directory(path); }
    };
    assert!(delegates_to_core(
        &best_effort,
        "sync_parent_directory",
        false
    ));
    let store_facade: syn::ItemFn = syn::parse_quote! {
        fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Error> {
            crate::store_util::atomic_write(path, bytes).map_err(Error::from)
        }
    };
    assert!(delegates_to_core(&store_facade, "write_atomic", true));
    let mut associated = Functions(Vec::new());
    associated.visit_file(&parse(&format!(
        "struct NewOwner; impl NewOwner {{ {} }}",
        renamed.to_token_stream()
    )));
    assert_eq!(associated.0.len(), 1);
    assert_eq!(
        helper_fingerprint(&source),
        helper_fingerprint(&associated.0[0]),
        "an associated helper must not hide the same copy"
    );
}

#[test]
fn confinement_policy_has_one_core_type_and_the_shared_function_uses_it() {
    const OWNER: &str = "crates/sceneworks-core/src/safe_path.rs";
    fn definitions(source: &str) -> Vec<String> {
        #[derive(Default)]
        struct Definitions(Vec<String>);
        impl<'ast> Visit<'ast> for Definitions {
            fn visit_item_enum(&mut self, item: &'ast syn::ItemEnum) {
                let variants: Vec<_> = item.variants.iter().map(|v| v.ident.to_string()).collect();
                if item.ident == "Confinement" || variants == ["Lexical", "SymlinkSafe"] {
                    self.0.push(item.ident.to_string());
                }
                visit::visit_item_enum(self, item);
            }
            fn visit_item_struct(&mut self, item: &'ast syn::ItemStruct) {
                if item.ident == "Confinement" {
                    self.0.push(item.ident.to_string());
                }
                visit::visit_item_struct(self, item);
            }
            fn visit_item_type(&mut self, item: &'ast syn::ItemType) {
                if item.ident == "Confinement" {
                    self.0.push(item.ident.to_string());
                }
                visit::visit_item_type(self, item);
            }
        }
        let mut definitions = Definitions::default();
        definitions.visit_file(&parse(source));
        definitions.0
    }
    let root = root();
    let mut owners = Vec::new();
    for member in read_manifest(&root.join("Cargo.toml"))["workspace"]["members"]
        .as_array()
        .unwrap()
    {
        for path in rust_sources(&root.join(member.as_str().unwrap()).join("src")) {
            for name in definitions(&std::fs::read_to_string(&path).unwrap()) {
                owners.push((
                    path.strip_prefix(&root)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/"),
                    name,
                ));
            }
        }
    }
    assert_eq!(
        owners,
        [(OWNER.to_owned(), "Confinement".to_owned())],
        "confinement policy must use the core type instead of a copied or renamed contract"
    );
    let _: fn(&Path, &str, sceneworks_core::safe_path::Confinement) -> std::io::Result<PathBuf> =
        sceneworks_core::safe_path::confine_under;
    assert_eq!(
        definitions("enum Renamed { Lexical, SymlinkSafe }"),
        ["Renamed"]
    );
    assert_eq!(
        definitions("mod nested { struct Confinement; }"),
        ["Confinement"]
    );
    assert!(
        definitions("const PROSE: &str = \"enum Confinement { Lexical, SymlinkSafe }\";")
            .is_empty()
    );
}
