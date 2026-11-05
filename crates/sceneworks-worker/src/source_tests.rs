//! Source-contract fixtures follow the authored facade instead of copying its file list.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use syn::{Item, UseTree};

/// Read a worker source and the modules it imports directly with a glob. Scoped modules and
/// cfg(test) module declarations retain their namespaces; string/comment lookalikes are not code.
pub(crate) fn facade_source(relative: &str) -> String {
    facade_source_at(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join(relative),
    )
}

fn direct_glob_modules(tree: &UseTree) -> Vec<String> {
    match tree {
        UseTree::Path(path) => {
            if matches!(&*path.tree, UseTree::Glob(_))
                || matches!(&*path.tree, UseTree::Group(group) if group.items.iter().any(|item| matches!(item, UseTree::Glob(_))))
            {
                vec![path.ident.to_string()]
            } else if path.ident == "self" {
                direct_glob_modules(&path.tree)
            } else {
                Vec::new()
            }
        }
        UseTree::Group(group) => group.items.iter().flat_map(direct_glob_modules).collect(),
        _ => Vec::new(),
    }
}

fn test_module(module: &syn::ItemMod) -> bool {
    module.attrs.iter().any(|attribute| {
        attribute.path().is_ident("cfg")
            && attribute
                .parse_args::<syn::Meta>()
                .is_ok_and(|meta| matches!(meta, syn::Meta::Path(path) if path.is_ident("test")))
    })
}

fn module_path(module: &syn::ItemMod, file: &Path) -> PathBuf {
    for attribute in &module.attrs {
        if let syn::Meta::NameValue(value) = &attribute.meta {
            if value.path.is_ident("path") {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(path),
                    ..
                }) = &value.value
                {
                    return file.parent().unwrap().join(path.value());
                }
            }
        }
    }
    let parent = file.parent().unwrap();
    let stem = file.file_stem().unwrap().to_str().unwrap();
    let directory = if ["lib", "main", "mod"].contains(&stem) {
        parent.to_path_buf()
    } else {
        parent.join(stem)
    };
    let direct = directory.join(format!("{}.rs", module.ident));
    if direct.is_file() {
        direct
    } else {
        directory.join(module.ident.to_string()).join("mod.rs")
    }
}

fn facade_source_at(file: &Path) -> String {
    fn visit(file: &Path, seen: &mut BTreeSet<PathBuf>, output: &mut String) {
        let file = file
            .canonicalize()
            .unwrap_or_else(|error| panic!("{}: {error}", file.display()));
        if !seen.insert(file.clone()) {
            return;
        }
        let source = std::fs::read_to_string(&file).unwrap();
        let syntax = syn::parse_file(&source).expect("source-contract input must be valid Rust");
        let globs: BTreeSet<_> = syntax
            .items
            .iter()
            .filter_map(|item| {
                if let Item::Use(item) = item {
                    Some(direct_glob_modules(&item.tree))
                } else {
                    None
                }
            })
            .flatten()
            .collect();
        output.push_str(&source);
        output.push('\n');
        for item in &syntax.items {
            if let Item::Mod(module) = item {
                if module.content.is_none()
                    && !test_module(module)
                    && globs.contains(&module.ident.to_string())
                {
                    visit(&module_path(module, &file), seen, output);
                }
            }
        }
    }
    let mut output = String::new();
    visit(file, &mut BTreeSet::new(), &mut output);
    output
}

#[test]
fn facade_view_follows_actual_imports_and_keeps_scoped_and_test_modules_out() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().join("root.rs");
    std::fs::write(
        &root,
        r###"
#[path="owner.rs"] mod owner; use owner::*;
#[path="scoped.rs"] mod scoped;
#[cfg(test)] #[path="test.rs"] mod hidden; use hidden::*;
const QUOTED: &str = r#"#[path="absent.rs"] mod phantom; use phantom::*;"#;
"###,
    )
    .unwrap();
    std::fs::write(temporary.path().join("owner.rs"), "fn actual_owner() {}\n").unwrap();
    std::fs::write(temporary.path().join("scoped.rs"), "fn scoped_owner() {}\n").unwrap();
    std::fs::write(
        temporary.path().join("test.rs"),
        "fn test_only_owner() {}\n",
    )
    .unwrap();
    let source = facade_source_at(&root);
    assert!(source.contains("fn actual_owner()"));
    assert!(!source.contains("fn scoped_owner()"));
    assert!(!source.contains("fn test_only_owner()"));
    std::fs::write(temporary.path().join("next.rs"), "fn next_owner() {}\n").unwrap();
    let mut source = std::fs::read_to_string(&root).unwrap();
    source.push_str("#[path=\"next.rs\"] mod next; pub(crate) use next::*;\n");
    std::fs::write(&root, source).unwrap();
    assert!(facade_source_at(&root).contains("fn next_owner()"));
}

#[test]
fn facade_view_reads_the_actual_worker_owner() {
    assert!(facade_source("candle_memory_strategy.rs").contains("fn evaluate_shared_image_priced("));
}
