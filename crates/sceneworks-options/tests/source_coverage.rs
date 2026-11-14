//! Static coverage of option reads. This test parses source; it never loads an engine.
//! Macro expansion and compiler name resolution are outside this guard. Source-owned
//! finite tables below cover the specifically documented dynamic training selectors.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};
use syn::{Expr, FnArg, Item, Lit, Meta, Pat};

const ROOTS: &[&str] = &[
    "crates/sceneworks-core/src",
    "crates/sceneworks-worker/src",
    "crates/sceneworks-mcp/src",
    "apps/rust-api/src",
];

// Evaluate only whether an item can exist in a production build. Unknown target/feature
// predicates remain possible. In particular any(native_lane, test) is production code.
fn possible(meta: &Meta, test: bool) -> bool {
    match meta {
        Meta::Path(path) if path.is_ident("test") => test,
        Meta::List(list) => {
            let nested = list
                .parse_args_with(
                    syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
                )
                .expect("valid cfg expression");
            if list.path.is_ident("all") {
                nested.iter().all(|it| possible(it, test))
            } else if list.path.is_ident("any") {
                nested.iter().any(|it| possible(it, test))
            } else if list.path.is_ident("not") && nested.len() == 1 {
                // Negated platform predicates are also possible on some production target.
                match nested.first().unwrap() {
                    Meta::Path(path) if path.is_ident("test") => !test,
                    _ => true,
                }
            } else {
                true
            }
        }
        _ => true,
    }
}

fn production(attrs: &[syn::Attribute]) -> bool {
    attrs
        .iter()
        .filter(|attr| attr.path().is_ident("cfg"))
        .all(|attr| possible(&attr.parse_args::<Meta>().expect("valid cfg"), false))
}

fn item_attrs(item: &Item) -> &[syn::Attribute] {
    match item {
        Item::Const(it) => &it.attrs,
        Item::Enum(it) => &it.attrs,
        Item::ExternCrate(it) => &it.attrs,
        Item::Fn(it) => &it.attrs,
        Item::ForeignMod(it) => &it.attrs,
        Item::Impl(it) => &it.attrs,
        Item::Macro(it) => &it.attrs,
        Item::Mod(it) => &it.attrs,
        Item::Static(it) => &it.attrs,
        Item::Struct(it) => &it.attrs,
        Item::Trait(it) => &it.attrs,
        Item::TraitAlias(it) => &it.attrs,
        Item::Type(it) => &it.attrs,
        Item::Union(it) => &it.attrs,
        Item::Use(it) => &it.attrs,
        _ => &[],
    }
}

fn unwrapped(expr: &Expr) -> &Expr {
    match expr {
        Expr::Reference(it) => unwrapped(&it.expr),
        Expr::Paren(it) => unwrapped(&it.expr),
        Expr::Group(it) => unwrapped(&it.expr),
        _ => expr,
    }
}

fn name(expr: &Expr) -> Option<String> {
    match unwrapped(expr) {
        Expr::Path(it) => it.path.segments.last().map(|it| it.ident.to_string()),
        _ => None,
    }
}

fn ident(pat: &Pat) -> Option<String> {
    match pat {
        Pat::Ident(it) => Some(it.ident.to_string()),
        Pat::Reference(it) => ident(&it.pat),
        Pat::Type(it) => ident(&it.pat),
        _ => None,
    }
}

#[derive(Default, Clone)]
struct Bindings {
    maps: BTreeSet<String>,
    descendants: BTreeSet<String>,
    strings: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Read {
    file: String,
    function: String,
    location: &'static str,
    key: String,
}

#[derive(Default)]
struct Coverage {
    reads: BTreeSet<Read>,
    dynamic: BTreeSet<(String, String, String)>,
    delegated: BTreeSet<(String, String, String)>,
    whole_map_reads: BTreeSet<(String, String, String)>,
    nested_reads: BTreeSet<(String, String, String)>,
    mutations: BTreeSet<(String, String, String)>,
    domains: BTreeMap<String, BTreeSet<String>>,
}

// These functions delegate a caller-supplied key; the call visitor below checks their key
// argument, including through the advanced module and renamed imports. Keep this explicit:
// adding another dynamic accessor requires reviewing its call contract here.
fn helper_key(name: &str) -> Option<usize> {
    match name {
        "flag"
        | "bool"
        | "f32"
        | "f32_clamped"
        | "i32"
        | "u32_clamped"
        | "str"
        | "technique_value"
        | "preflight_string"
        | "preflight_bool"
        | "preflight_u32"
        | "preflight_f32"
        | "preflight_i32"
        | "preflight_bounded_f64"
        | "optional_payload_string"
        | "advanced_bounded_number"
        | "number"
        | "text"
        | "menu_choice"
        | "number_in"
        | "unit_interval"
        | "bounded"
        | "advanced_string"
        | "advanced_u64"
        | "string_value"
        | "plan_advanced_bool"
        | "checkpoint_plan_f32_override"
        | "checkpoint_plan_u32_override"
        | "optional_advanced_name"
        | "strict_optional_number"
        | "advanced_float"
        | "advanced_opt_f32"
        | "advanced_opt_u32"
        | "svd_f32"
        | "svd_i32"
        | "resolve_advanced_or_manifest_u32"
        | "resolve_advanced_or_manifest_f32"
        | "resolve_advanced_or_manifest_u32_with"
        | "resolve_advanced_or_manifest_f32_with" => Some(1),
        _ => None,
    }
}

// Routing helpers accept a JSON object name and a child key. Their call sites are option
// reads whenever the object name is `advanced`, even though the helper definition uses a
// generic `object_key` parameter and never spells `advanced` itself.
fn nested_helper_key(name: &str) -> Option<usize> {
    match name {
        "has_nonempty_nested_array"
        | "has_nonempty_or_malformed_nested_array"
        | "has_nonnull_or_malformed_nested_carrier"
        | "has_malformed_optional_nested_number" => Some(2),
        _ => None,
    }
}

struct Scanner<'a> {
    file: &'a str,
    function: String,
    location: &'static str,
    bindings: Bindings,
    constants: &'a BTreeMap<String, BTreeSet<String>>,
    imports: BTreeMap<String, String>,
    delegated: BTreeSet<String>,
    closures: BTreeMap<String, usize>,
    writing: bool,
    coverage: &'a mut Coverage,
}

impl Scanner<'_> {
    fn map(&self, expr: &Expr) -> bool {
        match unwrapped(expr) {
            Expr::Field(it) => matches!(&it.member, syn::Member::Named(n) if n == "advanced"),
            Expr::Path(_) => {
                name(expr).is_some_and(|n| n == "advanced" || self.bindings.maps.contains(&n))
            }
            Expr::Index(it) => self
                .strings(&it.index)
                .is_some_and(|keys| keys == BTreeSet::from(["advanced".into()])),
            Expr::MethodCall(it) => {
                let method = it.method.to_string();
                if matches!(method.as_str(), "get" | "get_mut")
                    && it.args.len() == 1
                    && self
                        .strings(&it.args[0])
                        .is_some_and(|keys| keys == BTreeSet::from(["advanced".into()]))
                {
                    true
                } else {
                    ((method == "and_then"
                        && it.args.first().is_some_and(|arg| {
                            matches!(name(arg).as_deref(), Some("as_object" | "as_object_mut"))
                        }))
                        || matches!(
                            method.as_str(),
                            "as_object"
                                | "as_object_mut"
                                | "as_ref"
                                | "as_mut"
                                | "unwrap"
                                | "expect"
                                | "unwrap_or_default"
                                | "cloned"
                                | "clone"
                        ))
                        && self.map(&it.receiver)
                }
            }
            _ => false,
        }
    }

    /// A JSON value reached by reading a child of `advanced`. Follow only source-visible
    /// aliases and simple wrappers; typed domain structs are deliberately not aliases here.
    fn descendant(&self, expr: &Expr) -> bool {
        match unwrapped(expr) {
            Expr::Path(_) => {
                name(expr).is_some_and(|name| self.bindings.descendants.contains(&name))
            }
            Expr::Index(it) => self.map(&it.expr) || self.descendant(&it.expr),
            Expr::MethodCall(it) => {
                let method = it.method.to_string();
                if matches!(method.as_str(), "get" | "get_mut") {
                    self.map(&it.receiver) || self.descendant(&it.receiver)
                } else {
                    matches!(
                        method.as_str(),
                        "as_object"
                            | "as_object_mut"
                            | "as_array"
                            | "as_array_mut"
                            | "as_ref"
                            | "as_mut"
                            | "unwrap"
                            | "expect"
                            | "cloned"
                            | "clone"
                            | "iter"
                            | "iter_mut"
                            | "into_iter"
                            | "filter"
                            | "map"
                            | "and_then"
                    ) && self.descendant(&it.receiver)
                }
            }
            _ => false,
        }
    }

    fn strings(&self, expr: &Expr) -> Option<BTreeSet<String>> {
        match unwrapped(expr) {
            Expr::Lit(it) => match &it.lit {
                Lit::Str(it) => Some(BTreeSet::from([it.value()])),
                _ => None,
            },
            Expr::Unary(it) if matches!(it.op, syn::UnOp::Deref(_)) => self.strings(&it.expr),
            Expr::Array(it) => it
                .elems
                .iter()
                .map(|it| self.strings(it))
                .collect::<Option<Vec<_>>>()
                .map(|sets| sets.into_iter().flatten().collect()),
            Expr::If(it) => {
                let yes = it.then_branch.stmts.last().and_then(|stmt| match stmt {
                    syn::Stmt::Expr(expr, None) => self.strings(expr),
                    _ => None,
                })?;
                let no = self.strings(&it.else_branch.as_ref()?.1)?;
                Some(yes.union(&no).cloned().collect())
            }
            Expr::Block(it) => it.block.stmts.last().and_then(|stmt| match stmt {
                syn::Stmt::Expr(expr, None) => self.strings(expr),
                _ => None,
            }),
            Expr::Field(it) => match &it.member {
                syn::Member::Named(member) => {
                    self.constants.get(&format!("<field>:{member}")).cloned()
                }
                _ => None,
            },
            Expr::Path(_) => {
                let key = name(expr)?;
                self.bindings
                    .strings
                    .get(&key)
                    .or_else(|| self.constants.get(&key))
                    .cloned()
            }
            Expr::MethodCall(it)
                if matches!(
                    it.method.to_string().as_str(),
                    "to_owned" | "to_string" | "as_str" | "into"
                ) =>
            {
                self.strings(&it.receiver)
            }
            _ => None,
        }
    }

    fn read(&mut self, expr: &Expr) {
        if let Some(keys) = self.strings(expr) {
            for key in keys {
                self.coverage.reads.insert(Read {
                    file: self.file.into(),
                    function: self.function.clone(),
                    location: self.location,
                    key,
                });
            }
        } else {
            let read = (self.file.into(), self.function.clone(), describe(expr));
            if name(expr).is_some_and(|key| self.delegated.contains(&key)) {
                self.coverage.delegated.insert(read);
            } else {
                self.coverage.dynamic.insert(read);
            }
        }
    }

    fn function(&mut self, sig: &syn::Signature, body: &syn::Block) {
        let old_function = std::mem::replace(&mut self.function, sig.ident.to_string());
        let old_location = self.location;
        if self.file.contains("iris_curricula")
            || matches!(
                self.function.as_str(),
                "training_job_is_candle_eligible"
                    | "moss_training_plan_is_supported"
                    | "training_plan_is_lokr"
                    | "from_plan_syntax"
                    | "apply_training_retry_resume"
            )
        {
            self.location = "training.config.advanced";
        }
        let old_bindings = std::mem::take(&mut self.bindings);
        let old_delegated = std::mem::take(&mut self.delegated);
        let old_closures = std::mem::take(&mut self.closures);
        if let Some(index) = helper_key(&self.function) {
            if let Some(FnArg::Typed(arg)) = sig.inputs.iter().nth(index) {
                if let Some(key) = ident(&arg.pat) {
                    self.delegated.insert(key);
                }
            }
        }
        for arg in &sig.inputs {
            if let FnArg::Typed(it) = arg {
                if let Some(key) = ident(&it.pat) {
                    if key == "advanced" {
                        self.bindings.maps.insert(key);
                    }
                }
            }
        }
        self.visit_block(body);
        self.bindings = old_bindings;
        self.function = old_function;
        self.location = old_location;
        self.delegated = old_delegated;
        self.closures = old_closures;
    }

    fn bind_strings(&mut self, pat: &Pat, exprs: &[&Expr]) {
        if let Some(key) = ident(pat) {
            let values: Option<Vec<_>> = exprs.iter().map(|expr| self.strings(expr)).collect();
            if let Some(values) = values {
                self.bindings
                    .strings
                    .insert(key, values.into_iter().flatten().collect());
            }
        } else if let Pat::Tuple(tuple) = pat {
            for (index, field) in tuple.elems.iter().enumerate() {
                let fields: Option<Vec<_>> = exprs
                    .iter()
                    .map(|expr| match unwrapped(expr) {
                        Expr::Tuple(it) => it.elems.iter().nth(index),
                        _ => None,
                    })
                    .collect();
                if let Some(fields) = fields {
                    self.bind_strings(field, &fields);
                }
            }
        }
    }
}

fn describe(expr: &Expr) -> String {
    match unwrapped(expr) {
        Expr::Path(_) => name(expr).unwrap(),
        Expr::Field(it) => format!(
            "{}.{}",
            describe(&it.base),
            match &it.member {
                syn::Member::Named(it) => it.to_string(),
                syn::Member::Unnamed(it) => it.index.to_string(),
            }
        ),
        Expr::MethodCall(it) => format!("{}.{}(...)", describe(&it.receiver), it.method),
        _ => "computed expression".into(),
    }
}

impl<'ast> Visit<'ast> for Scanner<'_> {
    fn visit_expr_block(&mut self, expr: &'ast syn::ExprBlock) {
        if production(&expr.attrs) {
            visit::visit_expr_block(self, expr);
        }
    }

    fn visit_expr_assign(&mut self, expr: &'ast syn::ExprAssign) {
        if !production(&expr.attrs) {
            return;
        }
        let old = std::mem::replace(&mut self.writing, true);
        self.visit_expr(&expr.left);
        self.writing = old;
        self.visit_expr(&expr.right);
    }

    fn visit_file(&mut self, file: &'ast syn::File) {
        if production(&file.attrs) {
            visit::visit_file(self, file);
        }
    }

    fn visit_item(&mut self, item: &'ast Item) {
        if production(item_attrs(item)) {
            visit::visit_item(self, item);
        }
    }

    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        if let Some(values) = self.strings(&item.expr) {
            self.bindings.strings.insert(item.ident.to_string(), values);
        }
        let keys = if item.ident == "TECHNIQUE_KEYS" && self.file.contains("/training/") {
            self.strings(&item.expr)
        } else if item.ident == "IRIS_MICRO_STEP_FIELDS" {
            // The boolean separates top-level config fields from advanced-bag keys.
            if let Expr::Array(array) = unwrapped(&item.expr) {
                let mut keys = BTreeSet::new();
                for entry in &array.elems {
                    if let Expr::Tuple(tuple) = unwrapped(entry) {
                        if let Some(Expr::Lit(value)) = tuple.elems.iter().nth(1) {
                            if matches!(&value.lit, Lit::Bool(value) if !value.value) {
                                keys.extend(
                                    self.strings(&tuple.elems[0])
                                        .expect("finite micro-step key"),
                                );
                            }
                        }
                    }
                }
                Some(keys)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(keys) = keys {
            self.coverage
                .domains
                .entry(self.file.into())
                .or_default()
                .extend(keys.clone());
            for key in keys {
                self.coverage.reads.insert(Read {
                    file: self.file.into(),
                    function: item.ident.to_string(),
                    location: "training.config.advanced",
                    key,
                });
            }
        }
        visit::visit_item_const(self, item);
    }

    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.function(&item.sig, &item.block);
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        if production(&item.attrs) {
            self.function(&item.sig, &item.block);
        }
    }

    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        if production(&item.attrs) {
            if let Some(body) = &item.default {
                self.function(&item.sig, body);
            }
        }
    }

    fn visit_block(&mut self, block: &'ast syn::Block) {
        let old = self.bindings.clone();
        visit::visit_block(self, block);
        self.bindings = old;
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        if !production(&local.attrs) {
            return;
        }
        if let Some(init) = &local.init {
            if let (Some(local_name), Expr::Closure(closure)) =
                (ident(&local.pat), unwrapped(&init.expr))
            {
                // Only closures that actually read the advanced bag delegate an option key.
                let old = self.delegated.clone();
                let mut candidates = Vec::new();
                for (index, pat) in closure.inputs.iter().enumerate() {
                    if let Some(key) = ident(pat) {
                        if key == "key" {
                            self.delegated.insert(key.clone());
                            let record = (self.file.into(), self.function.clone(), key);
                            let existed = self.coverage.delegated.remove(&record);
                            candidates.push((index, record, existed));
                        }
                    }
                }
                self.visit_expr(&init.expr);
                for (index, record, existed) in candidates {
                    if self.coverage.delegated.contains(&record) {
                        self.closures.insert(local_name.clone(), index);
                    }
                    if existed {
                        self.coverage.delegated.insert(record);
                    }
                }
                self.delegated = old;
            } else {
                self.visit_expr(&init.expr);
            }
            if let Some(key) = ident(&local.pat) {
                let is_map = self.map(&init.expr);
                let is_descendant = self.descendant(&init.expr);
                let strings = self.strings(&init.expr);
                self.bindings.maps.remove(&key);
                self.bindings.descendants.remove(&key);
                self.bindings.strings.remove(&key);
                if let Some(strings) = strings {
                    self.bindings.strings.insert(key.clone(), strings);
                }
                if is_map {
                    self.bindings.maps.insert(key.clone());
                }
                if is_descendant {
                    self.bindings.descendants.insert(key);
                }
            }
            self.bind_strings(&local.pat, &[&init.expr]);
            if let Some((_, diverge)) = &init.diverge {
                self.visit_expr(diverge);
            }
        }
    }

    fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
        // The legacy FLUX.3 authoring parser passes this finite inline key table into a
        // delegated closure. Keep the dynamic closure read tied to its actual source keys.
        if self.file.ends_with("/image_request/flux3.rs") && self.function == "parse_with_advanced"
        {
            if let Some(keys) = self.strings(&expr.expr) {
                self.coverage
                    .domains
                    .entry(self.file.into())
                    .or_default()
                    .extend(keys.clone());
                for key in keys {
                    self.coverage.reads.insert(Read {
                        file: self.file.into(),
                        function: self.function.clone(),
                        location: "advanced",
                        key,
                    });
                }
            }
        }
        self.visit_expr(&expr.expr);
        let old = self.bindings.clone();
        if self.descendant(&expr.expr) {
            if let Some(key) = ident(&expr.pat) {
                self.bindings.descendants.insert(key);
            }
        }
        if let Expr::Array(array) = unwrapped(&expr.expr) {
            self.bind_strings(&expr.pat, &array.elems.iter().collect::<Vec<_>>());
        } else {
            self.bind_strings(&expr.pat, &[&expr.expr]);
        }
        self.visit_block(&expr.body);
        self.bindings = old;
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        if !production(&expr.attrs) {
            return;
        }
        if matches!(expr.method.to_string().as_str(), "pointer" | "pointer_mut") {
            if let Some(path) = expr.args.first().and_then(|arg| self.strings(arg)) {
                for path in path {
                    let parts: Vec<_> = path.split('/').collect();
                    if let Some(index) = parts.iter().position(|part| *part == "advanced") {
                        let Some(key) = parts.get(index + 1).filter(|key| !key.is_empty()) else {
                            continue;
                        };
                        let location = if self.location == "training.config.advanced"
                            || (index > 1 && parts[index - 1] == "config")
                        {
                            "training.config.advanced"
                        } else {
                            "advanced"
                        };
                        self.coverage.reads.insert(Read {
                            file: self.file.into(),
                            function: self.function.clone(),
                            location,
                            key: key.replace("~1", "/").replace("~0", "~"),
                        });
                    }
                }
            }
        }
        if self.map(&expr.receiver) {
            match expr.method.to_string().as_str() {
                "get" | "get_mut" | "get_key_value" | "contains_key" => {
                    if let Some(key) = expr.args.first() {
                        self.read(key);
                    }
                }
                "remove" | "entry" => {
                    if let Some(key) = expr.args.first() {
                        self.coverage.mutations.insert((
                            self.file.into(),
                            self.function.clone(),
                            describe(key),
                        ));
                    }
                }
                "iter" | "iter_mut" | "keys" | "values" | "values_mut" | "into_iter" | "clone"
                | "to_owned" => {
                    self.coverage.whole_map_reads.insert((
                        self.file.into(),
                        self.function.clone(),
                        format!("whole advanced map: {}", expr.method),
                    ));
                }
                _ => {}
            }
        }
        if self.descendant(&expr.receiver)
            && matches!(
                expr.method.to_string().as_str(),
                "get" | "get_mut" | "contains_key" | "iter" | "iter_mut" | "values" | "keys"
            )
        {
            self.coverage.nested_reads.insert((
                self.file.into(),
                self.function.clone(),
                format!("nested advanced JSON: {}", expr.method),
            ));
        }
        if let Some(Expr::Closure(closure)) = expr.args.first() {
            if matches!(expr.method.to_string().as_str(), "and_then" | "map")
                && (self.map(&expr.receiver) || self.descendant(&expr.receiver))
            {
                let old = self.bindings.clone();
                if let Some(key) = closure.inputs.first().and_then(ident) {
                    if self.map(&expr.receiver) {
                        self.bindings.maps.insert(key);
                    } else {
                        self.bindings.descendants.insert(key);
                    }
                }
                visit::visit_expr_method_call(self, expr);
                self.bindings = old;
                return;
            }
            if let Expr::MethodCall(iter) = unwrapped(&expr.receiver) {
                if matches!(iter.method.to_string().as_str(), "iter" | "into_iter") {
                    let old = self.bindings.clone();
                    if let Some(pat) = closure.inputs.first() {
                        if let Expr::Array(array) = unwrapped(&iter.receiver) {
                            self.bind_strings(pat, &array.elems.iter().collect::<Vec<_>>());
                        } else {
                            self.bind_strings(pat, &[&iter.receiver]);
                        }
                    }
                    visit::visit_expr_method_call(self, expr);
                    self.bindings = old;
                    return;
                }
            }
        }
        visit::visit_expr_method_call(self, expr);
    }

    fn visit_expr_index(&mut self, expr: &'ast syn::ExprIndex) {
        if !production(&expr.attrs) {
            return;
        }
        if !self.writing && self.map(&expr.expr) {
            self.read(&expr.index);
        } else if !self.writing && self.descendant(&expr.expr) {
            self.coverage.nested_reads.insert((
                self.file.into(),
                self.function.clone(),
                "nested advanced JSON: index".into(),
            ));
        }
        visit::visit_expr_index(self, expr);
    }

    fn visit_expr_call(&mut self, expr: &'ast syn::ExprCall) {
        if !production(&expr.attrs) {
            return;
        }
        if let Some(called) = name(&expr.func) {
            let called = self.imports.get(&called).cloned().unwrap_or(called);
            // The typed route binder uses a small generic `field(advanced, key, decode)`
            // parser. Track its literal key just like `advanced.get(key)`; otherwise moving a
            // raw read into that parser would make the inventory silently shrink.
            if called == "field" && expr.args.first().is_some_and(|arg| self.map(arg)) {
                if let Some(key) = expr.args.iter().nth(1) {
                    self.read(key);
                }
            }
            if called == "has_bound_aux_paths" && expr.args.first().is_some_and(|arg| self.map(arg))
            {
                self.coverage.whole_map_reads.insert((
                    self.file.into(),
                    self.function.clone(),
                    "delegated whole advanced map: has_bound_aux_paths".into(),
                ));
            }
            if called == "requested_reference_image_resize_mode" {
                self.coverage.whole_map_reads.insert((
                    self.file.into(),
                    self.function.clone(),
                    "delegated advanced lookup: requested_reference_image_resize_mode".into(),
                ));
            }
            if called == "bound_aux_path" && expr.args.first().is_some_and(|arg| self.map(arg)) {
                if let Some(key) = expr.args.iter().nth(1) {
                    self.coverage.whole_map_reads.insert((
                        self.file.into(),
                        self.function.clone(),
                        format!(
                            "delegated advanced lookup: bound_aux_path({})",
                            describe(key)
                        ),
                    ));
                }
            }
            if let Some(key_index) = nested_helper_key(&called) {
                if expr
                    .args
                    .iter()
                    .nth(1)
                    .and_then(|arg| self.strings(arg))
                    .is_some_and(|keys| keys == BTreeSet::from(["advanced".into()]))
                {
                    if let Some(key) = expr.args.iter().nth(key_index) {
                        self.read(key);
                    }
                }
            }
            if let Some(index) = self
                .closures
                .get(&called)
                .copied()
                .or_else(|| helper_key(&called))
            {
                let qualified_advanced = matches!(unwrapped(&expr.func), Expr::Path(it)
                    if it.path.segments.iter().any(|part| part.ident == "advanced"));
                let request_helper = called.starts_with("resolve_advanced_or_manifest")
                    || called.starts_with("checkpoint_plan_")
                    || called.starts_with("svd_")
                    || called.starts_with("advanced_opt_")
                    || matches!(
                        called.as_str(),
                        "advanced_bounded_number"
                            | "plan_advanced_bool"
                            | "advanced_string"
                            | "advanced_u64"
                    );
                if qualified_advanced
                    || request_helper
                    || self.closures.contains_key(&called)
                    || expr.args.first().is_some_and(|arg| self.map(arg))
                {
                    if let Some(key) = expr.args.iter().nth(index) {
                        self.read(key);
                    }
                }
            }
        }
        visit::visit_expr_call(self, expr);
    }
}

struct Constants(BTreeMap<String, BTreeSet<String>>);
impl<'ast> Visit<'ast> for Constants {
    fn visit_item(&mut self, item: &'ast Item) {
        if production(item_attrs(item)) {
            visit::visit_item(self, item);
        }
    }
    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        fn values(expr: &Expr) -> Option<BTreeSet<String>> {
            match unwrapped(expr) {
                Expr::Lit(it) => match &it.lit {
                    Lit::Str(it) => Some(BTreeSet::from([it.value()])),
                    _ => None,
                },
                Expr::Array(it) => it
                    .elems
                    .iter()
                    .map(values)
                    .collect::<Option<Vec<_>>>()
                    .map(|sets| sets.into_iter().flatten().collect()),
                _ => None,
            }
        }
        if let Some(values) = values(&item.expr) {
            self.0
                .entry(item.ident.to_string())
                .or_default()
                .extend(values);
        }
        visit::visit_item_const(self, item);
    }
    fn visit_expr_struct(&mut self, item: &'ast syn::ExprStruct) {
        if item
            .path
            .segments
            .last()
            .is_some_and(|it| it.ident == "LatentLossSpec")
        {
            for field in &item.fields {
                if let (syn::Member::Named(member), Expr::Lit(value)) =
                    (&field.member, unwrapped(&field.expr))
                {
                    if member.to_string().ends_with("_key") {
                        if let Lit::Str(value) = &value.lit {
                            self.0
                                .entry(format!("<field>:{member}"))
                                .or_default()
                                .insert(value.value());
                        }
                    }
                }
            }
        }
        visit::visit_expr_struct(self, item);
    }
}

fn import_aliases(tree: &syn::UseTree, imports: &mut BTreeMap<String, String>) {
    match tree {
        syn::UseTree::Path(it) => import_aliases(&it.tree, imports),
        syn::UseTree::Rename(it) => {
            imports.insert(it.rename.to_string(), it.ident.to_string());
        }
        syn::UseTree::Group(it) => {
            for item in &it.items {
                import_aliases(item, imports);
            }
        }
        _ => {}
    }
}

fn scan(sources: &[(String, syn::File)]) -> Coverage {
    let mut constants = Constants(BTreeMap::new());
    for (_, source) in sources {
        constants.visit_file(source);
    }
    let mut coverage = Coverage::default();
    for (file, source) in sources {
        let mut scoped_constants = constants.0.clone();
        let mut local_constants = Constants(BTreeMap::new());
        local_constants.visit_file(source);
        scoped_constants.extend(local_constants.0);
        let mut imports = BTreeMap::new();
        for item in &source.items {
            if let Item::Use(it) = item {
                import_aliases(&it.tree, &mut imports);
            }
        }
        Scanner {
            file,
            function: "<module>".into(),
            location: if file.contains("training") {
                "training.config.advanced"
            } else {
                "advanced"
            },
            bindings: Bindings::default(),
            delegated: BTreeSet::new(),
            closures: BTreeMap::new(),
            writing: false,
            constants: &scoped_constants,
            imports,
            coverage: &mut coverage,
        }
        .visit_file(source);
    }
    coverage
}

// Source labels are compared with repository-relative literals on every host.
fn source_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn source_files(root: &Path) -> Vec<(String, syn::File)> {
    fn load(
        root: &Path,
        file: PathBuf,
        seen: &mut BTreeSet<PathBuf>,
        sources: &mut Vec<(String, syn::File)>,
    ) {
        if !seen.insert(file.clone()) {
            return;
        }
        let text = std::fs::read_to_string(&file)
            .unwrap_or_else(|err| panic!("{}: {err}", file.display()));
        let source =
            syn::parse_file(&text).unwrap_or_else(|err| panic!("{}: {err}", file.display()));
        if !production(&source.attrs) {
            return;
        }
        let dir = if matches!(
            file.file_stem().unwrap().to_str().unwrap(),
            "lib" | "main" | "mod"
        ) {
            file.parent().unwrap().to_owned()
        } else {
            file.with_extension("")
        };
        modules(root, &source.items, &dir, &file, seen, sources);
        sources.push((source_path(file.strip_prefix(root).unwrap()), source));
    }
    fn modules(
        root: &Path,
        items: &[Item],
        dir: &Path,
        owner: &Path,
        seen: &mut BTreeSet<PathBuf>,
        sources: &mut Vec<(String, syn::File)>,
    ) {
        for item in items {
            if let Item::Mod(module) = item {
                if !production(&module.attrs) {
                    continue;
                }
                if let Some((_, items)) = &module.content {
                    modules(
                        root,
                        items,
                        &dir.join(module.ident.to_string()),
                        owner,
                        seen,
                        sources,
                    );
                } else {
                    let explicit = module.attrs.iter().find_map(|attr| {
                        if attr.path().is_ident("path") {
                            if let Meta::NameValue(it) = &attr.meta {
                                if let Expr::Lit(it) = &it.value {
                                    if let Lit::Str(it) = &it.lit {
                                        return Some(it.value());
                                    }
                                }
                            }
                        }
                        None
                    });
                    let path = if let Some(path) = explicit {
                        owner.parent().unwrap().join(path)
                    } else {
                        let direct = dir.join(format!("{}.rs", module.ident));
                        if direct.exists() {
                            direct
                        } else {
                            dir.join(module.ident.to_string()).join("mod.rs")
                        }
                    };
                    load(root, path, seen, sources);
                }
            }
        }
    }
    let mut sources = Vec::new();
    let mut seen = BTreeSet::new();
    for source_root in ROOTS {
        let dir = root.join(source_root);
        assert!(dir.exists(), "missing production root {}", dir.display());
        for entry in [dir.join("lib.rs"), dir.join("main.rs")] {
            if entry.exists() {
                load(root, entry, &mut seen, &mut sources);
            }
        }
        if dir.join("bin").exists() {
            for entry in std::fs::read_dir(dir.join("bin")).unwrap() {
                let path = entry.unwrap().path();
                if path.extension().is_some_and(|ext| ext == "rs") {
                    load(root, path, &mut seen, &mut sources);
                } else if path.is_dir() && path.join("main.rs").exists() {
                    load(root, path.join("main.rs"), &mut seen, &mut sources);
                }
            }
        }
    }
    sources
}

fn missing(coverage: &Coverage) -> Vec<String> {
    coverage
        .reads
        .iter()
        .filter(|read| sceneworks_options::by_wire(read.location, &read.key).is_none())
        .map(|read| {
            format!(
                "{}::{}: {}.{}",
                read.file, read.function, read.location, read.key
            )
        })
        .collect()
}

fn fixture(source: &str) -> Coverage {
    scan(&[("fixture.rs".into(), syn::parse_file(source).unwrap())])
}

#[test]
fn valid_fixture_and_test_only_exclusion() {
    let coverage = fixture(
        r#"
        fn production(request: Request) {
            request
                .advanced
                .get("steps");
            #[cfg(test)] { request.advanced.get("__unseen_test_block_key__"); }
            let knobs = &request.advanced;
            knobs.get_mut("guidanceScale");
            let other = knobs.clone();
            other["sampler"];
            advanced::f32(knobs, "schedulerShift", 0.0);
            for key in ["steps", "guidanceScale"] { knobs.get(key); }
        }
        #[cfg(test)] mod tests { fn ignored() { advanced.get("unseen_test_key"); } }
        #[cfg(all(test, feature = "test"))] fn ignored() { advanced.get("also_ignored"); }
        #[cfg(any(test, feature = "native"))] fn included() { advanced.get("steps"); }
    "#,
    );
    assert!(coverage.dynamic.is_empty(), "{:?}", coverage.dynamic);
    assert!(missing(&coverage).is_empty(), "{:?}", missing(&coverage));
    assert_eq!(coverage.reads.len(), 5);
}

#[test]
fn injecting_an_unseen_production_key_fails_coverage() {
    for body in [
        r#"request.advanced.get("__unseen_production_option__");"#,
        r#"let alias = &request.advanced; alias.get_mut("__unseen_production_option__");"#,
        r#"request.advanced["__unseen_production_option__"];"#,
        r#"advanced::f32(&request.advanced, "__unseen_production_option__", 0.0);"#,
        r#"let alias = &request.advanced; let alias = alias.clone(); alias.get("__unseen_production_option__");"#,
        r#"payload.get("advanced").and_then(Value::as_object).and_then(|knobs| knobs.get("__unseen_production_option__"));"#,
        r#"has_nonempty_nested_array(payload, "advanced", "__unseen_production_option__");"#,
        r#"has_nonnull_or_malformed_nested_carrier(payload, "advanced", "__unseen_production_option__");"#,
        r#"payload.pointer("/advanced/__unseen_production_option__");"#,
    ] {
        let coverage = fixture(&format!(
            "fn production() {{ {body} }} #[cfg(test)] mod tests {{}}"
        ));
        assert_eq!(missing(&coverage).len(), 1, "{body}");
    }
    assert_eq!(
        fixture("fn production(key: &str) { advanced.get(key); }")
            .dynamic
            .len(),
        1
    );
    assert_eq!(
        fixture("fn production() { let knobs = &request.advanced; for value in knobs.values() { consume(value); } }")
            .whole_map_reads
            .len(),
        1
    );
    assert_eq!(
        fixture("fn production() { let settings = request.advanced.clone(); persist(settings); }")
            .whole_map_reads
            .len(),
        1
    );
    assert_eq!(
        fixture(
            "fn production() { portable_aux_sources::has_bound_aux_paths(&request.advanced); }"
        )
        .whole_map_reads
        .len(),
        1
    );
    assert_eq!(
        fixture("fn production() { requested_reference_image_resize_mode(advanced); }")
            .whole_map_reads
            .len(),
        1
    );
    assert_eq!(
        fixture("fn production() { portable_aux_sources::bound_aux_path(&request.advanced, \"sft_cover\"); }")
            .whole_map_reads
            .len(),
        1
    );
}

#[test]
fn advanced_index_writes_are_not_execution_reads() {
    let coverage = fixture(
        r#"fn production() {
            body["advanced"]["filmHarness"]["seed"] = json!(7);
            let value = body["advanced"]["steps"];
        }"#,
    );
    assert_eq!(coverage.reads.len(), 1);
    assert!(coverage.reads.iter().any(|read| read.key == "steps"));
}

#[test]
fn nested_advanced_json_remains_visible_after_aliasing() {
    let coverage = fixture(
        r#"fn production() {
            let poses = request.advanced.get("poses").and_then(Value::as_array).unwrap();
            for pose in poses.iter() { pose.get("keypoints"); }
            let phases = request.advanced["phases"].as_array().unwrap();
            phases[0].get("steps");
        }"#,
    );
    assert!(coverage.reads.iter().any(|read| read.key == "poses"));
    assert!(coverage.reads.iter().any(|read| read.key == "phases"));
    assert!(
        coverage.nested_reads.len() >= 2,
        "{:?}",
        coverage.nested_reads
    );
}

#[test]
fn generic_typed_boundary_field_calls_remain_in_the_source_inventory() {
    let coverage = fixture(
        r#"fn from_payload_syntax(advanced: &JsonObject) {
            field(advanced, "poses", count);
            field(advanced, dynamic_key, count);
        }"#,
    );
    assert!(coverage.reads.iter().any(|read| read.key == "poses"));
    assert!(coverage
        .dynamic
        .iter()
        .any(|(_, _, key)| key == "dynamic_key"));
}

#[test]
fn adding_an_unregistered_key_to_a_dynamic_domain_fails_coverage() {
    let path = "crates/sceneworks-core/src/training/body_losses.rs";
    let valid = scan(&[(
        path.into(),
        syn::parse_file(
            r#"
        const TECHNIQUE_KEYS: &[&str] = &["gradientCheckpointing"];
        fn schedule(advanced: &JsonObject) { technique_value(advanced, every_key); }
    "#,
        )
        .unwrap(),
    )]);
    assert!(missing(&valid).is_empty());
    assert!(valid
        .domains
        .get(path)
        .is_some_and(|domain| domain.contains("gradientCheckpointing")));
    let mutated = scan(&[(
        path.into(),
        syn::parse_file(
            r#"
        const TECHNIQUE_KEYS: &[&str] = &["gradientCheckpointing", "__unseen_production_option__"];
        fn schedule(advanced: &JsonObject) { technique_value(advanced, every_key); }
    "#,
        )
        .unwrap(),
    )]);
    assert_eq!(missing(&mutated).len(), 1);
}

#[test]
fn common_constant_names_and_training_context_stay_scoped() {
    let sources = [
        (
            "first.rs".into(),
            syn::parse_file(
                r#"
            const KEYS: &[&str] = &["steps"];
            fn read() { KEYS.iter().any(|key| request.advanced.get(*key).is_some()); }
        "#,
            )
            .unwrap(),
        ),
        (
            "second.rs".into(),
            syn::parse_file(
                r#"
            const KEYS: &[&str] = &["sampler"];
            fn read() { KEYS.iter().any(|key| request.advanced.get(*key).is_some()); }
        "#,
            )
            .unwrap(),
        ),
        (
            "crates/sceneworks-worker/src/iris_curricula.rs".into(),
            syn::parse_file(
                r#"
            fn read() { request.advanced.get("gradientCheckpointing"); }
        "#,
            )
            .unwrap(),
        ),
    ];
    let coverage = scan(&sources);
    assert_eq!(coverage.reads.len(), 3);
    assert!(coverage
        .reads
        .iter()
        .any(|read| read.file == "first.rs" && read.key == "steps"));
    assert!(coverage
        .reads
        .iter()
        .any(|read| read.file == "second.rs" && read.key == "sampler"));
    assert!(coverage
        .reads
        .iter()
        .any(|read| read.file.contains("iris_curricula")
            && read.location == "training.config.advanced"));
}

// Exact dynamic sites whose finite domain lives in the source-owned declarations above.
// These are reported on every run; a new file/function/expression is never silently allowed.
// This is intentionally not a blanket exemption for a training module or function.
fn dynamic_reason(file: &str, function: &str, expression: &str) -> Option<&'static str> {
    match (file, function, expression) {
        ("crates/sceneworks-core/src/training.rs", "inverted_window_message", "max_key") =>
            Some("diagnostic-only max key supplied by technique parsers; their finite TECHNIQUE_KEYS are checked"),
        ("crates/sceneworks-core/src/training/body_losses.rs", "schedule", "every_key" | "min_key" | "max_key" | "loss.weight_key(...)") =>
            Some("BodyLoss formats the three loss prefixes; its explicit TECHNIQUE_KEYS table is checked"),
        ("crates/sceneworks-core/src/training/face_losses.rs", "schedule", "weight_key" | "min_key" | "max_key" | "every_key") =>
            Some("schedule destructures the caller's four-key array; the owning TECHNIQUE_KEYS table is checked"),
        ("crates/sceneworks-core/src/training/iris.rs", "validate_micro_steps", "field") =>
            Some("IRIS_MICRO_STEP_FIELDS selects sampleEvery/lrWarmupSteps from advanced; steps/saveEvery are top-level config fields"),
        ("crates/sceneworks-core/src/image_request/flux3.rs", "parse", "key") =>
            Some("legacy FLUX.3 parser delegates only the six literal noise-strength keys in parse_with_advanced"),
        _ => None,
    }
}

#[test]
fn production_advanced_reads_resolve_to_option_specs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let sources = source_files(root);
    let coverage = scan(&sources);
    let missing = missing(&coverage);
    assert!(
        missing.is_empty(),
        "Production reads lack OptionSpec declarations:\n{}",
        missing.join("\n")
    );
    let mut unresolved = Vec::new();
    for (file, function, expression) in &coverage.dynamic {
        if let Some(reason) = dynamic_reason(file, function, expression) {
            let domain_file = if function == "inverted_window_message" {
                "crates/sceneworks-core/src/training/face_losses.rs"
            } else {
                file.as_str()
            };
            assert!(coverage.domains.get(domain_file).is_some_and(|keys| !keys.is_empty()),
                "Dynamic read {file}::{function} ({expression}) lost its finite source-owned key table");
            eprintln!("Reviewed dynamic read {file}::{function} ({expression}): {reason}");
        } else {
            unresolved.push((file, function, expression));
        }
    }
    for (file, function, expression) in &coverage.delegated {
        eprintln!("Delegated dynamic read {file}::{function} ({expression}): checked at accessor/closure call sites");
    }
    assert!(unresolved.is_empty(), "Dynamic reads require structural coverage or a documented narrow exception:\n{unresolved:#?}");
    assert!(!coverage.reads.is_empty());
}

#[test]
#[ignore = "temporary SW-7038 inventory"]
fn print_production_binding_inventory() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let coverage = scan(&source_files(root));
    let mut by_file = BTreeMap::<&str, usize>::new();
    for read in &coverage.reads {
        *by_file.entry(&read.file).or_default() += 1;
    }
    eprintln!("TOTAL production advanced reads: {}", coverage.reads.len());
    eprintln!("Unresolved dynamic reads: {}", coverage.dynamic.len());
    eprintln!("Delegated dynamic reads: {}", coverage.delegated.len());
    eprintln!("Whole-map reads: {}", coverage.whole_map_reads.len());
    eprintln!(
        "Nested structured JSON reads: {}",
        coverage.nested_reads.len()
    );
    for site in &coverage.nested_reads {
        eprintln!("nested read: {site:?}");
    }
    for site in &coverage.whole_map_reads {
        eprintln!("whole-map read: {site:?}");
    }
    eprintln!("Raw-map mutations: {}", coverage.mutations.len());
    for site in &coverage.mutations {
        eprintln!("raw-map mutation: {site:?}");
    }
    for (file, count) in by_file {
        eprintln!("{count:>3} {file}");
    }
    for read in &coverage.reads {
        if read.file.ends_with("/film_compile.rs")
            || read.file.starts_with("apps/rust-api/src/")
            || read.file.ends_with("/image_jobs.rs")
            || read.file.ends_with("/workflow_parameters.rs")
            || read.file.ends_with("/workflow_resolution.rs")
            || read.file.ends_with("/workflow_share.rs")
        {
            eprintln!(
                "scoped read: {}::{} {}.{}",
                read.file, read.function, read.location, read.key
            );
        }
    }
}

// SW-7038 migration gate. Keep this explicit zero assertion next to the live inventory
// while family execution paths are converted. Only the exact one-time typed binder functions,
// authored persistence, and the historical upgrader may be excluded after site review.
// No path, directory, option family, or dynamic-accessor blanket exemption is permitted.
#[test]
#[ignore = "SW-7038 production binding migration in progress"]
fn production_execution_has_no_raw_advanced_reads() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let coverage = scan(&source_files(root));
    // These exact functions are the one-time typed parser boundary. Their registered keys stay
    // in the inventory and declaration coverage above, but execution consumers must be empty.
    let binding_boundary = |file: &str, function: &str| {
        file == "crates/sceneworks-core/src/resolved_options.rs"
            && matches!(function, "from_payload_syntax" | "from_plan_syntax")
    };
    // These exact clones write authored options into asset receipts. The functions only add
    // resolved telemetry and return the resulting settings object; no route or engine input reads
    // the clone. Keep each site explicit so a new execution clone cannot inherit the exception.
    let authored_receipt_clone = |file: &str, function: &str, expression: &str| {
        expression == "whole advanced map: clone"
            && matches!(
                (file, function),
                (
                    "crates/sceneworks-worker/src/sensenova_jobs.rs",
                    "write_interleaved_document"
                ) | (
                    "crates/sceneworks-worker/src/video_jobs/bernini.rs",
                    "bernini_raw_settings"
                ) | (
                    "crates/sceneworks-worker/src/video_jobs/candle.rs",
                    "candle_video_raw_settings"
                ) | (
                    "crates/sceneworks-worker/src/video_jobs/krea_realtime.rs",
                    "krea_realtime_raw_settings"
                ) | (
                    "crates/sceneworks-worker/src/video_jobs/ltx.rs",
                    "ltx_raw_settings"
                ) | (
                    "crates/sceneworks-worker/src/video_jobs/minimax_h3.rs",
                    "minimax_h3_raw_settings"
                ) | (
                    "crates/sceneworks-worker/src/video_jobs/mochi.rs",
                    "mochi_raw_settings"
                ) | (
                    "crates/sceneworks-worker/src/video_jobs/scail2.rs",
                    "scail2_raw_settings"
                ) | (
                    "crates/sceneworks-worker/src/video_jobs/svd.rs",
                    "svd_raw_settings"
                ) | (
                    "crates/sceneworks-worker/src/video_jobs/vace.rs",
                    "wan_vace_extend_raw_settings"
                ) | (
                    "crates/sceneworks-worker/src/video_jobs/vace.rs",
                    "wan_vace_raw_settings"
                ) | (
                    "crates/sceneworks-worker/src/video_jobs/wan.rs",
                    "wan_raw_settings"
                )
            )
    };
    // Workflow share and gallery rendering read the persisted authored envelope. These exact
    // fields never feed a provider or route decision; replay construction resolves the share
    // anew before execution.
    let authored_workflow_read = |read: &Read| {
        read.location == "advanced"
            && matches!(
                (
                    read.file.as_str(),
                    read.function.as_str(),
                    read.key.as_str()
                ),
                (
                    "crates/sceneworks-core/src/workflow_parameters.rs",
                    "lora_weight_is_fixed",
                    "phases"
                ) | (
                    "crates/sceneworks-core/src/workflow_parameters.rs",
                    "settings_line",
                    "guidanceMethod" | "guidanceScale" | "phases" | "sampler" | "steps"
                ) | (
                    "crates/sceneworks-core/src/workflow_resolution.rs",
                    "build_resolution_report",
                    "mossImageAdvanced"
                ) | (
                    "crates/sceneworks-core/src/workflow_share.rs",
                    "build_share_of_kind",
                    "mossImageAdvanced" | "mossImageEditExecutionReport"
                )
            )
    };
    // The film controller reads its persisted attempt key only to recover a job after a crash.
    // This key is not a generation option and never reaches the film compiler's model plan.
    let persisted_film_attempt_read = |read: &Read| {
        read.file == "apps/rust-api/src/film_harness/transport.rs"
            && read.function == "find_job_by_idempotency_key"
            && read.location == "advanced"
            && read.key == "filmHarness"
    };
    // The image worker reads these authored fields only while formatting a missing-reference
    // error or a sanitized share/PNG metadata envelope after execution has chosen its options.
    let authored_image_metadata_read = |read: &Read| {
        read.file == "crates/sceneworks-worker/src/image_jobs.rs"
            && read.location == "advanced"
            && matches!(
                (read.function.as_str(), read.key.as_str()),
                ("explain_image_reference_error", "imageReference")
                    | ("resolved_steps_for_share", "numInferenceSteps")
                    | ("write_image_asset", "phases")
            )
    };
    // The verified worker dispatch clone already has typed control-weight validity. This
    // get_mut only finds the object to replace its private path with a confined source path;
    // the exact rewrite site is pinned by raw_option_map_mutations_have_reviewed_sites.
    let trusted_control_weights_write = |read: &Read| {
        read.file == "crates/sceneworks-worker/src/external_library_runtime.rs"
            && read.function == "rewrite_loader_locator"
            && read.location == "advanced"
            && read.key == "controlWeights"
    };
    let direct = coverage
        .reads
        .iter()
        .filter(|read| {
            !binding_boundary(&read.file, &read.function)
                && !authored_workflow_read(read)
                && !persisted_film_attempt_read(read)
                && !authored_image_metadata_read(read)
                && !trusted_control_weights_write(read)
        })
        .map(|read| {
            format!(
                "{}::{} {}.{}",
                read.file, read.function, read.location, read.key
            )
        });
    let dynamic = coverage
        .dynamic
        .iter()
        .map(|(file, function, expr)| format!("{file}::{function} dynamic {expr}"));
    let delegated = coverage
        .delegated
        .iter()
        .map(|(file, function, expr)| format!("{file}::{function} delegated {expr}"));
    let whole_map = coverage
        .whole_map_reads
        .iter()
        .filter(|(file, function, expression)| !authored_receipt_clone(file, function, expression))
        .map(|(file, function, expr)| format!("{file}::{function} {expr}"));
    let nested = coverage
        .nested_reads
        .iter()
        .filter(|(file, function, expression)| {
            !(file == "crates/sceneworks-core/src/workflow_parameters.rs"
                && function == "settings_line"
                && expression == "nested advanced JSON: iter")
        })
        .map(|(file, function, expr)| format!("{file}::{function} {expr}"));
    let remaining: Vec<_> = direct
        .chain(dynamic)
        .chain(delegated)
        .chain(whole_map)
        .chain(nested)
        .collect();
    assert!(
        remaining.is_empty(),
        "Production execution still reads raw advanced options ({} sites):\n{}",
        remaining.len(),
        remaining.join("\n")
    );
}

/// Writes into authored or verified dispatch envelopes are a separate review surface from
/// execution reads. Pin the exact write sites so new rewrites require an explicit audit, while
/// the read gate can accurately require zero execution decisions from the raw option map.
#[test]
fn raw_option_map_mutations_have_reviewed_sites() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let coverage = scan(&source_files(root));
    let expected: BTreeSet<(String, String, String)> = [
        (
            "apps/rust-api/src/control_overlays.rs",
            "resolve_control_overlay_selection",
            "computed expression.to_owned(...)",
        ),
        (
            "apps/rust-api/src/generation.rs",
            "apply_recipe_preset_to_image_payload",
            "computed expression.to_owned(...)",
        ),
        (
            "apps/rust-api/src/generation.rs",
            "merge_preset_loras_into_payload",
            "computed expression",
        ),
        (
            "apps/rust-api/src/generation.rs",
            "normalize_ltx25_generation_controls",
            "key",
        ),
        (
            "apps/rust-api/src/iris_training.rs",
            "resolve_iris_run_sources",
            "options_key",
        ),
        (
            "apps/rust-api/src/jobs.rs",
            "apply_training_retry_resume",
            "computed expression",
        ),
        (
            "apps/rust-api/src/jobs.rs",
            "canonicalize_image_detail_dense_tier",
            "computed expression",
        ),
        (
            "apps/rust-api/src/jobs.rs",
            "canonicalize_image_model_payload",
            "key",
        ),
        (
            "apps/rust-api/src/jobs.rs",
            "validate_and_canonicalize_merged_generation_payload",
            "EDIT_OPTIONS_KEY",
        ),
        (
            "apps/rust-api/src/jobs.rs",
            "validate_and_canonicalize_merged_generation_payload",
            "KEY",
        ),
        (
            "apps/rust-api/src/jobs.rs",
            "validate_and_canonicalize_merged_generation_payload",
            "computed expression",
        ),
        (
            "apps/rust-api/src/jobs.rs",
            "validate_and_canonicalize_merged_generation_payload",
            "key",
        ),
        (
            "apps/rust-api/src/training.rs",
            "fresh_flux3_training_duplicate",
            "computed expression",
        ),
        (
            "crates/sceneworks-core/src/moss_image.rs",
            "scrub_reports",
            "EDIT_REPORT_KEY",
        ),
        (
            "crates/sceneworks-core/src/moss_image.rs",
            "scrub_reports",
            "REPORT_KEY",
        ),
        (
            "crates/sceneworks-core/src/moss_image.rs",
            "scrub_reports",
            "computed expression",
        ),
        (
            "crates/sceneworks-core/src/video_request.rs",
            "stamp_replayed_reference_image_resize_mode",
            "REFERENCE_IMAGE_RESIZE_MODE_KEY",
        ),
        (
            "crates/sceneworks-worker/src/external_library_runtime.rs",
            "begin_with_binding_cancelable",
            "computed expression",
        ),
        (
            "crates/sceneworks-worker/src/external_library_runtime.rs",
            "none",
            "computed expression",
        ),
        (
            "crates/sceneworks-worker/src/external_library_runtime.rs",
            "rewrite_loader_locator",
            "computed expression.to_owned(...)",
        ),
        (
            "crates/sceneworks-worker/src/iris_curricula.rs",
            "iris_stage_presets",
            "computed expression",
        ),
        (
            "crates/sceneworks-worker/src/iris_training.rs",
            "activation_checkpointing_hint",
            "IRIS_OPTIONS_KEY.to_owned(...)",
        ),
        (
            "crates/sceneworks-worker/src/sensenova_jobs.rs",
            "write_interleaved_document",
            "PREFIX_KV_CACHE_RESULT_KEY",
        ),
    ]
    .into_iter()
    .map(|(file, function, expression)| (file.into(), function.into(), expression.into()))
    .collect();
    assert_eq!(coverage.mutations, expected);
}

// A small lexical guard keeps Rust-only CI independent of Node/npm. It recognizes
// literal member accesses, optional chaining, same-file string constants, and simple
// local aliases. It does not claim JavaScript name resolution or macro expansion.
// Reassignment/destructuring aliases and imported computed keys require a guard update;
// regex literals are not parsed (an apparent access in one fails conservatively).
#[derive(Clone, Debug, PartialEq, Eq)]
enum JsToken {
    Word(String),
    Text(String),
    Punct(char),
    Optional,
}

fn js_tokens(source: &str) -> Vec<JsToken> {
    fn quoted(chars: &[char], at: &mut usize, quote: char) -> String {
        *at += 1;
        let mut value = String::new();
        while *at < chars.len() {
            let ch = chars[*at];
            *at += 1;
            if ch == quote {
                break;
            }
            if ch == '\\' && *at < chars.len() {
                let escaped = chars[*at];
                *at += 1;
                value.push(match escaped {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    other => other,
                });
            } else {
                value.push(ch);
            }
        }
        value
    }
    fn template(chars: &[char], at: &mut usize, out: &mut Vec<JsToken>) {
        *at += 1;
        while *at < chars.len() {
            match chars[*at] {
                '\\' => *at = (*at + 2).min(chars.len()),
                '`' => {
                    *at += 1;
                    break;
                }
                '$' if chars.get(*at + 1) == Some(&'{') => {
                    *at += 2;
                    code(chars, at, out, true);
                }
                _ => *at += 1,
            }
        }
    }
    fn code(chars: &[char], at: &mut usize, out: &mut Vec<JsToken>, interpolation: bool) {
        let mut braces = 0;
        while *at < chars.len() {
            let ch = chars[*at];
            if ch.is_whitespace() {
                *at += 1;
                continue;
            }
            if ch == '/' && chars.get(*at + 1) == Some(&'/') {
                *at += 2;
                while *at < chars.len() && chars[*at] != '\n' {
                    *at += 1;
                }
                continue;
            }
            if ch == '/' && chars.get(*at + 1) == Some(&'*') {
                *at += 2;
                while *at + 1 < chars.len() && !(chars[*at] == '*' && chars[*at + 1] == '/') {
                    *at += 1;
                }
                *at = (*at + 2).min(chars.len());
                continue;
            }
            if ch == '\'' || ch == '"' {
                out.push(JsToken::Text(quoted(chars, at, ch)));
                continue;
            }
            if ch == '`' {
                template(chars, at, out);
                continue;
            }
            if ch == '?' && chars.get(*at + 1) == Some(&'.') {
                out.push(JsToken::Optional);
                *at += 2;
                continue;
            }
            if ch.is_alphabetic() || ch == '_' || ch == '$' {
                let start = *at;
                *at += 1;
                while *at < chars.len()
                    && (chars[*at].is_alphanumeric() || matches!(chars[*at], '_' | '$'))
                {
                    *at += 1;
                }
                out.push(JsToken::Word(chars[start..*at].iter().collect()));
                continue;
            }
            if interpolation && ch == '}' && braces == 0 {
                *at += 1;
                return;
            }
            if ch == '{' {
                braces += 1;
            }
            if ch == '}' {
                braces -= 1;
            }
            out.push(JsToken::Punct(ch));
            *at += 1;
        }
    }
    let mut tokens = Vec::new();
    code(
        &source.chars().collect::<Vec<_>>(),
        &mut 0,
        &mut tokens,
        false,
    );
    tokens
}

fn js_word(token: Option<&JsToken>) -> Option<&str> {
    match token {
        Some(JsToken::Word(word)) => Some(word),
        _ => None,
    }
}

fn js_coverage(file: &str, source: &str) -> Coverage {
    let tokens = js_tokens(source);
    let mut constants = BTreeMap::new();
    for part in tokens.windows(4) {
        if matches!(js_word(part.first()), Some("const" | "let")) && part[2] == JsToken::Punct('=')
        {
            if let (JsToken::Word(name), JsToken::Text(value)) = (&part[1], &part[3]) {
                constants.insert(name.clone(), value.clone());
            }
        }
    }
    let location = if file.to_ascii_lowercase().contains("training") {
        "training.config.advanced"
    } else {
        "advanced"
    };
    let mut coverage = Coverage::default();
    let mut aliases = BTreeSet::from(["advanced".to_owned()]);
    let mut scopes = Vec::new();
    let mut function = "<javascript>".to_owned();
    let mut pending_function = None;
    for (index, token) in tokens.iter().enumerate() {
        if js_word(Some(token)) == Some("function") {
            pending_function = js_word(tokens.get(index + 1)).map(str::to_owned);
        }
        match token {
            JsToken::Punct('{') => {
                scopes.push((aliases.clone(), function.clone()));
                if let Some(name) = pending_function.take() {
                    function = name;
                }
            }
            JsToken::Punct('}') => {
                if let Some(outer) = scopes.pop() {
                    (aliases, function) = outer;
                }
            }
            _ => {}
        }
        if matches!(js_word(Some(token)), Some("const" | "let" | "var"))
            && tokens.get(index + 2) == Some(&JsToken::Punct('='))
        {
            if let Some(binding) = js_word(tokens.get(index + 1)) {
                let start = index + 3;
                let mut end = start;
                // A simple alias ends at a separator or nullish/default operator; calls and
                // arbitrary expressions are deliberately not inferred as map aliases.
                while end < tokens.len()
                    && matches!(
                        &tokens[end],
                        JsToken::Word(_) | JsToken::Punct('.') | JsToken::Optional
                    )
                {
                    end += 1;
                }
                let simple = end > start
                    && end - start <= 9
                    && !matches!(tokens.get(end), Some(JsToken::Punct('(')));
                let last = end.checked_sub(1).and_then(|at| js_word(tokens.get(at)));
                let map_alias = simple && last.is_some_and(|name| aliases.contains(name));
                if binding != "advanced" {
                    aliases.remove(binding);
                }
                if map_alias {
                    aliases.insert(binding.into());
                }
            }
        }
        let Some(receiver) = js_word(Some(token)) else {
            continue;
        };
        if !aliases.contains(receiver) {
            continue;
        }
        let mut next = index + 1;
        if matches!(
            tokens.get(next),
            Some(JsToken::Optional | JsToken::Punct('.'))
        ) {
            next += 1;
        } else if tokens.get(next) != Some(&JsToken::Punct('[')) {
            continue;
        }
        let key = if tokens.get(next) == Some(&JsToken::Punct('[')) {
            match (tokens.get(next + 1), tokens.get(next + 2)) {
                (Some(JsToken::Text(key)), Some(JsToken::Punct(']'))) => Some(key.clone()),
                (Some(JsToken::Word(key)), Some(JsToken::Punct(']'))) => {
                    constants.get(key).cloned()
                }
                _ => None,
            }
        } else {
            js_word(tokens.get(next)).map(str::to_owned)
        };
        if let Some(key) = key {
            // This hook's return value is UI state; these properties never belong to the
            // request's advanced map. Match exact path/key rather than excluding the file.
            if file == "apps/web/src/screens/characterPanels.jsx"
                && matches!(
                    key.as_str(),
                    "buildAdvanced" | "seedValue" | "supportsNegativePrompt"
                )
            {
                continue;
            }
            coverage.reads.insert(Read {
                file: file.into(),
                function: function.clone(),
                location,
                key,
            });
        } else {
            let expression = tokens[next..]
                .iter()
                .take_while(|it| **it != JsToken::Punct(']'))
                .map(|it| match it {
                    JsToken::Word(word) | JsToken::Text(word) => word.clone(),
                    JsToken::Punct(ch) => ch.to_string(),
                    JsToken::Optional => "?.".into(),
                })
                .collect::<String>();
            coverage
                .dynamic
                .insert((file.into(), function.clone(), expression));
        }
    }
    if file == "apps/web/src/training/trainingConfig.js" {
        // The computed read concatenates source-owned loss prefixes with auxLossKnobs.
        // Derive both inputs instead of approving the expression alone.
        let prefixes: BTreeSet<_> = tokens
            .windows(3)
            .filter_map(|part| match part {
                [JsToken::Word(name), JsToken::Punct(':'), JsToken::Text(value)]
                    if name == "prefix" =>
                {
                    Some(value.clone())
                }
                [JsToken::Word(name), JsToken::Punct('('), JsToken::Text(value)]
                    if name == "faceLossSpec" =>
                {
                    Some(value.clone())
                }
                _ => None,
            })
            .collect();
        let start = tokens.windows(4).position(|part| {
            js_word(part.first()) == Some("const")
                && js_word(part.get(1)) == Some("auxLossKnobs")
                && part[2] == JsToken::Punct('=')
                && part[3] == JsToken::Punct('[')
        });
        if let Some(start) = start {
            let knobs: Vec<_> = tokens[start + 4..]
                .iter()
                .take_while(|it| **it != JsToken::Punct(']'))
                .filter_map(|it| match it {
                    JsToken::Text(value) => Some(value),
                    _ => None,
                })
                .collect();
            assert!(
                !prefixes.is_empty() && !knobs.is_empty(),
                "training computed selector lost its finite domain"
            );
            for prefix in prefixes {
                for knob in &knobs {
                    coverage.reads.insert(Read {
                        file: file.into(),
                        function: "auxLossesDraft".into(),
                        location,
                        key: format!("{prefix}{knob}"),
                    });
                }
            }
            coverage.domains.insert(
                file.into(),
                coverage
                    .reads
                    .iter()
                    .filter(|read| read.function == "auxLossesDraft")
                    .map(|read| read.key.clone())
                    .collect(),
            );
        }
    }
    coverage
}

#[test]
fn javascript_mutations_require_registered_options() {
    for read in [
        r#"request.advanced.__unseen_production_option__"#,
        r#"request?.advanced?.__unseen_production_option__"#,
        r#"request.advanced['__unseen_production_option__']"#,
        r#"request.advanced?.["__unseen_production_option__"]"#,
        r#"const knobs = request.advanced; knobs.__unseen_production_option__"#,
        r#"const s = `${request.advanced.__unseen_production_option__}`"#,
    ] {
        assert_eq!(missing(&js_coverage("fixture.js", read)).len(), 1, "{read}");
    }
    let valid = js_coverage(
        "fixture.js",
        r#"
        // request.advanced.unseen
        /* advanced['unseen'] */
        const prose = "advanced.unseen";
        const template = `advanced.unseen`;
        const KEY = 'sampler';
        request.advanced.steps;
        const knobs = request?.advanced;
        knobs?.[KEY];
    "#,
    );
    assert_eq!(valid.reads.len(), 2);
    assert!(missing(&valid).is_empty());
    assert_eq!(js_coverage("fixture.js", "advanced[key]").dynamic.len(), 1);
    let mutated = js_coverage(
        "apps/web/src/training/trainingConfig.js",
        r#"
        const losses = [{prefix: "__unseen_production_option__"}];
        const auxLossKnobs = ["Weight"];
        function auxLossesDraft(advanced) { return advanced?.[prefix + knob]; }
    "#,
    );
    assert_eq!(missing(&mutated).len(), 1);
}

#[test]
fn windows_source_paths_preserve_scoped_rules_and_reject_unknown_keys() {
    let rust_file = source_path(Path::new(
        r"crates\sceneworks-core\src\training\body_losses.rs",
    ));
    assert_eq!(
        rust_file,
        "crates/sceneworks-core/src/training/body_losses.rs"
    );
    let rust = scan(&[(
        rust_file.clone(),
        syn::parse_file(
            r#"
            const TECHNIQUE_KEYS: &[&str] = &["bodyProportionEvery"];
            fn schedule(advanced: &JsonObject) { technique_value(advanced, every_key); }
        "#,
        )
        .unwrap(),
    )]);
    assert!(missing(&rust).is_empty());
    assert!(rust.domains[&rust_file].contains("bodyProportionEvery"));
    assert!(rust.dynamic.iter().all(|(file, function, expression)| {
        dynamic_reason(file, function, expression).is_some()
    }));
    assert!(!rust.dynamic.is_empty());
    assert!(dynamic_reason(&rust_file, "schedule", "unseen_selector").is_none());

    let character_file = source_path(Path::new(r"apps\web\src\screens\characterPanels.jsx"));
    let character = js_coverage(
        &character_file,
        "advanced.buildAdvanced(); advanced.seedValue; advanced.supportsNegativePrompt;",
    );
    assert!(character.reads.is_empty());
    let training_file = source_path(Path::new(r"apps\web\src\training\trainingConfig.js"));
    let training = js_coverage(
        &training_file,
        r#"
        const losses = [{ prefix: "bodyProportion" }];
        const auxLossKnobs = ["Weight"];
        function auxLossesDraft(advanced) { return advanced?.[prefix + knob]; }
    "#,
    );
    assert!(missing(&training).is_empty());
    assert!(training.domains[&training_file].contains("bodyProportionWeight"));
    assert!(training.dynamic.contains(&(
        "apps/web/src/training/trainingConfig.js".into(),
        "auxLossesDraft".into(),
        "[prefix+knob".into(),
    )));
    let workflow_file = source_path(Path::new(r"apps\web\src\workflowShare.js"));
    let workflow = js_coverage(
        &workflow_file,
        "function workflowSettingRows(share) { const settings = share?.advanced; return settings[key]; }",
    );
    assert!(workflow.dynamic.contains(&(
        "apps/web/src/workflowShare.js".into(),
        "workflowSettingRows".into(),
        "[key".into(),
    )));
    for file in [&character_file, &training_file, &workflow_file] {
        assert_eq!(
            missing(&js_coverage(file, "advanced.__unseen_production_option__")).len(),
            1,
            "{file}"
        );
    }
}

fn javascript_source_files(dir: &Path, sources: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            if !matches!(
                path.file_name().and_then(|it| it.to_str()),
                Some("testUtils" | "__tests__")
            ) {
                javascript_source_files(&path, sources);
            }
        } else {
            let name = path.file_name().unwrap().to_string_lossy();
            if matches!(
                path.extension().and_then(|it| it.to_str()),
                Some("js" | "jsx")
            ) && !name.contains(".test.")
                && !name.contains(".spec.")
            {
                sources.push(path);
            }
        }
    }
}

#[test]
fn javascript_advanced_reads_resolve_to_option_specs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let mut sources = Vec::new();
    javascript_source_files(&root.join("apps/web/src"), &mut sources);
    let mut absent = Vec::new();
    let mut dynamic = Vec::new();
    for path in sources {
        let file = source_path(path.strip_prefix(root).unwrap());
        let coverage = js_coverage(&file, &std::fs::read_to_string(&path).unwrap());
        absent.extend(missing(&coverage));
        for (file, function, expression) in coverage.dynamic {
            match (file.as_str(), function.as_str(), expression.as_str()) {
                ("apps/web/src/training/trainingConfig.js", "auxLossesDraft", "[prefix+knob") => {
                    assert!(coverage
                        .domains
                        .get(&file)
                        .is_some_and(|domain| !domain.is_empty()));
                    eprintln!("Reviewed JavaScript computed read {file}::{function}: prefix × knob source domain checked against registry");
                }
                ("apps/web/src/workflowShare.js", "workflowSettingRows", "[key") => {
                    // This inspector deliberately displays arbitrary imported keys, including
                    // unknown names, as unrestored. It never applies them to a generation request.
                    eprintln!("Reviewed JavaScript computed read {file}::{function}: Object.keys(settings) enumerates arbitrary imported settings for display");
                }
                _ => dynamic.push((file, function, expression)),
            }
        }
    }
    assert!(
        absent.is_empty(),
        "JavaScript accesses lack OptionSpec declarations:\n{}",
        absent.join("\n")
    );
    assert!(
        dynamic.is_empty(),
        "JavaScript computed accesses need explicit finite coverage:\n{dynamic:#?}"
    );
}

#[test]
#[ignore = "temporary SW-7038 web authoring/read inventory"]
fn print_javascript_binding_inventory() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let mut sources = Vec::new();
    javascript_source_files(&root.join("apps/web/src"), &mut sources);
    let mut by_file = BTreeMap::new();
    let mut total = 0;
    let mut dynamic = 0;
    for path in sources {
        let file = source_path(path.strip_prefix(root).unwrap());
        let coverage = js_coverage(&file, &std::fs::read_to_string(path).unwrap());
        if !coverage.reads.is_empty() || !coverage.dynamic.is_empty() {
            total += coverage.reads.len();
            dynamic += coverage.dynamic.len();
            by_file.insert(file, (coverage.reads.len(), coverage.dynamic.len()));
        }
    }
    eprintln!("JavaScript advanced property sites: {total}, computed: {dynamic}");
    for (file, (direct, computed)) in by_file {
        eprintln!("{direct:>3} direct {computed:>2} computed {file}");
    }
}

#[test]
fn timeline_context_reads_require_nested_declarations() {
    // This consumer names its nested advanced.timelineContext map `context`.
    // Reuse the JS lexer/reader guard with that binding isolated from the outer map.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let source = std::fs::read_to_string(root.join("apps/web/src/timelineGeneration.js")).unwrap();
    assert!(source.contains("const context = payload.advanced?.timelineContext ?? {}"));
    let nested_reads = |source: &str| {
        let source = source
            .replace("payload.advanced", "payload.settings")
            .replace("context.", "advanced.")
            .replace("context?.", "advanced?.")
            .replace("context[", "advanced[");
        let mut coverage = js_coverage("apps/web/src/timelineGeneration.js", &source);
        coverage.reads = coverage
            .reads
            .into_iter()
            .map(|mut read| {
                read.location = "advanced.timelineContext";
                read
            })
            .collect();
        coverage
    };
    let coverage = nested_reads(&source);
    assert!(!coverage.reads.is_empty());
    assert!(coverage.dynamic.is_empty(), "{:?}", coverage.dynamic);
    assert!(missing(&coverage).is_empty(), "{:?}", missing(&coverage));
    for expression in [
        "context.__unknown__",
        "context?.__unknown__",
        "context['__unknown__']",
    ] {
        assert_eq!(
            missing(&nested_reads(&format!("{source}\n{expression};"))).len(),
            1
        );
    }
}
