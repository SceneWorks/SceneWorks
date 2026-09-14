import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  RECORDS_PATH,
  STAGED_RESIDENCY_ENGAGED_IN_SAME_REQUEST,
  additionalPrerequisiteEdges,
  cfgProductionState,
  stagedResidencySupport,
  stripTestOnlyItems,
} from "./rung4-contract-prerequisites.mjs";

const records = async () =>
  JSON.parse(await readFile(new URL(`../${RECORDS_PATH}`, import.meta.url), "utf8"));

/** `[{strategy, conditional}]` -> `["Strategy"]` / `["Strategy?"]`, so a case reads in one line. */
const spell = (edges) =>
  edges.map(({ strategy, conditional }) => (conditional ? `${strategy}?` : strategy));

const EDGE = `MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            }`;

// The four constructions present at the pinned revision, each with the file it was taken from so a
// reader can check the fixture against the real thing rather than against this file's idea of it.
const SHAPES = [
  [
    "Vec::new — mlx-gen-z-image/src/memory_strategy.rs",
    "additional_prerequisites: Vec::new(),",
    [],
  ],
  [
    "array + map — candle-gen-flux/src/memory_strategy.rs",
    `additional_prerequisites: [
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency,
    ]
    .into_iter()
    .map(|strategy| {
        (
            strategy,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        )
    })
    .collect(),`,
    ["BoundedDecode", "BoundedAttention", "BoundedTransformerResidency"],
  ],
  [
    "push — mlx-gen-anima/src/memory_strategy.rs",
    `contract.additional_prerequisites.push((
        MemoryStrategy::BoundedTransformerResidency,
        MemoryStrategyPrerequisite::Rung {
            rung: MemoryStrategy::StagedResidency,
            scope: MemoryPrerequisiteScope::EngagedInSameRequest,
        },
    ));`,
    ["BoundedTransformerResidency"],
  ],
  [
    // `then_some` is conditional BY CONSTRUCTION — the provider appends the edge only when the load
    // is streamable — so the derived edge says so.
    "then_some — mlx-gen-krea/src/memory_strategy.rs",
    `additional_prerequisites: streamable_transformer
        .then_some((
            MemoryStrategy::BoundedTransformerResidency,
            mlx_gen::gen_core::MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: mlx_gen::gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ))
        .into_iter()
        .collect(),`,
    ["BoundedTransformerResidency?"],
  ],
  [
    // SHAPE 5 (sc-20799, inference ebcdc7da). One array of candidates, filtered down to the members
    // this build can execute. The filtered-out member is conditional; its siblings are not — a
    // distinction a shape that read the array alone would lose, and reading the filter as "keeps
    // everything" would present a streamable-only edge as unconditional.
    "filter + map — candle-gen-sana/src/memory_strategy.rs",
    `contract.additional_prerequisites = [
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency,
    ]
    .into_iter()
    .filter(|strategy| streamable || *strategy != MemoryStrategy::BoundedTransformerResidency)
    .map(|strategy| {
        (
            strategy,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        )
    })
    .collect();`,
    ["BoundedDecode", "BoundedAttention", "BoundedTransformerResidency?"],
  ],
];

test("every construction shape present at the pinned revision is read, individually", () => {
  for (const [name, source, expected] of SHAPES) {
    assert.deepEqual(spell(additionalPrerequisiteEdges(source, name)), expected, name);
  }
});

test("an unrecognised construction or edge fails closed rather than reading as no edges", () => {
  // The whole soundness argument for reading Rust with regular expressions. Each mutation is its own
  // assertion, because a single "some unknown shape throws" case would prove the set, not the members.
  assert.throws(
    () => additionalPrerequisiteEdges("additional_prerequisites: build_them(spec),", "call"),
    /unrecognised `additional_prerequisites` construction/,
  );
  assert.throws(
    () =>
      additionalPrerequisiteEdges(
        "additional_prerequisites: spec.edges().cloned().collect(),",
        "iterator",
      ),
    /unrecognised `additional_prerequisites` construction/,
  );
  assert.throws(
    () =>
      additionalPrerequisiteEdges(
        `contract.additional_prerequisites.push((
             MemoryStrategy::BoundedTransformerResidency,
             MemoryStrategyPrerequisite::LoadShape(LoadShape::DeferredMaterialization),
         ));`,
        "load-shape edge",
      ),
    /unrecognised `additional_prerequisites` construction/,
  );
  assert.throws(
    () =>
      additionalPrerequisiteEdges(
        `contract.additional_prerequisites.push((
             MemoryStrategy::BoundedTransformerResidency,
             MemoryStrategyPrerequisite::Rung {
                 rung: MemoryStrategy::BoundedDecode,
                 scope: MemoryPrerequisiteScope::EngagedInSameRequest,
             },
         ));`,
        "other rung",
      ),
    /unrecognised prerequisite edge/,
  );
  // Shape 5's filter is recognised for exactly ONE predicate. A predicate the extractor cannot read
  // must throw, not fall through to "keeps every member" — that would report a conditional edge as
  // unconditional, which is the same fail-open direction as reading a new construction as no edges.
  assert.throws(
    () =>
      additionalPrerequisiteEdges(
        `contract.additional_prerequisites = [
             MemoryStrategy::BoundedDecode,
             MemoryStrategy::BoundedTransformerResidency,
         ]
         .into_iter()
         .filter(|strategy| spec.supports(*strategy))
         .map(|strategy| {
             (
                 strategy,
                 MemoryStrategyPrerequisite::Rung {
                     rung: MemoryStrategy::StagedResidency,
                     scope: MemoryPrerequisiteScope::EngagedInSameRequest,
                 },
             )
         })
         .collect();`,
        "unreadable filter predicate",
      ),
    /unrecognised `additional_prerequisites` construction/,
  );
  // And a predicate that excludes a strategy the array never contained is not a shape the extractor
  // can attribute, so it fails closed too rather than guessing which member the filter governs.
  assert.throws(
    () =>
      additionalPrerequisiteEdges(
        `contract.additional_prerequisites = [
             MemoryStrategy::BoundedDecode,
         ]
         .into_iter()
         .filter(|strategy| streamable || *strategy != MemoryStrategy::BoundedTransformerResidency)
         .map(|strategy| {
             (
                 strategy,
                 MemoryStrategyPrerequisite::Rung {
                     rung: MemoryStrategy::StagedResidency,
                     scope: MemoryPrerequisiteScope::EngagedInSameRequest,
                 },
             )
         })
         .collect();`,
        "filter excludes an absent member",
      ),
    /which the mapped array does not contain/,
  );
  // A read is not a construction, and must contribute nothing rather than throw — three provider
  // test modules assert `is_empty()` on the vector.
  assert.deepEqual(
    spell(additionalPrerequisiteEdges("assert!(contract.additional_prerequisites.is_empty());", "read")),
    [],
  );
});

// ---------------------------------------------------------------------------
// The two SILENT-ZERO reproductions. Both derived `[]` at EXIT 0 from source that pushes the rung-1
// edge, and CI could not see either, because `--write` and `--check` share this extractor: a bug
// here writes a wrong record and then re-derives the same wrong record and agrees with itself. Each
// case therefore asserts the DERIVED EDGE, not merely that some output changed, and each carries the
// mutation that reds it.
// ---------------------------------------------------------------------------

test("a `#[cfg(not(test))]` module ships, and its edge is not stripped away as a test fixture", () => {
  // The attribute regex `/#\[cfg\((?:[^)]*\()?[^)]*\btest\b[^\]]*\]/` MATCHED `#[cfg(not(test))]`,
  // so this module was cut as though it were `mod tests`. The idiom is live at
  // `candle-gen-krea/src/transformer/block.rs:142` and `mlx-gen-sana/src/transformer.rs:115`, and
  // `candle-gen-catalog/src/lib.rs:227` documents it as one that "genuinely ships".
  //
  // MUTATION: restore the substring regex in `stripTestOnlyItems` and this derives `[]`.
  const source = `
#[cfg(not(test))]
mod production {
    pub fn contract() -> MemoryProviderContract {
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            ${EDGE},
        ));
    }
}
`;
  assert.deepEqual(spell(additionalPrerequisiteEdges(stripTestOnlyItems(source), "not(test)")), [
    "BoundedTransformerResidency",
  ]);
  // The classification the fix turns on, asserted directly so a reader can see WHY the case above
  // passes. `not(test)` is in every production build; `test` is in none; `any(feature, test)` is in
  // some, which is what `conditional` records.
  assert.equal(cfgProductionState("not(test)"), true);
  assert.equal(cfgProductionState("test"), false);
  assert.equal(cfgProductionState('any(feature = "cuda", test)'), "conditional");
  assert.equal(cfgProductionState("all(test, unix)"), false);
  assert.equal(cfgProductionState('not(any(feature = "cuda", test))'), "conditional");
  // A predicate the classifier cannot read is an error, not a guess in either direction.
  assert.throws(() => cfgProductionState("weird | thing"), /cannot parse cfg predicate/);
  assert.throws(() => cfgProductionState("some_combinator(test)"), /unrecognised cfg combinator/);
});

test("a block-comment opener inside a string literal cannot delete a construction", () => {
  // Comment stripping ran over RAW source, so a slash-star inside one string literal and a
  // star-slash inside a later one made `/\*[\s\S]*?\*\//` swallow everything between them —
  // including the push. Exit 0, `[]` derived, nothing to fail closed on.
  //
  // MUTATION (verified, sc-19542 review N2): make `blankLiterals` return `source` unchanged and the
  // SECOND leg below reds. The first leg does not — see why, because it matters.
  //
  // The recipe previously written here — "drop the `blankLiterals` call from `normalizeRust`" — was
  // measurably FALSE: run, it reddened 0 of 10 tests, because `normalizeRust` and its only caller
  // `canonical` had no call sites and graded nothing. Both are now deleted.
  //
  // Removing them also removed the last block-comment regex from the live path, which retired the
  // first leg's mutant along with it: with nothing downstream deleting comment spans, a slash-star
  // inside a literal can no longer swallow a construction, and the leg passes whatever
  // `blankLiterals` does. It is kept as a cheap fossil of the original defect, and labelled as one
  // rather than left to read as a live guard.
  //
  // What blankLiterals actually protects in the live path is every SPAN-BOUNDED read over the
  // scan — brace matching, `#[cfg]` item extents, function extents. The second leg exercises that
  // directly, so this test grades the code it runs on rather than the code it was written against.
  const glob = ["shards/", "*", ".safetensors"].join("");
  const ratio = ["peak", "*", "/base ratio"].join("");
  const source = `
const SHARD_GLOB: &str = "${glob}";

pub fn contract() -> MemoryProviderContract {
    contract.additional_prerequisites.push((
        MemoryStrategy::BoundedTransformerResidency,
        ${EDGE},
    ));
}

const RATIO_DOC: &str = "${ratio}";
`;
  assert.deepEqual(spell(additionalPrerequisiteEdges(source, "literal trap")), [
    "BoundedTransformerResidency",
  ]);
  // Non-vacuity: the trap really is armed. Remove the two constants and the same body derives the
  // same edge, so the case above is graded by the literal blanking and not by the push being found
  // for some other reason.
  const withoutTrap = source
    .replace(`const SHARD_GLOB: &str = "${glob}";`, "")
    .replace(`const RATIO_DOC: &str = "${ratio}";`, "");
  assert.deepEqual(spell(additionalPrerequisiteEdges(withoutTrap, "no trap")), [
    "BoundedTransformerResidency",
  ]);
  assert.ok(source.includes("/*") && source.includes("*/"), "the fixture must contain the trap");

  // The LIVE leg. A `#[cfg(test)]` spelled inside an ordinary string constant is not an attribute,
  // but `cfgAttributedItems` reads spans out of the scan and cannot tell the difference unless the
  // literal has been blanked first. Unblanked, the extractor decides a test-only item survived
  // stripping and throws — so a provider would derive nothing at all because one of its doc strings
  // quoted an attribute. Blanked, the constant is inert and the real construction is read.
  const attributeInAString = `
const CFG_DOC: &str = "#[cfg(test)] modules are stripped before extraction";

pub fn contract() -> MemoryProviderContract {
    contract.additional_prerequisites.push((
        MemoryStrategy::BoundedTransformerResidency,
        ${EDGE},
    ));
}
`;
  assert.deepEqual(
    spell(additionalPrerequisiteEdges(attributeInAString, "attribute in a literal")),
    ["BoundedTransformerResidency"],
  );
  // Non-vacuity, the same shape the first leg uses: strip the constant and the identical body
  // derives the identical edge, so the assertion above is graded by the blanking and not by the
  // push being found for some unrelated reason.
  const withoutAttribute = attributeInAString.replace(
    'const CFG_DOC: &str = "#[cfg(test)] modules are stripped before extraction";',
    "",
  );
  assert.deepEqual(spell(additionalPrerequisiteEdges(withoutAttribute, "no attribute")), [
    "BoundedTransformerResidency",
  ]);
  assert.ok(
    attributeInAString.includes("#[cfg(test)]"),
    "the fixture must contain the quoted attribute",
  );
});

test("`=` replaces the prerequisite vector and `.push` adds to it, scoped to one function", () => {
  // `contract.additional_prerequisites = [...]` is an ASSIGNMENT: it drops whatever came before.
  // Unioning it like the additive forms over-counts, and over-counting keeps the rung-1 proxy on
  // lanes the contract exempts — the defect sc-19542 exists to remove.
  //
  // MUTATION: make `resets` always false and the first case derives both strategies.
  const reset = `fn contract() -> MemoryProviderContract {
      contract.additional_prerequisites.push((MemoryStrategy::BoundedTransformerResidency, ${EDGE}));
      contract.additional_prerequisites = [MemoryStrategy::BoundedDecode]
          .into_iter()
          .map(|strategy| { (strategy, ${EDGE}) })
          .collect();
  }`;
  assert.deepEqual(spell(additionalPrerequisiteEdges(reset, "reset")), ["BoundedDecode"]);

  // ...but the reset is scoped to the enclosing function, and that is load-bearing rather than
  // decorative. `candle-gen-krea/src/lib.rs` assigns at :1700 inside a DIFFERENT function building a
  // DIFFERENT contract (`krea_2_turbo_control`) from the one initialised at :1396, so a file-flat
  // reset would erase the base provider's real edge — a silent zero reached by fixing an over-count.
  //
  // MUTATION: scope the reset to the file and this derives only `BoundedDecode`.
  const twoContracts = `fn base() -> MemoryProviderContract {
      contract.additional_prerequisites.push((MemoryStrategy::BoundedTransformerResidency, ${EDGE}));
  }
  fn control() -> MemoryProviderContract {
      contract.additional_prerequisites = [MemoryStrategy::BoundedDecode]
          .into_iter()
          .map(|strategy| { (strategy, ${EDGE}) })
          .collect();
  }`;
  assert.deepEqual(spell(additionalPrerequisiteEdges(twoContracts, "two scopes")), [
    "BoundedTransformerResidency",
    "BoundedDecode",
  ]);
});

test("a construction that does not reach every production build is recorded as conditional", () => {
  // `candle-gen-krea/src/lib.rs:1627`, `candle-gen-z-image/src/memory_strategy.rs:38` and
  // `candle-gen-lens/src/lib.rs` all build their contract inside `#[cfg(any(feature = "cuda",
  // test))]`. Presenting a feature-gated variant as the provider's unconditional declaration is the
  // over-count MAJOR 3 names; the gate still treats a conditional edge as PRESENT, which is the
  // fail-closed reading for an admission gate.
  //
  // MUTATION: drop the `conditional` filter over `cfgAttributedItems` and this reds.
  const gated = `#[cfg(any(feature = "cuda", test))]
  fn control() -> MemoryProviderContract {
      contract.additional_prerequisites.push((MemoryStrategy::BoundedTransformerResidency, ${EDGE}));
  }`;
  assert.deepEqual(spell(additionalPrerequisiteEdges(gated, "cfg-gated")), [
    "BoundedTransformerResidency?",
  ]);
  // The same construction outside any `#[cfg]` is unconditional, so `conditional` is a property of
  // the gate and not a constant this fixture would report either way.
  const ungated = `fn base() -> MemoryProviderContract {
      contract.additional_prerequisites.push((MemoryStrategy::BoundedTransformerResidency, ${EDGE}));
  }`;
  assert.deepEqual(spell(additionalPrerequisiteEdges(ungated, "ungated")), [
    "BoundedTransformerResidency",
  ]);
});

test("test-only items are excluded whatever shape they take, and their edges are not the provider's", () => {
  // Three regressions in one fixture: a `#[cfg(test)] use` before the production construction (a
  // first draft cut the file there and read 820 of `mlx-gen-krea`'s 30,249 bytes), a `#[cfg(test)]
  // fn` helper (the stripper only ever removed `mod`s, so a helper's fixture edges reached the
  // record), and a brace inside a string literal in the test module.
  const source = `
#[cfg(test)]
use gen_core::MemoryGeometry;

pub fn contract() -> MemoryProviderContract {
    contract.additional_prerequisites.push((MemoryStrategy::BoundedTransformerResidency, ${EDGE}));
}

#[cfg(test)]
fn fixture() -> MemoryProviderContract {
    contract.additional_prerequisites.push((MemoryStrategy::BoundedDecode, ${EDGE}));
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_brace_in_a_literal_does_not_end_the_module() {
        assert_eq!(render("}"), "}");
        assert_eq!(BRACE, '}');
        contract.additional_prerequisites.push((MemoryStrategy::BoundedAttention, ${EDGE}));
    }
}
`;
  assert.deepEqual(spell(additionalPrerequisiteEdges(stripTestOnlyItems(source), "stripped")), [
    "BoundedTransformerResidency",
  ]);
  // Non-vacuity: both fixture edges are really in the text and really removed by the stripping.
  const stripped = stripTestOnlyItems(source);
  for (const strategy of ["BoundedDecode", "BoundedAttention"]) {
    assert.ok(source.includes(strategy), `the fixture must carry a test-only ${strategy} edge`);
    assert.ok(!stripped.includes(strategy), `${strategy} must not survive stripping`);
  }
  // A caller that forgets to strip gets an error rather than a polluted record.
  assert.throws(
    () => additionalPrerequisiteEdges(source, "unstripped"),
    /a test-only item survived stripping/,
  );
});

test("the rung-1 support declaration is read only when the crate declares it unambiguously", () => {
  // gen-core's `validate_selection` satisfies an `EngagedInSameRequest` edge VACUOUSLY when the
  // prerequisite rung is `StructurallyNotApplicable`. That arm ADMITS, so a wrong positive admits a
  // cell the contract refuses — hence `null` on anything the extractor cannot read as one
  // declaration for the crate.
  //
  // MUTATION: accept a guarded arm, or take the first of several, and the last two cases red.
  assert.equal(
    stagedResidencySupport(
      // `mlx-gen-sensenova/src/memory_strategy.rs:479-483`, the only provider declaring it at the
      // pinned revision.
      `match capability.strategy {
           MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
           MemoryStrategy::StagedResidency => {
               MemoryStrategySupport::StructurallyNotApplicable {
                   reason: "one flat fused dual-path checkpoint".to_owned(),
               }
           }
       }`,
    ),
    "structurally_not_applicable",
  );
  assert.equal(
    stagedResidencySupport("MemoryStrategy::StagedResidency => MemoryStrategySupport::Implemented,"),
    "implemented",
  );
  assert.equal(
    stagedResidencySupport(
      "MemoryStrategy::StagedResidency if streamable => MemoryStrategySupport::StructurallyNotApplicable { reason: String::new() },",
    ),
    null,
    "a guarded arm is one of several and does not decide the declaration on its own",
  );
  assert.equal(
    stagedResidencySupport(
      `MemoryStrategy::StagedResidency => MemoryStrategySupport::StructurallyNotApplicable { reason: String::new() },
       MemoryStrategy::StagedResidency => MemoryStrategySupport::Implemented,`,
    ),
    null,
    "disagreeing arms must read as unknown rather than as N/A",
  );
  assert.equal(
    stagedResidencySupport("let staged = MemoryStrategy::StagedResidency;"),
    null,
    "a mention that is not a match arm declares nothing",
  );
});

test("the records retain source provenance and cite where each edge came from", async () => {
  const parsed = await records();
  assert.match(
    parsed.inferenceRevision ?? "",
    /^[0-9a-f]{40}$/,
    "the edges must remain dated to the full inference revision they were derived from",
  );

  for (const [group, family] of Object.entries(parsed.families)) {
    for (const [backend, record] of Object.entries(family.backends)) {
      const at = `${group}:${backend}`;
      assert.match(record.crate, /^crates\/media\/(mlx|candle)-gen\/(mlx|candle)-gen-[\w-]+$/, at);
      assert.ok(Array.isArray(record.additionalPrerequisites), at);
      assert.ok(
        ["implemented", "missing", "structurally_not_applicable", null].includes(
          record.stagedResidencySupport,
        ),
        `${at}: stagedResidencySupport must be a gen-core support variant or null, got ${JSON.stringify(record.stagedResidencySupport)}`,
      );
      for (const edge of record.additionalPrerequisites) {
        assert.deepEqual(
          { kind: edge.kind, rung: edge.rung, scope: edge.scope },
          STAGED_RESIDENCY_ENGAGED_IN_SAME_REQUEST,
          at,
        );
        assert.equal(
          typeof edge.conditional,
          "boolean",
          `${at}: an absent \`conditional\` would read as a claim that the construction is unconditional`,
        );
        assert.ok(
          edge.source.startsWith(`${record.crate}/src/`),
          `${at}: an edge must cite a file inside the crate the record names, got ${edge.source}`,
        );
      }
    }
  }
  // sc-22512 / E8: the "both partitions occupied" pin (`withEdge > 0 && withoutEdge > 0`) was
  // removed. It reds purely on a population emptying — the day every provider declares the edge, or
  // none does, which an inference pin can produce without a single record being wrong. The per-record
  // grading above is untouched and still reds on every malformed or mis-provenanced edge that IS
  // present.
});

test("the records file states the granularity its (family, backend) key cannot express", async () => {
  // One record per family, unioned over the crate, while FLUX.2 klein and dev share a crate and
  // `candle-gen-krea` declares a base contract and a control contract with different edge sets. A
  // limit a reader cannot see in the data has to be written down beside it.
  const parsed = await records();
  assert.ok(parsed.granularity, "the records must state their granularity limit");
  assert.match(parsed.granularity, /family/i);
  // Named cases, not an adjective: both live examples have to be identifiable from the note.
  assert.match(parsed.granularity, /flux2|FLUX\.2/i);
  assert.match(parsed.granularity, /krea/i);
  // sc-22512 / E8: the "some crate must back two families" assertion was removed. It reds on a
  // population emptying — a re-slice that gives each family its own crate leaves the note accurate
  // and the record set healthy, yet failed the check. The note's CONTENT is still graded above.
});
