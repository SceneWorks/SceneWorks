import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  RECORDS_PATH,
  STAGED_RESIDENCY_ENGAGED_IN_SAME_REQUEST,
  additionalPrerequisiteEdges,
  inferencePin,
  stripTestModules,
} from "./rung4-contract-prerequisites.mjs";

const records = async () =>
  JSON.parse(await readFile(new URL(`../${RECORDS_PATH}`, import.meta.url), "utf8"));

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
    ["BoundedTransformerResidency"],
  ],
];

test("every construction shape present at the pinned revision is read, individually", () => {
  for (const [name, source, expected] of SHAPES) {
    assert.deepEqual(additionalPrerequisiteEdges(source, name), expected, name);
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
  // A read is not a construction, and must contribute nothing rather than throw — three provider
  // test modules assert `is_empty()` on the vector.
  assert.deepEqual(
    additionalPrerequisiteEdges("assert!(contract.additional_prerequisites.is_empty());", "read"),
    [],
  );
});

test("test modules are excluded without taking the production declaration with them", () => {
  // The regression this guard exists for: `mlx-gen-krea/src/memory_strategy.rs` carries a
  // `#[cfg(test)] use` on line 13, and a first draft that cut the file at the first `#[cfg(test)]`
  // read 820 of its 30,249 bytes and derived zero edges from a provider that pushes one. A silent
  // zero is the fail-open direction, so the extent is matched rather than guessed at.
  const source = `
#[cfg(test)]
use gen_core::MemoryGeometry;

pub fn contract() -> MemoryProviderContract {
    contract.additional_prerequisites.push((
        MemoryStrategy::BoundedTransformerResidency,
        MemoryStrategyPrerequisite::Rung {
            rung: MemoryStrategy::StagedResidency,
            scope: MemoryPrerequisiteScope::EngagedInSameRequest,
        },
    ));
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_brace_in_a_literal_does_not_end_the_module() {
        assert_eq!(render("}"), "}");
        assert_eq!(BRACE, '}');
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedDecode,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
    }
}
`;
  assert.deepEqual(additionalPrerequisiteEdges(stripTestModules(source), "stripped"), [
    "BoundedTransformerResidency",
  ]);
  // Non-vacuity: the fixture really does carry an edge inside the test module, so the assertion
  // above is graded by the stripping rather than by there being only one construction.
  assert.deepEqual(additionalPrerequisiteEdges(source, "unstripped"), [
    "BoundedTransformerResidency",
    "BoundedDecode",
  ]);
});

test("the records are keyed to the live Cargo pin and cite where each edge came from", async () => {
  const parsed = await records();
  const pin = inferencePin(await readFile(new URL("../Cargo.toml", import.meta.url), "utf8"));
  assert.equal(
    parsed.inferenceRevision,
    pin,
    "the edges are a fact about one inference revision; a stale key means nobody re-derived them",
  );

  let withEdge = 0;
  let withoutEdge = 0;
  for (const [group, family] of Object.entries(parsed.families)) {
    for (const [backend, record] of Object.entries(family.backends)) {
      const at = `${group}:${backend}`;
      assert.match(record.crate, /^crates\/media\/(mlx|candle)-gen\/(mlx|candle)-gen-[\w-]+$/, at);
      assert.ok(Array.isArray(record.additionalPrerequisites), at);
      for (const edge of record.additionalPrerequisites) {
        assert.deepEqual(
          { kind: edge.kind, rung: edge.rung, scope: edge.scope },
          STAGED_RESIDENCY_ENGAGED_IN_SAME_REQUEST,
          at,
        );
        assert.ok(
          edge.source.startsWith(`${record.crate}/src/`),
          `${at}: an edge must cite a file inside the crate the record names, got ${edge.source}`,
        );
      }
      if (record.additionalPrerequisites.length) withEdge += 1;
      else withoutEdge += 1;
    }
  }
  // Both partitions occupied. This is the fact that makes the gate different from the blanket proxy
  // it replaced: if every provider appended the edge, consulting the record would be a longer way of
  // writing the proxy, and if none did, the rung-1 predicate would reach no rung-4 cell at all.
  assert.ok(withEdge > 0 && withoutEdge > 0, "the records must discriminate between providers");
});
