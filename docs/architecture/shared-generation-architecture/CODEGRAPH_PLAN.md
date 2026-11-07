# SGA CodeGraph plan

The SceneWorks project (`SW`) tracks the [reviewed requirements](REQUIREMENTS.md) and [implementation roadmap](IMPLEMENTATION_PLAN.md) in 25 features and 142 items. **SW-F331** holds the program index, full draft 0.7 source snapshots and the [closure review](evidence/reviews/round8-codex.md).

## Starting a session

Open the feature in CodeGraph and read its items, blockers and latest handoffs before claiming work. Planned items have status `ready` unless explicitly backlogged. Eligibility also requires completed prerequisites. CodeGraph is authoritative for subsequent changes.

At publication on November 7, 2026, the unblocked starting items were **SW-6931** (route coverage) and **SW-6932** (comparison baselines) in SW-F331, and **SW-7028** (pre-change catalog capture) in SW-F348. The engine and catalog work can start independently.

## Feature index

| Roadmap | CodeGraph | Feature | Direct prerequisites |
|---|---|---|---|
| ENG-1a | SW-F331 | Runner core + FLUX.1 migration | — |
| ENG-1b | SW-F332 | Low-res start (first Feature) | SW-F331 |
| ENG-2 | SW-F333 | Shared-driver image wave | SW-F332 |
| ENG-3 | SW-F334 | Shared-driver video wave | SW-F332 |
| ENG-4a | SW-F335 | Backend primitives + media/LLM Linear spine | SW-F333, SW-F334 |
| ENG-4b | SW-F336 | Request-scoped scales + per-phase adapters | SW-F335 |
| ENG-5 | SW-F337 | FamilyDrivers I | SW-F336 |
| ENG-6 | SW-F338 | FamilyDrivers II (multi-stream, autoregressive) | SW-F337 |
| ENG-7 | SW-F339 | Route records + capability export + registration | SW-F338 |
| ENG-8 | SW-F340 | Layer kit II | SW-F338 |
| ENG-9 | SW-F341 | Weights | SW-F339 |
| ENG-10 | SW-F342 | Latent codecs + R2 | SW-F340 |
| ENG-11 | SW-F343 | Encoders on llm + towers | SW-F340, SW-F341 |
| ENG-12 | SW-F344 | BlockStack + conditioning + R3/R4 | SW-F340, SW-F341 |
| ENG-13 | SW-F345 | Computed memory contracts + R1 + iOS | SW-F342, SW-F343, SW-F344 |
| ENG-14 | SW-F346 | Audio + LLM adoption | SW-F341, SW-F345 |
| ENG-15 | SW-F347 | DEC-1 (c) pilots | SW-F340, SW-F341, SW-F342, SW-F343, SW-F344 |
| APP-1 | SW-F348 | Model registry | — |
| APP-2 | SW-F349 | Options core | SW-F348 |
| APP-3 | SW-F350 | Generated panels | SW-F349 |
| ENG-16 | SW-F351 | Configurable chained denoise passes | SW-F339, SW-F350 |
| APP-4 | SW-F352 | Admission pipeline | SW-F338, SW-F348 |
| APP-5 | SW-F353 | Generation lane I | SW-F339, SW-F348, SW-F352 |
| APP-6 | SW-F354 | Generation lane II | SW-F346, SW-F353 |
| APP-7 | SW-F355 | Tooling | SW-F341, SW-F348 |

## Item index

### SW-F331 · ENG-1a · Runner core + FLUX.1 migration

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-6931 | task | Reconcile product dispatch and native coverage | — |
| SW-6932 | task | Establish the migration comparison baselines | — |
| SW-6933 | story | Run base FLUX.1 through the owner-local runner | SW-6931, SW-6932 |
| SW-6934 | task | Preserve solver history and native executor ownership | SW-6933 |
| SW-6935 | task | Admit bounded conditional stages | SW-6933 |
| SW-6936 | task | Migrate the remaining FLUX.1 providers | SW-6933, SW-6934, SW-6935 |
| SW-6937 | task | Integrate and accept Runner core + FLUX.1 migration | SW-6931, SW-6932, SW-6933, SW-6934, SW-6935, SW-6936 |

### SW-F332 · ENG-1b · Low-res start (first Feature)

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-6938 | story | Expose structural Feature applicability | SW-6937 |
| SW-6939 | story | Implement the two-phase low-res recipe | SW-6938 |
| SW-6940 | task | Qualify endpoint and previous-estimate policies | SW-6939 |
| SW-6941 | story | Bound the first real transition | SW-6939 |
| SW-6942 | task | Admit and expose low-res generation | SW-6939, SW-6941 |
| SW-6943 | story | Preserve low-res shares and exact replay | SW-6939 |
| SW-6944 | task | Integrate and accept Low-res start (first Feature) | SW-6938, SW-6939, SW-6940, SW-6941, SW-6942, SW-6943 |

### SW-F333 · ENG-2 · Shared-driver image wave

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-6945 | task | Migrate FLUX.2 and the shared image schedules | SW-6944 |
| SW-6946 | task | Qualify ordinary img2img initialization | SW-6945 |
| SW-6947 | task | Migrate remaining shared image families | SW-6945 |
| SW-6948 | task | Cut over Hires.fix for this image cohort | SW-6945, SW-6946, SW-6947 |
| SW-6949 | task | Integrate and accept Shared-driver image wave | SW-6945, SW-6946, SW-6947, SW-6948 |

### SW-F334 · ENG-3 · Shared-driver video wave

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-6950 | task | Run curated Wan through shared sampling | SW-6944 |
| SW-6951 | task | Migrate Mochi and SVD | SW-6950 |
| SW-6952 | task | Migrate SCAIL-2 and video conditioning | SW-6950 |
| SW-6953 | task | Bind migrated video admission to the plan | SW-6950, SW-6951, SW-6952 |
| SW-6954 | task | Integrate and accept Shared-driver video wave | SW-6950, SW-6951, SW-6952, SW-6953 |

### SW-F335 · ENG-4a · Backend primitives + media/LLM Linear spine

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-6955 | task | Extract lower-level backend primitives | SW-6949, SW-6954 |
| SW-6956 | task | Unify dense and quantized media Linears | SW-6955 |
| SW-6957 | task | Move LLM and Prism projections | SW-6955 |
| SW-6958 | task | Preserve target-specific bundle partitions | SW-6956, SW-6957 |
| SW-6959 | task | Integrate and accept Backend primitives + media/LLM Linear spine | SW-6955, SW-6956, SW-6957, SW-6958 |

### SW-F336 · ENG-4b · Request-scoped scales + per-phase adapters

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-6960 | task | Bind adapter scales to each request and phase | SW-6959 |
| SW-6961 | task | Migrate Krea multiphase and Candle residuals | SW-6960 |
| SW-6962 | task | Migrate LTX and Wan adapter filtering | SW-6960 |
| SW-6963 | task | Cut over imported Krea Hires.fix | SW-6961 |
| SW-6964 | task | Delete replaced Candle adapter graphs | SW-6961, SW-6962 |
| SW-6965 | task | Integrate and accept Request-scoped scales + per-phase adapters | SW-6960, SW-6961, SW-6962, SW-6963, SW-6964 |

### SW-F337 · ENG-5 · FamilyDrivers I

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-6966 | task | Migrate native SDXL and Kolors | SW-6965 |
| SW-6967 | task | Migrate imported SDXL refinement | SW-6966 |
| SW-6968 | task | Migrate Wan native and FLUX.3 flow-map | SW-6966 |
| SW-6969 | task | Migrate distillation and iOS drivers | SW-6966 |
| SW-6970 | task | Migrate Ideogram and Supra2 | SW-6966 |
| SW-6971 | task | Integrate and accept FamilyDrivers I | SW-6966, SW-6967, SW-6968, SW-6969, SW-6970 |

### SW-F338 · ENG-6 · FamilyDrivers II (multi-stream, autoregressive)

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-6972 | task | Run LTX multi-stream schedules | SW-6971 |
| SW-6973 | task | Preserve H3 joint execution and CUDA fit | SW-6972 |
| SW-6974 | task | Bind SenseNova feedback to the runner | SW-6972 |
| SW-6975 | task | Migrate Mage and Bernini | SW-6972 |
| SW-6976 | story | Deliver terminal chunk output for Krea realtime | SW-6972 |
| SW-6977 | task | Migrate SeedVR2 and PiD decode | SW-6972 |
| SW-6978 | task | Integrate and accept FamilyDrivers II (multi-stream, autoregressive) | SW-6972, SW-6973, SW-6974, SW-6975, SW-6976, SW-6977 |

### SW-F339 · ENG-7 · Route records + capability export + registration

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-6979 | task | Register a route from primary component declarations | SW-6978 |
| SW-6980 | task | Build descriptor-only platform exports | SW-6979 |
| SW-6981 | task | Type the request, outputs and execution receipts | SW-6979 |
| SW-6982 | task | Register residual bespoke and vector providers | SW-6979, SW-6981 |
| SW-6983 | task | Migrate all export consumers and bundle imports | SW-6980, SW-6981, SW-6982 |
| SW-6984 | task | Retire the final Hires.fix helper | SW-6982 |
| SW-6985 | task | Integrate and accept Route records + capability export + registration | SW-6979, SW-6980, SW-6981, SW-6982, SW-6983, SW-6984 |

### SW-F340 · ENG-8 · Layer kit II

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-6986 | task | Route attention through the backend operation | SW-6978 |
| SW-6987 | task | Migrate norms, rotary embeddings and timestep math | SW-6986 |
| SW-6988 | task | Support differentiable training through the same layers | SW-6986, SW-6987 |
| SW-6989 | task | Remove each verified duplicate trainer graph | SW-6988 |
| SW-6990 | task | Qualify the affected H3 attention seam | SW-6986 |
| SW-6991 | task | Integrate and accept Layer kit II | SW-6986, SW-6987, SW-6988, SW-6989, SW-6990 |

### SW-F341 · ENG-9 · Weights

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-6992 | task | Compile and bind a declarative weight plan | SW-6985 |
| SW-6993 | task | Migrate loader dialects and component tiers | SW-6992 |
| SW-6994 | task | Move adapter routing to canonical weight names | SW-6992, SW-6993 |
| SW-6995 | task | Unify load progress, cancellation and slot swaps | SW-6992 |
| SW-6996 | task | Integrate and accept Weights | SW-6992, SW-6993, SW-6994, SW-6995 |

### SW-F342 · ENG-10 · Latent codecs + R2

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-6997 | task | Share KL-2D codecs and bounded decode | SW-6991 |
| SW-6998 | task | Migrate Wan causal codecs | SW-6997 |
| SW-6999 | task | Move LTX, H3 and singleton codecs onto the kit | SW-6997 |
| SW-7000 | task | Qualify R2 choices and delete duplicate codecs | SW-6997, SW-6998, SW-6999 |
| SW-7001 | task | Integrate and accept Latent codecs + R2 | SW-6997, SW-6998, SW-6999, SW-7000 |

### SW-F343 · ENG-11 · Encoders on llm + towers

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-7002 | story | Expose bounded encoder taps on the LLM interface | SW-6991, SW-6996 |
| SW-7003 | task | Migrate causal encoder families and quant codecs | SW-7002 |
| SW-7004 | task | Share text and vision towers | SW-7002 |
| SW-7005 | task | Remove family-local encoder transformer copies | SW-7003, SW-7004 |
| SW-7006 | task | Integrate and accept Encoders on llm + towers | SW-7002, SW-7003, SW-7004, SW-7005 |

### SW-F344 · ENG-12 · BlockStack + conditioning + R3/R4

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-7007 | task | Move transformer iteration into BlockStack | SW-6991, SW-6996 |
| SW-7008 | task | Share grid-aligned conditioning modules | SW-7007 |
| SW-7009 | task | Relocate R3 bounded attention and R4 residency | SW-7007 |
| SW-7010 | task | Qualify feature applicability after conditioning cutover | SW-7008, SW-7009 |
| SW-7011 | task | Demonstrate one-change rung adoption | SW-7009 |
| SW-7012 | task | Integrate and accept BlockStack + conditioning + R3/R4 | SW-7007, SW-7008, SW-7009, SW-7010, SW-7011 |

### SW-F345 · ENG-13 · Computed memory contracts + R1 + iOS

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-7013 | task | Derive memory contracts from composed routes | SW-7001, SW-7006, SW-7012 |
| SW-7014 | task | Relocate R1 and retire MemoryRequestScope | SW-7013 |
| SW-7015 | task | Migrate iOS incremental and quantum admission | SW-7013 |
| SW-7016 | task | Bind consumer admission to computed contracts | SW-7013, SW-7014 |
| SW-7017 | task | Integrate and accept Computed memory contracts + R1 + iOS | SW-7013, SW-7014, SW-7015, SW-7016 |

### SW-F346 · ENG-14 · Audio + LLM adoption

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-7018 | task | Share audio loading, adapters and residency | SW-6996, SW-7017 |
| SW-7019 | task | Adopt the runner for diffusion and flow audio | SW-7018 |
| SW-7020 | task | Preserve autoregressive audio and unify LLM lifecycle | SW-7018 |
| SW-7021 | task | Close new audio recipe rows | SW-7019 |
| SW-7022 | task | Integrate and accept Audio + LLM adoption | SW-7018, SW-7019, SW-7020, SW-7021 |

### SW-F347 · ENG-15 · DEC-1 (c) pilots

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-7023 | task | Exercise the small-family generic pilot | SW-6991, SW-6996, SW-7001, SW-7006, SW-7012 |
| SW-7024 | task | Exercise the FLUX.3 generation pilot | SW-7023 |
| SW-7025 | task | Prove FLUX.3 LoRA training on the generic graph | SW-7023 |
| SW-7026 | task | Decide generic rollout from both pilot gates | SW-7023, SW-7024, SW-7025 |
| SW-7027 | task | Integrate and accept DEC-1 (c) pilots | SW-7023, SW-7024, SW-7025, SW-7026 |

### SW-F348 · APP-1 · Model registry

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-7028 | task | Capture the old public catalog before cutover | — |
| SW-7029 | task | Build the static ModelSpec registry | SW-7028 |
| SW-7030 | task | Preserve imported and trained identity across hosts | SW-7029 |
| SW-7031 | task | Upgrade manifests and migrate registry consumers | SW-7029, SW-7030 |
| SW-7032 | task | Retire capability digests and hand mirrors | SW-7029, SW-7031 |
| SW-7033 | task | Integrate and accept Model registry | SW-7028, SW-7029, SW-7030, SW-7031, SW-7032 |

### SW-F349 · APP-2 · Options core

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-7034 | task | Establish the shared option registry | SW-7033 |
| SW-7035 | task | Apply one validation path at all submission boundaries | SW-7034 |
| SW-7036 | story | Deliver portable browser validation | SW-7034, SW-7035 |
| SW-7037 | task | Upgrade persisted and external request shapes | SW-7034 |
| SW-7038 | task | Complete typed request binding | SW-7034, SW-7035, SW-7037 |
| SW-7039 | task | Integrate and accept Options core | SW-7034, SW-7035, SW-7036, SW-7037, SW-7038 |

### SW-F350 · APP-3 · Generated panels

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-7040 | task | Render generated image and simple studio panels | SW-7039 |
| SW-7041 | story | Migrate video, audio and training studios | SW-7040 |
| SW-7042 | task | Migrate character and structured editors | SW-7040 |
| SW-7043 | task | Remove backend-specific web gating | SW-7040, SW-7041, SW-7042 |
| SW-7044 | task | Integrate and accept Generated panels | SW-7040, SW-7041, SW-7042, SW-7043 |

### SW-F351 · ENG-16 · Configurable chained denoise passes

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-7045 | task | Compile user chains into the pass runner | SW-6985, SW-7044 |
| SW-7046 | task | Validate total expanded work before allocation | SW-7045 |
| SW-7047 | task | Price and execute supported chained routes | SW-7046 |
| SW-7048 | story | Expose chain controls through shared options | SW-7046 |
| SW-7049 | task | Qualify all applicable chain cohorts | SW-7047, SW-7048 |
| SW-7050 | task | Demonstrate one-change guidance adoption | SW-7048 |
| SW-7051 | task | Integrate and accept Configurable chained denoise passes | SW-7049, SW-7050 |

### SW-F352 · APP-4 · Admission pipeline

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-7052 | task | Route a real image job through Plan and Run admission | SW-6978, SW-7033 |
| SW-7053 | task | Unify budget probes and failure postures | SW-7052 |
| SW-7054 | task | Migrate all remaining admission paths | SW-7052, SW-7053 |
| SW-7055 | task | Centralize decisions, receipts and refusals | SW-7052, SW-7053, SW-7054 |
| SW-7056 | task | Integrate and accept Admission pipeline | SW-7052, SW-7053, SW-7054, SW-7055 |

### SW-F353 · APP-5 · Generation lane I

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-7057 | story | Drive a real image job through the shared lane | SW-6985, SW-7033, SW-7056 |
| SW-7058 | task | Migrate image and bespoke vector entrypoints | SW-7057 |
| SW-7059 | task | Bind output publication to the winning attempt | SW-7057 |
| SW-7060 | task | Remove duplicated lane and cache wiring | SW-7058, SW-7059 |
| SW-7061 | task | Integrate and accept Generation lane I | SW-7057, SW-7058, SW-7059, SW-7060 |

### SW-F354 · APP-6 · Generation lane II

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-7062 | task | Move video and interleaved documents to the common lane | SW-7022, SW-7061 |
| SW-7063 | task | Migrate audio and remaining utility lifecycles | SW-7062 |
| SW-7064 | task | Consolidate the platform module boundary | SW-7062 |
| SW-7065 | task | Separate API-only and desktop composition | SW-7064 |
| SW-7066 | task | Demonstrate addition of a model in an existing class | SW-7062, SW-7063, SW-7064, SW-7065 |
| SW-7067 | task | Integrate and accept Generation lane II | SW-7062, SW-7063, SW-7064, SW-7065, SW-7066 |

### SW-F355 · APP-7 · Tooling

| Item | Type | Outcome | Blocked by |
|---|---|---|---|
| SW-7068 | task | Run the memory adapter through registered routes | SW-6996, SW-7033 |
| SW-7069 | task | Migrate auxiliary binaries and workflow consumers | SW-7068 |
| SW-7070 | task | Remove generated memory mirrors | SW-7068, SW-7069 |
| SW-7071 | task | Prove memory-adapter equivalence | SW-7068, SW-7069 |
| SW-7072 | task | Integrate and accept Tooling | SW-7068, SW-7069, SW-7070, SW-7071 |
