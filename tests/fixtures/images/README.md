# Water bottle edit reference

`water-bottle.png` is the original blue bottle/oranges reference already used in
MOSS Edit validation (`inputs/B.png`), retained as a shared CPU test fixture.
PNG SHA-256: `c6ec194e6d7e04a2893449e530cf51d064c6adfd9847f0b028921b21e1c1bcb5`.
The UI, API and worker tests share `../water-bottle-edit-request.json`. The worker
uses a recording stub and stops before inference; these checks never load weights
or execute model layers. The real UI edit is manual validation, outside CI.
