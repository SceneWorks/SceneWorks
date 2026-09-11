# StarVector model provenance

SceneWorks installs the immutable
`starvector/starvector-1b-im2svg@380ab95d25a8e9ab1dc825debe238b4953ae13b9` snapshot for
native image-to-SVG generation, and the immutable
`starvector/starvector-8b-im2svg@518beea8dcb5f7a37c5911e92d1d62a76beee7f9` snapshot for its
higher-capacity sibling. Both upstream model cards declare Apache-2.0 and neither snapshot ships a
separate NOTICE file. SceneWorks installs each immutable model card plus the paired inference
contract's exact data-only runtime closure, excludes both repositories' Python modules, and does
not execute remote model code.
