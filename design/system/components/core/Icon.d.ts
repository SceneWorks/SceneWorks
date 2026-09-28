import React from "react";

export interface IconGlyphProps {
  /** Glyph size in px (width = height). Default 18. */
  size?: number;
  className?: string;
  style?: React.CSSProperties;
}

type Glyph = (props: IconGlyphProps) => React.ReactElement;

/**
 * SceneWorks icon set — a 24×24 stroked grid (1.7px, round caps, currentColor).
 * Access glyphs as members: `<Icon.Video />`, `<Icon.Sun size={16} />`.
 * @dsCard group="Components"
 */
export declare const Icon: {
  Library: Glyph; Image: Glyph; ImageEditor: Glyph; Video: Glyph; Editor: Glyph;
  Train: Glyph; Character: Glyph; Preset: Glyph; Model: Glyph; Queue: Glyph;
  Logs: Glyph; Search: Glyph; Sparkle: Glyph; Plus: Glyph; Sun: Glyph; Moon: Glyph;
  Bell: Glyph; Folder: Glyph; Book: Glyph; Info: Glyph; ChevDown: Glyph;
  Sliders: Glyph; Play: Glyph; Pause: Glyph; ArrowLeft: Glyph; ArrowRight: Glyph;
  Wand: Glyph; Star: (props: IconGlyphProps & { filled?: boolean }) => React.ReactElement; Stars: Glyph;
};
