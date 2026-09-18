import React from "react";

export interface LogoProps {
  /** Square size in px. Default 32. */
  size?: number;
  /** Accessible label + <title>. Default "SceneWorks". */
  title?: string;
  className?: string;
}

/**
 * SceneWorks scene-cut brand mark. Theme- and accent-aware via CSS variables.
 * @dsCard group="Components"
 * @startingPoint section="Brand" subtitle="Scene-cut mark + wordmark" viewport="700x150"
 */
export function Logo(props: LogoProps): React.ReactElement;

export interface WordmarkProps {
  size?: number;
}
/** Mark + "SceneWorks" wordmark lockup. */
export function Wordmark(props: WordmarkProps): React.ReactElement;
