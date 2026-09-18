import React from "react";

export interface MarkdownProps {
  /** Markdown source. Supports headings, lists, blockquotes, fenced code, and
   *  inline bold/italic/code/links. Safe (no dangerouslySetInnerHTML). */
  content: string;
}

/**
 * Dependency-free Markdown renderer used for prompt guides and help copy.
 * @dsCard group="Components"
 */
export function Markdown(props: MarkdownProps): React.ReactElement;
