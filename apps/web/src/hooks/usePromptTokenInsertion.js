import { useCallback, useLayoutEffect, useRef } from "react";
import { insertPromptToken } from "../imageReferenceLimits.js";

// SW-7101 follow-up — "Insert <Picture N>" lands at the prompt textarea's caret (replacing any
// selected text) instead of always at the end.
//
// Spread `textareaProps` onto the prompt <textarea>. Clicking an Insert button blurs the textarea,
// but a blurred textarea keeps its selectionStart/End, so the live values are read at click time.
// Until the user has placed a caret there (no focus or `select` event yet) the token is appended,
// which is the old behaviour and the only sensible target for a prompt the user has not touched.
export function usePromptTokenInsertion(prompt, setPrompt) {
  const textareaRef = useRef(null);
  const caretPlaced = useRef(false);
  const pendingCaret = useRef(null);
  const onSelect = useCallback(() => {
    caretPlaced.current = true;
  }, []);

  // Put focus back with the caret just after the inserted token, in the same commit that writes
  // the new value — so the user keeps typing (or inserts the next reference) where they were. A
  // layout effect rather than a timer, so nothing can move the caret after the user has.
  useLayoutEffect(() => {
    const caret = pendingCaret.current;
    const element = textareaRef.current;
    if (caret == null || !element) return;
    pendingCaret.current = null;
    element.focus();
    element.setSelectionRange(caret, caret);
  }, [prompt]);

  const insert = useCallback(
    (token) => {
      const element = textareaRef.current;
      const selection = element && caretPlaced.current
        ? { start: element.selectionStart, end: element.selectionEnd }
        : null;
      const { prompt: next, caret } = insertPromptToken(prompt, token, selection);
      if (element) caretPlaced.current = true;
      // Replacing a selected token with the same token leaves the prompt unchanged, so no commit
      // (and no layout effect) follows: restore the caret now rather than leave it pending for
      // some later, unrelated edit to apply.
      if (next === prompt) {
        if (element) {
          element.focus();
          element.setSelectionRange(caret, caret);
        }
        return;
      }
      if (element) pendingCaret.current = caret;
      setPrompt(next);
    },
    [prompt, setPrompt],
  );

  // Focusing the textarea places a caret as surely as moving it does.
  return { textareaProps: { ref: textareaRef, onSelect, onFocus: onSelect }, insert };
}
