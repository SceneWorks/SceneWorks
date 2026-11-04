import { useEffect } from "react";
import { isDesktop as isDesktopShell } from "../runtime.js";

// F-189: extracted from App along its existing concern boundary.
export function useWebView2RepaintPump() {
  // Desktop (WebView2 / Windows) only: opening a native file dialog from an
  // `<input type="file">` can strand the composited app shell painted blank.
  // When the modal dialog takes the foreground it disables the owner window and
  // Chromium stops presenting the promoted DirectComposition layers (`.sidebar`,
  // `.topbar`, `.workspace`); it does NOT restore them from ordinary window
  // messages (resize / minimize-restore) — only an internally-scheduled frame
  // recovers it, which is why disabling `CalculateNativeWinOcclusion` alone
  // doesn't help. So while a file picker is pending we pump a cheap, sub-pixel
  // transform on `.app` every frame to keep WebView2 presenting *during* the
  // dialog (not just after it closes, which was the visibly-flashing stopgap),
  // then do a final promote/demote once focus returns to guarantee recovery.
  // No-op off the desktop shell and on macOS WebKit, which never drops layers.
  useEffect(() => {
    if (!isDesktopShell) {
      return undefined;
    }

    let pumping = false;
    let rafId = 0;
    let timerId = 0;
    let deadlineId = 0;
    let phase = false;

    const shell = () => document.querySelector(".app");

    const tick = () => {
      const el = shell();
      if (el instanceof HTMLElement) {
        // The transform stays established the whole time (so `.app` never
        // toggles its containing-block status under any fixed child), but the
        // sub-pixel value changes each frame to force a fresh present. `.app`
        // fills the viewport at the origin, so the shift is imperceptible.
        phase = !phase;
        el.style.transform = phase ? "translate3d(0.02px, 0, 0)" : "translate3d(0, 0, 0)";
      }
      if (pumping) {
        rafId = requestAnimationFrame(tick);
      }
    };

    const stop = () => {
      if (!pumping) {
        return;
      }
      pumping = false;
      cancelAnimationFrame(rafId);
      window.clearInterval(timerId);
      window.clearTimeout(deadlineId);
      const el = shell();
      if (el instanceof HTMLElement) {
        // Promote-then-demote across two frames, then drop the inline transform
        // we borrowed, leaving the stylesheet in control again.
        requestAnimationFrame(() => {
          el.style.transform = "translateZ(0)";
          requestAnimationFrame(() => {
            el.style.transform = "";
          });
        });
      }
    };

    const start = () => {
      if (pumping) {
        return;
      }
      pumping = true;
      rafId = requestAnimationFrame(tick);
      // rAF is throttled while the window is occluded/deactivated — exactly our
      // failure window — so drive a timer fallback alongside it.
      timerId = window.setInterval(tick, 32);
      // Safety valve: never pump forever if a focus/change edge is missed.
      deadlineId = window.setTimeout(stop, 60000);
    };

    const onClick = (event) => {
      const target = event.target;
      if (target instanceof HTMLInputElement && target.type === "file") {
        start();
      }
    };

    // Only stop when visibility *returns*: opening the dialog can itself flip
    // the document to hidden, which must not tear down the pump we just armed.
    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        stop();
      }
    };

    // The dialog closing restores focus / flips visibility back and (on select)
    // fires `change`; any of those edges ends the pump. Capture phase so the
    // click arms before the input's own handler opens the dialog.
    document.addEventListener("click", onClick, true);
    window.addEventListener("focus", stop);
    document.addEventListener("visibilitychange", onVisibility);
    document.addEventListener("change", stop, true);

    return () => {
      stop();
      document.removeEventListener("click", onClick, true);
      window.removeEventListener("focus", stop);
      document.removeEventListener("visibilitychange", onVisibility);
      document.removeEventListener("change", stop, true);
    };
  }, []);
}
