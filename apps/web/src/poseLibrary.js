import { useEffect, useState } from "react";

// The bundled OpenPose pose library (apps/web/public/poses/index.json): normalized
// COCO-18 skeletons + preview thumbnails, grouped by category. Loaded once and cached;
// the selected poses' keypoints ride advanced.poses on a character-image job, where the
// InstantID adapter renders each skeleton and generates the character in that pose.
let cachePromise = null;

export function loadPoseLibrary() {
  if (!cachePromise) {
    // Promise.resolve().then(...) so a missing/throwing fetch becomes a rejection the
    // caller's .catch handles (rather than a synchronous throw at the call site).
    cachePromise = Promise.resolve()
      .then(() => {
        if (typeof fetch !== "function") {
          throw new Error("fetch unavailable");
        }
        return fetch("/poses/index.json");
      })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`pose library unavailable (${response.status})`);
        }
        return response.json();
      })
      .then((data) => {
        const poses = Array.isArray(data?.poses) ? data.poses : [];
        const categories = Array.isArray(data?.categories)
          ? data.categories
          : [...new Set(poses.map((pose) => pose.category))];
        const byId = Object.fromEntries(poses.map((pose) => [pose.id, pose]));
        return { poses, categories, byId };
      })
      .catch((error) => {
        cachePromise = null; // allow a retry on next mount
        throw error;
      });
  }
  return cachePromise;
}

export function usePoseLibrary() {
  const [state, setState] = useState({ poses: [], categories: [], byId: {}, loading: true, error: "" });
  useEffect(() => {
    let active = true;
    loadPoseLibrary()
      .then((library) => active && setState({ ...library, loading: false, error: "" }))
      .catch((error) => active && setState({ poses: [], categories: [], byId: {}, loading: false, error: String(error.message ?? error) }));
    return () => {
      active = false;
    };
  }, []);
  return state;
}
