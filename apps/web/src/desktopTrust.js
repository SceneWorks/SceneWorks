const invoke = (command, args) => window.__TAURI__.core.invoke(command, args);

let trustNoncePromise = null;

export async function getDesktopTrustNonce() {
  if (!trustNoncePromise) {
    trustNoncePromise = invoke("get_desktop_trust_nonce");
  }
  return trustNoncePromise;
}

export async function trustedDesktopInvoke(command, args = {}) {
  const trustNonce = await getDesktopTrustNonce();
  return invoke(command, { ...args, trustNonce });
}
