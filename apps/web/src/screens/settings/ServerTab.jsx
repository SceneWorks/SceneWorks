import React from "react";
import { Icon } from "../../components/Icons.jsx";

export function ServerTab({
  applyRemoteAccess,
  clearRemotePassword,
  copyRemoteUrl,
  remote,
  remotePassword,
  remotePasswordCandidate,
  remotePort,
  saveRemotePassword,
  setRemotePassword,
  setRemotePort,
}) {
  return (
<div className="settings-tab-body">
            <div className="settings-group-title">Remote access (LAN)</div>
            <div className="settings-row">
              <div>
                <div className="settings-row-title">Let other devices on this network in</div>
                <div className="settings-row-sub">
                  Generation still runs on this computer. Takes effect after a restart.
                </div>
              </div>
              <button
                aria-checked={Boolean(remote.enabled)}
                aria-label="Remote access"
                className={remote.enabled ? "settings-toggle on" : "settings-toggle"}
                disabled={!remote.passwordSet && !remote.enabled}
                onClick={() => applyRemoteAccess(!remote.enabled)}
                role="switch"
                type="button"
              >
                <span />
              </button>
            </div>

            {remote.url ? (
              <>
                <div className="settings-field-row">
                  <span className="settings-url">{remote.url}</span>
                  <button
                    className="settings-btn settings-btn--compact"
                    type="button"
                    onClick={copyRemoteUrl}
                  >
                    <Icon.Duplicate size={14} />
                    Copy
                  </button>
                </div>
                {remote.lanCandidates && remote.lanCandidates.length > 1 ? (
                  <div className="settings-note">
                    Also reachable at {remote.lanCandidates.slice(1).join(", ")}
                  </div>
                ) : null}
              </>
            ) : (
              <p className="settings-note">
                Couldn’t determine this computer’s LAN address — check your network connection.
              </p>
            )}

            <div className="settings-inset">
              <div className="settings-inset-title">Access</div>
              <div className="settings-field-row">
                <input
                  className="settings-grow"
                  type="password"
                  placeholder={remote.passwordSet ? "Change password" : "Set a password"}
                  value={remotePassword}
                  onChange={(event) => setRemotePassword(event.target.value)}
                  aria-label="Remote access password"
                />
                <button
                  className="settings-btn"
                  type="button"
                  onClick={saveRemotePassword}
                  disabled={!remotePasswordCandidate(remotePassword, remote)?.meetsMinimum}
                >
                  Save
                </button>
                {remote.passwordSet ? (
                  <button
                    className="settings-btn settings-btn--quiet"
                    type="button"
                    onClick={clearRemotePassword}
                  >
                    Clear
                  </button>
                ) : null}
              </div>
              <div className="settings-field-row">
                <span className="settings-note settings-grow">
                  Use at least {remote.minimumPasswordLength} characters.{" "}
                  {remote.passwordSet
                    ? "A password is set — remote browsers must enter it."
                    : "Set a password before enabling remote access."}
                </span>
                <label className="settings-note" htmlFor="remote-port">
                  Port
                </label>
                <input
                  className="settings-port"
                  id="remote-port"
                  type="number"
                  min="1024"
                  max="65535"
                  value={remotePort}
                  onChange={(event) => setRemotePort(event.target.value)}
                  aria-label="Remote access port"
                />
              </div>
            </div>

            {/* Security note + platform firewall guidance (story 11), consolidated into one notice. */}
            <div className="settings-notice">
              <Icon.Warning size={16} />
              <span>
                Trusted local networks only. Traffic is plain HTTP, so the password and your content
                travel unencrypted — never port-forward this or expose it to the internet.{" "}
                {remote.platform === "windows"
                  ? "The first time SceneWorks binds the port, allow it on Private networks in the Windows Security prompt."
                  : "The first time SceneWorks binds the port, macOS may ask to allow incoming connections — click Allow."}
              </span>
            </div>
          </div>
  );
}
