import React from "react";

export function FileDropzone({
  accept,
  active = false,
  busy = false,
  busyLabel = "Importing",
  className = "",
  hint,
  multiple = false,
  onActiveChange,
  onDrop,
  onFiles,
}) {
  return (
    <div
      className={`dataset-add-dropzone${active ? " active" : ""}${className ? ` ${className}` : ""}`}
      onDragLeave={() => onActiveChange?.(false)}
      onDragOver={(event) => {
        event.preventDefault();
        onActiveChange?.(true);
      }}
      onDrop={onDrop}
    >
      <p>{hint}</p>
      <label className="file-upload-button">
        <input
          accept={accept}
          disabled={busy}
          multiple={multiple}
          onChange={(event) => {
            onFiles?.(event.target.files);
            event.target.value = "";
          }}
          type="file"
        />
        {busy ? busyLabel : "Browse files"}
      </label>
    </div>
  );
}
