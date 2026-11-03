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
      onDrop={(event) => {
        event.preventDefault();
        onActiveChange?.(false);
        if (!busy && event.dataTransfer?.files?.length) onFiles?.(event.dataTransfer.files);
      }}
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
