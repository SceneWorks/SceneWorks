import React from "react";

export const LinkedLibraryStatus = React.memo(function LinkedLibraryStatus({ model, linkedFix, linkedStatus, deletingItem, relinkLinkedModel, rescanLinkedModel }) {
  return (
          <div
            aria-label={`${model.name ?? model.id} ${linkedFix.headline}`}
            className="model-card-linked-fix"
            role="group"
          >
            <p className="checkpoint-state-headline">{linkedFix.headline}</p>
            <p>{linkedFix.summary}</p>
            {linkedFix.detail ? <p className="checkpoint-import-detail">{linkedFix.detail}</p> : null}
            <button
              disabled={deletingItem === `relink:${linkedStatus.checkpointId}` || deletingItem === `rescan:${linkedStatus.checkpointId}`}
              onClick={() => (linkedFix.action === "relink" ? relinkLinkedModel(linkedStatus) : rescanLinkedModel(linkedStatus))}
              type="button"
            >
              {linkedFix.label}
            </button>
          </div>
  );
});
