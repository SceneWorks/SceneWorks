import React from "react";
import { OrderedReferenceList } from "./OrderedReferenceList.jsx";
import { ImageEditSourcePickerField } from "./AssetPicker.jsx";

// Each pick is a slot, not membership in a set. The same asset can occupy several ordinals.
export function MossReferencePicker({ assets = [], assetIds = [], onChange, characters = [], importAsset, projectId, max = 4, firstOrdinal = 1 }) {
  const labelFor = (id) => assets.find((asset) => asset.id === id)?.displayName ?? `Missing reference: ${id}`;
  return <fieldset><legend>Ordered MOSS references ({firstOrdinal}–{max + firstOrdinal - 1})</legend>
    {assetIds.map((id, index) => <div key={index}>
      <ImageEditSourcePickerField
        assets={assets}
        characters={characters}
        importAsset={importAsset}
        projectId={projectId}
        label={`Image ${index + firstOrdinal}`}
        buttonLabel="Select image"
        emptyLabel={id ? labelFor(id) : "No reference image selected"}
        value={id}
        onChange={(assetId) => onChange(assetIds.map((value, slot) => slot === index ? assetId : value))}
      />
      <button type="button" onClick={() => onChange(assetIds.filter((_, slot) => slot !== index))}>Remove image {index + firstOrdinal}</button>
      <button type="button" disabled={assetIds.length >= max} onClick={() => onChange([...assetIds.slice(0, index + 1), id, ...assetIds.slice(index + 1)])}>Repeat image {index + firstOrdinal}</button>
    </div>)}
    <button type="button" disabled={assetIds.length >= max} onClick={() => onChange([...assetIds, ""])}>Add reference image</button>
    <OrderedReferenceList assetIds={assetIds} onChange={onChange} labelFor={labelFor} firstOrdinal={firstOrdinal} />
  </fieldset>;
}
