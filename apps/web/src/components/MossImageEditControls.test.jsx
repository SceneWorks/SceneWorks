import React, { act } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot, click } from "../testUtils/dom.js";
import { MossImageEditControls } from "./MossImageEditControls.jsx";
import { apiFetch } from "../api.js";
vi.mock("../api.js",()=>({apiFetch:vi.fn()}));
let root,container;
afterEach(async()=>{if(root)await unmountRoot(root,container);vi.clearAllMocks();});
describe("conditional MOSS edit package installation",()=>{
  it("keeps optional NPY inputs collapsed without changing normal edit defaults", async () => {
    ({root,container}=mountRoot());
    const onAdvancedChange = vi.fn();
    await act(async () => root.render(<MossImageEditControls
      model={{id:"moss_image_base",conditionalPurposes:{moss_image_edit:{installState:"installed"}}}}
      options={{reference_max_edge:512,use_kv_cache:true}} onChange={vi.fn()}
      advanced={{output:"images"}} onAdvancedChange={onAdvancedChange} importAsset={vi.fn()}
    />));
    const details = container.querySelector("details");
    expect(details.open).toBe(false);
    expect(details.querySelector("summary").textContent).toBe("Advanced (optional)");
    expect(container.querySelector('input[accept=".npy,application/x-npy"]').closest("details")).toBe(details);
    expect(details.textContent).toContain("Normal editing only needs reference images and a prompt");
    expect(details.textContent).toContain("NPY files are optional");
    expect(details.querySelector("select").value).toBe("");
    expect(onAdvancedChange).not.toHaveBeenCalled();
  });
  it.each(["moss_image_base","moss_image_sft"])("installs the exact %s purpose with explicit source acknowledgment and actionable retries",async(id)=>{
    ({root,container}=mountRoot());
    const props={model:{id,conditionalPurposes:{moss_image_edit:{installState:"missing"}}},token:"owned-test",options:{reference_max_edge:512,use_kv_cache:true},onChange:vi.fn()};
    await act(async()=>root.render(<MossImageEditControls {...props}/>));
    const button=()=>[...container.querySelectorAll("button")].find((node)=>node.textContent==="Install MOSS edit adapter");
    expect(button().disabled).toBe(true);expect(container.textContent).toContain("Weight license: unspecified");
    await click(container.querySelector('input[type="checkbox"]'));
    apiFetch.mockRejectedValueOnce(new Error("pinned adapter hash mismatch"));await click(button());expect(container.textContent).toContain("pinned adapter hash mismatch");expect(button().disabled).toBe(false);
    apiFetch.mockResolvedValueOnce({jobs:[{id:"owned-download",status:"queued"}]});await click(button());
    expect(apiFetch).toHaveBeenLastCalledWith(`/api/v1/models/${id}/conditional-components/moss_image_edit/download`,"owned-test",{method:"POST",body:JSON.stringify({licenseAcknowledged:true})});
    expect(button().disabled).toBe(true);expect(container.textContent).toContain("queued");
    await act(async()=>root.render(<MossImageEditControls {...props} jobs={[{id:"owned-download",status:"failed",error:"corrupt adapter"}]}/>));
    expect(container.textContent).toContain("corrupt adapter");expect(button().disabled).toBe(false);
    await act(async()=>root.render(<MossImageEditControls {...props} model={{...props.model,conditionalPurposes:{moss_image_edit:{installState:"installed"}}}}/>));
    expect(container.textContent).toContain("Edit adapter: installed");expect(button()).toBeUndefined();
  });
});
