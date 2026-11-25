import { describe, expect, it } from "vitest";
import { isMossImage, modelPolicyId } from "./mossImageIdentity.js";
import { buildImageJobRequest } from "./imageJobRequest.js";

describe("imported model policy identity", () => {
  const selectedModel = { id: "serenity", policyModel: "moss_image_base" };
  it("uses the bound base for controls while preserving the imported id", () => {
    expect(modelPolicyId(selectedModel)).toBe("moss_image_base");
    expect(isMossImage(selectedModel)).toBe(true);
    expect(isMossImage({ id: "custom", family: "moss-image" })).toBe(false);
    const request = buildImageJobRequest({
      selectedModel, model: "serenity", mode: "text_to_image", promptToSend: "a pine forest",
      resolution: "512x512", width: 512, height: 512, count: 1, seed: "7",
      posePayload: [], referenceAssetIds: [], loras: [], sampler: "default", scheduler: "default",
      mossImage: { acceleration: "off", max_sequence_length: 2048 },
    });
    expect(request.model).toBe("serenity");
    expect(request.advanced.mossImage).toEqual({ acceleration: "off", max_sequence_length: 2048 });
  });
});
