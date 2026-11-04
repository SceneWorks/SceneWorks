import { JOB_PATHS, submitJobRequest } from "./jobs.js";

// image route ownership. Dynamic path identifiers are encoded as single segments;
// serialized query suffixes and RequestInit (including multipart and abort) stay intact.
export const imageApi = {
  imageJobs: (token, options = {}) =>
    submitJobRequest(JOB_PATHS.image, token, options),
};
