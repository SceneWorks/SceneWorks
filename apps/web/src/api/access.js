import { apiFetch } from "../api.js";

export const accessApi = {
  inspect: (token = "", options) => apiFetch("/api/v1/access", token, options),
  verify: (token, options) => apiFetch("/api/v1/auth/verify", token, options),
  mediaTicket: (token, options) => apiFetch("/api/v1/files/ticket", token, options),
};
