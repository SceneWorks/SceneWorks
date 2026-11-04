FROM node:22-alpine@sha256:fe4f6d74e20370acb04ad889077d14d0f64a9c8ea6a42bc60cddf1e844699914

WORKDIR /app
RUN chown node:node /app
ENV HOME=/home/node
USER node

COPY --chown=node:node apps/web/package*.json ./
RUN npm ci

COPY --chown=node:node apps/web ./
# /app is the bind-mounted apps/web directory, so its repository-relative build inputs live
# two levels above it. They remain outside Vite's development filesystem allowlist.
COPY --chown=node:node scripts /scripts
COPY --chown=node:node config /config
COPY --chown=node:node packages/schemas /packages/schemas
COPY --chown=node:node apps/desktop/licenses /desktop/licenses
COPY --chown=node:node docs/film-script-writing.md docs/film-editor.md /docs/

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
