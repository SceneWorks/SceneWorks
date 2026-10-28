FROM node:22-alpine

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

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
