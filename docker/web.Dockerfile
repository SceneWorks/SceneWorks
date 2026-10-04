FROM node:22-alpine

WORKDIR /app
RUN chown node:node /app
ENV HOME=/home/node
USER node

COPY --chown=node:node apps/web/package*.json ./
RUN npm ci

COPY --chown=node:node apps/web ./

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
