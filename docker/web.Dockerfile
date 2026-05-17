FROM node:22-alpine

WORKDIR /app

COPY apps/web/package*.json ./
RUN npm ci

COPY apps/web ./

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
