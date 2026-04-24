#!/bin/sh
set -e
mkdir -p /app/tmp/audio /app/tmp/uploads
chown -R nextjs:nodejs /app/tmp
exec su-exec nextjs:nodejs node server.js
